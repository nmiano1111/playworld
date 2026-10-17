package scenario

import (
	"strings"
	"testing"

	"github.com/nmiano/playworld/internal/sim"
)

var ccCast = []sim.AgentID{"mira", "dov", "ada", "rook"}
var ccNames = map[sim.AgentID]string{"mira": "Mira", "dov": "Dov", "ada": "Ada", "rook": "Rook"}

// buildCommunity builds and sets up a community environment with the full
// four-character cast. days/complication_day default small so tests don't
// have to step through seven days to reach the vote.
func buildCommunity(t *testing.T, params map[string]any) sim.Environment {
	t.Helper()
	env, err := Build("community", Spec{Participants: ccCast, Names: ccNames, Rounds: 8, Seed: 7, Params: params})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := env.Setup(ccCast); err != nil {
		t.Fatal(err)
	}
	return env
}

// fixedParams gives every numeric unknown a concrete, test-friendly value so
// assertions don't have to account for seed-driven ranges.
func fixedParams(overrides map[string]any) map[string]any {
	p := map[string]any{
		"days": 4, "complication_day": 0, "inspect_cost": 5,
		"structural_cost": 35, "complication_surcharge": 15, "rook_bonus_credits": 10,
	}
	for k, v := range overrides {
		p[k] = v
	}
	return p
}

// --- construction & private information -------------------------------------

func TestCommunityReproducibleForASeed(t *testing.T) {
	build := func() sim.Environment {
		env, err := Build("community", Spec{Participants: ccCast, Names: ccNames, Rounds: 8, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := env.Setup(ccCast); err != nil {
			t.Fatal(err)
		}
		return env
	}
	a, b := build(), build()
	for _, id := range ccCast {
		pa, pb := a.PrivateState(id), b.PrivateState(id)
		for k, v := range pa {
			if pb[k] != v {
				t.Errorf("%s private state %q differs across identical seeds: %v vs %v", id, k, v, pb[k])
			}
		}
	}
	oa, ob := a.Outcome(), b.Outcome()
	if oa.Metrics["structural_cost_initial"] != ob.Metrics["structural_cost_initial"] {
		t.Error("the true structural cost was not reproducible for a fixed seed")
	}
}

func TestCommunityDifferentSeedsProduceDifferentGroundTruth(t *testing.T) {
	// This is the fix for the lesson learned with the murder scenario: a
	// longitudinal rerun must not silently replay identical hidden facts.
	build := func(seed int64) sim.Environment {
		env, err := Build("community", Spec{Participants: ccCast, Names: ccNames, Rounds: 8, Seed: seed})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := env.Setup(ccCast); err != nil {
			t.Fatal(err)
		}
		return env
	}
	seen := map[int]bool{}
	for _, seed := range []int64{1, 2, 3, 4, 5, 6, 7, 8} {
		env := build(seed)
		cost := env.Outcome().Metrics["structural_cost_initial"].(int)
		seen[cost] = true
	}
	if len(seen) < 2 {
		t.Error("eight different seeds produced the same structural cost every time; ground truth is not actually varying with the seed")
	}
}

func TestCommunityPrivateInformationIsIsolated(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil))

	dov := env.PrivateState("dov")
	if dov["known_structural_cost"] != 35 {
		t.Errorf("dov's known structural cost = %v, want 35", dov["known_structural_cost"])
	}
	mira := env.PrivateState("mira")
	if _, ok := mira["market_insight_credits_per_season"]; !ok {
		t.Error("mira should have her own market insight")
	}
	ada := env.PrivateState("ada")
	if _, ok := ada["households_wanting_free_space"]; !ok {
		t.Error("ada should have her own demand insight")
	}
	rook := env.PrivateState("rook")
	if _, ok := rook["outside_funding_secured"]; !ok {
		t.Error("rook should know about his own outside funding")
	}

	// None of these facts leak into another agent's private state.
	for key := range dov {
		if key == "known_structural_cost" {
			if _, leaked := mira[key]; leaked {
				t.Error("mira's private state leaked dov's structural knowledge")
			}
		}
	}
	for _, other := range []map[string]any{mira, ada, rook} {
		if _, has := other["known_structural_cost"]; has {
			t.Errorf("a non-engineer's private state exposed the real structural cost: %v", other)
		}
	}
	if _, has := dov["outside_funding_secured"]; has {
		t.Error("dov's private state leaked rook's outside funding")
	}

	// And it never appears in the shared, environment-wide public state.
	pub := env.PublicState()
	if _, has := pub["known_structural_cost"]; has {
		t.Error("the true structural cost leaked into public state before any complication")
	}
}

func TestCommunityBriefNeverMentionsAnotherAgentsSecret(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil))
	dovBrief := env.Brief("dov")
	if !strings.Contains(dovBrief, "35 credits") {
		t.Errorf("dov's brief should state his own known cost:\n%s", dovBrief)
	}
	miraBrief := env.Brief("mira")
	if strings.Contains(miraBrief, "real repair cost: 35") {
		t.Errorf("mira's brief leaked dov's secret:\n%s", miraBrief)
	}
}

func TestCommunityStartingResourcesCreateRealTradeoffs(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil))
	total := 0
	for _, id := range ccCast {
		total += env.PrivateState(id)["your_credits"].(int)
	}
	// Nobody alone should be able to cover even a modest structural repair.
	for _, id := range ccCast {
		mine := env.PrivateState(id)["your_credits"].(int)
		if mine >= 35 {
			t.Errorf("%s alone has %d credits, enough to cover the 35-credit repair without anyone else", id, mine)
		}
	}
	if total < 35 {
		t.Errorf("the whole group's credits (%d) can't even cover the repair cost (35) together", total)
	}
}

// --- action validation --------------------------------------------------------

func TestCommunityPhaseGatingBeforeAndOnTheDeadline(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 3}))
	env.BeginRound(1)

	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "let's talk"}); err != nil {
		t.Errorf("talking should be legal before the deadline: %v", err)
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}}); err == nil {
		t.Error("voting before the deadline should be refused")
	}

	env.EndRound(1)
	env.BeginRound(2)
	env.EndRound(2)
	env.BeginRound(3) // the deadline (days=3)

	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "wait"}); err == nil {
		t.Error("talking on the deadline should be refused")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "x", "credits": 1, "labor": 1}}); err == nil {
		t.Error("project work on the deadline should be refused")
	}

	var kinds []string
	for _, spec := range env.Actions("mira") {
		kinds = append(kinds, string(spec.Kind)+":"+spec.Verb)
	}
	if contains2(kinds, "say:") || contains2(kinds, "do:propose") {
		t.Errorf("the deadline should only offer vote/pass: %v", kinds)
	}
	if !contains2(kinds, "do:vote") || !contains2(kinds, "pass:") {
		t.Errorf("deadline actions = %v", kinds)
	}
}

func TestCommunityProposeValidation(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil))
	env.BeginRound(1)

	cases := []struct {
		name    string
		args    map[string]any
		wantErr bool
	}{
		{"valid", map[string]any{"text": "a market", "credits": 40, "labor": 10}, false},
		{"missing text", map[string]any{"credits": 40, "labor": 10}, true},
		{"missing credits", map[string]any{"text": "x", "labor": 10}, true},
		{"missing labor", map[string]any{"text": "x", "credits": 40}, true},
		{"negative credits", map[string]any{"text": "x", "credits": -5, "labor": 10}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: c.args})
			if (err != nil) != c.wantErr {
				t.Errorf("Validate = %v, wantErr %v", err, c.wantErr)
			}
		})
	}

	if _, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a market", "credits": 40, "labor": 10}}); err != nil {
		t.Fatal(err)
	}
	// Only the original proposer may revise it.
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"proposal": "p1", "text": "y", "credits": 1, "labor": 1}}); err == nil {
		t.Error("a non-proposer revising someone else's proposal should be refused")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"proposal": "p1", "text": "a bigger market", "credits": 50, "labor": 15}}); err != nil {
		t.Errorf("the original proposer revising their own proposal should be legal: %v", err)
	}
}

func TestCommunitySupportIsNonBindingAndUnlimited(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil))
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 40, "labor": 10}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "free space", "credits": 30, "labor": 20}})

	if _, err := env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "support", Args: map[string]any{"proposal": "p1"}}); err != nil {
		t.Fatal(err)
	}
	// Supporting a second proposal too is explicitly allowed.
	if _, err := env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "support", Args: map[string]any{"proposal": "p2"}}); err != nil {
		t.Fatal(err)
	}
	// Support alone transfers nothing and commits nothing.
	if env.PrivateState("dov")["your_credits"] != env.(*community).startCredits["dov"] {
		t.Error("supporting a proposal should not cost any resources")
	}
}

func TestCommunityContributionValidationAndConservation(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 40, "labor": 10}})

	startCredits := env.credits["dov"]

	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": startCredits + 1}}); err == nil {
		t.Error("contributing more credits than you have should be refused")
	}
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 0, "labor": 0}}); err == nil {
		t.Error("contributing nothing at all should be refused")
	}

	events, err := env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 10, "labor": 2}})
	if err != nil {
		t.Fatal(err)
	}
	if !events[0].Visibility.Public {
		t.Error("a contribution is a real transfer into the shared project and should be publicly visible")
	}
	if env.credits["dov"] != startCredits-10 {
		t.Errorf("dov's remaining credits = %d, want %d", env.credits["dov"], startCredits-10)
	}
	p := env.findProposal("p1")
	if p.totalCredits() != 10 || p.totalLabor() != 2 {
		t.Errorf("proposal raised = %d credits / %d labor, want 10/2", p.totalCredits(), p.totalLabor())
	}
	// Conservation: what left dov's balance is exactly what the proposal gained.
	if (startCredits - env.credits["dov"]) != p.ContribCredits["dov"] {
		t.Error("credits vanished or were created somewhere between the agent and the proposal")
	}

	// Contributing beyond the now-reduced balance is refused.
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": startCredits}}); err == nil {
		t.Error("contributing more than the remaining balance should be refused")
	}
}

func TestCommunityInspectCostsCreditsAndRevealsOnlyToTheInspector(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"inspect_cost": 5})).(*community)
	env.BeginRound(1)

	before := env.credits["ada"]
	events, err := env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "inspect"})
	if err != nil {
		t.Fatal(err)
	}
	if env.credits["ada"] != before-5 {
		t.Errorf("inspection should cost exactly 5 credits, balance = %d, want %d", env.credits["ada"], before-5)
	}
	if len(events) != 2 {
		t.Fatalf("expected an announcement and a private result, got %d events", len(events))
	}
	if !events[0].Visibility.Public {
		t.Error("that an inspection happened should be public")
	}
	if strings.Contains(events[0].Text, "35") {
		t.Error("the public announcement must not contain the actual result")
	}
	if events[1].Visibility.Public || !events[1].Visibility.CanObserve("ada") {
		t.Errorf("the inspection result must be visible only to the inspector, got %s", events[1].Visibility.Describe())
	}
	if !strings.Contains(events[1].Text, "35") {
		t.Errorf("the private result should state the true cost: %q", events[1].Text)
	}
	if env.PrivateState("ada")["last_inspection_result"] != 35 {
		t.Errorf("ada's private state should record her inspection result")
	}
	if env.PrivateState("mira")["last_inspection_result"] != nil {
		t.Error("an inspection ada paid for should not appear in mira's private state")
	}

	if err := env.Validate("ada", sim.Action{}); err == nil {
		// not a real assertion, just keeping ada's remaining balance referenced
		_ = env.credits["ada"]
	}
}

// --- commitments --------------------------------------------------------------

func TestCommunityCommitmentDoesNotTransferResources(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 40, "labor": 10}})

	before := env.credits["rook"]
	events, err := env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 20}})
	if err != nil {
		t.Fatal(err)
	}
	if env.credits["rook"] != before {
		t.Errorf("a commitment must not move resources: balance = %d, want unchanged %d", env.credits["rook"], before)
	}
	if !events[0].Visibility.Public {
		t.Error("a commitment is a public promise")
	}
	if len(env.commitments) != 1 || env.commitments[0].Credits != 20 {
		t.Errorf("commitment was not recorded: %+v", env.commitments)
	}
}

func TestCommunityCommitmentCeilingIsTheStartingAllocation(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 40, "labor": 10}})

	ceiling := env.startCredits["ada"]
	if err := env.Validate("ada", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": ceiling + 1}}); err == nil {
		t.Error("committing more credits than you will ever have should be refused")
	}
	if err := env.Validate("ada", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": ceiling}}); err != nil {
		t.Errorf("committing exactly your full allocation should be legal: %v", err)
	}
}

func TestCommunityBrokenCommitmentIsVisibleInTheOutcome(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 10, "labor": 0}})
	// Rook promises 20 credits...
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 20}})
	// ...but only ever gives 5.
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 5}})
	env.EndRound(1)

	env.BeginRound(2) // the deadline
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	for _, id := range []sim.AgentID{"dov", "ada", "rook"} {
		env.Apply(id, sim.Action{Kind: sim.ActionPass})
	}
	env.EndRound(2)

	out := env.Outcome()
	if rate, ok := out.Metrics["commitment_fulfillment_rate"].(float64); !ok || rate != 0 {
		t.Errorf("commitment_fulfillment_rate = %v, want 0 (promised 20, gave 5)", out.Metrics["commitment_fulfillment_rate"])
	}
	detail, ok := out.Metrics["commitments_detail"].([]map[string]any)
	if !ok || len(detail) != 1 || detail[0]["fulfilled"] != false {
		t.Errorf("commitments_detail = %+v", out.Metrics["commitments_detail"])
	}
}

func TestCommunityFulfilledCommitmentIsVisibleInTheOutcome(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 10, "labor": 0}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 15}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 15}})
	env.EndRound(1)
	env.BeginRound(2)
	for _, id := range ccCast {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	}
	env.EndRound(2)

	out := env.Outcome()
	if rate, _ := out.Metrics["commitment_fulfillment_rate"].(float64); rate != 1 {
		t.Errorf("commitment_fulfillment_rate = %v, want 1", out.Metrics["commitment_fulfillment_rate"])
	}
}

// --- feasibility, the complication, and resolution ---------------------------

func TestCommunityWinningProposalMustBeSafe(t *testing.T) {
	// Structural cost is 30. A proposal that only budgets 20 wins the vote and
	// is fully funded at that level, but must still fail: its own declared
	// budget doesn't cover the real repair cost.
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "cheap market", "credits": 20, "labor": 0}})
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 20}})
	env.EndRound(1)
	env.BeginRound(2)
	for _, id := range ccCast {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	}
	env.EndRound(2)

	out := env.Outcome()
	if out.Metrics["feasible"] != false {
		t.Error("an underbudgeted proposal must not succeed just because it won the vote and was funded at its own (too low) level")
	}
	if !strings.Contains(out.Summary, "did not proceed") {
		t.Errorf("summary = %q", out.Summary)
	}
	for _, id := range ccCast {
		if out.Scores[id] != 0 {
			t.Errorf("%s scored %v on a failed proposal, want 0", id, out.Scores[id])
		}
	}
}

func TestCommunityWinningProposalMustBeFunded(t *testing.T) {
	// Budgeted safely (40 >= 30 structural cost) but never actually raised.
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 40, "labor": 0}})
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 10}})
	env.EndRound(1)
	env.BeginRound(2)
	for _, id := range ccCast {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	}
	env.EndRound(2)

	out := env.Outcome()
	if out.Metrics["feasible"] != false {
		t.Error("an underfunded proposal must not succeed merely for winning the vote")
	}
	if !strings.Contains(out.Summary, "did not proceed") {
		t.Errorf("summary = %q", out.Summary)
	}
}

func TestCommunityProposalThatIsFundedAndSafeSucceeds(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a real market", "credits": 40, "labor": 10}})
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 20, "labor": 5}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 15, "labor": 5}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 5}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "support", Args: map[string]any{"proposal": "p1"}})
	env.EndRound(1)

	env.BeginRound(2)
	for _, id := range ccCast {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	}
	events := env.EndRound(2)

	if len(events) != 1 || !events[0].Visibility.Public {
		t.Fatalf("the resolution must be announced publicly: %+v", events)
	}
	if !strings.Contains(events[0].Text, "proceeds") {
		t.Errorf("resolution text = %q", events[0].Text)
	}
	done, reason := env.Done()
	if !done || !strings.Contains(reason, "selected") {
		t.Fatalf("Done = %v, %q", done, reason)
	}

	out := env.Outcome()
	if out.Metrics["feasible"] != true || out.Metrics["winning_proposal"] != "p1" {
		t.Errorf("metrics = %+v", out.Metrics)
	}
	if out.Scores["mira"] != 100 {
		t.Errorf("the successful proposer should score 100, got %v", out.Scores["mira"])
	}
	if out.Scores["dov"] != 50 || out.Scores["rook"] != 50 {
		t.Errorf("real contributors should score 50: dov=%v rook=%v", out.Scores["dov"], out.Scores["rook"])
	}
	if out.Scores["ada"] != 25 {
		t.Errorf("a supporter who didn't contribute should score 25, got %v", out.Scores["ada"])
	}
}

func TestCommunityTiedVoteIsInconclusive(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "b", "credits": 10, "labor": 0}})
	env.EndRound(1)

	env.BeginRound(2)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p2"}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p2"}})
	env.EndRound(2)

	done, reason := env.Done()
	if !done || !strings.Contains(reason, "no majority") {
		t.Fatalf("Done = %v, %q", done, reason)
	}
	out := env.Outcome()
	if out.Metrics["winning_proposal"] != "" || out.Metrics["feasible"] != false {
		t.Errorf("metrics = %+v", out.Metrics)
	}
	for _, id := range ccCast {
		if out.Scores[id] != 0 {
			t.Errorf("a tie should score everyone 0, %s got %v", id, out.Scores[id])
		}
	}
}

func TestCommunityVoteIsSealedUntilTheReveal(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.EndRound(1)
	env.BeginRound(2)

	events, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	if err != nil {
		t.Fatal(err)
	}
	if events[0].Visibility.Public || events[0].Visibility.CanObserve("dov") {
		t.Fatalf("a vote must be sealed, got %s", events[0].Visibility.Describe())
	}
	if !events[0].Visibility.CanObserve("mira") {
		t.Error("the voter should see its own sealed vote")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}}); err == nil {
		t.Error("voting twice should be refused")
	}
}

func TestCommunityAbstainingOnTheDeadlineIsSealed(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)
	events, err := env.Apply("dov", sim.Action{Kind: sim.ActionPass})
	if err != nil {
		t.Fatal(err)
	}
	if events[0].Visibility.Public {
		t.Error("abstaining is still a sealed ballot, not a public pass")
	}
}

func TestCommunityComplicationChangesTheStructuralCostPublicly(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{
		"days": 4, "complication_day": 2, "structural_cost": 20, "complication_surcharge": 15,
	})).(*community)

	env.BeginRound(1)
	if _, has := env.PublicState()["public_repair_estimate"]; has {
		t.Error("the repair cost must not be public knowledge before the complication")
	}
	if env.structuralCost != 20 {
		t.Errorf("structural cost before the complication = %d, want 20", env.structuralCost)
	}
	env.EndRound(1)

	events := env.BeginRound(2)
	if len(events) == 0 || !strings.Contains(events[0].Text, "35") {
		t.Fatalf("the complication should publicly announce the new cost: %+v", events)
	}
	if !events[0].Visibility.Public {
		t.Error("the complication is a shared event, not a private one")
	}
	if env.structuralCost != 35 {
		t.Errorf("structural cost after the complication = %d, want 35 (20+15)", env.structuralCost)
	}
	if got := env.PublicState()["public_repair_estimate"]; got != 35 {
		t.Errorf("public_repair_estimate = %v, want 35", got)
	}

	// A proposal that was safe before the complication can become unsafe
	// afterward — the environment does not retroactively fix anyone's budget.
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "market", "credits": 25, "labor": 0}})
	p := env.findProposal("p1")
	if p.Credits >= env.structuralCost {
		t.Fatal("test setup is wrong: the proposal should be unsafe relative to the new cost")
	}
}

func TestCommunityAgentsCanAdaptAfterTheComplication(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{
		"days": 3, "complication_day": 2, "structural_cost": 20, "complication_surcharge": 15,
	})).(*community)

	env.BeginRound(1)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "cheap market", "credits": 20, "labor": 0}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "support", Args: map[string]any{"proposal": "p1"}}) // pre-complication support
	env.EndRound(1)

	env.BeginRound(2) // complication lands; new cost is 35
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a bigger plan", "credits": 40, "labor": 0}})
	env.EndRound(2)

	env.BeginRound(3) // deadline
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p2"}})
	// Ada abandons the now-unsafe plan she supported and votes for the new one.
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p2"}})
	env.Apply("rook", sim.Action{Kind: sim.ActionPass})
	env.EndRound(3)

	out := env.Outcome()
	if n, _ := out.Metrics["agents_adapted_after_complication"].(int); n != 1 {
		t.Errorf("agents_adapted_after_complication = %v, want 1 (ada)", out.Metrics["agents_adapted_after_complication"])
	}
}

// --- termination ---------------------------------------------------------

func TestCommunityTerminatesWithNoProposalsAtAll(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)
	for _, id := range ccCast {
		env.Apply(id, sim.Action{Kind: sim.ActionPass})
	}
	env.EndRound(2)

	done, _ := env.Done()
	if !done {
		t.Fatal("the scenario should be done once the deadline round has resolved")
	}
	out := env.Outcome()
	if !strings.Contains(out.Summary, "remains unused") {
		t.Errorf("summary = %q", out.Summary)
	}
}

func TestCommunityRunningPastTheDeadlineWithoutResolutionTerminates(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2})).(*community)
	env.BeginRound(5) // well past the deadline, defensive path
	done, reason := env.Done()
	if !done || !strings.Contains(reason, "no verdict") {
		t.Errorf("Done = %v, %q", done, reason)
	}
	if out := env.Outcome(); out.Summary == "" {
		t.Error("an unresolved scenario should still produce a usable outcome")
	}
}

// --- parameter validation -----------------------------------------------------

func TestCommunityRejectsBadParams(t *testing.T) {
	for name, params := range map[string]map[string]any{
		"non-numeric structural cost": {"structural_cost": "lots"},
		"bad credits shape":           {"credits": "mira=25"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Build("community", Spec{Participants: ccCast, Rounds: 4, Params: params}); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

func TestCommunityComplicationDayAtOrAfterDeadlineIsDisabled(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 3, "complication_day": 3, "structural_cost": 20})).(*community)
	for r := 1; r <= 3; r++ {
		env.BeginRound(r)
		env.EndRound(r)
	}
	if env.complicationApplied {
		t.Error("a complication day that lands on or after the deadline should be disabled rather than silently skipped at the wrong time")
	}
	if env.structuralCost != 20 {
		t.Errorf("structural cost = %d, want unchanged at 20", env.structuralCost)
	}
}

// --- the lock-in day ----------------------------------------------------------

func TestCommunityLockInDayClosesTalkingButKeepsProjectActionsOpen(t *testing.T) {
	// days=4 gives a lock-in day at day 3 (days-1); days<=3 has none.
	env := buildCommunity(t, fixedParams(map[string]any{"days": 4})).(*community)
	if got := env.lockInDayNumber(); got != 3 {
		t.Fatalf("lockInDayNumber() = %d, want 3", got)
	}

	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.EndRound(2)

	events := env.BeginRound(3) // the lock-in day
	if len(events) != 1 || !strings.Contains(events[0].Text, "talking is closed") {
		t.Fatalf("the lock-in day should be announced: %+v", events)
	}
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionSay, Content: "one more thing"}); err == nil {
		t.Error("talking should be closed on the lock-in day")
	}
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"ada"}}); err == nil {
		t.Error("whispering should be closed on the lock-in day")
	}
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "support", Args: map[string]any{"proposal": "p1"}}); err != nil {
		t.Errorf("project actions should stay open on the lock-in day: %v", err)
	}

	var kinds []string
	for _, spec := range env.Actions("dov") {
		kinds = append(kinds, string(spec.Kind)+":"+spec.Verb)
	}
	if contains2(kinds, "say:") || contains2(kinds, "whisper:") {
		t.Errorf("the lock-in day should not offer say/whisper: %v", kinds)
	}
	if !contains2(kinds, "do:support") || !contains2(kinds, "do:propose") {
		t.Errorf("lock-in day actions = %v", kinds)
	}
	if !strings.Contains(env.Brief("dov"), "Talking is closed today") {
		t.Errorf("the brief should explain the lock-in day:\n%s", env.Brief("dov"))
	}
}

func TestCommunityShortGamesHaveNoLockInDay(t *testing.T) {
	for _, days := range []int{2, 3} {
		env := buildCommunity(t, fixedParams(map[string]any{"days": days})).(*community)
		if got := env.lockInDayNumber(); got != 0 {
			t.Errorf("days=%d: lockInDayNumber() = %d, want 0 (no room for a dedicated lock-in day)", days, got)
		}
		// Every day up to the deadline should still allow talking.
		for r := 1; r < days; r++ {
			env.BeginRound(r)
			if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "hi"}); err != nil {
				t.Errorf("days=%d round=%d: talking should be legal, got %v", days, r, err)
			}
			env.EndRound(r)
		}
	}
}

// --- opportunities per day ----------------------------------------------------

func TestCommunityOpportunitiesPerDayMapping(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2, "opportunities_per_day": 3})).(*community)
	cases := []struct {
		round, wantDay, wantOpportunity int
	}{
		{1, 1, 1}, {2, 1, 2}, {3, 1, 3},
		{4, 2, 1}, {5, 2, 2}, {6, 2, 3},
	}
	for _, c := range cases {
		if got := env.dayOf(c.round); got != c.wantDay {
			t.Errorf("round %d: dayOf = %d, want %d", c.round, got, c.wantDay)
		}
		if got := env.opportunityOf(c.round); got != c.wantOpportunity {
			t.Errorf("round %d: opportunityOf = %d, want %d", c.round, got, c.wantOpportunity)
		}
	}
	if env.totalRounds() != 6 {
		t.Errorf("totalRounds() = %d, want 6", env.totalRounds())
	}
}

func TestCommunityDefaultOpportunityMappingIsIdentity(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	for r := 1; r <= env.days; r++ {
		if env.dayOf(r) != r {
			t.Errorf("dayOf(%d) = %d, want %d (opportunities_per_day=1 must be the identity)", r, env.dayOf(r), r)
		}
		if env.opportunityOf(r) != 1 {
			t.Errorf("opportunityOf(%d) = %d, want 1", r, env.opportunityOf(r))
		}
	}
}

func TestCommunityVoteOnlyOnTheLastOpportunityOfTheLastDay(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2, "opportunities_per_day": 2})).(*community)
	// Rounds 1-3 are day 1 (opps 1-2) and day 2's first opportunity: none of
	// these are the vote.
	for r := 1; r <= 3; r++ {
		env.round = r
		if env.isVoteDay() {
			t.Errorf("round %d should not be the vote round", r)
		}
	}
	env.round = 4 // day 2, opportunity 2 of 2: the vote
	if !env.isVoteDay() {
		t.Error("round 4 (last opportunity of the last day) should be the vote round")
	}
}

func TestCommunityComplicationFiresOnceRegardlessOfOpportunitiesPerDay(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{
		"days": 3, "opportunities_per_day": 2, "complication_day": 2, "structural_cost": 20, "complication_surcharge": 15,
	})).(*community)
	// Day 1 = rounds 1-2; day 2 = rounds 3-4; day 3 (vote) = rounds 5-6.
	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)
	env.EndRound(2)
	if env.complicationApplied {
		t.Fatal("the complication should not have fired before day 2")
	}

	events := env.BeginRound(3) // day 2, opportunity 1: complication fires here
	if len(events) != 2 {
		t.Fatalf("expected a complication event plus the day narration, got %d: %+v", len(events), events)
	}
	if env.structuralCost != 35 {
		t.Errorf("structural cost after the complication = %d, want 35", env.structuralCost)
	}
	env.EndRound(3)

	events = env.BeginRound(4) // day 2, opportunity 2: must not fire again
	for _, e := range events {
		if strings.Contains(e.Text, "structural damage") {
			t.Errorf("the complication fired a second time within the same day: %+v", events)
		}
	}
	if env.structuralCost != 35 {
		t.Errorf("structural cost changed a second time: %d", env.structuralCost)
	}
}

func TestCommunityNonFinalOpportunitiesOfTheVoteDayAreActionOnly(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 2, "opportunities_per_day": 2})).(*community)
	env.round = 3 // day 2 (the vote day), opportunity 1 of 2 — not the vote itself
	if env.isVoteDay() {
		t.Fatal("test setup: round 3 should not be the vote round")
	}
	if !env.isLockInDay() {
		t.Error("a non-final opportunity on the vote day should be treated as action-only, same as a lock-in day")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "hi"}); err == nil {
		t.Error("talking should be closed on a non-final opportunity of the vote day")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "inspect"}); err != nil {
		t.Errorf("project actions should stay open: %v", err)
	}
}

// --- lock-in toggle ------------------------------------------------------------

func TestCommunityLockInCanBeDisabled(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 4, "lock_in": "false"})).(*community)
	if got := env.lockInDayNumber(); got != 0 {
		t.Errorf("lockInDayNumber() = %d, want 0 with lock_in disabled", got)
	}
	env.round = 3 // would have been the lock-in day otherwise
	if env.isLockInDay() {
		t.Error("lock-in should not trigger when disabled")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "still talking"}); err != nil {
		t.Errorf("talking should stay legal with lock_in disabled: %v", err)
	}
}

func TestCommunityLockInDefaultsEnabled(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 4})).(*community)
	if got := env.lockInDayNumber(); got != 3 {
		t.Errorf("lockInDayNumber() = %d, want 3 (the default, unchanged)", got)
	}
}

// --- rich affordances ----------------------------------------------------------

func TestRichAffordancesExcludeUnaffordableInspect(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{
		"affordances": "rich", "inspect_cost": 5, "credits": map[string]any{"mira": 25, "dov": 3, "ada": 15, "rook": 20},
	})).(*community)
	env.round = 1

	var dovVerbs, miraVerbs []string
	for _, s := range env.Actions("dov") {
		dovVerbs = append(dovVerbs, s.Verb)
	}
	for _, s := range env.Actions("mira") {
		miraVerbs = append(miraVerbs, s.Verb)
	}
	if contains2(dovVerbs, "inspect") {
		t.Errorf("dov has only 3 credits and inspect costs 5; rich mode should exclude it: %v", dovVerbs)
	}
	if !contains2(miraVerbs, "inspect") {
		t.Errorf("mira can afford to inspect; rich mode should still offer it: %v", miraVerbs)
	}
}

func TestBasicAffordancesAlwaysOfferInspectRegardlessOfAffordability(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{
		"affordances": "basic", "inspect_cost": 5, "credits": map[string]any{"mira": 25, "dov": 3, "ada": 15, "rook": 20},
	})).(*community)
	env.round = 1
	var dovVerbs []string
	for _, s := range env.Actions("dov") {
		dovVerbs = append(dovVerbs, s.Verb)
	}
	if !contains2(dovVerbs, "inspect") {
		t.Error("basic mode must reproduce the original behavior: inspect is always offered regardless of affordability")
	}
}

func TestRichAffordancesExcludeProjectActionsWithNoProposals(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"affordances": "rich"})).(*community)
	env.round = 1
	var verbs []string
	for _, s := range env.Actions("mira") {
		verbs = append(verbs, s.Verb)
	}
	for _, unwanted := range []string{"support", "commit", "contribute"} {
		if contains2(verbs, unwanted) {
			t.Errorf("rich mode should exclude %q with no proposals on the table: %v", unwanted, verbs)
		}
	}
	if !contains2(verbs, "propose") {
		t.Error("propose must still be offered — it's how a proposal comes to exist in the first place")
	}
}

func TestRichAffordancesListRealProposalIDs(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"affordances": "rich"})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "b", "credits": 10, "labor": 0}})

	var supportUsage string
	for _, s := range env.Actions("ada") {
		if s.Verb == "support" {
			supportUsage = s.Usage
		}
	}
	if !strings.Contains(supportUsage, "p1") || !strings.Contains(supportUsage, "p2") {
		t.Errorf("rich mode's support usage should list real ids: %q", supportUsage)
	}
}

func TestRichAffordancesVoteDayWithNoProposalsOffersOnlyPass(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"affordances": "rich", "days": 2})).(*community)
	env.round = 2 // the vote day; no proposals were ever created
	specs := env.Actions("mira")
	if len(specs) != 1 || specs[0].Kind != sim.ActionPass {
		t.Errorf("Actions = %+v, want just pass", specs)
	}
}

// --- world-state grounding --------------------------------------------------

func TestBasicGroundingReproducesTheOriginalPublicStateExactly(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"grounding": "basic"})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 5}})

	pub := env.PublicState()
	proposals, _ := pub["proposals"].(string)
	if !strings.Contains(proposals, "raised 0c/0l") {
		t.Errorf("basic grounding should keep the original \"raised\" wording unchanged: %q", proposals)
	}
	if strings.Contains(proposals, "promised") || strings.Contains(proposals, "status:") {
		t.Errorf("basic grounding must not add promised/status fields: %q", proposals)
	}
	if _, has := env.PrivateState("dov")["your_commitments"]; has {
		t.Error("basic grounding must not add your_commitments to private state")
	}
}

func TestRichGroundingDistinguishesPromisedFromTransferred(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"grounding": "rich"})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	// dov promises 5 credits but only ever transfers 2.
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 5}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 2}})

	proposals, _ := env.PublicState()["proposals"].(string)
	if !strings.Contains(proposals, "promised 5c/0l") {
		t.Errorf("public state should report the promised total distinctly from what was transferred: %q", proposals)
	}
	if !strings.Contains(proposals, "transferred 2c/0l") {
		t.Errorf("public state should report only the actually transferred total: %q", proposals)
	}
	if !strings.Contains(proposals, "status: proposed") {
		t.Errorf("an unresolved proposal should report status \"proposed\": %q", proposals)
	}

	yours, _ := env.PrivateState("dov")["your_commitments"].(string)
	if !strings.Contains(yours, "promised 5c/0l") || !strings.Contains(yours, "transferred 2c/0l") || !strings.Contains(yours, "(outstanding)") {
		t.Errorf("your_commitments should show the gap between promise and transfer: %q", yours)
	}

	// Mira, who made no commitments, gets no your_commitments field at all.
	if _, has := env.PrivateState("mira")["your_commitments"]; has {
		t.Error("an agent with no commitments should not get an empty your_commitments field")
	}
}

func TestRichGroundingMarksACommitmentFulfilledOnceTransferCoversIt(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"grounding": "rich"})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 5}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 5}})

	yours, _ := env.PrivateState("dov")["your_commitments"].(string)
	if !strings.Contains(yours, "(fulfilled)") {
		t.Errorf("a commitment fully covered by actual transfer should read fulfilled: %q", yours)
	}
}

func TestRichGroundingReportsProposalStatusAfterResolution(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"grounding": "rich", "days": 2, "structural_cost": 5})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "b", "credits": 10, "labor": 0}})
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 10}})
	env.round = 2 // the vote day
	for _, id := range ccCast {
		if err := env.Validate(id, sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}}); err != nil {
			t.Fatalf("vote should be legal on the vote day: %v", err)
		}
		if _, err := env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "vote", Args: map[string]any{"proposal": "p1"}}); err != nil {
			t.Fatal(err)
		}
	}
	env.EndRound(2)

	proposals, _ := env.PublicState()["proposals"].(string)
	if !strings.Contains(proposals, "p1(") || !strings.Contains(proposals, "status: approved & completed") {
		t.Errorf("the winning, funded proposal should read approved & completed: %q", proposals)
	}
	if !strings.Contains(proposals, "status: not approved") {
		t.Errorf("the losing proposal should read not approved: %q", proposals)
	}
}

// --- feasibility metadata --------------------------------------------------

func TestFeasibilityMetadataComputedAtStartAndNeverLeaked(t *testing.T) {
	infeasible := buildCommunity(t, map[string]any{
		"days": 2, "structural_cost": 1000, "credits": map[string]any{"mira": 10, "dov": 10, "ada": 10, "rook": 10},
	}).(*community)
	if infeasible.feasibleAtStart {
		t.Error("40 total credits against a 1000-credit repair should be infeasible from the start")
	}
	if infeasible.totalCreditsAvailable != 40 {
		t.Errorf("totalCreditsAvailable = %d, want 40", infeasible.totalCreditsAvailable)
	}

	feasible := buildCommunity(t, fixedParams(map[string]any{"structural_cost": 20})).(*community)
	if !feasible.feasibleAtStart {
		t.Error("the default fixture's total credits comfortably exceed a 20-credit repair")
	}

	// Never exposed to any agent during play.
	for _, env := range []*community{infeasible, feasible} {
		for _, id := range ccCast {
			pub, priv, brief := env.PublicState(), env.PrivateState(id), env.Brief(id)
			if _, has := pub["feasible_outcome_possible_at_start"]; has {
				t.Error("feasibility metadata leaked into public state")
			}
			if _, has := priv["feasible_outcome_possible_at_start"]; has {
				t.Error("feasibility metadata leaked into private state")
			}
			if strings.Contains(brief, "feasible") {
				t.Error("feasibility metadata leaked into the brief")
			}
		}
	}

	// But fully reported at the end.
	out := infeasible.Outcome()
	if out.Metrics["feasible_outcome_possible_at_start"] != false || out.Metrics["total_credits_available"] != 40 {
		t.Errorf("metrics = %+v", out.Metrics)
	}
}

// --- first-proposal timing and per-proposal feasibility -----------------------

func TestFirstProposalTimingIsTracked(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 4, "opportunities_per_day": 2})).(*community)
	if out := env.Outcome(); out.Metrics["first_proposal_day"] != 0 || out.Metrics["first_proposal_opportunity"] != 0 {
		t.Errorf("before any proposal, metrics = %+v", out.Metrics)
	}

	env.round = 3 // day 2, opportunity 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.round = 4 // a later proposal must not overwrite the first
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "b", "credits": 10, "labor": 0}})

	out := env.Outcome()
	if out.Metrics["first_proposal_day"] != 2 || out.Metrics["first_proposal_opportunity"] != 1 {
		t.Errorf("first_proposal_day/opportunity = %v/%v, want 2/1", out.Metrics["first_proposal_day"], out.Metrics["first_proposal_opportunity"])
	}
}

func TestProposalFeasibilityAtCreationIsTrackedPerProposal(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"structural_cost": 30})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "cheap", "credits": 20, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "safe", "credits": 40, "labor": 0}})

	out := env.Outcome()
	detail, ok := out.Metrics["proposals_feasible_detail"].([]map[string]any)
	if !ok || len(detail) != 2 {
		t.Fatalf("proposals_feasible_detail = %+v", out.Metrics["proposals_feasible_detail"])
	}
	byID := map[string]map[string]any{}
	for _, d := range detail {
		byID[d["proposal"].(string)] = d
	}
	if byID["p1"]["feasible_when_created"] != false {
		t.Errorf("p1 (20 credits vs 30 cost) should be marked infeasible: %+v", byID["p1"])
	}
	if byID["p2"]["feasible_when_created"] != true {
		t.Errorf("p2 (40 credits vs 30 cost) should be marked feasible: %+v", byID["p2"])
	}

	// A revision re-evaluates feasibility.
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"proposal": "p1", "text": "cheap, revised", "credits": 35, "labor": 0}})
	out = env.Outcome()
	for _, d := range out.Metrics["proposals_feasible_detail"].([]map[string]any) {
		if d["proposal"] == "p1" && d["feasible_when_created"] != true {
			t.Errorf("p1 after revising to 35 credits should now be feasible: %+v", d)
		}
	}
}

// --- after-complication counters -----------------------------------------

func TestAfterComplicationCountersTrackOnlyPostComplicationActivity(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"days": 4, "complication_day": 2})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 5}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 5}})

	out := env.Outcome()
	if out.Metrics["proposals_revised_after_complication"] != 0 ||
		out.Metrics["commitments_made_after_complication"] != 0 ||
		out.Metrics["contributions_after_complication"] != 0 {
		t.Errorf("before the complication, after-counters should all be 0: %+v", out.Metrics)
	}

	env.BeginRound(2) // applies the complication
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"proposal": "p1", "text": "a, revised", "credits": 40, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 10}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 5}})

	out = env.Outcome()
	if out.Metrics["proposals_revised_after_complication"] != 1 {
		t.Errorf("proposals_revised_after_complication = %v, want 1", out.Metrics["proposals_revised_after_complication"])
	}
	if out.Metrics["commitments_made_after_complication"] != 1 {
		t.Errorf("commitments_made_after_complication = %v, want 1", out.Metrics["commitments_made_after_complication"])
	}
	if out.Metrics["contributions_after_complication"] != 1 {
		t.Errorf("contributions_after_complication = %v, want 1", out.Metrics["contributions_after_complication"])
	}
}

// --- commitment opportunity tracking -------------------------------------

func TestCommitmentOpportunityTrackingDistinguishesCapacityFromFollowThrough(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "b", "credits": 5, "labor": 0}})

	// Rook has his full starting balance when he promises 10 credits to p1,
	// then never actually gives any — he clearly had the means.
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 10}})
	// Ada spends her balance on a *different* proposal first (15 -> 5),
	// then promises p1 more than she now has left — Validate's ceiling only
	// checks her starting allocation, so the promise is legal, but she did
	// not have the means at the moment she made it, and she never
	// contributes anything further to p1.
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p2", "credits": 10}})
	env.Apply("ada", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 10}})

	out := env.Outcome()
	detail, _ := out.Metrics["commitments_detail"].([]map[string]any)
	byAgent := map[string]map[string]any{}
	for _, d := range detail {
		byAgent[d["by"].(string)] = d
	}
	if byAgent["rook"]["had_opportunity_when_promised"] != true {
		t.Errorf("rook's commitment detail = %+v, want had_opportunity_when_promised=true", byAgent["rook"])
	}
	if byAgent["ada"]["had_opportunity_when_promised"] != false {
		t.Errorf("ada's commitment detail = %+v, want had_opportunity_when_promised=false", byAgent["ada"])
	}
}

// --- commitment semantics (repeat / update vs. independent obligation) -------

func TestRepeatedIdenticalCommitmentUpdatesInPlace(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})

	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 14}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 14}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 14}})

	if len(env.commitments) != 1 {
		t.Fatalf("three identical commits to the same proposal should collapse to one record, got %d: %+v", len(env.commitments), env.commitments)
	}
	if env.commitments[0].Updates != 2 {
		t.Errorf("Updates = %d, want 2 (two repeats after the original)", env.commitments[0].Updates)
	}

	out := env.Outcome()
	if out.Metrics["unique_active_commitments"] != 1 {
		t.Errorf("unique_active_commitments = %v, want 1", out.Metrics["unique_active_commitments"])
	}
	if out.Metrics["repeated_identical_commitments"] != 2 {
		t.Errorf("repeated_identical_commitments = %v, want 2", out.Metrics["repeated_identical_commitments"])
	}
	if out.Metrics["commitment_updates"] != 0 {
		t.Errorf("commitment_updates = %v, want 0 (these were repeats, not revisions)", out.Metrics["commitment_updates"])
	}
	// The fulfillment-rate denominator must reflect the one real obligation,
	// not three inflated ones.
	if rate, _ := out.Metrics["commitment_fulfillment_rate"].(float64); rate != 0 {
		t.Errorf("fulfillment_rate = %v, want 0 (one unfulfilled obligation, denominator 1)", rate)
	}
}

func TestRevisedCommitmentReplacesTheAmount(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})

	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 14}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 20}})

	if len(env.commitments) != 1 {
		t.Fatalf("a revision should update the existing record, not add a second one, got %d", len(env.commitments))
	}
	if env.commitments[0].Credits != 20 {
		t.Errorf("the commitment should now reflect the revised amount, got %d", env.commitments[0].Credits)
	}
	if env.commitments[0].Updates != 1 {
		t.Errorf("Updates = %d, want 1", env.commitments[0].Updates)
	}

	out := env.Outcome()
	if out.Metrics["commitment_updates"] != 1 {
		t.Errorf("commitment_updates = %v, want 1", out.Metrics["commitment_updates"])
	}
	if out.Metrics["repeated_identical_commitments"] != 0 {
		t.Errorf("repeated_identical_commitments = %v, want 0 (this was a revision, not a repeat)", out.Metrics["repeated_identical_commitments"])
	}
	if out.Metrics["total_committed_credits"] != 20 {
		t.Errorf("total_committed_credits = %v, want 20 (the current amount, not 14+20)", out.Metrics["total_committed_credits"])
	}
}

func TestCommitmentsToDifferentProposalsAreIndependent(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "b", "credits": 10, "labor": 0}})

	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 10}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p2", "credits": 10}})

	if len(env.commitments) != 2 {
		t.Errorf("commitments to two different proposals must stay independent, got %d", len(env.commitments))
	}
}

func TestDifferentAgentsCommittingToTheSameProposalAreIndependent(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})

	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 10}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 5}})

	if len(env.commitments) != 2 {
		t.Errorf("two different agents committing to the same proposal must both count, got %d", len(env.commitments))
	}
}

func TestCommitmentAggregateMetricsUseCurrentAmountsOnly(t *testing.T) {
	env := buildCommunity(t, fixedParams(nil)).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "a", "credits": 10, "labor": 0}})

	// Rook commits and fully honours it.
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 10}})
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "contribute", Args: map[string]any{"proposal": "p1", "credits": 10}})
	// Dov commits and never follows through.
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1", "credits": 5}})

	out := env.Outcome()
	if out.Metrics["total_committed_credits"] != 15 {
		t.Errorf("total_committed_credits = %v, want 15 (10+5)", out.Metrics["total_committed_credits"])
	}
	if out.Metrics["total_fulfilled_credits"] != 10 {
		t.Errorf("total_fulfilled_credits = %v, want 10 (only rook's)", out.Metrics["total_fulfilled_credits"])
	}
	if out.Metrics["outstanding_obligations"] != 1 {
		t.Errorf("outstanding_obligations = %v, want 1 (dov's)", out.Metrics["outstanding_obligations"])
	}
}

// TestProposalsSafeDetailChecksEveryProposalAgainstTheFinalCost is a
// state-accuracy check (Section 7): unlike proposals_feasible_detail, which
// snapshots feasibility against the cost known at creation/revision time,
// this must reflect the FINAL real cost — including after a complication
// raises it — for every proposal, not only whichever one wins the vote.
func TestProposalsSafeDetailChecksEveryProposalAgainstTheFinalCost(t *testing.T) {
	env := buildCommunity(t, fixedParams(map[string]any{"structural_cost": 20})).(*community)
	env.round = 1
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "safe", "credits": 25, "labor": 0}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "unsafe", "credits": 10, "labor": 0}})

	detail, _ := env.Outcome().Metrics["proposals_safe_detail"].([]map[string]any)
	if len(detail) != 2 {
		t.Fatalf("proposals_safe_detail = %+v, want 2 entries", detail)
	}
	byID := map[string]map[string]any{}
	for _, d := range detail {
		byID[d["proposal"].(string)] = d
	}
	if byID["p1"]["safe_against_final_cost"] != true {
		t.Errorf("p1 (25 credits) should be safe against a 20-credit repair: %+v", byID["p1"])
	}
	if byID["p2"]["safe_against_final_cost"] != false {
		t.Errorf("p2 (10 credits) should be unsafe against a 20-credit repair: %+v", byID["p2"])
	}

	// The complication raises the real cost after the fact; the same
	// proposal's safety must track the FINAL cost, not the cost at creation.
	env.structuralCost += 15 // now 35; p1's 25 credits no longer cover it
	detail2, _ := env.Outcome().Metrics["proposals_safe_detail"].([]map[string]any)
	for _, d := range detail2 {
		if d["proposal"] == "p1" && d["safe_against_final_cost"] != false {
			t.Errorf("p1 should flip to unsafe once the final cost rises past its budget: %+v", d)
		}
	}
}
