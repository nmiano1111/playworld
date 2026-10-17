package scenario

import (
	"fmt"
	"strings"
	"testing"

	"github.com/nmiano/playworld/internal/sim"
)

var cast = []sim.AgentID{"mira", "dov", "rook"}

var names = map[sim.AgentID]string{"mira": "Mira", "dov": "Dov", "rook": "Rook"}

func buildEnv(t *testing.T, name string, params map[string]any) sim.Environment {
	t.Helper()
	env, err := Build(name, Spec{Participants: cast, Names: names, Rounds: 8, Seed: 1, Params: params})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := env.Setup(cast); err != nil {
		t.Fatal(err)
	}
	return env
}

func TestRegistryListsBothScenarios(t *testing.T) {
	got := Names()
	if len(got) < 2 {
		t.Fatalf("Names = %v, want at least social and split", got)
	}
	for _, want := range []string{"social", "split"} {
		if !contains2(got, want) {
			t.Errorf("%q is not registered: %v", want, got)
		}
	}
	if _, err := Lookup("nope"); err == nil {
		t.Error("unknown scenario should error")
	} else if !strings.Contains(err.Error(), "social") {
		t.Errorf("error should list the available scenarios, got %v", err)
	}
	for _, info := range List() {
		if info.Title == "" || info.Description == "" || info.DefaultRounds <= 0 {
			t.Errorf("scenario %q is under-documented: %+v", info.Name, info)
		}
	}
}

func TestBuildEnforcesAgentCounts(t *testing.T) {
	if _, err := Build("split", Spec{Participants: []sim.AgentID{"solo"}, Rounds: 4}); err == nil {
		t.Error("one agent should be refused by a 2-agent minimum")
	}
	many := make([]sim.AgentID, 12)
	for i := range many {
		many[i] = sim.AgentID(string(rune('a' + i)))
	}
	if _, err := Build("social", Spec{Participants: many, Rounds: 4}); err == nil {
		t.Error("twelve agents should exceed the maximum")
	}
}

// --- social ------------------------------------------------------------------

func TestSocialValidatesActions(t *testing.T) {
	env := buildEnv(t, "social", nil)

	cases := []struct {
		name    string
		action  sim.Action
		wantErr bool
	}{
		{"speech with content", sim.Action{Kind: sim.ActionSay, Content: "hello"}, false},
		{"empty speech", sim.Action{Kind: sim.ActionSay, Content: "  "}, true},
		{"whisper to a participant", sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"dov"}}, false},
		{"whisper with no recipient", sim.Action{Kind: sim.ActionWhisper, Content: "psst"}, true},
		{"whisper to a stranger", sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"ghost"}}, true},
		{"whisper to self", sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"mira"}}, true},
		{"propose with text", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "split evenly"}}, false},
		{"propose with no text", sim.Action{Kind: sim.ActionDo, Verb: "propose"}, true},
		{"invented verb", sim.Action{Kind: sim.ActionDo, Verb: "teleport"}, true},
		{"commit to nothing", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p9"}}, true},
		{"pass", sim.Action{Kind: sim.ActionPass}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := env.Validate("mira", c.action)
			if (err != nil) != c.wantErr {
				t.Errorf("Validate = %v, wantErr %v", err, c.wantErr)
			}
		})
	}
}

func TestSocialWhisperIsPrivateToSpeakerAndListener(t *testing.T) {
	env := buildEnv(t, "social", nil)
	events, err := env.Apply("mira", sim.Action{Kind: sim.ActionWhisper, Content: "don't trust Rook", To: []sim.AgentID{"dov"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %d", len(events))
	}
	vis := events[0].Visibility
	if vis.Public {
		t.Fatal("a whisper must not be public")
	}
	if !vis.CanObserve("mira") {
		t.Error("the speaker should see its own whisper")
	}
	if !vis.CanObserve("dov") {
		t.Error("the addressed agent should see the whisper")
	}
	if vis.CanObserve("rook") {
		t.Error("a third party must not observe a whisper")
	}
}

func TestSocialProposalNeedsSupportToBecomeAgreement(t *testing.T) {
	env := buildEnv(t, "social", map[string]any{"support_needed": 2})

	if _, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "mira takes the night shift"}}); err != nil {
		t.Fatal(err)
	}
	// Public state reflects the open proposal, so every agent can see it.
	if got := env.PublicState()["proposals_on_the_table"]; !strings.Contains(toString(got), "p1") {
		t.Errorf("public state = %v", env.PublicState())
	}

	events, err := env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 {
		t.Errorf("one commitment should not yet form an agreement, got %d events", len(events))
	}

	// Committing twice is refused rather than silently double-counted.
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1"}}); err == nil {
		t.Error("a second commitment from the same agent should be refused")
	}

	events, err = env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 2 {
		t.Fatalf("reaching the support threshold should also announce the agreement, got %d events", len(events))
	}
	if !strings.Contains(events[1].Text, "now an agreement") {
		t.Errorf("second event = %q", events[1].Text)
	}

	out := env.Outcome()
	if out.Metrics["agreements"] != 1 {
		t.Errorf("metrics = %+v", out.Metrics)
	}
	if !strings.Contains(out.Summary, "night shift") {
		t.Errorf("summary should name the agreement: %q", out.Summary)
	}
}

func TestSocialPrivateStateDoesNotLeakOthersPositions(t *testing.T) {
	env := buildEnv(t, "social", nil)
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "x"}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "commit", Args: map[string]any{"proposal": "p1"}})

	dovState := env.PrivateState("dov")
	if got := toString(dovState["you_have_committed_to"]); !strings.Contains(got, "p1") {
		t.Errorf("dov should know its own commitment, got %v", dovState)
	}
	if rookState := env.PrivateState("rook"); len(rookState) != 0 {
		t.Errorf("rook has committed to nothing and should be told nothing: %v", rookState)
	}
}

func TestSocialTracksInteractionShape(t *testing.T) {
	env := buildEnv(t, "social", nil)
	env.Apply("mira", sim.Action{Kind: sim.ActionSay, Content: "hello"})
	env.Apply("mira", sim.Action{Kind: sim.ActionWhisper, Content: "a", To: []sim.AgentID{"dov"}})
	env.Apply("mira", sim.Action{Kind: sim.ActionWhisper, Content: "b", To: []sim.AgentID{"dov"}})
	env.Apply("rook", sim.Action{Kind: sim.ActionPass})

	m := env.Outcome().Metrics
	if m["public_utterances"] != 1 || m["private_messages"] != 2 || m["passes"] != 1 {
		t.Errorf("metrics = %+v", m)
	}
	if got := toString(m["whisper_pairs"]); !strings.Contains(got, "mira→dov×2") {
		t.Errorf("whisper graph = %q", got)
	}
}

// --- split -------------------------------------------------------------------

func splitParams() map[string]any {
	return map[string]any{
		"pool": 100, "talk_rounds": 1, "attempts": 2, "bonus": 25,
		"needs": map[string]any{"mira": 30, "dov": 30, "rook": 50},
	}
}

func TestSplitPhasesGateWhichActionsAreLegal(t *testing.T) {
	env := buildEnv(t, "split", splitParams())

	// Round 1 is a talking round.
	env.BeginRound(1)
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "let's be sensible"}); err != nil {
		t.Errorf("talking should be legal in a talk round: %v", err)
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 30}}); err == nil {
		t.Error("claiming before the claim round should be refused")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "signal", Args: map[string]any{"amount": 30}}); err != nil {
		t.Errorf("signalling should be legal in a talk round: %v", err)
	}

	// Round 2 is the sealed-claim round.
	env.BeginRound(2)
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "one more thing"}); err == nil {
		t.Error("talking must be closed in a sealed-claim round")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionWhisper, Content: "x", To: []sim.AgentID{"dov"}}); err == nil {
		t.Error("whispering must be closed in a sealed-claim round")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 30}}); err != nil {
		t.Errorf("claiming should be legal now: %v", err)
	}

	// And the action list the agent is shown must match.
	verbs := map[string]bool{}
	for _, spec := range env.Actions("mira") {
		verbs[spec.Verb+string(spec.Kind)] = true
	}
	if !verbs["claimdo"] {
		t.Error("claim should be offered in a claim round")
	}
	if verbs["say"] {
		t.Error("say should not be offered in a claim round")
	}
}

func TestSplitClaimValidation(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	env.BeginRound(2) // claim round

	cases := []struct {
		name    string
		args    map[string]any
		wantErr bool
	}{
		{"in range", map[string]any{"amount": 40}, false},
		{"zero is allowed", map[string]any{"amount": 0}, false},
		{"the whole pool is allowed", map[string]any{"amount": 100}, false},
		{"over the pool", map[string]any{"amount": 101}, true},
		{"negative", map[string]any{"amount": -1}, true},
		{"fractional", map[string]any{"amount": 33.5}, true},
		{"missing", nil, true},
		{"not a number", map[string]any{"amount": "lots"}, true},
		{"numeric string is accepted", map[string]any{"amount": "40"}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: c.args})
			if (err != nil) != c.wantErr {
				t.Errorf("Validate = %v, wantErr %v", err, c.wantErr)
			}
		})
	}
}

func TestSplitClaimsAreSealedUntilResolution(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	env.BeginRound(2)

	events, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 30}})
	if err != nil {
		t.Fatal(err)
	}
	vis := events[0].Visibility
	if vis.Public || vis.CanObserve("dov") || vis.CanObserve("rook") {
		t.Fatalf("a sealed claim must be visible only to the claimer, got %s", vis.Describe())
	}
	if !vis.CanObserve("mira") {
		t.Error("the claimer should see its own claim")
	}
	// Nor may it leak through state.
	if dov := env.PrivateState("dov"); toString(dov["your_sealed_claim_this_attempt"]) != "" {
		t.Errorf("dov's private state mentions a claim it did not make: %v", dov)
	}
	if pub := env.PublicState(); strings.Contains(toString(pub), "30") {
		t.Errorf("public state leaked a sealed claim: %v", pub)
	}
	// Claiming twice in one attempt is refused.
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 10}}); err == nil {
		t.Error("a second claim in the same attempt should be refused")
	}
}

func TestSplitPrivateThresholdIsNotShared(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	if got := env.PrivateState("mira")["your_private_threshold"]; got != 30 {
		t.Errorf("mira's own threshold = %v, want 30", got)
	}
	if got := env.PrivateState("rook")["your_private_threshold"]; got != 50 {
		t.Errorf("rook's own threshold = %v, want 50", got)
	}
	// Mira's brief must not contain anyone else's number.
	brief := env.Brief("mira")
	if !strings.Contains(brief, "30") {
		t.Error("mira's brief should state her own threshold")
	}
	if strings.Contains(brief, "rook: 50") || strings.Contains(brief, "threshold is 50") {
		t.Errorf("mira's brief leaked rook's threshold:\n%s", brief)
	}
	if pub := toString(env.PublicState()); strings.Contains(pub, "threshold") {
		t.Errorf("public state leaked thresholds: %s", pub)
	}
}

func TestSplitSettlesWhenClaimsFit(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	env.BeginRound(1)
	env.EndRound(1)

	env.BeginRound(2)
	for id, amount := range map[sim.AgentID]int{"mira": 30, "dov": 30, "rook": 40} {
		if _, err := env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": amount}}); err != nil {
			t.Fatal(err)
		}
	}
	events := env.EndRound(2)
	if len(events) != 1 || !events[0].Visibility.Public {
		t.Fatalf("resolution must be public: %+v", events)
	}
	if !strings.Contains(events[0].Text, "they fit") {
		t.Errorf("resolution text = %q", events[0].Text)
	}

	done, why := env.Done()
	if !done {
		t.Fatal("a successful attempt should end the scenario")
	}
	if !strings.Contains(why, "fit the pool") {
		t.Errorf("termination reason = %q", why)
	}

	out := env.Outcome()
	// 30 units + the 25 bonus for meeting a threshold of 30.
	if out.Scores["mira"] != 55 {
		t.Errorf("mira score = %v, want 55", out.Scores["mira"])
	}
	// 40 units but a threshold of 50, so no bonus.
	if out.Scores["rook"] != 40 {
		t.Errorf("rook score = %v, want 40", out.Scores["rook"])
	}
	if out.Metrics["units_paid"] != 100 || out.Metrics["thresholds_met"] != 2 {
		t.Errorf("metrics = %+v", out.Metrics)
	}
}

func TestSplitBustPaysNobodyAndRevealsClaims(t *testing.T) {
	params := splitParams()
	params["attempts"] = 1
	env := buildEnv(t, "split", params)

	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)
	for id, amount := range map[sim.AgentID]int{"mira": 50, "dov": 50, "rook": 50} {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": amount}})
	}
	events := env.EndRound(2)
	if len(events) != 1 || !strings.Contains(events[0].Text, "BUST") {
		t.Fatalf("events = %+v", events)
	}
	// Everyone learns every claim, which is what lets an agent notice it was
	// misled and carry that forward.
	for _, id := range cast {
		if !strings.Contains(events[0].Text, string(id)) {
			t.Errorf("resolution should name %s's claim: %q", id, events[0].Text)
		}
	}
	if done, _ := env.Done(); !done {
		t.Error("the final attempt busting should end the scenario")
	}
	out := env.Outcome()
	for _, id := range cast {
		if out.Scores[id] != 0 {
			t.Errorf("%s scored %v after a bust, want 0", id, out.Scores[id])
		}
	}
	if out.Metrics["units_paid"] != 0 {
		t.Errorf("metrics = %+v", out.Metrics)
	}
}

func TestSplitAllowsASecondAttemptAfterABust(t *testing.T) {
	env := buildEnv(t, "split", splitParams()) // attempts: 2

	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)
	for _, id := range cast {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 50}})
	}
	env.EndRound(2)
	if done, _ := env.Done(); done {
		t.Fatal("a bust with an attempt remaining should not end the scenario")
	}

	// Round 3 is another talking round, and the bust is on the record.
	env.BeginRound(3)
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "that went badly"}); err != nil {
		t.Errorf("talking should reopen after a bust: %v", err)
	}
	if !strings.Contains(env.Brief("mira"), "BUST") {
		t.Errorf("the brief should record the failed attempt:\n%s", env.Brief("mira"))
	}

	env.BeginRound(4)
	for id, amount := range map[sim.AgentID]int{"mira": 30, "dov": 30, "rook": 30} {
		env.Apply(id, sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": amount}})
	}
	env.EndRound(4)
	if done, _ := env.Done(); !done {
		t.Error("the second attempt fitting should settle the scenario")
	}
	if env.Outcome().Metrics["attempts_used"] != 2 {
		t.Errorf("metrics = %+v", env.Outcome().Metrics)
	}
}

func TestSplitPassInAClaimRoundClaimsNothing(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	env.BeginRound(2)
	events, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 0}})
	if err != nil {
		t.Fatal(err)
	}
	_ = events
	// Passing is a real move with a real consequence, not a no-op.
	pass, err := env.Apply("dov", sim.Action{Kind: sim.ActionPass})
	if err != nil {
		t.Fatal(err)
	}
	if pass[0].Visibility.Public {
		t.Error("claiming nothing by passing is still a sealed claim")
	}
	if pass[0].Data["amount"] != 0 {
		t.Errorf("pass data = %+v", pass[0].Data)
	}
}

func TestSplitRecordsBrokenPromises(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	env.BeginRound(1)
	// Rook says 30 out loud, then claims 45.
	if _, err := env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "signal", Args: map[string]any{"amount": 30}}); err != nil {
		t.Fatal(err)
	}
	if _, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "signal", Args: map[string]any{"amount": 25}}); err != nil {
		t.Fatal(err)
	}
	env.EndRound(1)

	env.BeginRound(2)
	env.Apply("rook", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 45}})
	env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 25}})
	env.Apply("dov", sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": 30}})
	events := env.EndRound(2)

	// A stated intention is public, so the discrepancy is visible to everyone.
	if !strings.Contains(events[0].Text, "said 30") {
		t.Errorf("resolution should contrast the claim with what was said: %q", events[0].Text)
	}
	out := env.Outcome()
	if out.Metrics["broken_promises"] != 1 {
		t.Errorf("broken_promises = %v, want 1 (only rook claimed more than stated)", out.Metrics["broken_promises"])
	}
}

func TestSplitGeneratesConflictingNeedsFromSeed(t *testing.T) {
	env, err := Build("split", Spec{Participants: cast, Names: names, Rounds: 8, Seed: 42,
		Params: map[string]any{"pool": 100}})
	if err != nil {
		t.Fatal(err)
	}
	env.Setup(cast)

	total := 0
	for _, id := range cast {
		n, ok := env.PrivateState(id)["your_private_threshold"].(int)
		if !ok || n <= 0 {
			t.Fatalf("%s has no generated threshold: %v", id, env.PrivateState(id))
		}
		total += n
	}
	// The whole point: they cannot all be satisfied at once.
	if total <= 100 {
		t.Errorf("generated thresholds sum to %d, which fits the pool — there is no conflict to negotiate", total)
	}

	// Same seed, same thresholds.
	again, _ := Build("split", Spec{Participants: cast, Names: names, Rounds: 8, Seed: 42, Params: map[string]any{"pool": 100}})
	again.Setup(cast)
	for _, id := range cast {
		if env.PrivateState(id)["your_private_threshold"] != again.PrivateState(id)["your_private_threshold"] {
			t.Errorf("thresholds are not reproducible for seed 42 (%s)", id)
		}
	}
}

func TestSplitRejectsBadParams(t *testing.T) {
	for name, params := range map[string]map[string]any{
		"non-numeric pool": {"pool": "lots"},
		"zero pool":        {"pool": 0},
		"bad needs shape":  {"needs": "mira=30"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Build("split", Spec{Participants: cast, Rounds: 4, Params: params}); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

func TestSplitRunsOutOfRounds(t *testing.T) {
	env := buildEnv(t, "split", splitParams())
	env.BeginRound(99)
	done, why := env.Done()
	if !done || !strings.Contains(why, "ran out of rounds") {
		t.Errorf("Done = %v, %q", done, why)
	}
	// It still produces a usable outcome rather than panicking.
	if out := env.Outcome(); out.Summary == "" {
		t.Error("an unfinished scenario should still report an outcome")
	}
}

func contains2(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func toString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return strings.TrimSpace(strings.Join(strings.Fields(sprint(v)), " "))
}

func sprint(v any) string {
	type stringer interface{ String() string }
	if s, ok := v.(stringer); ok {
		return s.String()
	}
	return strings.TrimSpace(fmtSprint(v))
}

func fmtSprint(v any) string { return fmt.Sprint(v) }

func TestSocialClosingRoundForcesAnActionOrAPass(t *testing.T) {
	env, err := Build("social", Spec{Participants: cast, Names: names, Rounds: 3, Seed: 1,
		Params: map[string]any{"support_needed": 2}})
	if err != nil {
		t.Fatal(err)
	}
	env.Setup(cast)

	// Rounds 1-2 are ordinary talking rounds.
	env.BeginRound(1)
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "hello"}); err != nil {
		t.Errorf("talking should be legal in round 1: %v", err)
	}

	// The final round shuts the talking channel, which is what makes a model
	// emit the structured action instead of narrating a deal that never
	// actually happened.
	events := env.BeginRound(3)
	if len(events) != 1 || !strings.Contains(events[0].Text, "talking is over") {
		t.Fatalf("the closing round should be announced publicly: %+v", events)
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "one more thing"}); err == nil {
		t.Error("speech must be refused in the closing round")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"dov"}}); err == nil {
		t.Error("whispering must be refused in the closing round")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "propose", Args: map[string]any{"text": "mira takes it"}}); err != nil {
		t.Errorf("proposing must stay legal in the closing round: %v", err)
	}
	// Passing is still a real choice, not a forced action.
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionPass}); err != nil {
		t.Errorf("passing must stay legal: %v", err)
	}

	// And the agent is shown only the moves it can actually make.
	var kinds []string
	for _, spec := range env.Actions("mira") {
		kinds = append(kinds, string(spec.Kind)+":"+spec.Verb)
	}
	for _, unwanted := range []string{"say:", "whisper:"} {
		if contains2(kinds, unwanted) {
			t.Errorf("the closing round should not offer %q: %v", unwanted, kinds)
		}
	}
	if !contains2(kinds, "do:propose") || !contains2(kinds, "do:commit") || !contains2(kinds, "pass:") {
		t.Errorf("closing-round actions = %v", kinds)
	}
	if !strings.Contains(env.Brief("mira"), "talking is over") {
		t.Errorf("the brief should explain the closing round:\n%s", env.Brief("mira"))
	}
}

func TestSocialClosingRoundCanBeDisabled(t *testing.T) {
	env, err := Build("social", Spec{Participants: cast, Names: names, Rounds: 2, Seed: 1,
		Params: map[string]any{"closing_round": "false"}})
	if err != nil {
		t.Fatal(err)
	}
	env.Setup(cast)
	env.BeginRound(2)
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "still talking"}); err != nil {
		t.Errorf("with closing_round disabled, the last round stays conversational: %v", err)
	}
}

func TestSocialSingleRoundHasNoClosingRound(t *testing.T) {
	// A one-round run would otherwise be nothing but a closing round, leaving
	// no conversation at all.
	env, err := Build("social", Spec{Participants: cast, Names: names, Rounds: 1, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	env.Setup(cast)
	env.BeginRound(1)
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "hello"}); err != nil {
		t.Errorf("a one-round run should still allow talking: %v", err)
	}
}
