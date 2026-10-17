package scenario

import (
	"strings"
	"testing"

	"github.com/nmiano/playworld/internal/sim"
)

func murderParams(murderer string) map[string]any {
	return map[string]any{"talk_rounds": 1, "attempts": 2, "murderer": murderer}
}

func TestMurderAssignsRolesAndKeepsThemPrivate(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))

	// The murderer's own state and brief.
	if got := env.PrivateState("rook")["role"]; got != "murderer" {
		t.Errorf(`rook's role = %v, want "murderer"`, got)
	}
	rookBrief := env.Brief("rook")
	if !strings.Contains(rookBrief, "You did it") {
		t.Errorf("the murderer's brief should say so:\n%s", rookBrief)
	}
	if !strings.Contains(rookBrief, "If pressed, your story is") {
		t.Errorf("the murderer should be given an alibi:\n%s", rookBrief)
	}

	// Everyone else is told they are innocent and given their own evidence.
	for _, id := range []sim.AgentID{"mira", "dov"} {
		st := env.PrivateState(id)
		if st["role"] != "investigator" {
			t.Errorf("%s role = %v, want investigator", id, st["role"])
		}
		evidence, _ := st["your_evidence"].(string)
		if strings.TrimSpace(evidence) == "" {
			t.Errorf("%s was given no evidence", id)
		}
		brief := env.Brief(id)
		if !strings.Contains(brief, "You did not do it") {
			t.Errorf("%s's brief should say they are innocent:\n%s", id, brief)
		}
		if !strings.Contains(brief, evidence) {
			t.Errorf("%s's brief should contain their own evidence", id)
		}
	}

	// Nothing in the murderer's private state or brief leaks another agent's
	// evidence, and nothing in an investigator's private state reveals who
	// the murderer actually is.
	if rookBrief != env.Brief("rook") {
		t.Fatal("Brief should be stable across calls")
	}
	if strings.Contains(rookBrief, "You did not do it") {
		t.Error("the murderer's brief should not also claim innocence")
	}
	for k := range env.PrivateState("mira") {
		if k == "role" || k == "your_evidence" || k == "your_vote_this_round" {
			continue
		}
		t.Errorf("unexpected private-state key leaked to mira: %q", k)
	}
}

func TestMurderClueDistributionIsReproducibleForASeed(t *testing.T) {
	build := func() sim.Environment {
		env, err := Build("murder", Spec{Participants: cast, Names: names, Rounds: 8, Seed: 42,
			Params: map[string]any{"talk_rounds": 1, "attempts": 1}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := env.Setup(cast); err != nil {
			t.Fatal(err)
		}
		return env
	}
	a, b := build(), build()

	for _, id := range cast {
		if a.PrivateState(id)["role"] != b.PrivateState(id)["role"] {
			t.Errorf("%s got a different role across identical seeds", id)
		}
		if a.PrivateState(id)["your_evidence"] != b.PrivateState(id)["your_evidence"] {
			t.Errorf("%s got different evidence across identical seeds", id)
		}
	}
	if a.Outcome().Metrics["murderer"] != b.Outcome().Metrics["murderer"] {
		t.Error("the murderer was not reproducible for a fixed seed")
	}
}

func TestMurderRejectsAMurdererNotInTheCast(t *testing.T) {
	if _, err := Build("murder", Spec{Participants: cast, Names: names, Rounds: 8,
		Params: map[string]any{"murderer": "ghost"}}); err == nil {
		t.Error("an unknown murderer id should be rejected before the run starts")
	}
}

func TestMurderPhasesGateWhichActionsAreLegal(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))

	env.BeginRound(1) // talk
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "anyone see anything?"}); err != nil {
		t.Errorf("talking should be legal in a talk round: %v", err)
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "rook"}}); err == nil {
		t.Error("accusing before a vote should be refused")
	}

	env.EndRound(1)
	env.BeginRound(2) // vote
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "wait, one more thing"}); err == nil {
		t.Error("talking must be closed during a vote")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"dov"}}); err == nil {
		t.Error("whispering must be closed during a vote")
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "rook"}}); err != nil {
		t.Errorf("accusing should be legal during a vote: %v", err)
	}

	var kinds []string
	for _, spec := range env.Actions("mira") {
		kinds = append(kinds, string(spec.Kind)+":"+spec.Verb)
	}
	if contains2(kinds, "say:") || contains2(kinds, "whisper:") {
		t.Errorf("a vote round should not offer say/whisper: %v", kinds)
	}
	if !contains2(kinds, "do:accuse") || !contains2(kinds, "pass:") {
		t.Errorf("vote-round actions = %v", kinds)
	}
}

func TestMurderAccusationValidation(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))
	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2) // vote

	cases := []struct {
		name    string
		args    map[string]any
		wantErr bool
	}{
		{"accuse by id", map[string]any{"suspect": "rook"}, false},
		{"accuse by display name", map[string]any{"suspect": "Rook"}, false},
		{"missing suspect", nil, true},
		{"unknown suspect", map[string]any{"suspect": "ghost"}, true},
		{"self-accusation", map[string]any{"suspect": "mira"}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: c.args})
			if (err != nil) != c.wantErr {
				t.Errorf("Validate = %v, wantErr %v", err, c.wantErr)
			}
		})
	}

	if _, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "rook"}}); err != nil {
		t.Fatal(err)
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "dov"}}); err == nil {
		t.Error("voting twice in the same round should be refused")
	}
}

func TestMurderVotesAreSealedUntilTheReveal(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))
	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)

	events, err := env.Apply("mira", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "rook"}})
	if err != nil {
		t.Fatal(err)
	}
	vis := events[0].Visibility
	if vis.Public || vis.CanObserve("dov") || vis.CanObserve("rook") {
		t.Fatalf("a vote must be visible only to the voter, got %s", vis.Describe())
	}
	if !vis.CanObserve("mira") {
		t.Error("the voter should see its own sealed vote")
	}
	// It must not leak through any other agent's private state either.
	if dov := env.PrivateState("dov"); dov["your_vote_this_round"] != nil {
		t.Errorf("dov's private state should not reflect a vote it never cast: %v", dov)
	}
	if pub := env.PublicState(); contains2([]string{toString(pub["votes_submitted"])}, "rook") {
		t.Errorf("public state leaked a sealed vote: %v", pub)
	}
}

func TestMurderConclusiveVoteCatchesTheMurderer(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))
	env.BeginRound(1)
	env.EndRound(1)

	env.BeginRound(2)
	votes := map[sim.AgentID]sim.AgentID{"mira": "rook", "dov": "rook", "rook": "mira"}
	for voter, suspect := range votes {
		if _, err := env.Apply(voter, sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": string(suspect)}}); err != nil {
			t.Fatal(err)
		}
	}
	events := env.EndRound(2)
	if len(events) != 1 || !events[0].Visibility.Public {
		t.Fatalf("the reveal must be public: %+v", events)
	}
	if !strings.Contains(events[0].Text, "convicted") {
		t.Errorf("reveal text = %q", events[0].Text)
	}
	for _, voter := range []string{"mira", "dov", "rook"} {
		if !strings.Contains(events[0].Text, voter) {
			t.Errorf("the reveal should name every vote, missing %s: %q", voter, events[0].Text)
		}
	}

	done, why := env.Done()
	if !done || !strings.Contains(why, "convicted rook") {
		t.Fatalf("Done = %v, %q", done, why)
	}

	out := env.Outcome()
	if out.Metrics["caught"] != true || out.Metrics["convicted"] != "rook" {
		t.Errorf("metrics = %+v", out.Metrics)
	}
	if !strings.Contains(out.Summary, "correctly") {
		t.Errorf("summary = %q", out.Summary)
	}
	if out.Scores["rook"] != 0 {
		t.Errorf("the caught murderer should score 0, got %v", out.Scores["rook"])
	}
	// Both investigators voted for the real murderer and the group caught him.
	if out.Scores["mira"] != 100 || out.Scores["dov"] != 100 {
		t.Errorf("investigator scores = mira %v, dov %v, want 100 each", out.Scores["mira"], out.Scores["dov"])
	}
}

func TestMurderWrongConvictionLetsTheMurdererEscape(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))
	env.BeginRound(1)
	env.EndRound(1)

	env.BeginRound(2)
	// Dov correctly suspects Rook but is outvoted; Rook deflects onto Dov.
	votes := map[sim.AgentID]sim.AgentID{"mira": "dov", "dov": "rook", "rook": "dov"}
	for voter, suspect := range votes {
		env.Apply(voter, sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": string(suspect)}})
	}
	env.EndRound(2)

	done, why := env.Done()
	if !done || !strings.Contains(why, "convicted dov") {
		t.Fatalf("Done = %v, %q", done, why)
	}
	out := env.Outcome()
	if out.Metrics["caught"] != false || out.Metrics["convicted"] != "dov" {
		t.Errorf("metrics = %+v", out.Metrics)
	}
	if !strings.Contains(out.Summary, "wrongly") || !strings.Contains(out.Summary, "Rook") {
		t.Errorf("summary should name the real culprit: %q", out.Summary)
	}
	// The murderer escapes free.
	if out.Scores["rook"] != 100 {
		t.Errorf("rook (uncaught murderer) score = %v, want 100", out.Scores["rook"])
	}
	// Dov suspected correctly but the group got it wrong — partial credit.
	if out.Scores["dov"] != 25 {
		t.Errorf("dov (correct but outvoted) score = %v, want 25", out.Scores["dov"])
	}
	if out.Scores["mira"] != 0 {
		t.Errorf("mira (wrong and the group lost) score = %v, want 0", out.Scores["mira"])
	}
}

func TestMurderInconclusiveVoteBuysAnotherRoundOfTalk(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))
	env.BeginRound(1)
	env.EndRound(1)

	env.BeginRound(2) // vote 1: a tie
	votes := map[sim.AgentID]sim.AgentID{"mira": "rook", "dov": "mira", "rook": "dov"}
	for voter, suspect := range votes {
		env.Apply(voter, sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": string(suspect)}})
	}
	events := env.EndRound(2)
	if !strings.Contains(events[0].Text, "No majority") {
		t.Errorf("a tie should be reported as inconclusive: %q", events[0].Text)
	}
	if done, _ := env.Done(); done {
		t.Fatal("a tie with an attempt remaining should not end the scenario")
	}

	// Talking reopens for the extra round the inconclusive vote bought, and
	// the environment narrates why.
	narration := env.BeginRound(3)
	if len(narration) != 1 || !strings.Contains(narration[0].Text, "didn't settle it") {
		t.Errorf("round 3 should be narrated as a makeup round after the tie: %+v", narration)
	}
	if err := env.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "we need to talk this through again"}); err != nil {
		t.Errorf("talking should reopen after a tie: %v", err)
	}
	// The brief still carries the full voting history for context.
	if !strings.Contains(env.Brief("mira"), "no majority") {
		t.Errorf("the brief should carry the vote history:\n%s", env.Brief("mira"))
	}
	env.EndRound(3)

	env.BeginRound(4) // vote 2, the last attempt
	for voter := range votes {
		env.Apply(voter, sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "rook"}})
	}
	env.EndRound(4)
	done, why := env.Done()
	if !done || !strings.Contains(why, "convicted rook") {
		t.Fatalf("Done = %v, %q", done, why)
	}
	if env.Outcome().Metrics["votes_cast"] != 2 {
		t.Errorf("votes_cast = %v, want 2", env.Outcome().Metrics["votes_cast"])
	}
}

func TestMurderRunningOutOfAttemptsLetsTheMurdererEscapeByDefault(t *testing.T) {
	env := buildEnv(t, "murder", map[string]any{"talk_rounds": 1, "attempts": 1, "murderer": "rook"})
	env.BeginRound(1)
	env.EndRound(1)

	env.BeginRound(2) // the only vote, and it ties
	votes := map[sim.AgentID]sim.AgentID{"mira": "rook", "dov": "mira", "rook": "dov"}
	for voter, suspect := range votes {
		env.Apply(voter, sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": string(suspect)}})
	}
	env.EndRound(2)

	done, why := env.Done()
	if !done || !strings.Contains(why, "never reached a verdict") {
		t.Fatalf("Done = %v, %q", done, why)
	}
	out := env.Outcome()
	if out.Metrics["convicted"] != "" || out.Metrics["caught"] != false {
		t.Errorf("metrics = %+v", out.Metrics)
	}
	if out.Scores["rook"] != 100 {
		t.Errorf("an unconvicted murderer should score 100, got %v", out.Scores["rook"])
	}
	// Mira suspected correctly even though no verdict was reached.
	if out.Scores["mira"] != 25 {
		t.Errorf("mira score = %v, want 25", out.Scores["mira"])
	}
}

func TestMurderAbstainingIsARealSealedChoice(t *testing.T) {
	env := buildEnv(t, "murder", murderParams("rook"))
	env.BeginRound(1)
	env.EndRound(1)
	env.BeginRound(2)

	events, err := env.Apply("dov", sim.Action{Kind: sim.ActionPass})
	if err != nil {
		t.Fatal(err)
	}
	if events[0].Visibility.Public {
		t.Error("abstaining from a vote is still a sealed ballot, not a public pass")
	}
	if err := env.Validate("dov", sim.Action{Kind: sim.ActionDo, Verb: "accuse", Args: map[string]any{"suspect": "rook"}}); err == nil {
		t.Error("having abstained, dov should not get to vote again this round")
	}
}
