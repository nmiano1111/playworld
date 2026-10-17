package agent

import (
	"strings"
	"testing"

	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

func promptTurn() sim.Turn {
	return sim.Turn{
		RunID: "r1", Scenario: "split", Round: 2, MaxRound: 2,
		Self: "mira", Participants: []sim.AgentID{"mira", "dov"},
		Brief: "the rules of the world",
		Actions: []sim.ActionSpec{
			{Kind: sim.ActionSay, Usage: `{"kind":"say"}`, Description: "speak aloud"},
			{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper"}`, Description: "speak privately"},
			{Kind: sim.ActionDo, Verb: "claim", Usage: `{"kind":"do","verb":"claim"}`, Description: "claim units"},
			{Kind: sim.ActionDo, Verb: "signal", Usage: `{"kind":"do","verb":"signal"}`, Description: "state an intention"},
			{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "stay quiet"},
		},
		PublicState:  map[string]any{"pool": 100},
		PrivateState: map[string]any{"your_private_threshold": 35},
	}
}

func TestSystemPromptCarriesIdentityRulesAndProtocol(t *testing.T) {
	p := testPersona("mira", "Mira")
	p.Role = "a fixer"
	roster := []rosterEntry{{ID: "mira", Name: "Mira"}, {ID: "dov", Name: "Dov"}}

	got := buildSystem(p, promptTurn(), roster, false, false, false, false)

	for _, want := range []string{
		"You are Mira, a fixer",
		"secret goal of mira",
		"the rules of the world",
		`- mira (Mira) — you`,
		"- dov (Dov)",
		`{"kind":"whisper"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("system prompt missing %q:\n%s", want, got)
		}
	}
	// The distinction between speech and action has to be explicit, or models
	// narrate commitments in prose and never emit a structured action.
	if !strings.Contains(got, "Saying a thing does not do it") {
		t.Error("the protocol does not distinguish speech from action")
	}
	// A roster entry must never carry another agent's persona or goals.
	if strings.Count(got, "scripted participant") > 0 {
		t.Error("the roster leaked another agent's persona")
	}
}

func TestUserPromptSeparatesSeenFromNewAndListsOptions(t *testing.T) {
	p := testPersona("mira", "Mira")
	turn := promptTurn()
	turn.New = []sim.Observation{{Round: 2, Text: "Dov (dov) said: just now"}}
	earlier := []sim.Observation{{Round: 1, Text: "Dov (dov) said: earlier"}}

	st := memory.NewState("mira", "Mira")
	st.SetBelief("dov", "blunt but honest")
	st.AddCommitment("r0", "take the night shift")
	st.AddEpisode(memory.Episode{RunID: "r0", Text: "Rook lied last time"}, 0)

	got := buildUser(p, turn, st, earlier, "r1", false, false)

	for _, want := range []string{
		"blunt but honest",
		"take the night shift",
		"Rook lied last time",
		"Known to everyone",
		"pool: 100",
		"Known only to you",
		"your_private_threshold: 35",
		"earlier",
		"Since your last turn",
		"just now",
		"Round 2 of 2 — the last one",
		"Open to you right now: say, whisper, do claim, do signal, pass",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("user prompt missing %q:\n%s", want, got)
		}
	}
	// Nothing should appear twice: the newest line belongs in one section only.
	if strings.Count(got, "just now") != 1 {
		t.Errorf("a new observation was repeated in the prompt:\n%s", got)
	}
}

func TestUserPromptHandlesAnEmptyWorld(t *testing.T) {
	p := testPersona("mira", "Mira")
	turn := sim.Turn{Self: "mira", Round: 1, MaxRound: 3}
	got := buildUser(p, turn, memory.NewState("mira", "Mira"), nil, "r1", false, false)

	if !strings.Contains(got, "nothing yet") {
		t.Errorf("an agent with no observations should be told so:\n%s", got)
	}
	if !strings.Contains(got, "(none — you can only pass)") {
		t.Errorf("an agent with no legal actions should be told so:\n%s", got)
	}
	// With no memory there should be no empty recall heading.
	if strings.Contains(got, "WHAT YOU REMEMBER FROM BEFORE") {
		t.Errorf("an agent with no memory should get no recall section:\n%s", got)
	}
}

func TestRecallExcludesTheCurrentRun(t *testing.T) {
	p := testPersona("mira", "Mira")
	st := memory.NewState("mira", "Mira")
	st.AddEpisode(memory.Episode{RunID: "r0", Text: "from a previous evening"}, 0)
	st.AddEpisode(memory.Episode{RunID: "r1", Text: "from tonight"}, 0)

	got := renderRecall(p, st, "r1")
	if !strings.Contains(got, "from a previous evening") {
		t.Errorf("recall is missing a prior session:\n%s", got)
	}
	if strings.Contains(got, "from tonight") {
		t.Errorf("recall duplicated the current run, which is already in the transcript:\n%s", got)
	}
}

func TestRenderValueFormatsCleanly(t *testing.T) {
	for in, want := range map[any]string{
		"text":       "text",
		float64(42):  "42",
		float64(0.5): "0.50",
		12:           "12",
		true:         "true",
	} {
		if got := renderValue(in); got != want {
			t.Errorf("renderValue(%v) = %q, want %q", in, got, want)
		}
	}
	// A nested value falls back to JSON rather than Go's %v syntax.
	if got := renderValue(map[string]int{"a": 1}); got != `{"a":1}` {
		t.Errorf("renderValue(map) = %q", got)
	}
}
