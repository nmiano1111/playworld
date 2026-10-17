package agent

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

func TestExtractIntentions(t *testing.T) {
	raw := `{"communication":null,"world_action":null,
		"intentions":[{"when":"tomorrow 09:00","note":"ask rook about a loan"},{"when":"","note":"no time"},{"at":"in 2h","what":"check stalls"}],
		"cancel_intentions":["i3"]}`
	got := extractIntentions(raw)
	if len(got) != 3 {
		t.Fatalf("want 2 intentions and 1 cancel, got %+v", got)
	}
	if got[0].Kind != sim.ActionIntend || got[0].Args["when"] != "tomorrow 09:00" || got[0].Args["note"] != "ask rook about a loan" {
		t.Errorf("first = %+v", got[0])
	}
	if got[1].Args["when"] != "in 2h" || got[1].Args["note"] != "check stalls" {
		t.Errorf("synonyms not accepted: %+v", got[1])
	}
	if got[2].Args["cancel"] != "i3" {
		t.Errorf("cancel = %+v", got[2])
	}
	// A single object instead of a list is tolerated.
	if one := extractIntentions(`{"kind":"pass","intentions":{"when":"14:00","note":"x"}}`); len(one) != 1 {
		t.Errorf("single-object intentions: %+v", one)
	}
	if none := extractIntentions(`{"kind":"pass"}`); len(none) != 0 {
		t.Errorf("no intentions field should yield nothing: %+v", none)
	}
}

func TestIntentionsAreReturnedOnlyWhenEnabled(t *testing.T) {
	reply := `{"communication":{"type":"say","message":"hi"},"world_action":null,"intentions":[{"when":"in 1h","note":"follow up"}]}`
	for _, enabled := range []bool{false, true} {
		a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: llm.ScriptedFake(reply), RunID: "r", DualAction: true, Intentions: enabled})
		if err != nil {
			t.Fatal(err)
		}
		a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
		acts, err := a.DecideMany(context.Background(), basicTurn())
		if err != nil {
			t.Fatal(err)
		}
		intents := 0
		for _, x := range acts {
			if x.Kind == sim.ActionIntend {
				intents++
			}
		}
		if enabled && intents != 1 || !enabled && intents != 0 {
			t.Errorf("enabled=%v: got %d intentions in %+v", enabled, intents, acts)
		}
	}
	// The single-action protocol carries them too.
	a, _ := New(Options{Persona: testPersona("mira", "Mira"), Provider: llm.ScriptedFake(`{"kind":"pass","intentions":[{"when":"in 1h","note":"x"}]}`), RunID: "r", Intentions: true})
	acts, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil || len(acts) != 2 || acts[1].Kind != sim.ActionIntend {
		t.Fatalf("single-action intentions: %+v, %v", acts, err)
	}
}

func TestClockedTurnsSpeakInTimeNotRounds(t *testing.T) {
	fake := llm.ScriptedFake(`{"kind":"pass"}`)
	a := newTestAgent(t, "mira", "Mira", fake, nil)
	a.Observe(sim.Observation{Text: "Dov said: morning", Stamp: "Day 2 08:05"})
	turn := basicTurn()
	turn.Clock = "Day 2, 08:10 (morning)"
	if _, err := a.Decide(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	a.Observe(sim.Observation{Text: "Dov left", Stamp: "Day 2 08:20"})
	if _, err := a.Decide(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	p := promptText(fake.Calls()[1])
	if !strings.Contains(p, "[Day 2 08:05] Dov said: morning") {
		t.Errorf("transcript should be stamped with simulated time:\n%s", p)
	}
	if !strings.Contains(p, "It is Day 2, 08:10 (morning).") || strings.Contains(p, "Round 1 of 3") {
		t.Errorf("a clocked turn should name the time, not a round:\n%s", p)
	}
}

func TestResumeTranscriptRestoresRecentHistoryAsSeen(t *testing.T) {
	store := memory.NewMemStore()
	first, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: llm.ScriptedFake(`{"kind":"pass"}`), Store: store, RunID: "world-town"})
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range []string{"Dov said: one", "Dov said: two", "Dov said: three"} {
		first.Observe(sim.Observation{Text: text, Stamp: "Day 1 0" + string(rune('7'+i)) + ":00", Time: time.Unix(int64(i), 0)})
	}
	if err := first.Persist(); err != nil {
		t.Fatal(err)
	}

	// A different process, same world: the transcript comes back from memory.
	fake := llm.ScriptedFake(`{"kind":"pass"}`)
	second, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: fake, Store: store, RunID: "world-town"})
	if err != nil {
		t.Fatal(err)
	}
	second.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
	second.ResumeTranscript()
	if _, err := second.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	p := promptText(fake.Calls()[0])
	if !strings.Contains(p, "[Day 1 09:00] Dov said: three") {
		t.Errorf("resumed transcript missing:\n%s", p)
	}
	if strings.Contains(p, "Since your last turn") {
		t.Errorf("restored history was already seen and must not be presented as new:\n%s", p)
	}
}

func TestLongLivedAgentsSeeTheirOwnNotesAndCommitments(t *testing.T) {
	fake := llm.ScriptedFake(
		`{"kind":"pass","remember":"Rook owes me 3 credits","plan":{"primary_goal":"open a stall","commitments":["build Ada a shelf by Day 3"]}}`,
		`{"kind":"pass"}`)
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: fake, RunID: "world-town",
		Planning: true, Intentions: true, RecallOwnNotes: true})
	if err != nil {
		t.Fatal(err)
	}
	var planSeen *memory.Plan
	notes := 0
	a.OnPlan = func(p *memory.Plan) { planSeen = p }
	a.OnMemory = func(kind, _, _ string) {
		if kind == "note" {
			notes++
		}
	}
	turn := basicTurn()
	turn.Clock = "Day 2, 09:30 (morning)"
	for i := 0; i < 2; i++ {
		if _, err := a.DecideMany(context.Background(), turn); err != nil {
			t.Fatal(err)
		}
	}
	if planSeen == nil || len(planSeen.Commitments) != 1 || notes != 1 {
		t.Fatalf("hooks: plan=%+v notes=%d", planSeen, notes)
	}
	calls := fake.Calls()
	sys, user := calls[1].Messages[0].Content, calls[1].Messages[1].Content
	if !strings.Contains(sys, `"commitments"`) {
		t.Error("world agents with planning should be told about plan commitments")
	}
	if !strings.Contains(user, "[Day 2 09:30] Rook owes me 3 credits") {
		t.Errorf("own notes should be recalled with their stamp:\n%s", user)
	}
	if !strings.Contains(user, "build Ada a shelf by Day 3") {
		t.Errorf("commitments should be shown back in the plan:\n%s", user)
	}

	// Round-based agents are unaffected.
	plain := newTestAgent(t, "mira", "Mira", llm.ScriptedFake(`{"kind":"pass"}`), nil)
	plain.planning = true
	if _, err := plain.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
}
