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

var grounded = Cognition{CompactTranscript: true, SalientRecall: 6, PlanContext: true, RepairSlots: true, TypedBeliefs: true}

func obs(stamp, text string, minute int) sim.Observation {
	return sim.Observation{Stamp: stamp, Text: text, Time: time.Date(2000, 1, 1, 9, minute, 0, 0, time.UTC)}
}

func TestCompactTranscriptFoldsRepeatsBySource(t *testing.T) {
	earlier := []sim.Observation{
		obs("Day 3 11:04", "Dov (dov) said: The scrap pile is empty for today. I will salvage tomorrow.", 1),
		obs("Day 3 11:22", "Ada (ada) said: Let's check the pile together.", 2),
		obs("Day 3 11:24", "Dov (dov) said: The scrap pile is empty for today. I will salvage tomorrow.", 3),
		obs("Day 3 11:30", "✔ You offered Ada 20 credits and 4 materials in exchange for 8 materials (offer o11).", 4),
		obs("Day 3 11:44", "✔ You offered Ada 20 credits and 4 materials in exchange for 8 materials (offer o12).", 5),
		obs("Day 3 11:50", "✔ You offered Ada 22 credits and 4 materials in exchange for 8 materials (offer o13).", 6),
		obs("Day 3 11:52", "Ada (ada) said: The scrap pile is empty for today. I will salvage tomorrow.", 7),
	}
	fresh := []sim.Observation{
		obs("Day 3 12:04", "Dov (dov) said: The scrap pile is empty today. I will salvage tomorrow!", 8),
	}
	e, f := compactTranscript(earlier, fresh)
	var all []string
	for _, l := range append(e, f...) {
		all = append(all, l.text)
	}
	joined := strings.Join(all, "\n")
	if len(f) != 1 || !strings.Contains(f[0].text, "(3 times, first Day 3 11:04)") {
		t.Errorf("the fresh repeat should carry the count:\n%s", joined)
	}
	if strings.Count(joined, "Dov (dov) said") != 1 {
		t.Errorf("Dov's repeats should fold into one line:\n%s", joined)
	}
	if !strings.Contains(joined, "(offer o12). (2 times, first Day 3 11:30)") {
		t.Errorf("identical offers should fold at the latest one:\n%s", joined)
	}
	if !strings.Contains(joined, "22 credits") {
		t.Error("an offer with different numbers is different and must stay")
	}
	if !strings.Contains(joined, "Ada (ada) said: The scrap pile is empty") {
		t.Error("the same words from a different speaker are a different line")
	}
	if !strings.Contains(joined, "Ada (ada) said: Let's check the pile together.") {
		t.Error("unrelated lines are kept")
	}
}

func TestCompactTranscriptNeverMergesClaimsWithVerifiedEvents(t *testing.T) {
	earlier := []sim.Observation{
		obs("Day 1 09:00", "Rook (rook) said: I gave Ada 20 credits.", 1),
		obs("Day 1 09:01", "✔ Rook (rook) gave Ada 20 credits as a gift — nothing was asked or given in return.", 2),
	}
	e, _ := compactTranscript(earlier, nil)
	if len(e) != 2 {
		t.Errorf("a claim and the verified event must both stay: %+v", e)
	}
}

func TestSlotRepairMovesSpeechOutOfTheWorldSlot(t *testing.T) {
	reply := `{"communication":null,"world_action":{"type":"say","args":{"message":"Let's wait for Dov."}}}`
	for _, repair := range []bool{false, true} {
		a, err := New(Options{Persona: testPersona("ada", "Ada"), Provider: llm.ScriptedFake(reply), RunID: "r", DualAction: true,
			Cognition: Cognition{RepairSlots: repair}})
		if err != nil {
			t.Fatal(err)
		}
		a.Bind([]sim.AgentID{"ada", "dov"}, map[sim.AgentID]string{"ada": "Ada", "dov": "Dov"})
		acts, err := a.DecideMany(context.Background(), basicTurn())
		if err != nil {
			t.Fatal(err)
		}
		if len(acts) != 1 {
			t.Fatalf("acts = %+v", acts)
		}
		if repair && (acts[0].Kind != sim.ActionSay || acts[0].Content != "Let's wait for Dov." || a.Stats().SlotRepairs != 1) {
			t.Errorf("with repair: %+v (repairs %d)", acts[0], a.Stats().SlotRepairs)
		}
		if !repair && (acts[0].Kind != sim.ActionDo || acts[0].Verb != "say") {
			t.Errorf("without repair the action is passed through for the world to refuse: %+v", acts[0])
		}
	}
	// A world verb is never turned into speech, and a full communication
	// slot is never overwritten.
	both := `{"communication":{"type":"say","message":"hi"},"world_action":{"type":"say","message":"again"}}`
	a, _ := New(Options{Persona: testPersona("ada", "Ada"), Provider: llm.ScriptedFake(both), RunID: "r", DualAction: true, Cognition: Cognition{RepairSlots: true}})
	a.Bind([]sim.AgentID{"ada", "dov"}, map[sim.AgentID]string{"ada": "Ada", "dov": "Dov"})
	acts, _ := a.DecideMany(context.Background(), basicTurn())
	if len(acts) != 2 || acts[0].Content != "hi" || acts[1].Kind != sim.ActionDo {
		t.Errorf("with both slots used nothing is repaired: %+v", acts)
	}
}

func TestZeroCognitionPromptIsUnchanged(t *testing.T) {
	st := memory.NewState("mira", "Mira")
	st.SetPlan("r", memory.Plan{PrimaryGoal: "open a stall", NextSteps: []string{"buy materials"}})
	st.SetBelief("rook", "pushy")
	tr := []sim.Observation{obs("Day 1 09:00", "Rook (rook) said: hi", 1), obs("Day 1 09:01", "Rook (rook) said: hi", 2)}
	turn := basicTurn()
	if a, b := buildUser(testPersona("mira", "Mira"), turn, st, tr, "r", true, true), buildUserWith(testPersona("mira", "Mira"), turn, st, tr, "r", true, true, Cognition{}); a != b {
		t.Errorf("zero Cognition must reproduce the original prompt exactly:\n%s\n---\n%s", a, b)
	}
}

func TestSalientEventsOutliveTheTranscriptWindow(t *testing.T) {
	st := memory.NewState("ada", "Ada")
	base := time.Date(2000, 1, 2, 9, 0, 0, 0, time.UTC)
	add := func(min int, text string, importance float64) {
		st.AddEpisode(memory.Episode{RunID: "r", Kind: memory.KindObservation, Time: base.Add(time.Duration(min) * time.Minute),
			Stamp: "Day 2 09:" + string(rune('0'+min/10)) + string(rune('0'+min%10)), Text: text, Importance: importance}, 0)
	}
	add(1, "✔ Rook (rook) gave you 20 credits and 4 materials as a gift — nothing was asked or given in return.", 2)
	add(2, "Dov (dov) said: the pile is empty", 0)
	add(3, "✔ Rook (rook) gave you 20 credits and 4 materials as a gift — nothing was asked or given in return.", 2) // exact repeat
	add(4, "✔ Mira (mira) declined your offer o3.", 2)
	add(50, "Dov (dov) said: the pile is empty again", 0) // inside the window below

	window := []sim.Observation{{Stamp: "Day 2 09:50", Text: "Dov (dov) said: the pile is empty again", Time: base.Add(50 * time.Minute)}}
	turn := basicTurn()
	turn.Now = base.Add(60 * time.Minute)
	user := buildUserWith(testPersona("ada", "Ada"), turn, st, window, "r", true, false, grounded)
	section := user[strings.Index(user, "--- EARLIER, YOU SAW"):strings.Index(user, "--- WHAT YOU HAVE SEEN")]
	if strings.Count(section, "gave you 20 credits") != 1 || !strings.Contains(section, "declined your offer o3") {
		t.Errorf("consequential events (once each) should be recalled:\n%s", section)
	}
	if strings.Contains(section, "the pile is empty") {
		t.Errorf("routine chatter is not salient:\n%s", section)
	}
}

func TestPlanContextShowsAgeProgressAndDoneSteps(t *testing.T) {
	st := memory.NewState("ada", "Ada")
	written := time.Date(2000, 1, 3, 7, 2, 0, 0, time.UTC)
	st.SetPlan("r", memory.Plan{PrimaryGoal: "settle the dispute", NextSteps: []string{"let Dov finish salvaging"}, Done: []string{"asked Rook what he needs"}, At: written, Stamp: "Day 3 07:02"})
	seen := []sim.Observation{
		{Text: "✔ Dov (dov) finished salvaging and came away with 2 materials.", Time: written.Add(2 * time.Hour)},
		{Text: "Rook (rook) said: liar", Time: written.Add(3 * time.Hour)},
		{Text: "✔ Old news.", Time: written.Add(-time.Hour)},
	}
	got := renderPlanContext(st, "r", seen)
	for _, want := range []string{"(Written Day 3 07:02; 1 thing you saw happen since.)", "let Dov finish salvaging", "Done so far:\n  - asked Rook what he needs", "rests on something no longer true"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
}

func TestPlanDoneAndIntentionReplacesAreDecoded(t *testing.T) {
	plan, _, err := extractPlan(`{"plan":{"primary_goal":"g","next_steps":["b"],"done":["a"]}}`)
	if err != nil || len(plan.Done) != 1 || plan.Done[0] != "a" {
		t.Errorf("plan = %+v, %v", plan, err)
	}
	got := extractIntentions(`{"intentions":[{"when":"in 2h","note":"check again","replaces":"i4"}]}`)
	if len(got) != 1 || got[0].Args["replaces"] != "i4" {
		t.Errorf("intentions = %+v", got)
	}
}

func TestTypedBeliefsAreLabelledAsViewsWithBasisAndTime(t *testing.T) {
	reply := `{"communication":null,"world_action":null,"opinions":{"dov":{"note":"may be hiding scrap","basis":"suspect"},"the scrap pile":"empty today"}}`
	a, err := New(Options{Persona: testPersona("rook", "Rook"), Provider: llm.ScriptedFake(reply, reply), RunID: "r", DualAction: true, Intentions: true, Cognition: grounded})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"rook", "dov"}, map[sim.AgentID]string{"rook": "Rook", "dov": "Dov"})
	turn := basicTurn()
	turn.Clock = "Day 3, 11:00 (late morning)"
	if _, err := a.DecideMany(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	b := a.State().Beliefs["dov"]
	if b.Basis != "suspect" || b.Stamp != "Day 3 11:00" {
		t.Errorf("belief = %+v", b)
	}
	if a.State().Beliefs["the scrap pile"].Note != "empty today" {
		t.Error("an impression about a situation should be kept too")
	}
	msgs := a.PreviewPrompt(turn)
	if !strings.Contains(msgs[0].Content, "saw|heard|inferred|suspect") {
		t.Error("the system prompt should describe the basis categories")
	}
	if !strings.Contains(msgs[1].Content, "not facts") || !strings.Contains(msgs[1].Content, "dov: may be hiding scrap [suspect; as of Day 3 11:00]") {
		t.Errorf("impressions should be rendered as dated views:\n%s", msgs[1].Content)
	}
}

func TestPreviewPromptMatchesWhatIsSent(t *testing.T) {
	fake := llm.ScriptedFake(`{"communication":null,"world_action":null}`)
	a, _ := New(Options{Persona: testPersona("mira", "Mira"), Provider: fake, RunID: "r", DualAction: true, Intentions: true, Cognition: grounded})
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
	a.Observe(sim.Observation{Text: "Dov (dov) said: hello", Stamp: "Day 1 09:00"})
	turn := basicTurn()
	preview := a.PreviewPrompt(turn)
	if _, err := a.DecideMany(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	sent := fake.Calls()[0].Messages
	if len(sent) != len(preview) || sent[0].Content != preview[0].Content || sent[1].Content != preview[1].Content {
		t.Error("the preview must be exactly the prompt that is sent")
	}
}
