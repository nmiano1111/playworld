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

var tracked = Cognition{CompactTranscript: true, SalientRecall: 6, PlanContext: true, RepairSlots: true, TypedBeliefs: true,
	ExplicitAffordances: true, PlanTracking: true}

func worldTurn(clock string, specs ...sim.ActionSpec) sim.Turn {
	t := basicTurn()
	t.Scenario = "world"
	t.Clock = clock
	t.Actions = append(specs, sim.ActionSpec{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "wait"})
	return t
}

var (
	sayOK     = sim.ActionSpec{Kind: sim.ActionSay, Usage: `{"kind":"say"}`, Description: "speak"}
	whisperOK = sim.ActionSpec{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper"}`, Description: "note"}
	goOK      = sim.ActionSpec{Kind: sim.ActionDo, Verb: "go", Params: []string{"to", "for"}, Description: "walk"}
)

func trackedAgent(t *testing.T, cog Cognition, replies ...string) *Agent {
	t.Helper()
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: llm.ScriptedFake(replies...), RunID: "r",
		DualAction: true, Planning: true, Intentions: true, Cognition: cog})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
	return a
}

func TestSpeechAvailabilityIsSaidPlainly(t *testing.T) {
	alone := worldTurn("Day 1, 09:00 (morning)", whisperOK)
	a := trackedAgent(t, tracked, `{"communication":null,"world_action":null}`)
	msgs := a.PreviewPrompt(alone)
	if !strings.Contains(msgs[1].Content, `nobody is with you, so a "say" would be refused`) {
		t.Errorf("an unavailable say should be named next to the actions:\n%s", msgs[1].Content)
	}
	if !strings.Contains(msgs[0].Content, "People in town (address others by their id; who is with you right now is under HERE)") ||
		strings.Contains(msgs[0].Content, "Present (address") {
		t.Error("the roster should not read as who is present")
	}
	if !strings.Contains(msgs[0].Content, "what you expect it to lead to") {
		t.Error("reasoning should invite the expected outcome")
	}
	if user := a.PreviewPrompt(worldTurn("Day 1, 09:00 (morning)", sayOK, whisperOK))[1].Content; strings.Contains(user, "would be refused") {
		t.Error("no warning when someone is there to hear")
	}
	off := trackedAgent(t, grounded, `{}`)
	if p := off.PreviewPrompt(alone); strings.Contains(p[1].Content, "would be refused") || !strings.Contains(p[0].Content, "Present (address") {
		t.Error("without the refinement the prompt is as before")
	}
}

func TestWalkWithoutPurposeKeepsItsReasoning(t *testing.T) {
	a := trackedAgent(t, tracked,
		`{"communication":null,"world_action":{"type":"go","args":{"to":"market"}},"reasoning":"buy the last material before it sells out"}`,
		`{"communication":null,"world_action":{"type":"go","args":{"to":"square","for":"meet Ada"}},"reasoning":"Ada asked"}`)
	turn := worldTurn("Day 1, 09:00 (morning)", goOK)
	acts, err := a.DecideMany(context.Background(), turn)
	if err != nil || len(acts) != 1 {
		t.Fatalf("acts = %+v, %v", acts, err)
	}
	if acts[0].Verb != "go" || acts[0].Args["to"] != "market" || acts[0].Args["for"] != "buy the last material before it sells out" {
		t.Errorf("the walk keeps its destination and gains its stated reason: %+v", acts[0])
	}
	acts, _ = a.DecideMany(context.Background(), turn)
	if acts[0].Args["for"] != "meet Ada" {
		t.Errorf("an explicit purpose is never overwritten: %+v", acts[0])
	}
	if a.Stats().Counts["trip_purpose_from_reasoning"] != 1 {
		t.Errorf("counts = %v", a.Stats().Counts)
	}
}

func TestGoalAndObstacleAgesCarryAcrossRewrites(t *testing.T) {
	a := trackedAgent(t, tracked,
		`{"world_action":null,"plan":{"primary_goal":"Finish the Community Garden Bed","obstacles":["one material is missing"]}}`,
		`{"world_action":null}`,
		`{"world_action":null,"plan":{"primary_goal":"Finish the Community Garden Bed today","next_steps":["ask Dov"],"obstacles":["still one material missing","Rook wants 20 credits"],"done":["labor"]}}`,
		`{"world_action":null,"plan":{"primary_goal":"Start a stall of my own","obstacles":["no credits"]}}`)
	clocks := []string{"Day 3, 10:22 (morning)", "Day 3, 11:00 (morning)", "Day 3, 14:50 (afternoon)", "Day 4, 09:30 (morning)"}
	for i, c := range clocks[:3] {
		a.now = time.Date(2000, 1, 3, 10+i, 0, 0, 0, time.UTC)
		turn := worldTurn(c)
		turn.Now = a.now
		if _, err := a.DecideMany(context.Background(), turn); err != nil {
			t.Fatal(err)
		}
	}
	p := a.State().ActivePlan("r")
	if p.GoalStamp != "Day 3 10:22" || p.Turns != 3 {
		t.Errorf("the reworded goal is the same goal: since %q, %d turns", p.GoalStamp, p.Turns)
	}
	if p.ObstacleSince[0] != "Day 3 10:22" || p.ObstacleTurns[0] != 3 || p.ObstacleTurns[1] != 1 {
		t.Errorf("obstacles: %v %v", p.ObstacleSince, p.ObstacleTurns)
	}
	if a.Stats().Counts["plan_steps_done"] != 1 || a.Stats().Counts["goal_changes"] != 0 {
		t.Errorf("counts = %v", a.Stats().Counts)
	}
	turn := worldTurn(clocks[3])
	if _, err := a.DecideMany(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	p = a.State().ActivePlan("r")
	if p.Turns != 1 || p.GoalStamp != "Day 4 09:30" || a.Stats().Counts["goal_changes"] != 1 {
		t.Errorf("a new goal starts fresh: %+v %v", p, a.Stats().Counts)
	}
}

func TestPersistentObstacleInvitesReconsiderationWithoutDictating(t *testing.T) {
	stuck := &memory.Plan{PrimaryGoal: "Finish the garden bed", Obstacles: []string{"one material missing", "rain"},
		ObstacleSince: []string{"Day 3 10:22", "Day 3 14:00"}, ObstacleTurns: []int{5, 1}, GoalStamp: "Day 3 08:02", Turns: 7,
		At: time.Date(2000, 1, 3, 14, 0, 0, 0, time.UTC), Stamp: "Day 3 14:00"}
	got := renderPlanTracking(stuck, nil)
	for _, want := range []string{
		"You have held this goal since Day 3 08:02, through 7 of your decisions.",
		`"one material missing" — on your plan since Day 3 10:22, through 5 decisions`,
		"Whether to keep at it the same way, wait, come at it some other way, or put it aside is up to you.",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if strings.Contains(got, `"rain"`) {
		t.Error("a fresh obstacle is not called lasting")
	}
	// The note never names a particular remedy.
	for _, remedy := range []string{"salvage", "market", "buy", "ask", "work", "trade", "Rook", "Dov"} {
		if strings.Contains(got, remedy) && !strings.Contains(stuck.PrimaryGoal+strings.Join(stuck.Obstacles, ""), remedy) {
			t.Errorf("the reconsideration note suggests %q:\n%s", remedy, got)
		}
	}

	// Below the thresholds, or with no obstacle at all, nothing is said: a
	// quiet agent with no productive goal is never nudged.
	for _, p := range []*memory.Plan{
		{PrimaryGoal: "g", Obstacles: []string{"x"}, ObstacleTurns: []int{2}, Turns: 7},
		{PrimaryGoal: "g", Obstacles: []string{"x"}, ObstacleTurns: []int{5}, Turns: 3},
		{PrimaryGoal: "enjoy the afternoon", Turns: 12},
	} {
		if planStuck(p) {
			t.Errorf("not stuck: %+v", p)
		}
	}

	a := trackedAgent(t, tracked, `{"world_action":{"type":"pass"}}`)
	a.State().SetPlan("r", *stuck)
	if _, err := a.DecideMany(context.Background(), worldTurn("Day 3, 15:00 (afternoon)")); err != nil {
		t.Fatal(err)
	}
	if a.Stats().Counts["reconsider_notes"] != 1 {
		t.Errorf("shown notes are counted: %v", a.Stats().Counts)
	}
	idle := trackedAgent(t, tracked, `{"world_action":{"type":"pass"}}`)
	acts, err := idle.DecideMany(context.Background(), worldTurn("Day 3, 15:00 (afternoon)"))
	if err != nil || len(acts) != 1 || acts[0].Kind != sim.ActionPass || idle.Stats().Counts["reconsider_notes"] != 0 {
		t.Errorf("an agent with no plan may simply pass: %+v %v", acts, err)
	}
}

func TestPlanPointsAtChangesToWhatItMentions(t *testing.T) {
	written := time.Date(2000, 1, 4, 7, 54, 0, 0, time.UTC)
	plan := &memory.Plan{PrimaryGoal: "Finish the garden bed today", NextSteps: []string{"wait for Rook to answer o21"}, At: written, Stamp: "Day 4 07:54"}
	at := func(min int, text string, imp int) sim.Observation {
		return sim.Observation{Text: text, Time: written.Add(time.Duration(min) * time.Minute), Stamp: "Day 4 x", Importance: imp}
	}
	seen := []sim.Observation{
		at(-30, `✔ News: Ada's project "Community Garden Bed" at the square is planned.`, 2),
		at(10, "✔ Ada (ada) accepted Rook's offer o21 — it is done.", 2),
		at(20, `✔ Dov (dov) contributed 2 materials to "Rook's Hoard" (p1).`, 2),
		at(30, `Mira (mira) said: the "Community Garden Bed" is done!`, 0),
		at(70, `✔ News: Ada's project "Community Garden Bed" at the square is complete. It now stands there for good.`, 2),
	}
	got := planChanges(plan, seen, 4)
	if len(got) != 2 || !strings.Contains(got[0].Text, "o21") || !strings.Contains(got[1].Text, "is complete") {
		t.Errorf("only later, consequential, related events: %+v", got)
	}
	text := renderPlanTracking(plan, seen)
	if !strings.Contains(text, "Since you wrote it, these happened to things it mentions:") || !strings.Contains(text, "that is yours to judge") {
		t.Errorf("rendered:\n%s", text)
	}
}

func TestDroppingAFinishedGoalKeepsItsHistory(t *testing.T) {
	a := trackedAgent(t, tracked,
		`{"world_action":null,"plan":{"primary_goal":"Finish the garden bed"}}`,
		`{"world_action":null,"plan":null}`)
	turn := worldTurn("Day 4, 09:00 (morning)")
	if _, err := a.DecideMany(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	a.Observe(sim.Observation{Text: `✔ News: Ada's project "Community Garden Bed" at the square is complete.`, Stamp: "Day 4 09:04", Importance: 2,
		Time: time.Date(2000, 1, 4, 9, 4, 0, 0, time.UTC)})
	if _, err := a.DecideMany(context.Background(), worldTurn("Day 4, 09:10 (morning)")); err != nil {
		t.Fatal(err)
	}
	if a.State().ActivePlan("r") != nil || a.Stats().Counts["plans_dropped"] != 1 {
		t.Error("the agent may retire a finished goal")
	}
	kept := false
	for _, e := range a.State().Episodes {
		kept = kept || (strings.Contains(e.Text, "is complete") && e.Importance == 2)
	}
	if !kept {
		t.Error("the completion stays in memory after the goal is dropped")
	}
}

func TestDifferentEventsWithTheSameWordsAreNotFolded(t *testing.T) {
	e, _ := compactTranscript([]sim.Observation{
		obs("Day 5 06:00", "✔ Offer o10 from Mira to Ada expired unanswered.", 1),
		obs("Day 5 06:00", "✔ Offer o12 from Ada to Mira expired unanswered.", 2),
		obs("Day 5 06:00", "✔ Offer o13 from Ada to Mira expired unanswered.", 3),
	}, nil)
	if len(e) != 2 || !strings.Contains(e[1].text, "(2 times") {
		t.Errorf("role-swapped events stay apart, identical ones fold: %+v", e)
	}
}
