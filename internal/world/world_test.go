package world

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/sim"
)

// --- test doubles ------------------------------------------------------------

// scripted is an actor whose decisions come from a function of its turn, so a
// test can say "when activated for this reason, do that".
type scripted struct {
	id       sim.AgentID
	decide   func(t sim.Turn, n int) []sim.Action
	turns    []sim.Turn
	observed []sim.Observation
}

func (s *scripted) ID() sim.AgentID           { return s.id }
func (s *scripted) Name() string              { return strings.ToUpper(string(s.id[:1])) + string(s.id[1:]) }
func (s *scripted) Observe(o sim.Observation) { s.observed = append(s.observed, o) }
func (s *scripted) Decide(ctx context.Context, t sim.Turn) (sim.Action, error) {
	return sim.Action{Kind: sim.ActionPass}, nil
}
func (s *scripted) DecideMany(ctx context.Context, t sim.Turn) ([]sim.Action, error) {
	s.turns = append(s.turns, t)
	if s.decide == nil {
		return nil, nil
	}
	return s.decide(t, len(s.turns)-1), nil
}
func (s *scripted) CallCount() int { return len(s.turns) }

func (s *scripted) saw(substr string) bool {
	for _, o := range s.observed {
		if strings.Contains(o.Text, substr) {
			return true
		}
	}
	return false
}

// why is the reasons an activation carried, in either cognition mode.
func why(t sim.Turn) string {
	if s, ok := t.PrivateState["why you are deciding now"].(string); ok {
		return s
	}
	for _, l := range strings.Split(t.Situation, "\n") {
		if r, ok := strings.CutPrefix(l, "Why you are deciding now: "); ok {
			return r
		}
	}
	return ""
}

func do(verb string, args map[string]any) sim.Action {
	return sim.Action{Kind: sim.ActionDo, Verb: verb, Args: args}
}

func newWorld(t *testing.T, ids ...string) *World {
	t.Helper()
	cast := make([]sim.AgentID, 0, len(ids))
	for _, id := range ids {
		cast = append(cast, sim.AgentID(id))
	}
	names := map[sim.AgentID]string{}
	for _, id := range cast {
		names[id] = strings.ToUpper(string(id[:1])) + string(id[1:])
	}
	w, err := New("test", 1, DefaultSettings(), cast, names)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func engine(t *testing.T, w *World, actors ...*scripted) *Engine {
	t.Helper()
	as := make([]sim.Actor, 0, len(actors))
	for _, a := range actors {
		as = append(as, a)
	}
	e, err := NewEngine(Config{World: w, SessionID: "s", RunID: "world-test", Agents: as})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// --- scheduler ---------------------------------------------------------------

func TestQueueOrdersByTimeThenKindThenID(t *testing.T) {
	q := newQueue(nil)
	t0 := At(1, 9, 0)
	q.Push(&Item{ID: 3, At: t0, Kind: KindActivate})
	q.Push(&Item{ID: 4, At: t0, Kind: KindActivityDone})
	q.Push(&Item{ID: 1, At: t0.Add(time.Minute), Kind: KindDawn})
	q.Push(&Item{ID: 2, At: t0, Kind: KindActivate})
	q.Push(&Item{ID: 5, At: t0, Kind: KindIdle})

	var got []int64
	for q.Len() > 0 {
		got = append(got, q.Pop().ID)
	}
	if want := []int64{4, 5, 2, 3, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v, want %v (activity before idle before activations, ties by id)", got, want)
	}
}

func TestQueueRemoveAndReschedule(t *testing.T) {
	q := newQueue(nil)
	for i := int64(1); i <= 4; i++ {
		q.Push(&Item{ID: i, At: At(1, 8+int(i), 0), Kind: KindIdle})
	}
	q.Remove(2)
	q.Remove(99) // not pending: no-op
	q.Reschedule(4, At(1, 7, 0))
	var got []int64
	for q.Len() > 0 {
		got = append(got, q.Pop().ID)
	}
	if want := []int64{4, 1, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order = %v, want %v", got, want)
	}
}

func TestParseWhen(t *testing.T) {
	now := At(2, 10, 30)
	cases := map[string]time.Time{
		"in 45m":           now.Add(45 * time.Minute),
		"in 2 hours":       now.Add(2 * time.Hour),
		"1h30m":            now.Add(90 * time.Minute),
		"14:00":            At(2, 14, 0),
		"09:00":            At(3, 9, 0), // already passed today
		"2pm":              At(2, 14, 0),
		"evening":          At(2, 18, 0),
		"tomorrow 09:00":   At(3, 9, 0),
		"tomorrow morning": At(3, 8, 0),
		"tomorrow":         At(3, 8, 0),
		"day 4 08:30":      At(4, 8, 30),
		"Day 3":            At(3, 8, 0),
		"at 11:15":         At(2, 11, 15),
	}
	for in, want := range cases {
		got, err := ParseWhen(in, now)
		if err != nil {
			t.Errorf("ParseWhen(%q): %v", in, err)
			continue
		}
		if !got.Equal(want) {
			t.Errorf("ParseWhen(%q) = %s, want %s", in, Stamp(got), Stamp(want))
		}
	}
	for _, bad := range []string{"", "soonish", "day 1 08:00", "25:00", "2h of rest"} {
		if _, err := ParseWhen(bad, now); err == nil {
			t.Errorf("ParseWhen(%q) should fail", bad)
		}
	}
}

func TestParseSpan(t *testing.T) {
	for in, want := range map[string]time.Duration{"2d": 48 * time.Hour, "36h": 36 * time.Hour, "1d12h": 36 * time.Hour, "90m": 90 * time.Minute} {
		if got, err := ParseSpan(in); err != nil || got != want {
			t.Errorf("ParseSpan(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
}

// --- activation policy -------------------------------------------------------

func TestTriggersCoalesceIntoOneActivation(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Now = At(1, 9, 0)
	w.Trigger(Trigger{Agent: "mira", At: w.Now, Reason: Reason{Kind: "message", Text: "note from rook"}})
	w.Trigger(Trigger{Agent: "mira", At: w.Now.Add(time.Minute), Reason: Reason{Kind: "offer", Text: "offer from rook"}})
	w.Trigger(Trigger{Agent: "mira", At: w.Now, Reason: Reason{Kind: "message", Text: "note from rook"}}) // duplicate

	var acts []*Item
	for _, it := range w.queue.Sorted() {
		if it.Kind == KindActivate {
			acts = append(acts, it)
		}
	}
	if len(acts) != 1 {
		t.Fatalf("want one pending activation, got %d", len(acts))
	}
	if len(acts[0].Reasons) != 2 {
		t.Fatalf("want 2 coalesced reasons, got %+v", acts[0].Reasons)
	}
}

func TestCooldownBusyAndNightDeferActivations(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	mira := w.Residents["mira"]
	w.Now = At(1, 9, 0)

	mira.LastActivation = w.Now
	if got := w.eligible(mira, w.Now, false); !got.Equal(At(1, 9, 20)) {
		t.Errorf("cooldown: eligible = %s, want Day 1 09:20", Stamp(got))
	}
	mira.Activity = &Activity{Kind: "work", End: At(1, 12, 0)}
	if got := w.eligible(mira, w.Now, false); !got.Equal(At(1, 12, 0)) {
		t.Errorf("busy: eligible = %s, want the end of the shift", Stamp(got))
	}
	mira.Activity = nil
	mira.LastActivation = time.Time{}
	night := At(1, 23, 30)
	if got := w.eligible(mira, night, false); !got.Equal(At(2, 7, 0)) {
		t.Errorf("night: eligible = %s, want Day 2 07:00", Stamp(got))
	}
	if got := w.eligible(mira, night, true); !got.Equal(night) {
		t.Errorf("explicit wakeups are honoured at night: got %s", Stamp(got))
	}
	rook := w.Residents["rook"]
	if got := w.eligible(rook, night, false); !got.Equal(At(2, 7, 7)) {
		t.Errorf("mornings are staggered by roster position: got %s", Stamp(got))
	}
	mira.ActDay, mira.ActCount = 1, w.Settings.Scheduling.MaxActivationsPerDay
	if got := w.eligible(mira, w.Now, true); !got.Equal(At(2, 7, 0)) {
		t.Errorf("daily cap: eligible = %s, want next morning", Stamp(got))
	}
}

func TestQuietWorldStillGivesEveryoneOpportunities(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	mira, rook := &scripted{id: "mira"}, &scripted{id: "rook"}
	res := engine(t, w, mira, rook).Run(context.Background(), w.Now.Add(48*time.Hour))
	if res.StopReason != "reached Day 3 07:00" {
		t.Fatalf("stop = %q", res.StopReason)
	}
	// Nothing ever happens and mira always declines, so the idle interval
	// backs off: 07:00, then +3h, +6h, then +9h (the cap) which lands at
	// night and moves to the next morning — never a call at night, and
	// never a stream of calls announcing that nothing is happening.
	var got []string
	for _, turn := range mira.turns {
		got = append(got, turn.Clock)
		if !strings.Contains(why(turn), "quiet moment") {
			t.Errorf("unexpected reason %q", why(turn))
		}
		if h := mustClockHour(t, turn.Clock); h < 7 || h >= 22 {
			t.Errorf("mira was woken at night: %s", turn.Clock)
		}
	}
	want := []string{"Day 1, 07:00 (morning)", "Day 1, 10:00 (morning)", "Day 1, 16:00 (afternoon)", "Day 2, 07:00 (morning)", "Day 2, 16:00 (afternoon)", "Day 3, 07:00 (morning)"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("quiet activations:\n got %v\nwant %v", got, want)
	}
	if len(rook.turns) == 0 {
		t.Error("rook should get opportunities too")
	}
	if !w.Now.Equal(At(3, 7, 0)) {
		t.Errorf("clock = %s, want exactly the end of the window", Stamp(w.Now))
	}
}

func TestDoingSomethingResetsTheIdleBackoff(t *testing.T) {
	w := newWorld(t, "mira")
	mira := &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
		if n == 2 {
			return []sim.Action{do("go", map[string]any{"to": "square"})}
		}
		return nil
	}}
	engine(t, w, mira).Run(context.Background(), At(1, 16, 10))
	if len(mira.turns) != 3 {
		t.Fatalf("want activations at 07:00, 10:00, 16:00; got %d", len(mira.turns))
	}
	if w.Residents["mira"].IdleStreak != 0 {
		t.Errorf("acting should reset the idle streak, got %d", w.Residents["mira"].IdleStreak)
	}
}

func mustClockHour(t *testing.T, clock string) int {
	t.Helper()
	var day, h, m int
	if _, err := fmtSscanf(clock, "Day %d, %d:%d", &day, &h, &m); err != nil {
		t.Fatalf("clock %q: %v", clock, err)
	}
	return h
}

// --- conversation and visibility ---------------------------------------------

func TestSpeechIsLocalAndWhispersReachAnywhere(t *testing.T) {
	w := newWorld(t, "mira", "rook", "ada")
	w.Residents["rook"].Location = "market"
	w.Residents["mira"].Location = "market"
	ada := &scripted{id: "ada"}
	rook := &scripted{id: "rook"}
	mira := &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
		if n == 0 {
			return []sim.Action{
				{Kind: sim.ActionSay, Content: "Rook, are the stalls busy today?"},
			}
		}
		if n == 1 {
			return []sim.Action{{Kind: sim.ActionWhisper, To: []sim.AgentID{"ada"}, Content: "Come to the market?"}}
		}
		return nil
	}}
	e := engine(t, w, mira, rook, ada)
	e.Run(context.Background(), At(1, 11, 0))

	if !rook.saw("are the stalls busy") {
		t.Error("rook, at the market, should hear mira")
	}
	if ada.saw("are the stalls busy") {
		t.Error("ada, at home, must not hear speech at the market")
	}
	if !ada.saw("Come to the market?") {
		t.Error("a whisper should reach ada at home")
	}
	if rook.saw("Come to the market?") {
		t.Error("rook must not see a whisper to ada")
	}
	// Being named woke rook (a reaction, coalesced with nothing else).
	woken := false
	for _, turn := range rook.turns {
		woken = woken || strings.Contains(why(turn), "Mira spoke to you")
	}
	if !woken {
		t.Error("rook was addressed by name and should have been activated by it")
	}
	gotNote := false
	for _, turn := range ada.turns {
		gotNote = gotNote || strings.Contains(why(turn), "private note from Mira")
	}
	if !gotNote {
		t.Error("ada should have been activated by the note")
	}
}

func TestUnaddressedSpeechInAGroupDoesNotWakeListeners(t *testing.T) {
	w := newWorld(t, "mira", "rook", "ada")
	for _, id := range []sim.AgentID{"mira", "rook", "ada"} {
		w.Residents[id].Location = "square"
	}
	w.Now = At(1, 9, 0)
	eff, err := w.Apply("mira", sim.Action{Kind: sim.ActionSay, Content: "Lovely morning."})
	if err != nil {
		t.Fatal(err)
	}
	if len(eff.Triggers) != 0 {
		t.Fatalf("speech addressed to nobody in a group woke %d listeners", len(eff.Triggers))
	}
	eff, _ = w.Apply("mira", sim.Action{Kind: sim.ActionSay, Content: "ada, have you eaten?"})
	if len(eff.Triggers) != 1 || eff.Triggers[0].Agent != "ada" {
		t.Fatalf("only the named listener should wake: %+v", eff.Triggers)
	}
}

func TestSayingToAnEmptyRoomIsRefused(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	if err := w.Validate("mira", sim.Action{Kind: sim.ActionSay, Content: "hello?"}); err == nil {
		t.Fatal("nobody is home with mira; say should be refused")
	}
}

// --- agent-driven scheduling -------------------------------------------------

func TestIntentionsWakeTheAgentAndCanBeCancelled(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	mira := &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
		switch n {
		case 0:
			return []sim.Action{
				{Kind: sim.ActionIntend, Args: map[string]any{"when": "10:00", "note": "approach Rook about financing"}},
				{Kind: sim.ActionIntend, Args: map[string]any{"when": "in 5h", "note": "check the market"}},
			}
		}
		if strings.Contains(why(t), "approach Rook") {
			return []sim.Action{{Kind: sim.ActionIntend, Args: map[string]any{"cancel": "i2"}}}
		}
		return nil
	}}
	rook := &scripted{id: "rook"}
	engine(t, w, mira, rook).Run(context.Background(), At(1, 20, 0))

	fired := false
	for _, turn := range mira.turns {
		if strings.Contains(why(turn), "approach Rook about financing") {
			fired = true
			if turn.Clock != "Day 1, 10:00 (morning)" {
				t.Errorf("intention fired at %s, want 10:00", turn.Clock)
			}
		}
		if strings.Contains(why(turn), "check the market") {
			t.Error("cancelled intention still fired")
		}
	}
	if !fired {
		t.Fatal("intention never woke mira")
	}
	if w.Intentions[1].Status != "cancelled" {
		t.Errorf("i2 status = %s", w.Intentions[1].Status)
	}
}

func TestIntentionHorizonsAreEnforced(t *testing.T) {
	w := newWorld(t, "mira")
	w.Now = At(1, 9, 0)
	notes := []string{"check the market prices", "visit the workshop pile", "ask rook about his debt", "post a notice", "count my savings", "plan a project"}
	n := 0
	intend := func(when string) error {
		// Distinct notes: identical ones would revise each other instead
		// of accumulating (see TestRevisedIntentionsReplaceRatherThanPileUp).
		_, err := w.Apply("mira", sim.Action{Kind: sim.ActionIntend, Args: map[string]any{"when": when, "note": notes[n]}})
		n++
		return err
	}
	if err := intend("in 1m"); err != nil {
		t.Fatal(err)
	}
	if got := w.Intentions[0].At; !got.Equal(At(1, 9, 20)) {
		t.Errorf("too-soon intention scheduled at %s, want clamped to the minimum (09:20)", Stamp(got))
	}
	if err := intend("day 9 08:00"); err == nil {
		t.Error("an intention beyond the maximum horizon should be refused")
	}
	for i := 0; i < 3; i++ {
		if err := intend("in 2h"); err != nil {
			t.Fatal(err)
		}
	}
	if err := intend("in 3h"); err == nil {
		t.Error("pending intentions beyond the cap should be refused")
	}
}

func TestAppointmentsNotifyAndRemindEveryone(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	mira := &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
		if n == 0 {
			return []sim.Action{do("arrange", map[string]any{"with": "rook", "place": "square", "when": "18:00", "what": "talk about a loan"})}
		}
		return nil
	}}
	rook := &scripted{id: "rook"}
	engine(t, w, mira, rook).Run(context.Background(), At(1, 19, 0))
	if !rook.saw("arranged a meeting with Rook at the square on Day 1 18:00") {
		t.Error("rook should be told about the meeting")
	}
	for _, a := range []*scripted{mira, rook} {
		reminded := false
		for _, turn := range a.turns {
			if strings.Contains(why(turn), "talk about a loan") && turn.Clock == "Day 1, 18:00 (evening)" {
				reminded = true
			}
		}
		if !reminded {
			t.Errorf("%s was not activated when the meeting came due", a.id)
		}
	}
}

// --- economy and persistent consequences --------------------------------------

func TestWorkTakesTimeAndPays(t *testing.T) {
	w := newWorld(t, "mira")
	mira := &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
		switch n {
		case 0:
			return []sim.Action{do("go", map[string]any{"to": "market"})}
		case 1:
			return []sim.Action{do("work", nil)}
		}
		return nil
	}}
	engine(t, w, mira).Run(context.Background(), At(1, 12, 0))
	if got := w.Residents["mira"].Goods.Credits; got != 16 {
		t.Errorf("credits = %d, want 10 + 6", got)
	}
	// She arrives at 07:15 but was activated at 07:00, so the 20-minute
	// cooldown holds the arrival activation until 07:20.
	if mira.turns[1].Clock != "Day 1, 07:20 (morning)" {
		t.Errorf("arrival activation at %s, want 07:20", mira.turns[1].Clock)
	}
	if mira.turns[2].Clock != "Day 1, 10:20 (morning)" {
		t.Errorf("a 3-hour shift from 07:20 ends at 10:20; next activation was %s", mira.turns[2].Clock)
	}
	if w.Stock.Shifts["market"] != w.Settings.Economy.ShiftsPerDay-1 {
		t.Errorf("shifts left = %d", w.Stock.Shifts["market"])
	}
}

func TestProjectsCompleteAndPayTheirOwner(t *testing.T) {
	w := newWorld(t, "mira", "dov")
	w.Residents["mira"].Location = "square"
	w.Residents["dov"].Location = "square"
	w.Now = At(1, 9, 0)
	mustApply(t, w, "mira", do("start_project", map[string]any{"name": "bench", "credits": 8, "materials": 1, "labor": 2}))
	if err := w.Validate("dov", do("build", map[string]any{"project": "p1"})); err != nil {
		t.Fatalf("dov should be able to build at the square: %v", err)
	}
	eff := mustApply(t, w, "dov", do("contribute", map[string]any{"project": "p1", "credits": 50}))
	if got := w.Residents["dov"].Goods.Credits; got != 2 {
		t.Errorf("contribution should be capped at what is needed: dov has %d credits, want 2", got)
	}
	if len(eff.Triggers) != 1 || eff.Triggers[0].Agent != "mira" {
		t.Errorf("the owner should be told: %+v", eff.Triggers)
	}
	mustApply(t, w, "mira", do("contribute", map[string]any{"project": "bench", "materials": 1}))
	mustApply(t, w, "mira", do("build", map[string]any{"project": "p1"}))
	done := w.queue.Get(w.Residents["mira"].Activity.ItemID)
	w.queue.Remove(done.ID)
	w.Now = done.At
	eff = w.Process(done)
	p := w.Projects[0]
	if !p.Done() {
		t.Fatalf("project should be complete: %+v", p)
	}
	public := false
	for _, ev := range eff.Events {
		public = public || (ev.Visibility.Public && strings.Contains(ev.Text, "is complete"))
	}
	if !public {
		t.Error("completion should be public news")
	}
	if f := w.Location("square").Features; len(f) != 1 || !strings.Contains(f[0], "bench") {
		t.Errorf("the completed project should become a permanent feature: %v", f)
	}
	before := w.Residents["mira"].Goods.Credits
	w.Now = At(2, 6, 0)
	w.dawn()
	if got := w.Residents["mira"].Goods.Credits - before; got != w.Settings.Economy.ProjectYield {
		t.Errorf("dawn yield = %d", got)
	}
	if err := w.Validate("dov", do("build", map[string]any{"project": "p1"})); err == nil {
		t.Error("cannot build a completed project")
	}
}

func TestTradesMoveNothingUntilAccepted(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Residents["mira"].Location = "market"
	w.Now = At(1, 9, 0)
	mustApply(t, w, "rook", do("offer", map[string]any{"to": "mira", "give_credits": "4", "want_materials": 1}))
	if w.Residents["rook"].Goods.Credits != 10 {
		t.Fatal("an offer must not move resources")
	}
	if err := w.Validate("rook", do("accept", map[string]any{"offer": "o1"})); err == nil {
		t.Error("only the recipient can accept")
	}
	mustApply(t, w, "mira", do("accept", map[string]any{"offer": "o1"}))
	if m, r := w.Residents["mira"].Goods, w.Residents["rook"].Goods; m != (Goods{14, 0}) || r != (Goods{6, 2}) {
		t.Errorf("after trade mira=%v rook=%v", m, r)
	}
	if err := w.Validate("mira", do("accept", map[string]any{"offer": "o1"})); err == nil {
		t.Error("an offer can only be accepted once")
	}
}

func TestBeliefsCannotChangeReality(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Now = At(1, 9, 0)
	for name, a := range map[string]sim.Action{
		"build elsewhere":   do("build", map[string]any{"project": "p9"}),
		"work at home":      do("work", nil),
		"give to absentee":  do("give", map[string]any{"to": "rook", "credits": 1}),
		"spend too much":    do("offer", map[string]any{"to": "rook", "give_credits": 999}),
		"unknown verb":      do("teleport", nil),
		"walk to nowhere":   do("go", map[string]any{"to": "the moon"}),
		"oversized project": do("start_project", map[string]any{"name": "castle", "credits": 999}),
	} {
		if err := w.Validate("mira", a); err == nil {
			t.Errorf("%s: should be refused", name)
		}
	}
	if w.Residents["mira"].Goods != w.Settings.Starting {
		t.Error("validation must never change state")
	}
}

func mustApply(t *testing.T, w *World, id sim.AgentID, a sim.Action) Effects {
	t.Helper()
	eff, err := w.Apply(id, a)
	if err != nil {
		t.Fatalf("%s %s: %v", id, a.Verb, err)
	}
	return eff
}

// --- persistence -------------------------------------------------------------

func TestSaveLoadResumesTheSameFuture(t *testing.T) {
	dir := t.TempDir()
	store := Store{Root: dir}

	script := func(id sim.AgentID) *scripted {
		return &scripted{id: id, decide: func(t sim.Turn, n int) []sim.Action {
			if id == "mira" && n == 0 {
				return []sim.Action{{Kind: sim.ActionIntend, Args: map[string]any{"when": "day 2 09:30", "note": "visit the workshop"}}}
			}
			return nil
		}}
	}

	// Session 1 runs to Day 1 20:00 and is saved.
	w := newWorld(t, "mira", "rook")
	engine(t, w, script("mira"), script("rook")).Run(context.Background(), At(1, 20, 0))
	if err := store.Save(w); err != nil {
		t.Fatal(err)
	}
	pendingBefore := describeQueue(w.queue)

	// Session 2 is a different process: load from disk, new actors.
	w2, err := store.Load("test")
	if err != nil {
		t.Fatal(err)
	}
	if got := describeQueue(w2.queue); got != pendingBefore {
		t.Fatalf("queue changed across save/load:\n%s\nvs\n%s", pendingBefore, got)
	}
	if !w2.Now.Equal(At(1, 20, 0)) {
		t.Fatalf("resumed clock = %s", Stamp(w2.Now))
	}
	mira := script("mira")
	mira.decide = func(sim.Turn, int) []sim.Action { return nil }
	engine(t, w2, mira, script("rook")).Run(context.Background(), At(2, 12, 0))
	fired := false
	for _, turn := range mira.turns {
		if strings.Contains(why(turn), "visit the workshop") {
			fired = turn.Clock == "Day 2, 09:30 (morning)"
		}
	}
	if !fired {
		t.Fatal("a wakeup scheduled in session 1 should fire in session 2 at its time")
	}
}

func describeQueue(q *Queue) string {
	var b strings.Builder
	for _, it := range q.Sorted() {
		raw, _ := json.Marshal(it)
		b.Write(raw)
		b.WriteByte('\n')
	}
	return b.String()
}

func TestRunsAreDeterministic(t *testing.T) {
	run := func() string {
		w := newWorld(t, "mira", "rook", "ada")
		chatty := func(id sim.AgentID, other sim.AgentID) *scripted {
			return &scripted{id: id, decide: func(t sim.Turn, n int) []sim.Action {
				if n%2 == 0 {
					return []sim.Action{{Kind: sim.ActionWhisper, To: []sim.AgentID{other}, Content: "hi again"}}
				}
				return []sim.Action{do("go", map[string]any{"to": "square"})}
			}}
		}
		engine(t, w, chatty("mira", "rook"), chatty("rook", "ada"), chatty("ada", "mira")).Run(context.Background(), At(2, 7, 0))
		raw, _ := json.Marshal(w.History)
		return string(raw)
	}
	if a, b := run(), run(); a != b {
		t.Fatal("identical scripted runs produced different histories")
	}
}

func TestChattyAgentsAreBounded(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	ping := func(id, other sim.AgentID) *scripted {
		return &scripted{id: id, decide: func(sim.Turn, int) []sim.Action {
			return []sim.Action{{Kind: sim.ActionWhisper, To: []sim.AgentID{other}, Content: "ping"}}
		}}
	}
	mira, rook := ping("mira", "rook"), ping("rook", "mira")
	engine(t, w, mira, rook).Run(context.Background(), At(2, 7, 0))
	max := w.Settings.Scheduling.MaxActivationsPerDay
	for _, a := range []*scripted{mira, rook} {
		day1 := 0
		for _, turn := range a.turns {
			if strings.HasPrefix(turn.Clock, "Day 1,") {
				day1++
			}
		}
		if day1 > max {
			t.Errorf("%s activated %d times on Day 1; the cap is %d", a.id, day1, max)
		}
	}
	// Consecutive activations of one agent are at least the cooldown apart.
	var last time.Time
	for _, turn := range mira.turns {
		var d, h, m int
		fmtSscanf(turn.Clock, "Day %d, %d:%d", &d, &h, &m)
		at := At(d, h, m)
		if !last.IsZero() && at.Sub(last) < time.Duration(w.Settings.Scheduling.CooldownMinutes)*time.Minute {
			t.Fatalf("activations %s and %s violate the cooldown", Stamp(last), Stamp(at))
		}
		last = at
	}
}

func TestBudgetsStopASessionCleanly(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	e, err := NewEngine(Config{World: w, SessionID: "s", Agents: []sim.Actor{&scripted{id: "mira"}, &scripted{id: "rook"}},
		Limits: Limits{MaxLLMCalls: 3}, Sink: explog.Discard{}})
	if err != nil {
		t.Fatal(err)
	}
	res := e.Run(context.Background(), At(5, 0, 0))
	if res.LLMCalls != 3 || !strings.Contains(res.StopReason, "LLM call budget") {
		t.Fatalf("calls=%d stop=%q", res.LLMCalls, res.StopReason)
	}
	if w.Now.After(At(1, 18, 0)) {
		t.Errorf("a session stopped by a budget must not skip the clock ahead: now %s", Stamp(w.Now))
	}
}

func fmtSscanf(s, format string, args ...any) (int, error) { return fmt.Sscanf(s, format, args...) }

func TestReplyChainsDieOutInsteadOfLoopingForever(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Residents["mira"].Location = "square"
	w.Residents["rook"].Location = "square"
	// Both always answer the other by name, each time with something new
	// (a repeat would not re-trigger at all; that is tested separately): a
	// pure feedback loop only the reply-chain limit can end.
	reply := func(id, other string) *scripted {
		return &scripted{id: sim.AgentID(id), decide: func(_ sim.Turn, n int) []sim.Action {
			return []sim.Action{{Kind: sim.ActionSay, Content: fmt.Sprintf("%s, what do you think of offer number %d?", other, n)}}
		}}
	}
	mira, rook := reply("mira", "rook"), reply("rook", "mira")
	engine(t, w, mira, rook).Run(context.Background(), At(1, 9, 59))

	// By 10:00 nobody has had an idle opportunity since their first one, so
	// every activation after those is a reaction. The chain allows
	// MaxReplyChain of them, then the exchange rests.
	reactive := 0
	for _, a := range []*scripted{mira, rook} {
		for _, turn := range a.turns {
			if strings.Contains(why(turn), "spoke to you") && !strings.Contains(why(turn), "quiet moment") {
				reactive++
			}
		}
	}
	if max := w.Settings.Scheduling.MaxReplyChain; reactive > max {
		t.Fatalf("%d reactive activations; the reply chain should cap them at %d", reactive, max)
	}
	if w.Totals["reactions_damped"] == 0 {
		t.Error("the loop should have been damped at least once")
	}
	// Damped speech is still heard.
	if !mira.saw("Rook (rook) said: mira, what do you think of offer number") {
		t.Error("damped speech must still be delivered")
	}
}

func TestTwoPeopleCannotBuildTheSameLastHours(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Residents["mira"].Location = "square"
	w.Residents["rook"].Location = "square"
	w.Now = At(1, 9, 0)
	mustApply(t, w, "mira", do("start_project", map[string]any{"name": "bench", "labor": 2}))
	mustApply(t, w, "mira", do("build", map[string]any{"project": "p1"}))
	if err := w.Validate("rook", do("build", map[string]any{"project": "p1"})); err == nil {
		t.Fatal("the remaining labor is already underway; a second stint should be refused")
	}
	for _, s := range w.Actions("rook") {
		if s.Verb == "build" {
			t.Error("build should not be offered when the remaining labor is already underway")
		}
	}
}

func TestMeetingsCannotBeUsedToBypassTheReplyChain(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Now = At(1, 9, 0)
	mustApply(t, w, "mira", do("arrange", map[string]any{"with": "rook", "when": "in 10m", "what": "chat"}))
	if got := w.Appointments[0].At; !got.Equal(At(1, 10, 0)) {
		t.Errorf("a meeting 10 minutes out should be pushed to the 60-minute minimum, got %s", Stamp(got))
	}
	if err := w.Validate("rook", do("arrange", map[string]any{"with": "mira", "when": "in 3h", "what": "again"})); err == nil {
		t.Error("a second pending meeting between the same pair should be refused")
	}
}

func totalGoods(w *World) Goods {
	var g Goods
	for _, r := range w.Residents {
		g = g.Plus(r.Goods)
	}
	for _, p := range w.Projects {
		g = g.Plus(p.Raised)
	}
	return g
}

func TestTransfersConserveResources(t *testing.T) {
	w := newWorld(t, "mira", "rook", "ada")
	for _, r := range w.Residents {
		r.Location = "square"
	}
	w.Now = At(1, 9, 0)
	before := totalGoods(w)
	steps := []struct {
		who sim.AgentID
		a   sim.Action
	}{
		{"mira", do("give", map[string]any{"to": "rook", "credits": 3})},
		{"rook", do("offer", map[string]any{"to": "ada", "give_credits": 5, "want_materials": 1})},
		{"ada", do("accept", map[string]any{"offer": "o1"})},
		{"ada", do("start_project", map[string]any{"name": "well", "credits": 20, "materials": 3, "labor": 4})},
		{"rook", do("contribute", map[string]any{"project": "p1", "credits": 4, "materials": 1})},
		{"mira", do("contribute", map[string]any{"project": "well", "credits": 7})},
		{"mira", do("offer", map[string]any{"to": "rook", "give_materials": 1, "want_credits": 2})},
		{"mira", do("give", map[string]any{"to": "ada", "materials": 1})}, // mira now cannot honour o2
		{"rook", do("accept", map[string]any{"offer": "o2"})},             // refused at accept time
	}
	for i, s := range steps {
		if _, err := w.Apply(s.who, s.a); err != nil && i != len(steps)-1 {
			t.Fatalf("step %d (%s %s): %v", i, s.who, s.a.Verb, err)
		}
		if got := totalGoods(w); got != before {
			t.Fatalf("after step %d (%s %s) resources are %v, started with %v", i, s.who, s.a.Verb, got, before)
		}
		for id, r := range w.Residents {
			if r.Goods.Credits < 0 || r.Goods.Materials < 0 {
				t.Fatalf("%s went negative: %v", id, r.Goods)
			}
		}
	}
	if w.Totals["credits_given"] != 3 || w.Totals["credits_traded"] != 5 || w.Totals["credits_contributed"] != 11 {
		t.Errorf("transfer metrics: %v", w.Totals)
	}
	if w.Residents["mira"].Totals["credits_given_out"] != 3 || w.Residents["rook"].Totals["credits_given_in"] != 3 {
		t.Errorf("per-agent transfer metrics: mira %v rook %v", w.Residents["mira"].Totals, w.Residents["rook"].Totals)
	}
}

func TestNothingRunsTwiceAcrossAResume(t *testing.T) {
	store := Store{Root: t.TempDir()}
	w := newWorld(t, "mira")
	w.Residents["mira"].Location = "market"
	worker := func() *scripted {
		return &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
			if t.Clock == "Day 1, 07:00 (morning)" {
				return []sim.Action{do("work", nil)}
			}
			return nil
		}}
	}
	// Session 1 stops mid-shift.
	engine(t, w, worker()).Run(context.Background(), At(1, 9, 0))
	if w.Residents["mira"].Activity == nil {
		t.Fatal("mira should be mid-shift at the save")
	}
	if err := store.Save(w); err != nil {
		t.Fatal(err)
	}
	w2, err := store.Load("test")
	if err != nil {
		t.Fatal(err)
	}
	m2 := worker()
	engine(t, w2, m2).Run(context.Background(), At(1, 14, 0))
	if got := w2.Residents["mira"].Goods.Credits; got != 16 {
		t.Errorf("credits = %d: the shift should pay exactly once (10 + 6)", got)
	}
	finished := 0
	for _, h := range w2.History {
		if strings.Contains(h.Text, "finished a shift") {
			finished++
		}
	}
	if finished != 1 {
		t.Errorf("the shift finished %d times across the resume", finished)
	}
	if w2.Stock.Shifts["market"] != w2.Settings.Economy.ShiftsPerDay-1 {
		t.Errorf("shift stock = %d", w2.Stock.Shifts["market"])
	}
	for _, turn := range m2.turns {
		if turn.Clock == "Day 1, 07:00 (morning)" {
			t.Error("the 07:00 activation ran again after the resume")
		}
	}
	if w2.History[0].Seq != w.History[0].Seq || w2.EventSeq <= w.EventSeq {
		t.Error("event numbering should continue, not restart")
	}
}

func TestStartingKnowledgeIsDeliveredOnceAndPrivately(t *testing.T) {
	s := DefaultSettings()
	s.PerAgent = map[string]ResidentSpec{"mira": {Knows: []string{"the corner lot is empty"}, Start: "square"}}
	w, err := New("k", 1, s, []sim.AgentID{"mira", "rook"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if w.Residents["mira"].Location != "square" {
		t.Errorf("start location ignored")
	}
	mira, rook := &scripted{id: "mira"}, &scripted{id: "rook"}
	engine(t, w, mira, rook).Run(context.Background(), At(1, 8, 0))
	engine(t, w, mira, rook).Run(context.Background(), At(1, 9, 0)) // a second session
	n := 0
	for _, o := range mira.observed {
		if strings.Contains(o.Text, "the corner lot is empty") {
			n++
		}
	}
	if n != 1 {
		t.Errorf("knowledge delivered %d times", n)
	}
	if rook.saw("corner lot") {
		t.Error("one resident's knowledge leaked to another")
	}
}

func TestVerifiedEventsAreMarkedAndClaimsAreNot(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Residents["mira"].Location = "square"
	w.Residents["rook"].Location = "square"
	mira := &scripted{id: "mira", decide: func(t sim.Turn, n int) []sim.Action {
		if n == 0 {
			return []sim.Action{{Kind: sim.ActionSay, Content: "I gave you 10 credits."}, do("give", map[string]any{"to": "rook", "credits": 2})}
		}
		return nil
	}}
	rook := &scripted{id: "rook"}
	engine(t, w, mira, rook).Run(context.Background(), At(1, 7, 30))
	if !rook.saw("✔ Mira (mira) gave Rook 2 credits as a gift") {
		t.Error("the actual transfer should be marked verified")
	}
	if !rook.saw("Mira (mira) said: I gave you 10 credits.") || rook.saw("✔ Mira (mira) said") {
		t.Error("speech is a claim and must not be marked verified")
	}
	if w.Residents["rook"].Goods.Credits != 12 {
		t.Errorf("only the action moves credits: rook has %d", w.Residents["rook"].Goods.Credits)
	}
}

func TestRepeatedlyInvalidAgentsCoolOff(t *testing.T) {
	w := newWorld(t, "mira")
	mira := &scripted{id: "mira", decide: func(sim.Turn, int) []sim.Action {
		return []sim.Action{do("teleport", nil)}
	}}
	e := engine(t, w, mira)
	e.Run(context.Background(), At(1, 22, 0))
	if w.Totals["invalid_cooloffs"] == 0 {
		t.Fatal("an agent refused every time should be cooled off")
	}
	if len(mira.turns) > 8 {
		t.Errorf("%d activations of a consistently invalid agent in a day", len(mira.turns))
	}
}

func TestEventBudgetStopsTheSession(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Residents["mira"].Location = "square"
	w.Residents["rook"].Location = "square"
	talk := func(id sim.AgentID) *scripted {
		return &scripted{id: id, decide: func(sim.Turn, int) []sim.Action {
			return []sim.Action{{Kind: sim.ActionSay, Content: "hello"}}
		}}
	}
	e, err := NewEngine(Config{World: w, Agents: []sim.Actor{talk("mira"), talk("rook")}, Limits: Limits{MaxEvents: 3}})
	if err != nil {
		t.Fatal(err)
	}
	res := e.Run(context.Background(), At(3, 0, 0))
	if !strings.Contains(res.StopReason, "event budget") {
		t.Fatalf("stop = %q", res.StopReason)
	}
}

func TestOwnersCanCancelMeetingsAndWithdrawOffers(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	w.Now = At(1, 9, 0)
	mustApply(t, w, "mira", do("arrange", map[string]any{"with": "rook", "when": "15:00", "what": "talk"}))
	if err := w.Validate("rook", do("cancel_meeting", map[string]any{"meeting": "a1"})); err == nil {
		t.Error("only the arranger can cancel")
	}
	pending := w.queue.Len()
	eff := mustApply(t, w, "mira", do("cancel_meeting", map[string]any{"meeting": "a1"}))
	if w.Appointments[0].Status != "cancelled" || w.queue.Len() != pending-1 {
		t.Error("cancelling should remove the scheduled reminder")
	}
	if len(eff.Triggers) != 1 || eff.Triggers[0].Agent != "rook" {
		t.Error("the invitee should be told")
	}
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "rook", "give_credits": 1}))
	mustApply(t, w, "mira", do("withdraw", map[string]any{"offer": "o1"}))
	if err := w.Validate("rook", do("accept", map[string]any{"offer": "o1"})); err == nil {
		t.Error("a withdrawn offer cannot be accepted")
	}
	if err := w.Validate("mira", do("wait", nil)); err != nil {
		t.Errorf("wait should be a valid way to do nothing: %v", err)
	}
}
