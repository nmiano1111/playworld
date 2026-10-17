package world

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/orchestrator"
	"github.com/nmiano/playworld/internal/sim"
)

// Limits bound one execution session. The simulated window is the primary
// bound; these are the safety nets that keep a session from consuming the
// machine if agents get chatty.
type Limits struct {
	// MaxLLMCalls caps inference calls in this session. Zero disables.
	MaxLLMCalls int
	// MaxActivations caps agent activations in this session. Zero disables.
	MaxActivations int
	// Timeout bounds wall-clock duration. Zero disables.
	Timeout time.Duration
	// MaxAgentErrors stops the session once one agent fails this many
	// activations in a row.
	MaxAgentErrors int
	// MaxEvents caps events delivered in this session. Zero disables.
	MaxEvents int
}

// Config wires an Engine.
type Config struct {
	World *World
	// SessionID names this session in logs.
	SessionID string
	// RunID is what agent-facing turns carry; it is stable across sessions
	// so an agent's plan and memory scope survive a resume.
	RunID  string
	Agents []sim.Actor
	Sink   explog.Sink
	Limits Limits
	// Checkpoint, if set, is called after every activation so a session
	// interrupted at any point loses at most the activation in flight. The
	// runner uses it to save the world and the agents' memory.
	Checkpoint func() error
	// Clock is wall time for log records. Defaults to time.Now.
	Clock func() time.Time
	// Introspect logs, at every activation, the agent's view of the world
	// alongside a privileged snapshot for the experimenter (RecView).
	Introspect bool
}

// Result reports what a session did.
type Result struct {
	From, To    time.Time
	StopReason  string
	Activations int
	LLMCalls    int
	// Metrics are this session's counters (world totals accrued during it,
	// plus activation-reason and action-verb breakdowns).
	Metrics map[string]int
	// PerAgent breaks the same counters down by resident.
	PerAgent map[sim.AgentID]map[string]int
	Duration time.Duration
	// Err is a fatal error that ended the session early, if any.
	Err error
}

// Engine drives agents through a World by processing its event queue in
// simulated-time order. Inference happens only for KindActivate items and is
// strictly sequential: one model call in flight at a time, whatever the
// logical concurrency of the world.
type Engine struct {
	cfg    Config
	w      *World
	sink   explog.Sink
	clock  func() time.Time
	actors map[sim.AgentID]sim.Actor
	names  map[sim.AgentID]string

	activations int
	events      int
	errors      map[sim.AgentID]int
	metrics     map[string]int
	perAgent    map[sim.AgentID]map[string]int
	baseCalls   int
}

// NewEngine validates the cast against the world's residents.
func NewEngine(cfg Config) (*Engine, error) {
	if cfg.World == nil {
		return nil, fmt.Errorf("world engine: no world")
	}
	if cfg.Sink == nil {
		cfg.Sink = explog.Discard{}
	}
	if cfg.Clock == nil {
		cfg.Clock = time.Now
	}
	e := &Engine{
		cfg: cfg, w: cfg.World, sink: cfg.Sink, clock: cfg.Clock,
		actors: map[sim.AgentID]sim.Actor{}, names: cfg.World.Names(),
		errors: map[sim.AgentID]int{}, metrics: map[string]int{},
		perAgent: map[sim.AgentID]map[string]int{},
	}
	for _, a := range cfg.Agents {
		if _, ok := e.w.Residents[a.ID()]; !ok {
			return nil, fmt.Errorf("world engine: %s is not a resident of %s", a.ID(), e.w.Name)
		}
		e.actors[a.ID()] = a
	}
	for _, id := range e.w.Order {
		if _, ok := e.actors[id]; !ok {
			return nil, fmt.Errorf("world engine: resident %s has no agent", id)
		}
	}
	e.baseCalls = e.countCalls()
	return e, nil
}

// Run processes events until simulated time reaches until, a limit is hit,
// or ctx is cancelled. The clock jumps from event to event; nothing sleeps.
// When the window is reached the world clock is set to until exactly, so the
// next session starts where this one ended.
func (e *Engine) Run(ctx context.Context, until time.Time) *Result {
	start := e.clock()
	if t := e.cfg.Limits.Timeout; t > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, t)
		defer cancel()
	}
	res := &Result{From: e.w.Now}
	before := copyCounts(e.w.Totals)
	beforeAgent := map[sim.AgentID]map[string]int{}
	for _, id := range e.w.Order {
		beforeAgent[id] = copyCounts(e.w.Residents[id].Totals)
	}
	e.brief()

	for {
		if err := ctx.Err(); err != nil {
			res.StopReason = ctxReason(err)
			break
		}
		next := e.w.queue.Peek()
		if next == nil {
			res.StopReason = "nothing left scheduled"
			e.w.Now = until
			break
		}
		if next.At.After(until) {
			res.StopReason = "reached " + Stamp(until)
			e.w.Now = until
			break
		}
		if reason, hit := e.limitHit(next); hit {
			res.StopReason = reason
			break
		}
		it := e.w.queue.Pop()
		if it.At.After(e.w.Now) {
			e.w.Now = it.At
		}
		if it.Kind != KindActivate {
			e.apply(e.w.Process(it))
			continue
		}
		if !e.w.BeginActivation(it) {
			continue
		}
		if err := e.activate(ctx, it); err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				res.StopReason = ctxReason(ctxErr)
				break
			}
			var u *sim.Unrecoverable
			if errors.As(err, &u) {
				res.Err = err
				res.StopReason = "fatal error: " + err.Error()
				break
			}
		}
		if e.cfg.Checkpoint != nil {
			if err := e.cfg.Checkpoint(); err != nil {
				res.Err = err
				res.StopReason = "could not save: " + err.Error()
				break
			}
		}
	}

	res.To = e.w.Now
	res.Activations = e.activations
	res.LLMCalls = e.countCalls() - e.baseCalls
	res.Duration = e.clock().Sub(start)
	res.Metrics = map[string]int{}
	for k, v := range e.w.Totals {
		if d := v - before[k]; d != 0 {
			res.Metrics[k] = d
		}
	}
	for k, v := range e.metrics {
		res.Metrics[k] = v
	}
	res.Metrics["llm_calls"] = res.LLMCalls
	res.Metrics["events"] = e.events
	res.PerAgent = map[sim.AgentID]map[string]int{}
	for _, id := range e.w.Order {
		m := map[string]int{}
		for k, v := range e.w.Residents[id].Totals {
			if d := v - beforeAgent[id][k]; d != 0 {
				m[k] = d
			}
		}
		for k, v := range e.perAgent[id] {
			m[k] = v
		}
		res.PerAgent[id] = m
	}
	return res
}

// brief delivers each resident's configured starting knowledge, once in
// the world's life, as private observations.
func (e *Engine) brief() {
	for _, id := range e.w.Order {
		r := e.w.Residents[id]
		if r.Briefed {
			continue
		}
		var evs []sim.Event
		for _, k := range r.Knows {
			evs = append(evs, sim.Event{Type: sim.EventSystem, Text: "You know: " + k, Visibility: sim.Only(id)})
		}
		e.deliver(evs)
		r.Briefed = true
	}
}

func (e *Engine) bump(id sim.AgentID, key string) {
	e.metrics[key]++
	m := e.perAgent[id]
	if m == nil {
		m = map[string]int{}
		e.perAgent[id] = m
	}
	m[key]++
}

// limitHit checks the safety nets before an item that would cost inference.
// Non-activation items are always processed: they are free, and leaving them
// half-done would make the world inconsistent.
func (e *Engine) limitHit(next *Item) (string, bool) {
	if next.Kind != KindActivate {
		return "", false
	}
	l := e.cfg.Limits
	if l.MaxActivations > 0 && e.activations >= l.MaxActivations {
		return fmt.Sprintf("reached the activation budget (%d)", l.MaxActivations), true
	}
	if l.MaxLLMCalls > 0 && e.countCalls()-e.baseCalls >= l.MaxLLMCalls {
		return fmt.Sprintf("reached the LLM call budget (%d)", l.MaxLLMCalls), true
	}
	if l.MaxEvents > 0 && e.events >= l.MaxEvents {
		return fmt.Sprintf("reached the event budget (%d)", l.MaxEvents), true
	}
	return "", false
}

// slotOf classifies a proposed action into the three independent parts of
// one decision: what the agent said, what it did, and what it scheduled.
func slotOf(a sim.Action) string {
	switch a.Kind {
	case sim.ActionSay, sim.ActionWhisper:
		return "communication"
	case sim.ActionIntend:
		return "intention"
	}
	return "world"
}

// activate gives one agent its moment to decide, and executes what it chose.
func (e *Engine) activate(ctx context.Context, it *Item) error {
	id := it.Agent
	actor := e.actors[id]
	r := e.w.Residents[id]
	e.activations++
	e.bump(id, "activations")
	for _, rs := range it.Reasons {
		e.bump(id, "activated_by_"+rs.Kind)
	}
	texts := make([]string, 0, len(it.Reasons))
	idleOnly := true
	for _, rs := range it.Reasons {
		texts = append(texts, rs.Text)
		idleOnly = idleOnly && rs.Kind == "idle"
	}
	rec := e.record(explog.RecActivation)
	rec.Activation = &explog.Activation{Agent: id, Index: e.activations, SimTime: e.w.Now, Stamp: Stamp(e.w.Now), Reasons: texts}
	e.sink.Emit(rec)

	e.w.Look(id)
	turn := e.w.TurnFor(e.cfg.RunID, id, it.Reasons)
	turn.Round = e.activations
	if e.cfg.Introspect {
		view := e.record(explog.RecView)
		view.View = &explog.View{Agent: id, Situation: turn.Situation, Actions: renderOptions(turn.Actions), Privileged: e.w.privileged()}
		if turn.Situation == "" {
			view.View.Situation = legacyView(turn)
		}
		e.sink.Emit(view)
	}

	proposed, err := decide(ctx, actor, turn)
	if err != nil {
		e.w.EndActivation(id, false)
		e.errors[id]++
		e.bump(id, "decide_errors")
		fail := e.record(explog.RecError)
		fail.Failure = &explog.Failure{Agent: id, Stage: "decide", Err: err.Error()}
		e.sink.Emit(fail)
		if max := e.cfg.Limits.MaxAgentErrors; max > 0 && e.errors[id] >= max {
			return sim.Fatal(fmt.Errorf("agent %s failed %d activations in a row: %w", id, e.errors[id], err))
		}
		return err
	}
	e.errors[id] = 0
	// A pass alongside something the agent did is not a decision of its
	// own, just an empty world slot; only a pass on its own is recorded.
	var acted []sim.Action
	for _, a := range proposed {
		if a.Kind != sim.ActionPass {
			acted = append(acted, a)
		}
	}
	if len(acted) == 0 {
		acted = []sim.Action{{Kind: sim.ActionPass}}
	}
	proposed = acted

	e.w.SetChain(chainDepth(it.Reasons))
	defer e.w.SetChain(0)

	// After a setback, did the agent try the same kind of move again or
	// something else? Counted once, on the next decision.
	if failed := r.AfterSetback; failed != "" {
		r.AfterSetback = ""
		same := false
		for _, a := range proposed {
			same = same || moveLabel(a) == failed
		}
		if same {
			e.bump(id, "after_setback_same")
		} else {
			e.bump(id, "after_setback_changed")
		}
	}

	used := map[string]int{}
	rejected := false
	for _, action := range proposed {
		slot := slotOf(action)
		limit := 1
		if slot == "intention" {
			limit = e.w.Settings.Scheduling.MaxIntentionsPerActivation
		}
		if used[slot] >= limit {
			continue
		}
		used[slot]++

		valErr := e.w.Validate(id, action)
		dec := e.record(explog.RecDecision)
		dec.Decision = &explog.Decision{Agent: id, Action: action, Accepted: valErr == nil, Slot: slot}
		e.sink.Emit(dec)
		if valErr != nil {
			rejected = true
			e.bump(id, "rejected")
			e.bump(id, "rejected_"+RefusalCode(valErr))
			if offered(turn.Actions, action) {
				e.bump(id, "rejected_offered")
			} else {
				e.bump(id, "rejected_unoffered")
			}
			if e.w.NoteRefusal(id, describe(action), valErr) {
				e.bump(id, "repeat_refusals")
			}
			r.AfterSetback = moveLabel(action)
			rej := e.record(explog.RecRejected)
			rej.Rejected = &explog.Rejection{Agent: id, Action: action, Reason: valErr.Error(), Code: RefusalCode(valErr), Slot: slot}
			e.sink.Emit(rej)
			e.deliver([]sim.Event{{Type: sim.EventRejected, Actor: id,
				Text: fmt.Sprintf("Your %s was refused: %s", describe(action), valErr.Error()), Visibility: sim.Only(id)}})
			continue
		}
		eff, err := e.w.Apply(id, action)
		if err != nil {
			fail := e.record(explog.RecError)
			fail.Failure = &explog.Failure{Agent: id, Stage: "apply", Err: err.Error()}
			e.sink.Emit(fail)
			continue
		}
		e.bump(id, "action_"+actionKey(action))
		e.apply(eff)
	}

	quiet := idleOnly && !rejected && len(proposed) == 1 && proposed[0].Kind == sim.ActionPass
	e.w.EndActivation(id, quiet)

	if rejected {
		r.RejectStreak++
		sc := e.w.Settings.Scheduling
		switch {
		case sc.MaxConsecutiveInvalid > 0 && r.RejectStreak >= sc.MaxConsecutiveInvalid:
			next := e.w.CoolOff(id)
			e.deliver([]sim.Event{{Type: sim.EventSystem, Actor: id, Visibility: sim.Only(),
				Text: fmt.Sprintf("%s: %d activations in a row ended in a refused action; next chance deferred to %s", id, r.RejectStreak, Stamp(next))}})
			r.RejectStreak = 0
		case r.RejectStreak <= sc.MaxRejectRetriggers:
			// Same depth as this activation: a refusal is a chance to
			// correct course, not a fresh stimulus that resets the chain.
			e.schedule(e.w.Trigger(Trigger{Agent: id, At: e.w.Now, Reason: Reason{Kind: "refused", Text: "your last action was refused", Depth: chainDepth(it.Reasons)}}),
				Trigger{Agent: id, Reason: Reason{Kind: "refused", Text: "your last action was refused"}})
		}
	} else {
		r.RejectStreak = 0
	}
	return nil
}

// TurnFor assembles what id is allowed to know when it decides now, with
// reasons saying why. Grounded worlds describe the situation as ordered text;
// legacy worlds use the original key/value state.
func (w *World) TurnFor(runID string, id sim.AgentID, reasons []Reason) sim.Turn {
	t := sim.Turn{
		RunID: runID, Scenario: "world",
		Self: id, Participants: w.Order,
		Brief:   w.Brief(id),
		Actions: w.Actions(id),
		Clock:   Clock(w.Now),
		Now:     w.Now,
	}
	if w.Settings.Grounded() {
		t.Situation = w.Situation(id, reasons)
	} else {
		t.PublicState = w.PublicState()
		t.PrivateState = w.PrivateState(id, reasons)
	}
	return t
}

// PendingReasons are the reasons id's next activation will carry, if one is
// queued (for previewing what the agent would see).
func (w *World) PendingReasons(id sim.AgentID) []Reason {
	r := w.Residents[id]
	if r == nil {
		return nil
	}
	if it := w.queue.Get(r.ActivateItem); it != nil {
		return it.Reasons
	}
	return nil
}

// privileged is the experimenter's one-line truth for an introspection
// record: everyone's location and holdings. Agents never see it.
func (w *World) privileged() string {
	var parts []string
	for _, id := range w.Order {
		r := w.Residents[id]
		where := r.Location
		if where == "" {
			where = "on the road"
		}
		parts = append(parts, fmt.Sprintf("%s@%s %s", id, where, r.Goods))
	}
	return strings.Join(parts, " | ")
}

// legacyView renders a legacy turn's key/value state for introspection.
func legacyView(t sim.Turn) string {
	var b strings.Builder
	for _, m := range []map[string]any{t.PublicState, t.PrivateState} {
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(&b, "%s: %v\n", k, m[k])
		}
	}
	return b.String()
}

func renderOptions(specs []sim.ActionSpec) []string {
	var out []string
	for _, s := range specs {
		label := string(s.Kind)
		if s.Kind == sim.ActionDo {
			label = s.Verb
		}
		out = append(out, label)
	}
	return out
}

func decide(ctx context.Context, actor sim.Actor, t sim.Turn) ([]sim.Action, error) {
	if md, ok := actor.(sim.MultiDecider); ok {
		return md.DecideMany(ctx, t)
	}
	a, err := actor.Decide(ctx, t)
	if err != nil {
		return nil, err
	}
	return []sim.Action{a}, nil
}

// apply delivers an effect's events, then its triggers, in that order: an
// agent woken by an event must already have observed it.
func (e *Engine) apply(eff Effects) {
	e.deliver(eff.Events)
	for _, t := range eff.Triggers {
		e.schedule(e.w.Trigger(t), t)
	}
}

// schedule logs what a trigger did, so "Rook will decide at 09:35" and
// damped reactions are visible in the trace.
func (e *Engine) schedule(res TriggerResult, t Trigger) {
	rec := e.record(explog.RecScheduled)
	rec.Scheduled = &explog.Scheduled{Agent: t.Agent, At: res.At, Reason: t.Reason.Text, Kind: t.Reason.Kind,
		Depth: t.Reason.Depth, Damped: res.Damped, Coalesced: res.Coalesced}
	if !res.At.IsZero() {
		rec.Scheduled.Stamp = Stamp(res.At)
	}
	e.sink.Emit(rec)
}

// deliver stamps events, records them, and routes each to exactly the
// residents its visibility permits — the single enforcement point for what an
// agent can know, as in the round-based orchestrator. Delivering an event is
// free: it lands in the agent's transcript and memory, and the agent reads it
// at its next activation. Only a Trigger can cause one.
func (e *Engine) deliver(events []sim.Event) {
	for _, ev := range events {
		e.w.EventSeq++
		ev.Seq = e.w.EventSeq
		ev.Round = e.activations
		if ev.Time.IsZero() {
			ev.Time = e.w.Now
		}
		e.w.record(ev)
		e.events++
		rec := e.record(explog.RecEvent)
		rec.Event = &ev
		e.sink.Emit(rec)

		stamp := Stamp(ev.Time)
		for _, id := range e.w.Order {
			if !ev.Visibility.CanObserve(id) {
				continue
			}
			text := orchestrator.RenderFor(id, ev, e.names)
			if verified(ev.Type) {
				text = verifiedMark + text
			}
			e.actors[id].Observe(sim.Observation{
				Round: ev.Round, Seq: ev.Seq, Time: ev.Time, Type: ev.Type,
				Private: !ev.Visibility.Public, Stamp: stamp, Text: text, Importance: ev.Importance,
			})
		}
	}
}

func (e *Engine) record(t explog.RecordType) explog.Record {
	return explog.Record{Type: t, RunID: e.cfg.SessionID, Time: e.clock(), Round: e.activations, SimStamp: Stamp(e.w.Now)}
}

// verifiedMark prefixes, in an agent's own view, events the world itself
// confirms happened — an action taking effect, a state change, world news —
// so they read differently from what someone merely said or wrote, which
// stays a claim however confidently it is put.
const verifiedMark = "✔ "

func verified(t sim.EventType) bool {
	return t == sim.EventAction || t == sim.EventState || t == sim.EventSystem
}

func (e *Engine) countCalls() int {
	total := 0
	for _, a := range e.actors {
		if m, ok := a.(sim.Metered); ok {
			total += m.CallCount()
		}
	}
	return total
}

// Outcome summarises a session for the log and terminal. There is no score:
// an open-ended world has no winner, only what happened.
func (w *World) Outcome(res *Result) sim.Outcome {
	m := map[string]any{}
	keys := make([]string, 0, len(res.Metrics))
	for k := range res.Metrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		m[k] = res.Metrics[k]
	}
	m["sim_hours"] = fmt.Sprintf("%.1f", res.To.Sub(res.From).Hours())
	if len(res.PerAgent) > 0 {
		m["per_agent"] = res.PerAgent
	}
	if h := res.To.Sub(res.From).Hours(); h > 0 {
		m["llm_calls_per_sim_day"] = fmt.Sprintf("%.1f", float64(res.LLMCalls)*24/h)
	}
	open, done := 0, 0
	for _, p := range w.Projects {
		if p.Done() {
			done++
		} else {
			open++
		}
	}
	summary := fmt.Sprintf("%s → %s: %d activations, %d LLM calls. The town now has %d completed and %d open project(s); %d events are scheduled ahead.",
		Stamp(res.From), Stamp(res.To), res.Activations, res.LLMCalls, done, open, w.queue.Len())
	return sim.Outcome{Summary: summary, Metrics: m}
}

func describe(a sim.Action) string {
	switch a.Kind {
	case sim.ActionDo:
		return `"` + a.Verb + `"`
	case sim.ActionIntend:
		return "intention"
	}
	return string(a.Kind)
}

func actionKey(a sim.Action) string {
	if a.Kind == sim.ActionDo {
		return a.Verb
	}
	if a.Kind == sim.ActionIntend {
		if _, ok := a.Args["cancel"]; ok {
			return "cancel_intention"
		}
	}
	return strings.ToLower(string(a.Kind))
}

func copyCounts(m map[string]int) map[string]int {
	out := make(map[string]int, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func ctxReason(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return "hit the wall-clock timeout"
	}
	return "cancelled"
}

// moveLabel names the kind of move an action is: its verb for world
// actions, otherwise its kind (say, whisper, pass).
func moveLabel(a sim.Action) string {
	if a.Kind == sim.ActionDo {
		return a.Verb
	}
	return string(a.Kind)
}

// offered reports whether a's kind (and verb) was among the actions the
// agent was shown as available this turn — telling a refusal the agent was
// warned about from one the affordances did not foresee.
func offered(specs []sim.ActionSpec, a sim.Action) bool {
	for _, s := range specs {
		if s.Kind == a.Kind && (a.Kind != sim.ActionDo || s.Verb == a.Verb) {
			return true
		}
	}
	return false
}
