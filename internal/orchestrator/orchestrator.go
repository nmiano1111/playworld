// Package orchestrator drives a scenario: it takes turns, routes events to the
// agents permitted to observe them, enforces run limits, and records
// everything. It knows nothing about any particular LLM provider or
// environment — only the sim.Actor and sim.Environment interfaces.
package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/sim"
)

// Limits bound a run so an experiment cannot consume the machine. Every limit
// is checked between turns, so a run always stops at a clean boundary.
type Limits struct {
	// MaxRounds is the hard cap on simulation rounds.
	MaxRounds int
	// MaxLLMCalls caps total inference calls across all agents. Zero disables.
	MaxLLMCalls int
	// MaxActions caps total executed actions. Zero disables.
	MaxActions int
	// Timeout bounds wall-clock duration. Zero disables.
	Timeout time.Duration
	// MaxConsecutivePasses stops a run once this many turns in a row are
	// passes or failures, which is how a dead conversation ends early rather
	// than burning the whole round budget.
	MaxConsecutivePasses int
	// MaxAgentErrors stops the run once one agent has failed this many times.
	MaxAgentErrors int
}

// DefaultLimits are deliberately conservative for a laptop.
func DefaultLimits() Limits {
	return Limits{
		MaxRounds:            6,
		MaxLLMCalls:          120,
		MaxActions:           400,
		Timeout:              20 * time.Minute,
		MaxConsecutivePasses: 8,
		MaxAgentErrors:       3,
	}
}

// Config configures a run.
type Config struct {
	RunID  string
	Env    sim.Environment
	Agents []sim.Actor
	Limits Limits
	Sink   explog.Sink
	Seed   int64
	// Shuffle randomises turn order each round using Seed. Off by default so
	// that scheduling is deterministic and runs are comparable.
	Shuffle bool
	// Clock is injectable for tests. Defaults to time.Now.
	Clock func() time.Time
}

// Result is what a finished run reports back to the caller.
type Result struct {
	RunID      string
	Rounds     int
	StopReason string
	Outcome    sim.Outcome
	LLMCalls   int
	Actions    map[string]int
	Rejections int
	Duration   time.Duration
	// Err is a fatal error that ended the run, if any.
	Err error
}

// Orchestrator coordinates agents and an environment.
type Orchestrator struct {
	cfg   Config
	env   sim.Environment
	sink  explog.Sink
	clock func() time.Time

	actors map[sim.AgentID]sim.Actor
	order  []sim.AgentID
	names  map[sim.AgentID]string

	// eventSeq numbers events in the world's history. Log-line numbering is a
	// separate concern, handled by explog.Sequencer.
	eventSeq   int
	round      int
	llmCalls   int
	actions    map[string]int
	rejections int
	errors     map[sim.AgentID]int
	passStreak int
}

// New builds an orchestrator.
func New(cfg Config) (*Orchestrator, error) {
	if cfg.Env == nil {
		return nil, fmt.Errorf("orchestrator: no environment")
	}
	if len(cfg.Agents) == 0 {
		return nil, fmt.Errorf("orchestrator: no agents")
	}
	if cfg.Sink == nil {
		cfg.Sink = explog.Discard{}
	}
	if cfg.Clock == nil {
		cfg.Clock = time.Now
	}
	if cfg.Limits.MaxRounds <= 0 {
		cfg.Limits.MaxRounds = DefaultLimits().MaxRounds
	}

	o := &Orchestrator{
		cfg:     cfg,
		env:     cfg.Env,
		sink:    cfg.Sink,
		clock:   cfg.Clock,
		actors:  map[sim.AgentID]sim.Actor{},
		names:   map[sim.AgentID]string{},
		actions: map[string]int{},
		errors:  map[sim.AgentID]int{},
	}
	for _, a := range cfg.Agents {
		if _, dup := o.actors[a.ID()]; dup {
			return nil, fmt.Errorf("orchestrator: duplicate agent id %q", a.ID())
		}
		o.actors[a.ID()] = a
		o.order = append(o.order, a.ID())
		o.names[a.ID()] = a.Name()
	}
	return o, nil
}

// Names maps agent ids to display names.
func (o *Orchestrator) Names() map[sim.AgentID]string { return o.names }

// Run executes the simulation until the environment terminates, a limit is
// reached, or the context is cancelled. A cancelled or limited run is a normal
// outcome, not an error: the caller still gets a Result and an Outcome.
func (o *Orchestrator) Run(ctx context.Context) (*Result, error) {
	start := o.clock()
	if o.cfg.Limits.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.cfg.Limits.Timeout)
		defer cancel()
	}

	setup, err := o.env.Setup(o.order)
	if err != nil {
		return nil, fmt.Errorf("orchestrator: environment setup: %w", err)
	}
	o.deliver(setup)

	rng := rand.New(rand.NewSource(o.cfg.Seed))
	stop := ""
	var fatal error

rounds:
	for round := 1; round <= o.cfg.Limits.MaxRounds; round++ {
		o.round = round
		o.sink.Emit(o.record(explog.RecRoundStart))

		o.deliver(o.env.BeginRound(round))
		if done, why := o.env.Done(); done {
			stop = why
			break
		}

		turnOrder := append([]sim.AgentID(nil), o.order...)
		if o.cfg.Shuffle {
			rng.Shuffle(len(turnOrder), func(i, j int) {
				turnOrder[i], turnOrder[j] = turnOrder[j], turnOrder[i]
			})
		}

		for _, id := range turnOrder {
			if err := ctx.Err(); err != nil {
				stop = stopReasonForCtx(err)
				break rounds
			}
			if reason, hit := o.limitHit(); hit {
				stop = reason
				break rounds
			}
			// An agent may be removed from play by the environment mid-round.
			if !contains(o.env.Participants(), id) {
				continue
			}

			if err := o.takeTurn(ctx, id); err != nil {
				// A cancelled or timed-out context is a normal way for a run
				// to end, so report it as such rather than as a failure.
				if ctxErr := ctx.Err(); ctxErr != nil {
					stop = stopReasonForCtx(ctxErr)
					break rounds
				}
				if isFatal(err) {
					fatal = err
					stop = "fatal error: " + err.Error()
					break rounds
				}
			}

			if done, why := o.env.Done(); done {
				stop = why
				break rounds
			}
			if o.cfg.Limits.MaxConsecutivePasses > 0 && o.passStreak >= o.cfg.Limits.MaxConsecutivePasses {
				stop = fmt.Sprintf("stalled: %d turns in a row with no action", o.passStreak)
				break rounds
			}
		}

		o.deliver(o.env.EndRound(round))
		if done, why := o.env.Done(); done {
			stop = why
			break
		}
	}

	if stop == "" {
		if err := ctx.Err(); err != nil {
			stop = stopReasonForCtx(err)
		} else {
			stop = fmt.Sprintf("reached the round limit (%d)", o.cfg.Limits.MaxRounds)
		}
	}

	outcome := o.env.Outcome()
	// The outcome is public knowledge: every agent learns how it ended.
	o.deliver([]sim.Event{{
		Type:       sim.EventOutcome,
		Text:       outcome.Summary,
		Data:       outcome.Metrics,
		Visibility: sim.PublicTo(),
	}})
	rec := o.record(explog.RecOutcome)
	rec.Outcome = &outcome
	o.sink.Emit(rec)

	return &Result{
		RunID:      o.cfg.RunID,
		Rounds:     o.round,
		StopReason: stop,
		Outcome:    outcome,
		LLMCalls:   o.countCalls(),
		Actions:    o.actions,
		Rejections: o.rejections,
		Duration:   o.clock().Sub(start),
		Err:        fatal,
	}, nil
}

// maxActionRetries bounds how many times a rejected action may be re-proposed
// within a single turn. One retry matters because some phases give an agent a
// single turn: the environment's complaint has to reach it in time to be acted
// on, not a round later when the chance has gone.
const maxActionRetries = 1

// slot classifies a proposed action as the communication half of a turn or
// the world-action half. An agent using the original single-action protocol
// only ever proposes one action per turn, so exactly one slot is ever used —
// this is what makes the single-slot and dual-slot cases the same code path.
type slot int

const (
	slotComm slot = iota
	slotWorld
)

func slotOf(a sim.Action) slot {
	if a.Kind == sim.ActionSay || a.Kind == sim.ActionWhisper {
		return slotComm
	}
	return slotWorld
}

func (s slot) String() string {
	if s == slotComm {
		return "communication"
	}
	return "world"
}

// decideActions gets this turn's proposed actions from an actor, dispatching
// through sim.MultiDecider when the actor implements it (every *agent.Agent
// does, regardless of whether it is configured for dual-action mode — see
// Agent.DecideMany) and otherwise falling back to the plain sim.Actor.Decide
// method unchanged, wrapped as a single-element slice. This is the one seam
// between the two protocols; everything below it operates on a slice of 0-2
// actions whichever path produced them.
func decideActions(ctx context.Context, actor sim.Actor, t sim.Turn) ([]sim.Action, error) {
	if md, ok := actor.(sim.MultiDecider); ok {
		return md.DecideMany(ctx, t)
	}
	a, err := actor.Decide(ctx, t)
	if err != nil {
		return nil, err
	}
	return []sim.Action{a}, nil
}

// takeTurn gives one agent its turn: 0, 1, or 2 proposed actions (one
// communication, one world action — a single-action agent only ever proposes
// one), each validated and applied independently. A slot rejected on the
// first attempt gets exactly one retry, scoped to that slot alone: a slot
// that already succeeded is never reconsidered, even if a retry response
// mentions it again, and a slot that runs out of retries simply does not
// happen this turn. Errors from the agent's own inference (as opposed to an
// environment rejection) are not retried and end the turn immediately, same
// as before.
func (o *Orchestrator) takeTurn(ctx context.Context, id sim.AgentID) error {
	actor := o.actors[id]

	accepted := map[slot]bool{}
	appliedAny := false
	appliedNonPass := false

	for attempt := 0; attempt <= maxActionRetries; attempt++ {
		turn := sim.Turn{
			RunID:        o.cfg.RunID,
			Scenario:     o.env.Name(),
			Round:        o.round,
			MaxRound:     o.cfg.Limits.MaxRounds,
			Self:         id,
			Participants: o.env.Participants(),
			Brief:        o.env.Brief(id),
			Actions:      o.env.Actions(id),
			PublicState:  o.env.PublicState(),
			PrivateState: o.env.PrivateState(id),
		}

		proposed, err := decideActions(ctx, actor, turn)
		if err != nil {
			o.errors[id]++
			o.passStreak++
			rec := o.record(explog.RecError)
			rec.Failure = &explog.Failure{Agent: id, Stage: "decide", Err: err.Error()}
			o.sink.Emit(rec)
			if o.cfg.Limits.MaxAgentErrors > 0 && o.errors[id] >= o.cfg.Limits.MaxAgentErrors {
				return sim.Fatal(fmt.Errorf("agent %s failed %d consecutive times: %w", id, o.errors[id], err))
			}
			return err
		}
		o.errors[id] = 0

		if len(proposed) == 0 {
			// Both slots null (or a single-action agent passing outright):
			// an explicit choice to do nothing, not a failure to produce one.
			proposed = []sim.Action{{Kind: sim.ActionPass}}
		}

		anyRejectedThisAttempt := false
		for _, action := range proposed {
			s := slotOf(action)
			if accepted[s] {
				// This slot already succeeded in an earlier attempt within
				// this same turn; a retry response touching it again (e.g.
				// repeating or changing what it said) is not re-applied, so
				// one turn can never execute more than one world action.
				continue
			}

			valErr := o.env.Validate(id, action)
			dec := o.record(explog.RecDecision)
			dec.Decision = &explog.Decision{Agent: id, Action: action, Accepted: valErr == nil, Attempt: attempt, Slot: s.String()}
			o.sink.Emit(dec)

			if valErr != nil {
				o.rejections++
				anyRejectedThisAttempt = true
				willRetry := attempt < maxActionRetries
				rec := o.record(explog.RecRejected)
				rec.Rejected = &explog.Rejection{Agent: id, Action: action, Reason: valErr.Error(), Attempt: attempt, Slot: s.String(), Retried: willRetry}
				o.sink.Emit(rec)

				msg := fmt.Sprintf("Your %s action was refused: %s", s, valErr.Error())
				if other := otherSlot(s); accepted[other] {
					msg += fmt.Sprintf(" Your %s already went through this turn — leave it out (null) and only resend this one.", other)
				}
				// Tell the agent privately why it failed. Nobody else sees
				// that it tried something illegal.
				o.deliver([]sim.Event{{
					Type:       sim.EventRejected,
					Actor:      id,
					Text:       msg,
					Visibility: sim.Only(id),
				}})
				continue
			}

			events, err := o.env.Apply(id, action)
			if err != nil {
				rec := o.record(explog.RecError)
				rec.Failure = &explog.Failure{Agent: id, Stage: "apply", Err: err.Error()}
				o.sink.Emit(rec)
				return err
			}

			accepted[s] = true
			appliedAny = true
			if action.Kind != sim.ActionPass {
				appliedNonPass = true
			}
			o.actions[string(action.Kind)]++
			// The agent learns what it did the same way it learns everything
			// else: through deliver, which echoes its own action back to it.
			// There is no separate write path, so memory can never contain
			// something the agent was not entitled to observe.
			o.deliver(events)
		}

		if !anyRejectedThisAttempt {
			break
		}
	}

	if appliedNonPass {
		o.passStreak = 0
	} else if appliedAny {
		o.passStreak++
	} else {
		// Every proposed slot was rejected and retries are exhausted: the
		// turn is spent with nothing to show for it, same as a pass.
		o.passStreak++
	}
	return nil
}

// otherSlot returns the slot on the other side of a turn from s, for
// phrasing a rejection message that tells the agent what it does not need to
// resend.
func otherSlot(s slot) slot {
	if s == slotComm {
		return slotWorld
	}
	return slotComm
}

// deliver stamps events, logs them and routes them to every agent the
// visibility rules permit. This is the single enforcement point for private
// information: an agent learns nothing except through this function.
func (o *Orchestrator) deliver(events []sim.Event) {
	for _, e := range events {
		o.eventSeq++
		e.Seq = o.eventSeq
		if e.Round == 0 {
			e.Round = o.round
		}
		if e.Time.IsZero() {
			e.Time = o.clock()
		}

		rec := o.record(explog.RecEvent)
		rec.Round = e.Round
		rec.Event = &e
		o.sink.Emit(rec)

		for _, id := range o.order {
			if !e.Visibility.CanObserve(id) {
				continue
			}
			actor := o.actors[id]
			actor.Observe(sim.Observation{
				Round:   e.Round,
				Seq:     e.Seq,
				Time:    e.Time,
				Type:    e.Type,
				Private: !e.Visibility.Public,
				Text:    renderFor(id, e, o.names),
			})
		}
	}
}

// RenderFor is renderFor exported, so other drivers (the event-driven world
// engine) phrase events for their observers exactly as the orchestrator does.
func RenderFor(observer sim.AgentID, e sim.Event, names map[sim.AgentID]string) string {
	return renderFor(observer, e, names)
}

// renderFor phrases an event from one observer's point of view, so an agent
// reads "you" rather than its own name and can tell a whisper to itself from a
// whisper it merely sent.
func renderFor(observer sim.AgentID, e sim.Event, names map[sim.AgentID]string) string {
	name := func(id sim.AgentID) string {
		if id == observer {
			return "you"
		}
		if n := names[id]; n != "" {
			return fmt.Sprintf("%s (%s)", n, id)
		}
		return string(id)
	}

	switch e.Type {
	case sim.EventSay:
		return fmt.Sprintf("%s said: %s", name(e.Actor), e.Text)
	case sim.EventWhisper:
		targets := make([]string, 0, len(e.Targets))
		for _, t := range e.Targets {
			targets = append(targets, name(t))
		}
		return fmt.Sprintf("%s whispered to %s: %s", name(e.Actor), strings.Join(targets, ", "), e.Text)
	case sim.EventPass:
		return fmt.Sprintf("%s stayed silent.", name(e.Actor))
	case sim.EventAction:
		if e.Actor == observer {
			return "You " + e.Text
		}
		return fmt.Sprintf("%s %s", name(e.Actor), e.Text)
	default:
		return e.Text
	}
}

func (o *Orchestrator) record(t explog.RecordType) explog.Record {
	return explog.Record{Type: t, RunID: o.cfg.RunID, Time: o.clock(), Round: o.round}
}

// countCalls totals the inference calls reported by agents that implement
// sim.Metered. Agents that do not simply contribute nothing to the budget.
func (o *Orchestrator) countCalls() int {
	total := 0
	for _, a := range o.actors {
		if m, ok := a.(sim.Metered); ok {
			total += m.CallCount()
		}
	}
	return total
}

func (o *Orchestrator) limitHit() (string, bool) {
	l := o.cfg.Limits
	o.llmCalls = o.countCalls()
	if l.MaxLLMCalls > 0 && o.llmCalls >= l.MaxLLMCalls {
		return fmt.Sprintf("reached the LLM call budget (%d)", l.MaxLLMCalls), true
	}
	total := 0
	for _, n := range o.actions {
		total += n
	}
	if l.MaxActions > 0 && total >= l.MaxActions {
		return fmt.Sprintf("reached the action budget (%d)", l.MaxActions), true
	}
	return "", false
}

func stopReasonForCtx(err error) string {
	if errors.Is(err, context.DeadlineExceeded) {
		return "hit the wall-clock timeout"
	}
	return "cancelled"
}

// isFatal reports whether a turn error should end the run.
func isFatal(err error) bool {
	var u *sim.Unrecoverable
	return errors.As(err, &u) ||
		errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func contains(ids []sim.AgentID, id sim.AgentID) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}
