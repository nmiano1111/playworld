// Package sim holds the shared domain vocabulary of the harness: identities,
// actions, events, visibility rules, and the two interfaces that let the
// orchestrator drive arbitrary agents through arbitrary environments.
//
// It deliberately contains no LLM, no IO and no orchestration logic so that
// agents, environments and the scheduler can all depend on it without
// depending on each other.
package sim

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"
)

// AgentID is a stable, human-readable identifier for an agent ("mira").
type AgentID string

// ActionKind enumerates the structured choices an agent may make on its turn.
// The harness distinguishes communication from environment effects from
// deliberate inaction, so a "pass" is a real, logged decision.
type ActionKind string

const (
	// ActionSay is public speech: every participant observes it.
	ActionSay ActionKind = "say"
	// ActionWhisper is private speech addressed to specific agents.
	ActionWhisper ActionKind = "whisper"
	// ActionDo is an environment-specific action, named by Verb.
	ActionDo ActionKind = "do"
	// ActionPass is an explicit decision to do nothing this turn.
	ActionPass ActionKind = "pass"
	// ActionIntend asks for a future opportunity to decide: "wake me at this
	// time, about this". It changes nothing in the world by itself. Only
	// environments with a notion of time (the open-ended world mode) offer it;
	// an agent never produces one unless it was configured to. Args carry
	// "when" and "note", or "cancel" naming a pending intention to withdraw.
	ActionIntend ActionKind = "intend"
)

// Action is what an agent proposes. The environment validates and executes it;
// nothing here is trusted until Environment.Validate accepts it.
type Action struct {
	Kind ActionKind `json:"kind"`

	// Content carries the utterance for Say and Whisper.
	Content string `json:"content,omitempty"`
	// To lists recipients for Whisper.
	To []AgentID `json:"to,omitempty"`

	// Verb and Args describe an ActionDo, interpreted by the environment.
	Verb string         `json:"verb,omitempty"`
	Args map[string]any `json:"args,omitempty"`

	// Reasoning is an optional self-reported rationale. It is recorded purely
	// as commentary and is never shown to other agents nor used by the
	// environment: we do not rely on models exposing hidden chain-of-thought.
	Reasoning string `json:"reasoning,omitempty"`

	// Remember is an optional note the agent wants kept in its own episodic
	// memory beyond what it observed.
	Remember string `json:"remember,omitempty"`
	// Opinions are updates to the agent's beliefs about others, keyed by agent
	// id or name.
	Opinions map[string]string `json:"opinions,omitempty"`
	// OpinionBasis and OpinionConfidence are optional provenance for the
	// matching entry in Opinions (same keys): why the agent believes it, and
	// how sure it is. Both are exactly what the agent itself reported —
	// never inferred or corrected — and are empty when it didn't say.
	OpinionBasis      map[string]string `json:"opinion_basis,omitempty"`
	OpinionConfidence map[string]string `json:"opinion_confidence,omitempty"`
}

// ArgFloat reads a numeric argument, tolerating the json.Number-ish shapes an
// LLM may produce (float64, int, numeric string).
func (a Action) ArgFloat(key string) (float64, bool) {
	v, ok := a.Args[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		var f float64
		if _, err := fmt.Sscanf(strings.TrimSpace(n), "%g", &f); err == nil {
			return f, true
		}
	}
	return 0, false
}

// ArgString reads a string argument.
func (a Action) ArgString(key string) (string, bool) {
	v, ok := a.Args[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// ActionSpec documents one action available to an agent. Environments publish
// these so the prompt can describe the rules without the agent package knowing
// anything about any particular scenario.
type ActionSpec struct {
	Kind ActionKind `json:"kind"`
	// Verb is set for ActionDo specs.
	Verb string `json:"verb,omitempty"`
	// Usage is a one-line signature, e.g. `do claim {"amount": <int>}`. This
	// is authored against the single-action wire shape; prompt rendering
	// that uses a different envelope (such as the dual-action protocol)
	// should synthesize its own example from Kind/Verb/Params rather than
	// printing Usage verbatim, so the envelope's own field names are never
	// contradicted by an example written for a different one.
	Usage string `json:"usage"`
	// Description explains when and why to use it.
	Description string `json:"description"`
	// Params optionally names this ActionDo's own argument keys (e.g.
	// ["text","credits","labor"]), independent of any particular wire
	// envelope. Empty for actions that take no arguments, or where only
	// Usage's prose documents them.
	Params []string `json:"params,omitempty"`
}

// EventType classifies what happened in the world.
type EventType string

const (
	EventSystem   EventType = "system"   // narration from the environment
	EventSay      EventType = "say"      // public utterance
	EventWhisper  EventType = "whisper"  // private utterance
	EventAction   EventType = "action"   // an environment action took effect
	EventState    EventType = "state"    // environment state changed
	EventPass     EventType = "pass"     // an agent deliberately did nothing
	EventRejected EventType = "rejected" // an invalid action was refused
	EventError    EventType = "error"    // inference or harness failure
	EventOutcome  EventType = "outcome"  // final results
)

// Visibility decides who may observe an event. An empty Visibility is private
// to nobody, which is a useful default for records that are logged but not
// observed by any agent.
type Visibility struct {
	// Public means every participant observes the event.
	Public bool `json:"public"`
	// Audience lists the observers when Public is false. The actor should be
	// included if it is meant to see its own action echoed.
	Audience []AgentID `json:"audience,omitempty"`
}

// PublicTo returns a Visibility visible to all participants.
func PublicTo() Visibility { return Visibility{Public: true} }

// Only returns a Visibility restricted to the given agents.
func Only(ids ...AgentID) Visibility {
	return Visibility{Audience: append([]AgentID(nil), ids...)}
}

// CanObserve reports whether id is permitted to see the event.
func (v Visibility) CanObserve(id AgentID) bool {
	if v.Public {
		return true
	}
	for _, a := range v.Audience {
		if a == id {
			return true
		}
	}
	return false
}

// Describe renders the visibility for logs and terminal output.
func (v Visibility) Describe() string {
	if v.Public {
		return "public"
	}
	if len(v.Audience) == 0 {
		return "hidden"
	}
	ids := make([]string, 0, len(v.Audience))
	for _, a := range v.Audience {
		ids = append(ids, string(a))
	}
	sort.Strings(ids)
	return "private:" + strings.Join(ids, ",")
}

// Event is an immutable record of something that happened. Events are the only
// channel through which agents learn about the world, so their Visibility is
// the enforcement point for information boundaries.
type Event struct {
	Seq        int            `json:"seq"`
	Round      int            `json:"round"`
	Time       time.Time      `json:"time"`
	Type       EventType      `json:"type"`
	Actor      AgentID        `json:"actor,omitempty"`
	Targets    []AgentID      `json:"targets,omitempty"`
	Text       string         `json:"text"`
	Data       map[string]any `json:"data,omitempty"`
	Visibility Visibility     `json:"visibility"`
	// Importance optionally marks an event as consequential (2+) — a
	// transfer, a settled offer, a project milestone — as opposed to routine
	// (0/1). Environments that set it let an agent's memory prefer such
	// events when older context has to be dropped. Zero means unspecified.
	Importance int `json:"importance,omitempty"`
}

// Observation is an event as presented to one particular agent.
type Observation struct {
	Round   int       `json:"round"`
	Seq     int       `json:"seq"`
	Time    time.Time `json:"time"`
	Type    EventType `json:"type"`
	Private bool      `json:"private"`
	// Text is the rendered line the agent reads, e.g. `Mira (to you): ...`.
	Text string `json:"text"`
	// Stamp optionally labels when the observation happened in the
	// environment's own terms (e.g. "Day 2 09:30"). Round-based scenarios
	// leave it empty and the prompt falls back to the round number.
	Stamp string `json:"stamp,omitempty"`
	// Importance is copied from the event (see Event.Importance).
	Importance int `json:"importance,omitempty"`
}

// Turn is everything an agent is permitted to know when it decides what to do.
// The orchestrator assembles it; the environment supplies the scenario-specific
// parts via Brief, PublicState and PrivateState.
type Turn struct {
	RunID    string
	Scenario string
	Round    int
	MaxRound int

	Self         AgentID
	Participants []AgentID

	// Brief is the environment's rules and this agent's private instructions.
	Brief string
	// Actions are the moves currently legal for this agent.
	Actions []ActionSpec

	PublicState  map[string]any
	PrivateState map[string]any

	// New are observations delivered since this agent last acted.
	New []Observation

	// Clock optionally describes the current simulated time ("Day 2, 09:30").
	// When set, the prompt speaks in terms of time instead of "round N of M",
	// which is meaningless in an open-ended, event-driven world.
	Clock string
	// Now is the simulated time Clock describes (zero for round-based runs).
	Now time.Time
	// Situation, when set, is the environment's own ordered, sectioned
	// account of what this agent can currently see. It is rendered verbatim
	// in place of PublicState/PrivateState, which stay available for
	// environments that prefer key/value state.
	Situation string
}

// Actor is the decision-making side of an agent. Returning an error lets the
// orchestrator record the failure and move on rather than aborting the run.
type Actor interface {
	ID() AgentID
	Name() string
	// Observe delivers an event the agent is permitted to see.
	Observe(Observation)
	// Decide chooses an action for this turn.
	Decide(ctx context.Context, t Turn) (Action, error)
}

// Outcome is the measurable result of a finished scenario.
type Outcome struct {
	Summary string              `json:"summary"`
	Scores  map[AgentID]float64 `json:"scores,omitempty"`
	Metrics map[string]any      `json:"metrics,omitempty"`
}

// Environment is a simulation scenario. It owns all state, decides what each
// agent may know, validates proposed actions and declares when the run is over.
type Environment interface {
	Name() string
	Description() string

	// Setup binds participants and produces the opening events.
	Setup(participants []AgentID) ([]Event, error)
	Participants() []AgentID

	// Brief returns the rules plus any private instructions for one agent.
	Brief(id AgentID) string
	// Actions lists the moves legal for this agent right now.
	Actions(id AgentID) []ActionSpec

	// Validate rejects an illegal action with an explanatory error, which is
	// fed back to the agent as an observation.
	Validate(id AgentID, a Action) error
	// Apply executes a validated action and returns the resulting events.
	Apply(id AgentID, a Action) ([]Event, error)

	PublicState() map[string]any
	PrivateState(id AgentID) map[string]any

	// BeginRound lets the environment advance phases and narrate.
	BeginRound(round int) []Event
	// EndRound runs after every agent has acted in a round.
	EndRound(round int) []Event

	// Done reports whether the scenario has terminated, and why.
	Done() (bool, string)
	Outcome() Outcome
}

// Unrecoverable marks an error that means the run cannot usefully continue —
// the inference backend is down, a model is missing, or an agent has failed
// repeatedly. The orchestrator stops on these and carries on past everything
// else, which keeps error classification out of string matching.
type Unrecoverable struct{ Err error }

func (u *Unrecoverable) Error() string { return u.Err.Error() }
func (u *Unrecoverable) Unwrap() error { return u.Err }

// Fatal wraps err as Unrecoverable, returning nil for a nil error.
func Fatal(err error) error {
	if err == nil {
		return nil
	}
	return &Unrecoverable{Err: err}
}

// Metered is an optional Actor capability: report how many inference calls the
// agent has made so far. The orchestrator uses it to enforce a call budget
// without knowing anything about the inference layer.
type Metered interface {
	CallCount() int
}

// MultiDecider is an optional Actor capability: propose more than one action
// for a single turn — at most one communication action (Say/Whisper) and at
// most one world action (Do/Pass) — from a single underlying decision.
//
// This exists so an agent can choose independently whether to communicate and
// whether to act, without the harness forcing a choice between the two. The
// returned slice holds 0, 1 or 2 actions; an empty slice means the agent
// chose to do nothing at all this turn (the orchestrator treats that as an
// explicit Pass). An Actor that does not implement MultiDecider is driven
// through the plain Decide method exactly as before — this capability is
// additive, not a replacement.
type MultiDecider interface {
	DecideMany(ctx context.Context, t Turn) ([]Action, error)
}
