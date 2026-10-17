package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

// Agent is a persistent autonomous entity: an identity, a private memory, an
// observation history and its own model configuration. Several agents can share
// one Provider (and therefore one loaded model) while remaining completely
// separate minds — nothing is shared between them but the inference backend.
type Agent struct {
	persona  Persona
	provider llm.Provider
	store    memory.Store
	state    *memory.State

	// transcript is the bounded short-term context of the current run: the
	// observations this agent was permitted to see, newest last. Older entries
	// age out of the prompt but remain in episodic memory.
	transcript []sim.Observation
	// observed counts every observation ever delivered; seen counts those the
	// agent has already been shown. The difference is what is new since its
	// last turn, which lets the prompt split "what you have seen" from "what
	// just happened" without repeating a single line in both.
	observed int
	seen     int

	roster []rosterEntry
	byName map[string]sim.AgentID

	runID string
	stats Stats

	// dualAction selects the two-slot protocol (independent optional
	// communication plus optional world action) for DecideMany. Decide itself
	// is unaffected by this flag — it always speaks the single-action
	// protocol, which is what lets DecideMany delegate to it unchanged when
	// dual mode is off.
	dualAction bool

	// planning enables the optional, self-authored, persistent "plan" field:
	// the protocol mentions it, the prompt shows the agent its own current
	// plan for this run, and a plan sent back is saved into memory. Off by
	// default, in which case no plan is ever read, shown, or written —
	// strategy stays implicit in the conversation exactly as before.
	planning bool

	// beliefProvenance enables the optional, richer "opinions" shape: a basis
	// and confidence alongside each note. The protocol mentions it and
	// SetBeliefProvenance is used when applying a reply's opinions. Off by
	// default, in which case opinions are recorded exactly as before — plain
	// notes with no provenance.
	beliefProvenance bool

	// intentions enables the optional "intentions"/"cancel_intentions"
	// fields: requests for a future opportunity to decide, returned from
	// DecideMany as sim.ActionIntend actions alongside the reply's other
	// actions. Off by default; only time-aware environments offer it.
	intentions bool
	// pendingIntents carries intentions decoded by Decide (the
	// single-action path) up to DecideMany, which is the only caller that
	// can return more than one action.
	pendingIntents []sim.Action

	// recallNotes surfaces the agent's own notes from this run in every
	// prompt (long-lived worlds), not only from earlier runs.
	recallNotes bool
	cog         Cognition
	// now is the simulated time of the decision in progress (zero for
	// round-based runs).
	now time.Time
	// noteStamp labels the notes written during the current decision in
	// simulated time (empty for round-based runs).
	noteStamp string

	// OnPlan, when set, is told whenever the agent replaces (non-nil) or
	// abandons (nil) its plan; OnMemory whenever it writes a note
	// ("note", "", text) or updates an impression ("opinion", subject,
	// text). Like Hook, these exist so logging stays outside the agent.
	OnPlan   func(p *memory.Plan)
	OnMemory func(kind, subject, text string)

	// Hook, when set, receives the raw model exchange for each decision. The
	// orchestrator uses it for logging without the agent knowing about logs.
	Hook func(Trace)
}

// Stats is per-agent inference accounting.
type Stats struct {
	Calls        int           `json:"calls"`
	SlotRepairs  int           `json:"slot_repairs,omitempty"`
	// Counts are other cognition counters (goal changes, steps done,
	// reconsideration notes shown...), keyed by metric name.
	Counts map[string]int `json:"counts,omitempty"`
	Retries      int           `json:"retries"`
	Errors       int           `json:"errors"`
	PromptTokens int           `json:"prompt_tokens"`
	OutputTokens int           `json:"output_tokens"`
	Latency      time.Duration `json:"latency_total"`
}

// Trace is one model exchange, reported for observability. Thinking is kept
// separate from the chosen action and is never treated as fact.
type Trace struct {
	Agent    sim.AgentID
	Round    int
	Model    llm.ModelConfig
	Attempt  int
	Request  []llm.Message
	Raw      string
	Thinking string
	Usage    llm.Usage
	Latency  time.Duration
	Err      error
}

// Options configure a new Agent.
type Options struct {
	Persona  Persona
	Provider llm.Provider
	Store    memory.Store
	// RunID scopes episodic memory written during this run.
	RunID string
	// DualAction switches this agent to the two-slot protocol: an
	// independent, optional communication action alongside an independent,
	// optional world action, decided together in one inference call. Off by
	// default, which reproduces the original single-action behavior exactly.
	DualAction bool
	// Planning enables the self-authored, persistent plan field. Off by
	// default.
	Planning bool
	// BeliefProvenance enables the optional basis/confidence shape for
	// "opinions". Off by default.
	BeliefProvenance bool
	// Intentions enables the optional self-scheduling fields (see
	// intentionsAddendum). Off by default.
	Intentions bool
	// RecallOwnNotes shows the agent its own notes from the current run in
	// every prompt. Off by default; long-lived worlds turn it on.
	RecallOwnNotes bool
	// Cognition selects the context refinements for long-lived agents. The
	// zero value reproduces the original prompt exactly.
	Cognition Cognition
}

// Cognition toggles how a long-lived agent's context is assembled. Each
// refinement is deterministic and costs no extra model call; all of them
// only change what the agent is shown, never what it may decide.
type Cognition struct {
	// CompactTranscript folds a line that repeats an earlier line from the
	// same source (the same thing said again, the same offer made again)
	// into its latest occurrence with a count, so a loop reads as one
	// repeated stance rather than crowding everything else out.
	CompactTranscript bool
	// SalientRecall is how many consequential events (transfers, settled
	// offers, project milestones — whatever the environment marked
	// important) that have scrolled out of the transcript are still shown.
	SalientRecall int
	// PlanContext shows when the plan was last written, how much has
	// happened since, and the steps the agent recorded as done.
	PlanContext bool
	// RepairSlots accepts a say/whisper sent in the world_action slot as
	// the reply's communication when that slot was left empty. It is the
	// same action in the wrong envelope, not a different action; every
	// repair is counted.
	RepairSlots bool
	// TypedBeliefs describes and renders the basis of an impression as one
	// of saw / heard / inferred / suspect, with when it was last revised.
	TypedBeliefs bool
	// RecentLimit, when positive, shows at most this many of the most recent
	// transcript lines (SalientRecall keeps what matters from older ones).
	RecentLimit int
	// ExplicitAffordances says plainly, each turn, when "say" cannot be
	// heard; calls the roster the people in town rather than "present";
	// asks the optional reasoning to include what the agent expects; and,
	// when a walk carries no stated purpose, records the reply's reasoning
	// as the walk's private purpose (counted).
	ExplicitAffordances bool
	// PlanTracking keeps the plan's goal and obstacles dated across
	// rewrites, shows how long each has lasted, points at consequential
	// events since the plan was written that concern things it mentions,
	// and says so plainly when an obstacle has outlasted several decisions.
	PlanTracking bool
}

// New builds an agent and loads its persisted memory. A nil Store means the
// agent is ephemeral.
func New(opts Options) (*Agent, error) {
	if opts.Provider == nil {
		return nil, fmt.Errorf("agent %s: no llm provider", opts.Persona.ID)
	}
	a := &Agent{
		persona:          opts.Persona,
		provider:         opts.Provider,
		store:            opts.Store,
		runID:            opts.RunID,
		dualAction:       opts.DualAction,
		planning:         opts.Planning,
		beliefProvenance: opts.BeliefProvenance,
		intentions:       opts.Intentions,
		recallNotes:      opts.RecallOwnNotes,
		cog:              opts.Cognition,
	}
	if opts.Store != nil {
		st, err := opts.Store.Load(opts.Persona.ID)
		if err != nil {
			return nil, err
		}
		a.state = st
	} else {
		a.state = memory.NewState(opts.Persona.ID, opts.Persona.Name)
	}
	a.state.Name = opts.Persona.Name
	a.state.Persona = oneLine(opts.Persona.Personality, 240)
	a.state.NoteRun(opts.RunID)
	return a, nil
}

func (a *Agent) ID() sim.AgentID        { return a.persona.AgentID() }
func (a *Agent) Name() string           { return a.persona.Name }
func (a *Agent) Persona() Persona       { return a.persona }
func (a *Agent) Model() llm.ModelConfig { return a.persona.Model }
func (a *Agent) Stats() Stats {
	st := a.stats
	st.Counts = make(map[string]int, len(a.stats.Counts))
	for k, v := range a.stats.Counts {
		st.Counts[k] = v
	}
	return st
}

// count bumps one of the agent's cognition counters.
func (a *Agent) count(key string, n int) {
	if n == 0 {
		return
	}
	if a.stats.Counts == nil {
		a.stats.Counts = map[string]int{}
	}
	a.stats.Counts[key] += n
}

// CallCount implements sim.Metered so the orchestrator can enforce an
// inference budget without knowing how inference works.
func (a *Agent) CallCount() int       { return a.stats.Calls }
func (a *Agent) State() *memory.State { return a.state }

// Bind tells the agent who else is present. Only ids and display names are
// shared — never another agent's persona, goals or memory.
func (a *Agent) Bind(participants []sim.AgentID, names map[sim.AgentID]string) {
	a.roster = a.roster[:0]
	a.byName = map[string]sim.AgentID{}
	for _, id := range participants {
		name := names[id]
		if name == "" {
			name = string(id)
		}
		a.roster = append(a.roster, rosterEntry{ID: id, Name: name})
		a.byName[strings.ToLower(string(id))] = id
		a.byName[strings.ToLower(name)] = id
	}
}

// Observe records an event this agent was permitted to see. It feeds both the
// bounded short-term transcript and durable episodic memory, so memory reflects
// only what the agent could actually observe.
func (a *Agent) Observe(o sim.Observation) {
	a.transcript = append(a.transcript, o)
	if max := a.persona.Context.RecentObservations; max > 0 && len(a.transcript) > max {
		a.transcript = a.transcript[len(a.transcript)-max:]
	}
	a.observed++

	kind := memory.KindObservation
	if o.Type == sim.EventOutcome {
		kind = memory.KindOutcome
	}
	a.state.AddEpisode(memory.Episode{
		RunID:      a.runID,
		Round:      o.Round,
		Time:       o.Time,
		Kind:       kind,
		Text:       o.Text,
		Private:    o.Private,
		Stamp:      o.Stamp,
		Importance: float64(o.Importance),
		Type:       string(o.Type),
	}, a.persona.Context.MaxEpisodes)
}

// ResumeTranscript rebuilds the short-term transcript from observations this
// agent already persisted under its current run id, and marks them all as
// already seen. It exists for long-lived environments resumed across
// processes (the persistent world): without it, an agent returning to a world
// it has lived in for days would start every session with an empty
// recollection of what just happened. Round-based runs never call it.
func (a *Agent) ResumeTranscript() {
	var obs []sim.Observation
	for _, e := range a.state.Episodes {
		if e.RunID != a.runID || (e.Kind != memory.KindObservation && e.Kind != memory.KindOutcome) {
			continue
		}
		obs = append(obs, sim.Observation{
			Round: e.Round, Time: e.Time, Text: e.Text, Private: e.Private, Stamp: e.Stamp,
			Type: sim.EventType(e.Type), Importance: int(e.Importance),
		})
	}
	if max := a.persona.Context.RecentObservations; max > 0 && len(obs) > max {
		obs = obs[len(obs)-max:]
	}
	a.transcript = obs
	a.observed = len(obs)
	a.seen = a.observed
}

// maxDecodeRetries bounds how many times we re-ask a model that produced an
// unparseable reply. One retry keeps inference economical while tolerating the
// occasional malformed object from a small model.
const maxDecodeRetries = 1

// Decide asks the model to choose this agent's action for the turn.
func (a *Agent) Decide(ctx context.Context, t sim.Turn) (sim.Action, error) {
	if len(a.roster) == 0 {
		names := map[sim.AgentID]string{}
		a.Bind(t.Participants, names)
	}
	msgs := a.messages(t, false)
	a.noteStamp, a.now = stampOf(t), t.Now
	a.countShown()

	var lastErr error
	for attempt := 0; attempt <= maxDecodeRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return sim.Action{}, err
		}
		req := llm.Request{Model: a.persona.Model, Messages: msgs, Format: llm.FormatJSON}

		resp, err := a.provider.Chat(ctx, req)
		a.stats.Calls++
		if attempt > 0 {
			a.stats.Retries++
		}
		if err != nil {
			a.stats.Errors++
			a.report(Trace{Agent: a.ID(), Round: t.Round, Model: a.persona.Model, Attempt: attempt, Request: msgs, Err: err})
			if llm.Fatal(err) {
				// The backend itself is unusable: tell the orchestrator to stop
				// rather than letting every agent fail in turn.
				return sim.Action{}, sim.Fatal(err)
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return sim.Action{}, err
			}
			lastErr = err
			continue
		}
		a.stats.PromptTokens += resp.Usage.PromptTokens
		a.stats.OutputTokens += resp.Usage.CompletionTokens
		a.stats.Latency += resp.Latency

		action, decErr := decodeAction(resp.Content, a.byName, a.ID())
		a.report(Trace{
			Agent: a.ID(), Round: t.Round, Model: a.persona.Model, Attempt: attempt,
			Request: msgs, Raw: resp.Content, Thinking: resp.Thinking,
			Usage: resp.Usage, Latency: resp.Latency, Err: decErr,
		})
		if decErr == nil {
			a.seen = a.observed
			a.noteOwnDecision(t, action)
			a.applyPlanUpdate(resp.Content)
			a.pendingIntents = a.decodeIntentions(resp.Content)
			return action, nil
		}
		a.stats.Errors++
		lastErr = decErr
		// Re-ask with the model's own bad reply plus the specific complaint.
		msgs = append(msgs,
			llm.Assistant(resp.Content),
			llm.User(fmt.Sprintf("That could not be used: %v\nReply again with exactly one JSON action object and nothing else.", decErr)),
		)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no usable action produced")
	}
	return sim.Action{}, lastErr
}

// DecideMany implements sim.MultiDecider. When the agent is not configured
// for dual-action mode, it is exactly Decide wrapped as a one-element slice —
// single-action agents are unaffected by this method existing at all. In dual
// mode, it asks the model once for an independent, optional communication and
// an independent, optional world action, and returns whichever of the two
// (zero, one, or both) the model actually chose to use.
func (a *Agent) DecideMany(ctx context.Context, t sim.Turn) ([]sim.Action, error) {
	if !a.dualAction {
		a.pendingIntents = nil
		action, err := a.Decide(ctx, t)
		if err != nil {
			return nil, err
		}
		out := append([]sim.Action{action}, a.pendingIntents...)
		a.pendingIntents = nil
		return out, nil
	}
	return a.decideDual(ctx, t)
}

// decideDual is DecideMany's dual-protocol implementation. It mirrors
// Decide's structure closely — same bounded parse-retry budget, same stats
// and trace reporting, same error classification — differing only in which
// protocol text it sends and how it decodes the reply.
func (a *Agent) decideDual(ctx context.Context, t sim.Turn) ([]sim.Action, error) {
	if len(a.roster) == 0 {
		names := map[sim.AgentID]string{}
		a.Bind(t.Participants, names)
	}
	msgs := a.messages(t, true)
	a.noteStamp, a.now = stampOf(t), t.Now
	a.countShown()

	var lastErr error
	for attempt := 0; attempt <= maxDecodeRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		req := llm.Request{Model: a.persona.Model, Messages: msgs, Format: llm.FormatJSON}

		resp, err := a.provider.Chat(ctx, req)
		a.stats.Calls++
		if attempt > 0 {
			a.stats.Retries++
		}
		if err != nil {
			a.stats.Errors++
			a.report(Trace{Agent: a.ID(), Round: t.Round, Model: a.persona.Model, Attempt: attempt, Request: msgs, Err: err})
			if llm.Fatal(err) {
				return nil, sim.Fatal(err)
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			lastErr = err
			continue
		}
		a.stats.PromptTokens += resp.Usage.PromptTokens
		a.stats.OutputTokens += resp.Usage.CompletionTokens
		a.stats.Latency += resp.Latency

		decoded, decErr := decodeDualAction(resp.Content, a.byName, a.ID())
		a.report(Trace{
			Agent: a.ID(), Round: t.Round, Model: a.persona.Model, Attempt: attempt,
			Request: msgs, Raw: resp.Content, Thinking: resp.Thinking,
			Usage: resp.Usage, Latency: resp.Latency, Err: decErr,
		})
		if decErr == nil {
			a.seen = a.observed
			a.noteMemory(t, decoded.Remember, decoded.Opinions, decoded.OpinionBasis, decoded.OpinionConfidence)
			a.applyPlanUpdate(resp.Content)
			if a.cog.RepairSlots && decoded.Communication == nil && decoded.World != nil {
				if comm, ok := speechInWorldSlot(*decoded.World, a.byName, a.ID()); ok {
					decoded.Communication, decoded.World = &comm, nil
					a.stats.SlotRepairs++
				}
			}
			if a.cog.ExplicitAffordances && decoded.World != nil {
				a.notePurpose(decoded.World, decoded.Reasoning)
			}
			var out []sim.Action
			if decoded.Communication != nil {
				out = append(out, *decoded.Communication)
			}
			if decoded.World != nil {
				out = append(out, *decoded.World)
			}
			out = append(out, a.decodeIntentions(resp.Content)...)
			return out, nil
		}
		a.stats.Errors++
		lastErr = decErr
		msgs = append(msgs,
			llm.Assistant(resp.Content),
			llm.User(fmt.Sprintf("That could not be used: %v\nReply again with exactly one JSON object with \"communication\" and \"world_action\" (either may be null) and nothing else.", decErr)),
		)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("no usable action produced")
	}
	return nil, lastErr
}

// messages assembles the full request for a turn: the stable system prompt
// and this turn's context. It has no side effects, which is what lets
// PreviewPrompt show exactly what a decision would send.
func (a *Agent) messages(t sim.Turn, dual bool) []llm.Message {
	if len(a.roster) == 0 {
		a.Bind(t.Participants, map[sim.AgentID]string{})
	}
	earlier, fresh := a.splitContext()
	if n := a.cog.RecentLimit; n > 0 && len(earlier)+len(fresh) > n {
		earlier = earlier[max(len(earlier)-max(n-len(fresh), 0), 0):]
	}
	t.New = fresh
	typed := a.cog.TypedBeliefs && a.intentions
	system := buildSystem(a.persona, t, a.roster, dual, a.planning, a.beliefProvenance && !typed, a.intentions)
	if typed {
		system += typedBeliefsAddendum
	}
	if a.cog.PlanContext && a.planning && a.intentions {
		system += planDoneAddendum
	}
	if a.cog.ExplicitAffordances && t.Clock != "" {
		system = strings.Replace(system, "\nPresent (address others by their id):\n", "\nPeople in town (address others by their id; who is with you right now is under HERE):\n", 1)
		system = strings.Replace(system, `"reasoning": "<one short sentence on why>"`, `"reasoning": "<one short sentence: why, and what you expect it to lead to>"`, 1)
	}
	user := buildUserWith(a.persona, t, a.state, earlier, a.runID, dual, a.planning, a.cog)
	if a.recallNotes {
		user = renderOwnNotes(a.state, a.runID, a.persona.Context.RecallEpisodes) + user
	}
	return []llm.Message{llm.System(system), llm.User(user)}
}

// PreviewPrompt returns the messages the agent would send for t, without
// calling the model or changing anything. It exists for inspection: what
// exactly does this agent see right now?
func (a *Agent) PreviewPrompt(t sim.Turn) []llm.Message {
	return a.messages(t, a.dualAction)
}

// splitContext divides the bounded transcript into what the agent has already
// been shown and what has arrived since its last turn. Both are bounded because
// the transcript is.
func (a *Agent) splitContext() (earlier, fresh []sim.Observation) {
	n := a.observed - a.seen
	if n > len(a.transcript) {
		n = len(a.transcript)
	}
	if n < 0 {
		n = 0
	}
	cut := len(a.transcript) - n
	return a.transcript[:cut], a.transcript[cut:]
}

// noteOwnDecision folds the agent's own self-reported memory updates into
// persistent state. These come from the action, not from hidden reasoning.
func (a *Agent) noteOwnDecision(t sim.Turn, act sim.Action) {
	a.noteMemory(t, act.Remember, act.Opinions, act.OpinionBasis, act.OpinionConfidence)
}

// noteMemory is noteOwnDecision's underlying logic, taken directly rather than
// pulled off a single sim.Action — the dual-action protocol carries
// "remember" and "opinions" once per reply, not once per action slot. basis
// and confidence are only applied when belief provenance is enabled; both are
// safe to pass even when empty or nil.
func (a *Agent) noteMemory(t sim.Turn, remember string, opinions, basis, confidence map[string]string) {
	if remember != "" {
		a.state.AddEpisode(memory.Episode{
			RunID: a.runID, Scenario: t.Scenario, Round: t.Round,
			Time: time.Now().UTC(), Kind: memory.KindNote, Text: remember, Private: true,
			Stamp: a.noteStamp,
		}, a.persona.Context.MaxEpisodes)
		if a.OnMemory != nil {
			a.OnMemory("note", "", remember)
		}
	}
	for subject, note := range opinions {
		resolved := subject
		if id, ok := a.byName[strings.ToLower(strings.TrimSpace(subject))]; ok && id != a.ID() {
			resolved = string(id)
		}
		a.state.SetBelief(resolved, note)
		a.state.StampBelief(resolved, a.noteStamp)
		if a.OnMemory != nil {
			a.OnMemory("opinion", resolved, note)
		}
		if a.beliefProvenance || a.cog.TypedBeliefs {
			a.state.SetBeliefProvenance(resolved, basis[subject], confidence[subject])
		}
	}
}

// applyPlanUpdate parses and applies the optional "plan" field from a reply
// whose action(s) already decoded successfully. A no-op when planning is
// disabled, or when the field was omitted (the existing plan simply carries
// forward unchanged). A malformed "plan" is swallowed rather than spent on
// a retry — the action itself is already good, and the plan is a bonus, not
// something worth the turn's one retry over.
func (a *Agent) applyPlanUpdate(raw string) {
	if !a.planning {
		return
	}
	plan, present, err := extractPlan(raw)
	prev := a.state.ActivePlan(a.runID)
	if err != nil || !present {
		if a.cog.PlanTracking {
			tickPlan(prev)
		}
		return
	}
	if plan == nil {
		had := prev != nil
		a.state.ClearPlan(a.runID)
		if had {
			a.count("plans_dropped", 1)
		}
		if had && a.OnPlan != nil {
			a.OnPlan(nil)
		}
		return
	}
	plan.At, plan.Stamp = a.now, a.noteStamp
	if a.cog.PlanTracking {
		if carryPlan(prev, plan, a.now, a.noteStamp) {
			a.count("goal_changes", 1)
		}
		a.count("plan_steps_done", newlyDone(prev, plan))
	}
	a.state.SetPlan(a.runID, *plan)
	if a.OnPlan != nil {
		a.OnPlan(a.state.ActivePlan(a.runID))
	}
}

// countShown counts, for the decision about to be made, the plan
// annotations the agent is shown (computed from state, so a preview of the
// same prompt counts nothing).
func (a *Agent) countShown() {
	if !a.cog.PlanTracking || !a.planning {
		return
	}
	p := a.state.ActivePlan(a.runID)
	if p == nil {
		return
	}
	if planStuck(p) {
		a.count("reconsider_notes", 1)
	}
	earlier, fresh := a.splitContext()
	if len(planChanges(p, append(append([]sim.Observation(nil), earlier...), fresh...), 4)) > 0 {
		a.count("plan_change_notes", 1)
	}
}

// notePurpose gives a walk with no stated purpose the reply's own private
// reasoning as its purpose, so the agent is reminded on arrival why it came.
// The walk itself — where to — is untouched.
func (a *Agent) notePurpose(w *sim.Action, reasoning string) {
	if w.Kind != sim.ActionDo || w.Verb != "go" || strings.TrimSpace(reasoning) == "" {
		return
	}
	for _, k := range []string{"for", "purpose", "why", "to_do"} {
		if v, ok := w.Args[k]; ok && strings.TrimSpace(fmt.Sprint(v)) != "" {
			return
		}
	}
	if w.Args == nil {
		w.Args = map[string]any{}
	}
	w.Args["for"] = reasoning
	a.count("trip_purpose_from_reasoning", 1)
}

// stampOf turns a clocked turn ("Day 2, 09:30 (morning)") into the compact
// stamp used for notes ("Day 2 09:30"); empty for round-based turns.
func stampOf(t sim.Turn) string {
	if t.Clock == "" {
		return ""
	}
	c := t.Clock
	if i := strings.Index(c, " ("); i >= 0 {
		c = c[:i]
	}
	return strings.Replace(c, ",", "", 1)
}

// decodeIntentions reads the optional self-scheduling fields from a reply
// whose action(s) already decoded. Like the plan, a malformed entry is
// dropped rather than spent on a retry: the actions are already good.
func (a *Agent) decodeIntentions(raw string) []sim.Action {
	if !a.intentions {
		return nil
	}
	return extractIntentions(raw)
}

// RecordOwnAction stores an executed action in episodic memory. The
// orchestrator calls this after the environment accepts the action, so memory
// reflects what actually happened rather than what was merely proposed.
func (a *Agent) RecordOwnAction(round int, scenario, text string) {
	a.state.AddEpisode(memory.Episode{
		RunID: a.runID, Scenario: scenario, Round: round,
		Time: time.Now().UTC(), Kind: memory.KindOwnAction, Text: text,
	}, a.persona.Context.MaxEpisodes)
}

// Commit records a commitment in durable memory.
func (a *Agent) Commit(text string) { a.state.AddCommitment(a.runID, text) }

// Persist writes the agent's memory back to its store.
func (a *Agent) Persist() error {
	if a.store == nil {
		return nil
	}
	return a.store.Save(a.state)
}

func (a *Agent) report(tr Trace) {
	if a.Hook != nil {
		a.Hook(tr)
	}
}
