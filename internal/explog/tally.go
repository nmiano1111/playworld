package explog

import (
	"sort"
	"time"

	"github.com/nmiano/playworld/internal/sim"
)

// Tally is a Sink that accumulates metrics as a run proceeds and produces the
// end-of-run Summary. Deriving the summary from the same record stream the log
// and terminal see means the three can never disagree.
type Tally struct {
	order        []sim.AgentID
	info         map[sim.AgentID]*AgentInfo
	acc          map[sim.AgentID]*AgentSummary
	agentLatency map[sim.AgentID]time.Duration

	calls      int
	callErrors int
	retries    int
	prompt     int
	output     int
	latency    time.Duration
	rejections int
	actions    map[string]int
	rounds     int
	outcome    *sim.Outcome
	started    time.Time

	// turnSlots tracks, per (agent, round), which action slots were ever
	// attempted — the raw material for ActionSelection's turn-level
	// breakdown. A turn can appear more than once across retries; the slot
	// flags only ever turn on, never off.
	turnSlots map[turnKey]*turnSlotFlags
	sel       ActionSelection
}

// turnKey identifies one agent's turn within a round, which is the unit
// ActionSelection's per-turn stats are computed over.
type turnKey struct {
	agent sim.AgentID
	round int
}

type turnSlotFlags struct {
	comm, world bool
}

// NewTally returns an empty tally.
func NewTally() *Tally {
	return &Tally{
		info:         map[sim.AgentID]*AgentInfo{},
		acc:          map[sim.AgentID]*AgentSummary{},
		agentLatency: map[sim.AgentID]time.Duration{},
		actions:      map[string]int{},
		turnSlots:    map[turnKey]*turnSlotFlags{},
	}
}

func (t *Tally) agent(id sim.AgentID) *AgentSummary {
	a, ok := t.acc[id]
	if !ok {
		a = &AgentSummary{ID: id, Name: string(id)}
		t.acc[id] = a
		t.order = append(t.order, id)
	}
	return a
}

func (t *Tally) Emit(r Record) {
	switch r.Type {
	case RecRunStart:
		if r.Run != nil {
			t.started = r.Run.Started
		}
	case RecAgent:
		if r.Agent == nil {
			return
		}
		info := *r.Agent
		t.info[info.ID] = &info
		a := t.agent(info.ID)
		a.Name = info.Name
		a.Model = info.Model.String()
	case RecRoundStart, RecActivation:
		if r.Round > t.rounds {
			t.rounds = r.Round
		}
	case RecLLMCall:
		if r.Call == nil {
			return
		}
		t.calls++
		if r.Call.Attempt > 0 {
			t.retries++
		}
		t.prompt += r.Call.PromptTokens
		t.output += r.Call.OutputTokens
		t.latency += time.Duration(r.Call.LatencyMS) * time.Millisecond
		a := t.agent(r.Call.Agent)
		a.Calls++
		a.PromptTokens += r.Call.PromptTokens
		a.OutputTokens += r.Call.OutputTokens
		t.agentLatency[r.Call.Agent] += time.Duration(r.Call.LatencyMS) * time.Millisecond
		if r.Call.Err != "" {
			t.callErrors++
			a.Errors++
		}
	case RecDecision:
		if r.Decision == nil {
			return
		}
		t.noteActionSelection(r.Round, *r.Decision)
		if !r.Decision.Accepted {
			return
		}
		a := t.agent(r.Decision.Agent)
		switch r.Decision.Action.Kind {
		case sim.ActionSay:
			a.Said++
		case sim.ActionWhisper:
			a.Whispered++
		case sim.ActionDo:
			a.Acted++
		case sim.ActionPass:
			a.Passed++
		}
		t.actions[string(r.Decision.Action.Kind)]++
	case RecRejected:
		t.rejections++
	case RecPlan:
		if r.Plan != nil {
			t.agent(r.Plan.Agent).PlanUpdates++
		}
	case RecMemory:
		if r.Memory != nil {
			a := t.agent(r.Memory.Agent)
			if r.Memory.Kind == "note" {
				a.NotesWritten++
			} else {
				a.OpinionUpdates++
			}
		}
	case RecOutcome:
		t.outcome = r.Outcome
	}
}

// noteActionSelection folds one proposed action into the running
// ActionSelection tallies and marks which slot its turn used. "Slot" is
// trusted when the orchestrator set it (both single- and dual-action turns
// populate it); records from an older log without it fall back to
// classifying by the action's own Kind, so historical logs still produce a
// sensible (if slot-less) breakdown.
func (t *Tally) noteActionSelection(round int, d Decision) {
	t.sel.ProposedActions++
	if d.Accepted {
		t.sel.AcceptedActions++
	} else {
		t.sel.RejectedActions++
	}
	if d.Attempt > 0 {
		t.sel.RetriedSlots++
		if d.Accepted {
			t.sel.RetriesSucceeded++
		}
	}

	slot := d.Slot
	if slot == "" {
		switch d.Action.Kind {
		case sim.ActionSay, sim.ActionWhisper:
			slot = "communication"
		default:
			slot = "world"
		}
	}

	key := turnKey{agent: d.Agent, round: round}
	flags, ok := t.turnSlots[key]
	if !ok {
		flags = &turnSlotFlags{}
		t.turnSlots[key] = flags
	}
	switch slot {
	case "communication":
		flags.comm = true
	case "world":
		flags.world = true
	}
}

// finishActionSelection turns the per-turn slot flags accumulated so far into
// the aggregate counts a Summary reports. Idempotent, so it is safe to call
// every time Summary is built.
func (t *Tally) finishActionSelection() ActionSelection {
	out := t.sel
	out.Turns = len(t.turnSlots)
	for _, flags := range t.turnSlots {
		switch {
		case flags.comm && flags.world:
			out.TurnsWithBoth++
		case flags.comm:
			out.TurnsWithCommunication++
		case flags.world:
			out.TurnsWithWorldAction++
		default:
			out.TurnsWithNeither++
		}
	}
	return out
}

func (t *Tally) Close() error { return nil }

// Summary finalises the report. The caller supplies the facts only it knows:
// why the run stopped, how long it took, and where the log was written.
func (t *Tally) Summary(runID, scenario, stopReason string, duration time.Duration, logPath string) Summary {
	s := Summary{
		RunID:      runID,
		Scenario:   scenario,
		Rounds:     t.rounds,
		StopReason: stopReason,
		Duration:   FormatDuration(duration),
		DurationMS: duration.Milliseconds(),
		LLMCalls:   t.calls,
		LLMErrors:  t.callErrors,
		Retries:    t.retries,
		Prompt:     t.prompt,
		Output:     t.output,
		Latency:    FormatDuration(t.latency),
		Actions:    t.actions,
		Rejections: t.rejections,
		Outcome:    t.outcome,
		LogPath:    logPath,

		ActionSelection: t.finishActionSelection(),
	}
	for _, id := range t.order {
		a := *t.acc[id]
		if a.Calls > 0 {
			a.AvgLatencyMS = int64(float64(t.perAgentLatency(id)) / float64(a.Calls))
		}
		if t.outcome != nil {
			if score, ok := t.outcome.Scores[id]; ok {
				v := score
				a.Score = &v
			}
		}
		s.PerAgent = append(s.PerAgent, a)
	}
	sort.Slice(s.PerAgent, func(i, j int) bool { return s.PerAgent[i].ID < s.PerAgent[j].ID })
	return s
}

// perAgentLatency keeps a running total per agent without a second map.
func (t *Tally) perAgentLatency(id sim.AgentID) int64 {
	return t.agentLatency[id].Milliseconds()
}

// SetAgentCounts lets the caller fold in per-agent facts the record stream does
// not carry, such as how much memory an agent ended up holding.
func (t *Tally) SetAgentCounts(id sim.AgentID, commitments, beliefs int) {
	a := t.agent(id)
	a.Commitments = commitments
	a.Beliefs = beliefs
}

// SetPlanAndBeliefCounts folds in the goal-directed-execution and belief-
// provenance signals from Sections 3 and 4: whether a self-authored plan
// was still active at the end of the run, how many times it was revised,
// and how many of the agent's beliefs carry a basis or confidence. All stay
// zero/false when planning or belief provenance are disabled.
func (t *Tally) SetPlanAndBeliefCounts(id sim.AgentID, hasPlan bool, planRevisions, beliefsWithProvenance int) {
	a := t.agent(id)
	a.HasPlan = hasPlan
	a.PlanRevisions = planRevisions
	a.BeliefsWithProvenance = beliefsWithProvenance
}
