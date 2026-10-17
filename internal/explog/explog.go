// Package explog records an experiment twice: once as a machine-readable JSONL
// stream that can reconstruct the whole run, and once as readable live terminal
// output. Both are fed by the same Record stream, so they cannot drift.
package explog

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

// RecordType classifies a log line.
type RecordType string

const (
	RecRunStart   RecordType = "run_start"
	RecAgent      RecordType = "agent" // one per agent: identity + model config
	RecRoundStart RecordType = "round_start"
	RecEvent      RecordType = "event"    // something happened in the world
	RecDecision   RecordType = "decision" // an agent proposed an action
	RecLLMCall    RecordType = "llm_call" // one inference, with usage + latency
	RecRejected   RecordType = "rejected" // an action failed validation
	RecError      RecordType = "error"    // inference or harness failure
	RecRoundEnd   RecordType = "round_end"
	RecOutcome    RecordType = "outcome"
	RecSummary    RecordType = "summary"
	// RecActivation marks an agent being given a moment to decide in the
	// event-driven world mode, with the simulated time and what prompted it.
	// Round-based runs never emit it.
	RecActivation RecordType = "activation"
	// RecScheduled reports that a trigger gave an agent a (new or earlier)
	// pending activation, or was damped and woke nobody. World mode only.
	RecScheduled RecordType = "scheduled"
	// RecPlan records an agent replacing or abandoning its own plan.
	RecPlan RecordType = "plan"
	// RecMemory records an agent writing a note or updating an impression.
	RecMemory RecordType = "memory"
	// RecView is an introspection record (world mode, opt-in): what one
	// agent was shown at an activation, next to a privileged snapshot for
	// the experimenter that the agent never saw.
	RecView RecordType = "view"
)

// Record is one line of the experiment log. Exactly one payload field is set.
type Record struct {
	Type  RecordType `json:"type"`
	RunID string     `json:"run_id"`
	Seq   int        `json:"seq"`
	Time  time.Time  `json:"time"`
	Round int        `json:"round,omitempty"`

	Run      *RunInfo     `json:"run,omitempty"`
	Agent    *AgentInfo   `json:"agent,omitempty"`
	Event    *sim.Event   `json:"event,omitempty"`
	Decision *Decision    `json:"decision,omitempty"`
	Call     *Call        `json:"call,omitempty"`
	Rejected *Rejection   `json:"rejected,omitempty"`
	Failure  *Failure     `json:"failure,omitempty"`
	Outcome  *sim.Outcome `json:"outcome,omitempty"`
	Summary  *Summary     `json:"summary,omitempty"`
	Note     string       `json:"note,omitempty"`

	Activation *Activation   `json:"activation,omitempty"`
	Scheduled  *Scheduled    `json:"scheduled,omitempty"`
	Plan       *PlanUpdate   `json:"plan,omitempty"`
	Memory     *MemoryUpdate `json:"memory,omitempty"`
	View       *View         `json:"view,omitempty"`
	// SimStamp is the simulated time of the record in world mode
	// ("Day 2 09:30"); empty for round-based runs.
	SimStamp string `json:"sim_stamp,omitempty"`
}

// Scheduled is a trigger's effect on the scheduler.
type Scheduled struct {
	Agent  sim.AgentID `json:"agent"`
	At     time.Time   `json:"at"`
	Stamp  string      `json:"stamp"`
	Reason string      `json:"reason"`
	Kind   string      `json:"kind"`
	Depth  int         `json:"depth,omitempty"`
	// Damped means the trigger was past the reply-chain limit and woke
	// nobody (the event itself was still delivered).
	Damped bool `json:"damped,omitempty"`
	// Coalesced means it joined an activation that was already pending.
	Coalesced bool `json:"coalesced,omitempty"`
}

// PlanUpdate is an agent's own plan after a change; Plan is nil when the
// agent abandoned it.
type PlanUpdate struct {
	Agent sim.AgentID  `json:"agent"`
	Plan  *memory.Plan `json:"plan"`
}

// MemoryUpdate is a note or impression an agent chose to record.
type MemoryUpdate struct {
	Agent   sim.AgentID `json:"agent"`
	Kind    string      `json:"kind"` // note | opinion
	Subject string      `json:"subject,omitempty"`
	Text    string      `json:"text"`
}

// Activation is one agent opportunity in the event-driven world mode.
type Activation struct {
	Agent sim.AgentID `json:"agent"`
	// Index numbers activations within a session; it is also the record's
	// Round, so per-turn breakdowns keyed on Round keep working.
	Index   int       `json:"index"`
	SimTime time.Time `json:"sim_time"`
	Stamp   string    `json:"stamp"`
	// Reasons are the coalesced triggers that led to this activation.
	Reasons []string `json:"reasons"`
}

// View is what an agent saw at one activation (Situation and the moves
// offered), plus Privileged: the experimenter-only truth at that moment,
// which is never part of any agent's prompt.
type View struct {
	Agent      sim.AgentID `json:"agent"`
	Situation  string      `json:"situation"`
	Actions    []string    `json:"actions,omitempty"`
	Privileged string      `json:"privileged,omitempty"`
}

// WorldSession describes the slice of a persistent world a session covers.
type WorldSession struct {
	World   string `json:"world"`
	Session int    `json:"session"`
	From    string `json:"from"`
	To      string `json:"to"`
}

// RunInfo captures how a run was configured, so a log is self-describing.
type RunInfo struct {
	RunID      string         `json:"run_id"`
	Scenario   string         `json:"scenario"`
	Started    time.Time      `json:"started"`
	Seed       int64          `json:"seed"`
	MaxRounds  int            `json:"max_rounds"`
	MaxCalls   int            `json:"max_llm_calls"`
	MaxActions int            `json:"max_actions"`
	Timeout    string         `json:"timeout,omitempty"`
	Params     map[string]any `json:"params,omitempty"`
	MemoryDir  string         `json:"memory_dir,omitempty"`
	Harness    string         `json:"harness_version,omitempty"`
	// World is set for an event-driven world session instead of a
	// round-based scenario run.
	World *WorldSession `json:"world,omitempty"`
}

// AgentInfo records an agent's identity and model configuration.
type AgentInfo struct {
	ID       sim.AgentID     `json:"id"`
	Name     string          `json:"name"`
	Role     string          `json:"role,omitempty"`
	Persona  string          `json:"persona,omitempty"`
	Goals    []string        `json:"goals,omitempty"`
	Model    llm.ModelConfig `json:"model"`
	Episodes int             `json:"recalled_episodes"`
}

// Decision is an action an agent chose, before the environment ran it.
type Decision struct {
	Agent  sim.AgentID `json:"agent"`
	Action sim.Action  `json:"action"`
	// Accepted reports whether the environment allowed it.
	Accepted bool `json:"accepted"`
	// Attempt is which try within the turn produced this decision: 0 for the
	// first attempt, 1 for a retry after a rejection. Lets a log reader see
	// whether a retry succeeded.
	Attempt int `json:"attempt,omitempty"`
	// Slot distinguishes which independent action this was, for agents using
	// the dual-action protocol: "communication" or "world". Empty for an
	// agent using the original single-action protocol, where there is only
	// ever one slot per turn.
	Slot string `json:"slot,omitempty"`
}

// Call is one inference, with the observability the user asked for: counts,
// token usage where available, and latency. Prompt and reply text are included
// only at high verbosity to keep logs manageable.
type Call struct {
	Agent        sim.AgentID   `json:"agent"`
	Model        string        `json:"model"`
	Provider     string        `json:"provider"`
	Attempt      int           `json:"attempt"`
	PromptTokens int           `json:"prompt_tokens"`
	OutputTokens int           `json:"output_tokens"`
	LatencyMS    int64         `json:"latency_ms"`
	Err          string        `json:"error,omitempty"`
	Raw          string        `json:"raw_reply,omitempty"`
	Prompt       []llm.Message `json:"prompt,omitempty"`
	// Thinking is any separately-reported reasoning text. Recorded as
	// commentary, kept distinct from the action actually taken.
	Thinking string `json:"thinking,omitempty"`
}

// Rejection records an action the environment refused.
type Rejection struct {
	Agent  sim.AgentID `json:"agent"`
	Action sim.Action  `json:"action"`
	Reason string      `json:"reason"`
	// Code classifies the refusal for metrics, when the environment gives
	// one (world mode: "cannot_afford", "offer_closed", ...).
	Code    string `json:"code,omitempty"`
	Attempt int    `json:"attempt,omitempty"`
	Slot    string `json:"slot,omitempty"`
	// Retried reports whether a further attempt was made for this slot
	// within the same turn (regardless of whether that retry succeeded).
	Retried bool `json:"retried,omitempty"`
}

// Failure records an error that did not stop the run.
type Failure struct {
	Agent sim.AgentID `json:"agent,omitempty"`
	Stage string      `json:"stage"`
	Err   string      `json:"error"`
}

// Summary is the end-of-run report.
type Summary struct {
	RunID    string `json:"run_id"`
	Scenario string `json:"scenario"`
	Rounds   int    `json:"rounds"`
	// StopReason explains why the run ended: termination condition, round
	// limit, call budget, timeout or cancellation.
	StopReason string         `json:"stop_reason"`
	Duration   string         `json:"duration"`
	DurationMS int64          `json:"duration_ms"`
	LLMCalls   int            `json:"llm_calls"`
	LLMErrors  int            `json:"llm_errors"`
	Retries    int            `json:"retries"`
	Prompt     int            `json:"prompt_tokens"`
	Output     int            `json:"output_tokens"`
	Latency    string         `json:"inference_time"`
	Actions    map[string]int `json:"actions"`
	Rejections int            `json:"rejected_actions"`
	PerAgent   []AgentSummary `json:"per_agent"`
	Outcome    *sim.Outcome   `json:"outcome,omitempty"`
	LogPath    string         `json:"log_path,omitempty"`
	// ActionSelection breaks down how agents used the communication and
	// world-action slots — see ActionSelection for what each field means.
	ActionSelection ActionSelection `json:"action_selection"`
}

// ActionSelection summarises, across the whole run, how agents used the two
// action slots: whether a turn included a communication, a world action,
// both, or neither, and how often a proposed action was rejected or retried.
// It is computed from Decision records' Slot/Attempt/Accepted fields, so it
// reflects what agents actually attempted — including rejected attempts —
// not only what ended up taking effect. For an agent using the original
// single-action protocol there is only ever one slot per turn, so
// TurnsWithBoth stays zero; the rest of the breakdown is still meaningful.
type ActionSelection struct {
	Turns                  int `json:"turns"`
	TurnsWithCommunication int `json:"turns_with_communication"`
	TurnsWithWorldAction   int `json:"turns_with_world_action"`
	TurnsWithBoth          int `json:"turns_with_both"`
	TurnsWithNeither       int `json:"turns_with_neither"`
	ProposedActions        int `json:"proposed_actions"`
	AcceptedActions        int `json:"accepted_actions"`
	RejectedActions        int `json:"rejected_actions"`
	RetriedSlots           int `json:"retried_slots"`
	RetriesSucceeded       int `json:"retries_succeeded"`
}

// AgentSummary is per-agent accounting at the end of a run.
type AgentSummary struct {
	ID           sim.AgentID `json:"id"`
	Name         string      `json:"name"`
	Model        string      `json:"model"`
	Calls        int         `json:"calls"`
	Errors       int         `json:"errors"`
	PromptTokens int         `json:"prompt_tokens"`
	OutputTokens int         `json:"output_tokens"`
	AvgLatencyMS int64       `json:"avg_latency_ms"`
	Said         int         `json:"said"`
	Whispered    int         `json:"whispered"`
	Acted        int         `json:"acted"`
	Passed       int         `json:"passed"`
	Score        *float64    `json:"score,omitempty"`
	Commitments  int         `json:"commitments,omitempty"`
	Beliefs      int         `json:"beliefs,omitempty"`
	// HasPlan and PlanRevisions are goal-directed-execution signals: whether
	// the agent still held an active, self-authored plan at the end of the
	// run, and how many times it replaced one (see memory.Plan.Revisions).
	// Both stay zero/false when planning is disabled.
	HasPlan       bool `json:"has_plan,omitempty"`
	PlanRevisions int  `json:"plan_revisions,omitempty"`
	// BeliefsWithProvenance counts Beliefs that carry a Basis or Confidence —
	// zero unless belief provenance is enabled.
	BeliefsWithProvenance int `json:"beliefs_with_provenance,omitempty"`
	// PlanUpdates, NotesWritten and OpinionUpdates count self-authored
	// memory changes as they happened (from plan/memory records).
	PlanUpdates    int `json:"plan_updates,omitempty"`
	NotesWritten   int `json:"notes_written,omitempty"`
	OpinionUpdates int `json:"opinion_updates,omitempty"`
}

// Sink consumes records. The orchestrator writes to one Sink and does not care
// whether it is a file, a terminal, both, or nothing.
type Sink interface {
	Emit(Record)
	Close() error
}

// Multi fans a record out to several sinks.
type Multi []Sink

func (m Multi) Emit(r Record) {
	for _, s := range m {
		s.Emit(r)
	}
}

func (m Multi) Close() error {
	var first error
	for _, s := range m {
		if err := s.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Discard swallows records.
type Discard struct{}

func (Discard) Emit(Record)  {}
func (Discard) Close() error { return nil }

// JSONL writes one JSON object per line.
type JSONL struct {
	mu   sync.Mutex
	w    *bufio.Writer
	f    *os.File
	path string
	// Verbose includes prompts and raw replies in llm_call records.
	Verbose bool
}

// NewJSONL creates (or truncates) a JSONL log at path.
func NewJSONL(path string, verbose bool) (*JSONL, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("explog: create log dir: %w", err)
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("explog: create %s: %w", path, err)
	}
	return &JSONL{w: bufio.NewWriter(f), f: f, path: path, Verbose: verbose}, nil
}

// Path is the file being written.
func (j *JSONL) Path() string { return j.path }

func (j *JSONL) Emit(r Record) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if !j.Verbose && r.Call != nil {
		c := *r.Call
		c.Prompt = nil
		// Keep the raw reply only when it failed to parse: that is the case
		// where a reader needs to see what the model actually said.
		if c.Err == "" {
			c.Raw = ""
		}
		r.Call = &c
	}
	raw, err := json.Marshal(r)
	if err != nil {
		return
	}
	j.w.Write(raw)
	j.w.WriteByte('\n')
	// Flush eagerly: a run that is interrupted should still leave a usable log.
	j.w.Flush()
}

func (j *JSONL) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.f == nil {
		return nil
	}
	if err := j.w.Flush(); err != nil {
		j.f.Close()
		j.f = nil
		return err
	}
	err := j.f.Close()
	j.f = nil
	return err
}

// SortedKeys is a small helper shared by the console renderer and summaries.
func SortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Sequencer stamps every record with a monotonic log-line number as it passes
// through to the wrapped sink. This is deliberately separate from sim.Event.Seq,
// which counts events in the world: one orders the log, the other orders
// history, and conflating them makes a log that cannot be replayed in order.
type Sequencer struct {
	mu   sync.Mutex
	n    int
	sink Sink
}

// NewSequencer wraps a sink so that every record it receives is numbered.
func NewSequencer(s Sink) *Sequencer {
	if s == nil {
		s = Discard{}
	}
	return &Sequencer{sink: s}
}

func (s *Sequencer) Emit(r Record) {
	s.mu.Lock()
	s.n++
	r.Seq = s.n
	s.mu.Unlock()
	s.sink.Emit(r)
}

func (s *Sequencer) Close() error { return s.sink.Close() }

// Count reports how many records have been numbered.
func (s *Sequencer) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.n
}
