// Package runner composes the harness: it builds agents from personas, wires
// them to an environment through the orchestrator, streams records to the
// configured sinks and produces the end-of-run summary.
//
// It exists so the CLI and the integration tests drive exactly the same code
// path; nothing here parses flags or touches os.Args.
package runner

import (
	"context"
	"fmt"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/orchestrator"
	"github.com/nmiano/playworld/internal/scenario"
	"github.com/nmiano/playworld/internal/sim"
)

// Options is a fully-resolved experiment. The caller has already applied every
// default except the few noted below.
type Options struct {
	// RunID identifies the experiment. Generated from the scenario and the
	// clock if empty.
	RunID string
	// Scenario is a registered scenario name.
	Scenario string
	// Personas are the agents to run, in turn order.
	Personas []agent.Persona
	// Providers resolves each persona's configured provider name.
	Providers *llm.Registry
	// Store persists agent memory. Nil means the agents are ephemeral.
	Store memory.Store

	Rounds int
	Seed   int64
	Params map[string]any

	// Limits bound the run. MaxRounds is taken from Rounds.
	Limits orchestrator.Limits
	// Shuffle randomises turn order each round using Seed.
	Shuffle bool
	// DualAction switches every agent in this run to the two-slot protocol
	// (an independent, optional communication plus an independent, optional
	// world action per turn). Off by default, which reproduces the original
	// single-action behavior exactly.
	DualAction bool
	// Planning enables every agent's self-authored, persistent plan field.
	// Off by default.
	Planning bool
	// BeliefProvenance enables every agent's optional basis/confidence shape
	// for its opinions of others. Off by default.
	BeliefProvenance bool

	// Sinks receive the record stream. The runner adds its own tally.
	Sinks []explog.Sink
	// LogPath is reported in the summary, so a reader knows where the
	// machine-readable log went.
	LogPath string
	// MemoryDir is recorded in the run header for the same reason.
	MemoryDir string
	// Harness is the harness version recorded in the log.
	Harness string

	// Preflight checks each distinct model before any inference begins.
	Preflight bool
}

// Result is what a completed experiment reports.
type Result struct {
	RunID   string
	Summary explog.Summary
	Outcome sim.Outcome
	// Run holds the orchestrator's accounting, including a fatal error if the
	// run was cut short.
	Run *orchestrator.Result
	// Agents are the live agents, so a caller can inspect their final memory.
	Agents []*agent.Agent
}

// Execute runs one experiment.
func Execute(ctx context.Context, opts Options) (*Result, error) {
	if opts.Scenario == "" {
		return nil, fmt.Errorf("runner: no scenario")
	}
	if len(opts.Personas) == 0 {
		return nil, fmt.Errorf("runner: no agents")
	}
	if opts.Providers == nil {
		return nil, fmt.Errorf("runner: no provider registry")
	}
	if opts.Rounds <= 0 {
		info, err := scenario.Lookup(opts.Scenario)
		if err != nil {
			return nil, err
		}
		opts.Rounds = info.DefaultRounds
	}
	if opts.RunID == "" {
		opts.RunID = fmt.Sprintf("%s-%s", opts.Scenario, time.Now().Format("20060102-150405"))
	}

	if opts.Preflight {
		if err := Preflight(ctx, opts.Providers, opts.Personas); err != nil {
			return nil, err
		}
	}

	tally := explog.NewTally()
	// Every record — the runner's headers and summary included — passes through
	// one sequencer, so the log can be replayed in the order it happened.
	sink := explog.NewSequencer(explog.Multi(append(append([]explog.Sink(nil), opts.Sinks...), tally)))

	agents, actors, err := buildAgents(opts, sink, false, opts.RunID)
	if err != nil {
		return nil, err
	}

	ids := make([]sim.AgentID, 0, len(agents))
	names := map[sim.AgentID]string{}
	for _, a := range agents {
		ids = append(ids, a.ID())
		names[a.ID()] = a.Name()
	}
	// Agents learn who else is present: ids and display names only, never
	// another agent's persona, goals or memory.
	for _, a := range agents {
		a.Bind(ids, names)
	}

	env, err := scenario.Build(opts.Scenario, scenario.Spec{
		Participants: ids, Names: names, Rounds: opts.Rounds, Seed: opts.Seed, Params: opts.Params,
	})
	if err != nil {
		return nil, err
	}

	limits := opts.Limits
	limits.MaxRounds = opts.Rounds
	if limits.MaxLLMCalls == 0 {
		// Enough for every agent to act every round, plus slack for retries.
		limits.MaxLLMCalls = len(agents)*opts.Rounds*2 + 10
	}
	if limits.MaxActions == 0 {
		limits.MaxActions = orchestrator.DefaultLimits().MaxActions
	}

	orch, err := orchestrator.New(orchestrator.Config{
		RunID: opts.RunID, Env: env, Agents: actors, Limits: limits,
		Sink: sink, Seed: opts.Seed, Shuffle: opts.Shuffle,
	})
	if err != nil {
		return nil, err
	}

	// Announce the run and every agent's configuration first, so the log is
	// self-describing before a single turn is taken.
	started := time.Now()
	sink.Emit(explog.Record{
		Type: explog.RecRunStart, RunID: opts.RunID, Time: started,
		Run: &explog.RunInfo{
			RunID: opts.RunID, Scenario: opts.Scenario, Started: started, Seed: opts.Seed,
			MaxRounds: limits.MaxRounds, MaxCalls: limits.MaxLLMCalls, MaxActions: limits.MaxActions,
			Timeout: durString(limits.Timeout), Params: opts.Params,
			MemoryDir: opts.MemoryDir, Harness: opts.Harness,
		},
	})
	for _, a := range agents {
		p := a.Persona()
		sink.Emit(explog.Record{
			Type: explog.RecAgent, RunID: opts.RunID, Time: time.Now(),
			Agent: &explog.AgentInfo{
				ID: a.ID(), Name: a.Name(), Role: p.Role,
				Persona: truncate(p.Personality, 200), Goals: p.Goals,
				Model: p.Model, Episodes: len(a.State().PriorRuns(opts.RunID, 0)),
			},
		})
	}

	runResult, err := orch.Run(ctx)
	if err != nil {
		return nil, err
	}

	// Persist memory even when the run was cut short: a partial experiment is
	// still an experiment, and the agents genuinely lived through it.
	var persistErrs []error
	for _, a := range agents {
		if err := a.Persist(); err != nil {
			persistErrs = append(persistErrs, err)
		}
		tally.SetAgentCounts(a.ID(), len(a.State().Commitments), len(a.State().Beliefs))
		plan := a.State().ActivePlan(opts.RunID)
		provenance := 0
		for _, b := range a.State().Beliefs {
			if b.Basis != "" || b.Confidence != "" {
				provenance++
			}
		}
		tally.SetPlanAndBeliefCounts(a.ID(), plan != nil, planRevisions(plan), provenance)
	}

	summary := tally.Summary(opts.RunID, opts.Scenario, runResult.StopReason, runResult.Duration, opts.LogPath)
	sink.Emit(explog.Record{Type: explog.RecSummary, RunID: opts.RunID, Time: time.Now(), Summary: &summary})
	for _, err := range persistErrs {
		sink.Emit(explog.Record{
			Type: explog.RecError, RunID: opts.RunID, Time: time.Now(),
			Failure: &explog.Failure{Stage: "persist", Err: err.Error()},
		})
	}

	return &Result{
		RunID: opts.RunID, Summary: summary, Outcome: runResult.Outcome,
		Run: runResult, Agents: agents,
	}, nil
}

// buildAgents constructs each agent and wires its inference trace into the
// record stream, so the agent itself knows nothing about logging. longLived
// enables the world-mode options (self-scheduling, own-note recall);
// logRunID is the run id stamped
// on the agents' llm_call records, which in world mode is the session rather
// than the memory scope.
func buildAgents(opts Options, sink explog.Sink, longLived bool, logRunID string) ([]*agent.Agent, []sim.Actor, error) {
	return buildAgentsWith(opts, sink, longLived, logRunID, agent.Cognition{})
}

func buildAgentsWith(opts Options, sink explog.Sink, longLived bool, logRunID string, cog agent.Cognition) ([]*agent.Agent, []sim.Actor, error) {
	agents := make([]*agent.Agent, 0, len(opts.Personas))
	actors := make([]sim.Actor, 0, len(opts.Personas))

	for _, p := range opts.Personas {
		provider, err := opts.Providers.Get(p.Model.Provider)
		if err != nil {
			return nil, nil, fmt.Errorf("agent %s: %w", p.ID, err)
		}
		a, err := agent.New(agent.Options{
			Persona: p, Provider: provider, Store: opts.Store, RunID: opts.RunID,
			DualAction: opts.DualAction, Planning: opts.Planning, BeliefProvenance: opts.BeliefProvenance,
			Intentions: longLived, RecallOwnNotes: longLived, Cognition: cog,
		})
		if err != nil {
			return nil, nil, err
		}
		providerName := p.Model.Provider
		runID := logRunID
		a.Hook = func(tr agent.Trace) {
			rec := explog.Record{
				Type: explog.RecLLMCall, RunID: runID, Round: tr.Round, Time: time.Now(),
				Call: &explog.Call{
					Agent: tr.Agent, Model: tr.Model.Model, Provider: providerName,
					Attempt: tr.Attempt, PromptTokens: tr.Usage.PromptTokens,
					OutputTokens: tr.Usage.CompletionTokens, LatencyMS: tr.Latency.Milliseconds(),
					Raw: tr.Raw, Prompt: tr.Request, Thinking: tr.Thinking,
				},
			}
			if tr.Err != nil {
				rec.Call.Err = tr.Err.Error()
			}
			sink.Emit(rec)
		}
		agents = append(agents, a)
		actors = append(actors, a)
	}
	return agents, actors, nil
}

// Preflight checks every distinct model the cast references, so a missing model
// fails fast with a fixable message instead of mid-experiment.
func Preflight(ctx context.Context, reg *llm.Registry, personas []agent.Persona) error {
	type key struct{ provider, model, endpoint string }
	seen := map[key]bool{}
	for _, p := range personas {
		k := key{p.Model.Provider, p.Model.Model, p.Model.Endpoint}
		if seen[k] {
			continue
		}
		seen[k] = true
		provider, err := reg.Get(p.Model.Provider)
		if err != nil {
			return fmt.Errorf("agent %s: %w", p.ID, err)
		}
		if err := provider.Preflight(ctx, p.Model); err != nil {
			return err
		}
	}
	return nil
}

// planRevisions reports a nil plan's revision count as 0 rather than
// requiring every caller to nil-check memory.Plan itself.
func planRevisions(p *memory.Plan) int {
	if p == nil {
		return 0
	}
	return p.Revisions
}

func durString(d time.Duration) string {
	if d <= 0 {
		return ""
	}
	return d.String()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
