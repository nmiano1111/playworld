package runner

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/world"
)

// WorldOptions is one execution session over a persistent world.
type WorldOptions struct {
	Store world.Store
	Name  string
	// Personas must cover exactly the world's residents; the caller loads
	// them from the world's roster (or an override).
	Personas  []agent.Persona
	Providers *llm.Registry
	// Span is how much simulated time to advance.
	Span   time.Duration
	Limits world.Limits

	// Overrides for the protocol options stored with the world. Nil keeps
	// the world's own setting.
	DualAction *bool
	Planning   *bool
	Beliefs    *bool
	// Cognition overrides the world's cognition mode for this session only
	// ("grounded" or "legacy"); the stored setting is left unchanged.
	Cognition string
	// Introspect logs each activation's agent view (see explog.RecView) and
	// keeps prompts and raw replies in the log.
	Introspect bool

	// Sinks receive the record stream in addition to the session's own
	// JSONL log, which is always written under the world's logs directory.
	Sinks []explog.Sink
	// VerboseLog keeps prompts and raw replies in the JSONL log.
	VerboseLog bool
	Harness    string
	Preflight  bool
}

// WorldResult is what a session reports.
type WorldResult struct {
	World   *world.World
	Session world.Session
	Summary explog.Summary
	Run     *world.Result
	Agents  []*agent.Agent
}

// WorldRunID is the run id every resident's memory is scoped to. It is the
// same for every session of a world, which is what lets an agent's plan and
// short-term recollection carry across a resume.
func WorldRunID(name string) string { return "world-" + name }

// ExecuteWorld loads a world, runs one bounded session over it, and saves
// everything back: the world (including its pending queue) and each
// resident's memory. A session cut short by a limit or Ctrl-C is still saved
// and resumable.
func ExecuteWorld(ctx context.Context, opts WorldOptions) (*WorldResult, error) {
	if opts.Providers == nil {
		return nil, fmt.Errorf("runner: no provider registry")
	}
	if opts.Span <= 0 {
		return nil, fmt.Errorf("runner: session length must be positive")
	}
	w, err := opts.Store.Load(opts.Name)
	if err != nil {
		return nil, err
	}
	storedCognition := w.Settings.Cognition
	if opts.Cognition != "" {
		switch opts.Cognition {
		case world.CognitionGrounded, world.CognitionLegacy:
		default:
			return nil, fmt.Errorf(`cognition must be "grounded" or "legacy", not %q`, opts.Cognition)
		}
		w.Settings.Cognition = opts.Cognition
	}
	personas, err := worldPersonas(w, opts.Personas)
	if err != nil {
		return nil, err
	}
	if opts.Preflight {
		if err := Preflight(ctx, opts.Providers, personas); err != nil {
			return nil, err
		}
	}
	return executeWorld(ctx, opts, w, personas, storedCognition)
}

// worldPersonas orders personas as the world's residents, requiring one for
// each.
func worldPersonas(w *world.World, given []agent.Persona) ([]agent.Persona, error) {
	byID := map[sim.AgentID]agent.Persona{}
	for _, p := range given {
		byID[p.AgentID()] = p
	}
	personas := make([]agent.Persona, 0, len(w.Order))
	for _, id := range w.Order {
		p, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("world %s: no persona for resident %q", w.Name, id)
		}
		personas = append(personas, p)
	}
	return personas, nil
}

// AgentCognition is the agent-side counterpart of a world's cognition mode.
func AgentCognition(s world.Settings) agent.Cognition {
	if !s.Grounded() {
		return agent.Cognition{}
	}
	return agent.Cognition{CompactTranscript: true, SalientRecall: 6, PlanContext: true, RepairSlots: true, TypedBeliefs: true, RecentLimit: 24,
		ExplicitAffordances: true, PlanTracking: true}
}

func executeWorld(ctx context.Context, opts WorldOptions, w *world.World, personas []agent.Persona, storedCognition string) (*WorldResult, error) {
	settings := w.Settings
	pick := func(override *bool, stored bool) bool {
		if override != nil {
			return *override
		}
		return stored
	}
	dual, planning, beliefs := pick(opts.DualAction, settings.DualAction), pick(opts.Planning, settings.Planning), pick(opts.Beliefs, settings.Beliefs)

	number := len(w.Sessions) + 1
	sessionID := fmt.Sprintf("%s-s%03d-%s", w.Name, number, time.Now().Format("20060102-150405"))
	logPath := filepath.Join(opts.Store.LogsDir(w.Name), sessionID+".jsonl")
	jsonl, err := explog.NewJSONL(logPath, opts.VerboseLog || opts.Introspect)
	if err != nil {
		return nil, err
	}
	defer jsonl.Close()

	tally := explog.NewTally()
	sink := explog.NewSequencer(explog.Multi(append(append([]explog.Sink{jsonl}, opts.Sinks...), tally)))

	store, err := memory.NewFileStore(opts.Store.MemoryDir(w.Name))
	if err != nil {
		return nil, err
	}
	runID := WorldRunID(w.Name)
	agents, actors, err := buildAgentsWith(Options{
		Personas: personas, Providers: opts.Providers, Store: store, RunID: runID,
		DualAction: dual, Planning: planning, BeliefProvenance: beliefs,
	}, sink, true, sessionID, AgentCognition(settings))
	if err != nil {
		return nil, err
	}
	names := w.Names()
	for _, a := range agents {
		a.Bind(w.Order, names)
		a.ResumeTranscript()
		id := a.ID()
		a.OnPlan = func(p *memory.Plan) {
			var snapshot *memory.Plan
			if p != nil {
				cp := *p
				snapshot = &cp
			}
			sink.Emit(explog.Record{Type: explog.RecPlan, RunID: sessionID, Time: time.Now(), SimStamp: world.Stamp(w.Now),
				Plan: &explog.PlanUpdate{Agent: id, Plan: snapshot}})
		}
		a.OnMemory = func(kind, subject, text string) {
			sink.Emit(explog.Record{Type: explog.RecMemory, RunID: sessionID, Time: time.Now(), SimStamp: world.Stamp(w.Now),
				Memory: &explog.MemoryUpdate{Agent: id, Kind: kind, Subject: subject, Text: text}})
		}
	}

	persist := func() error {
		for _, a := range agents {
			if err := a.Persist(); err != nil {
				return err
			}
		}
		// A session-only cognition override is never written back.
		session := w.Settings.Cognition
		w.Settings.Cognition = storedCognition
		defer func() { w.Settings.Cognition = session }()
		return opts.Store.Save(w)
	}
	statsBefore := make([]agent.Stats, len(agents))
	for i, a := range agents {
		statsBefore[i] = a.Stats()
	}

	limits := opts.Limits
	if limits.MaxAgentErrors == 0 {
		limits.MaxAgentErrors = 3
	}
	days := opts.Span.Hours() / 24
	if limits.MaxActivations == 0 {
		// Every resident using its full daily allowance, which in practice
		// is never reached; the allowance itself is the real bound.
		limits.MaxActivations = int(float64(len(agents)*settings.Scheduling.MaxActivationsPerDay)*max(days, 1/24.0)) + len(agents)
	}
	if limits.MaxLLMCalls == 0 {
		// A laptop-sized session: about 100 calls, fewer for short windows.
		limits.MaxLLMCalls = min(100, limits.MaxActivations*2+10)
	}
	if limits.MaxEvents == 0 {
		limits.MaxEvents = 50 * limits.MaxLLMCalls
	}

	engine, err := world.NewEngine(world.Config{
		World: w, SessionID: sessionID, RunID: runID, Agents: actors, Sink: sink,
		Limits: limits, Checkpoint: persist, Introspect: opts.Introspect,
	})
	if err != nil {
		return nil, err
	}

	from, until := w.Now, w.Now.Add(opts.Span)
	started := time.Now()
	sink.Emit(explog.Record{
		Type: explog.RecRunStart, RunID: sessionID, Time: started,
		Run: &explog.RunInfo{
			RunID: sessionID, Scenario: "world", Started: started, Seed: w.Seed,
			MaxRounds: limits.MaxActivations, MaxCalls: limits.MaxLLMCalls,
			Timeout: durString(limits.Timeout), MemoryDir: store.Dir, Harness: opts.Harness,
			Params: map[string]any{"dual_actions": dual, "planning": planning, "beliefs": beliefs, "cognition": cognitionName(settings)},
			World:  &explog.WorldSession{World: w.Name, Session: number, From: world.Stamp(from), To: world.Stamp(until)},
		},
	})
	for _, a := range agents {
		p := a.Persona()
		sink.Emit(explog.Record{
			Type: explog.RecAgent, RunID: sessionID, Time: time.Now(),
			Agent: &explog.AgentInfo{
				ID: a.ID(), Name: a.Name(), Role: p.Role,
				Persona: truncate(p.Personality, 200), Goals: p.Goals,
				Model: p.Model, Episodes: len(a.State().Episodes),
			},
		})
	}

	res := engine.Run(ctx, until)
	for i, a := range agents {
		st := a.Stats()
		res.Metrics["prompt_tokens"] += st.PromptTokens - statsBefore[i].PromptTokens
		res.Metrics["output_tokens"] += st.OutputTokens - statsBefore[i].OutputTokens
		if n := st.SlotRepairs - statsBefore[i].SlotRepairs; n > 0 {
			res.Metrics["slot_repairs"] += n
			res.PerAgent[a.ID()]["slot_repairs"] = n
		}
		for k, v := range st.Counts {
			if n := v - statsBefore[i].Counts[k]; n > 0 {
				res.Metrics[k] += n
				res.PerAgent[a.ID()][k] = n
			}
		}
	}
	res.Metrics["sim_minutes"] = int(res.To.Sub(res.From).Minutes())
	// Marks a session that records the decision-coherence counters, so a
	// comparison can tell their zero from their absence.
	res.Metrics["counters_v2"] = 1

	session := world.Session{
		Number: number, ID: sessionID, Started: started.UTC(), Ended: time.Now().UTC(),
		SimFrom: res.From, SimTo: res.To, StopReason: res.StopReason,
		Activations: res.Activations, LLMCalls: res.LLMCalls, LogPath: logPath, Metrics: res.Metrics,
	}
	w.Sessions = append(w.Sessions, session)
	saveErr := persist()

	outcome := w.Outcome(res)
	sink.Emit(explog.Record{Type: explog.RecOutcome, RunID: sessionID, Time: time.Now(), Outcome: &outcome})
	for _, a := range agents {
		tally.SetAgentCounts(a.ID(), len(a.State().Commitments), len(a.State().Beliefs))
		plan := a.State().ActivePlan(runID)
		provenance := 0
		for _, b := range a.State().Beliefs {
			if b.Basis != "" || b.Confidence != "" {
				provenance++
			}
		}
		tally.SetPlanAndBeliefCounts(a.ID(), plan != nil, planRevisions(plan), provenance)
	}
	summary := tally.Summary(sessionID, "world", res.StopReason, res.Duration, logPath)
	sink.Emit(explog.Record{Type: explog.RecSummary, RunID: sessionID, Time: time.Now(), Summary: &summary})
	if saveErr != nil {
		return nil, fmt.Errorf("world %s: session ran but could not be saved: %w", w.Name, saveErr)
	}
	if err := jsonl.Close(); err != nil {
		return nil, err
	}
	return &WorldResult{World: w, Session: session, Summary: summary, Run: res, Agents: agents}, nil
}

func cognitionName(s world.Settings) string {
	if s.Grounded() {
		return world.CognitionGrounded
	}
	return world.CognitionLegacy
}

// PreviewWorld renders, without calling any model or changing anything on
// disk, the exact prompt one resident would be sent if it were activated
// now — for the experimenter's inspection. cognition, if set, previews
// under that mode instead of the world's own.
func PreviewWorld(store world.Store, name string, personas []agent.Persona, who sim.AgentID, cognition string) ([]llm.Message, error) {
	w, err := store.Load(name)
	if err != nil {
		return nil, err
	}
	if cognition != "" {
		w.Settings.Cognition = cognition
	}
	if _, ok := w.Residents[who]; !ok {
		return nil, fmt.Errorf("world %s has no resident %q", name, who)
	}
	personas, err = worldPersonas(w, personas)
	if err != nil {
		return nil, err
	}
	var persona agent.Persona
	for _, p := range personas {
		if p.AgentID() == who {
			persona = p
		}
	}
	mem, err := memory.NewFileStore(store.MemoryDir(name))
	if err != nil {
		return nil, err
	}
	// The agent is built over a read-only view of its memory: it is never
	// persisted, so nothing it does here reaches disk.
	a, err := agent.New(agent.Options{
		Persona: persona, Provider: previewProvider{}, Store: readOnly{mem}, RunID: WorldRunID(name),
		DualAction: w.Settings.DualAction, Planning: w.Settings.Planning, BeliefProvenance: w.Settings.Beliefs,
		Intentions: true, RecallOwnNotes: true, Cognition: AgentCognition(w.Settings),
	})
	if err != nil {
		return nil, err
	}
	a.Bind(w.Order, w.Names())
	a.ResumeTranscript()
	return a.PreviewPrompt(w.TurnFor(WorldRunID(name), who, w.PendingReasons(who))), nil
}

// previewProvider is never called; PreviewPrompt does no inference.
type previewProvider struct{}

func (previewProvider) Name() string { return "preview" }
func (previewProvider) Chat(context.Context, llm.Request) (*llm.Response, error) {
	return nil, fmt.Errorf("preview does not run inference")
}
func (previewProvider) Preflight(context.Context, llm.ModelConfig) error { return nil }

// readOnly is a memory store that loads but refuses to write.
type readOnly struct{ memory.Store }

func (readOnly) Save(*memory.State) error { return fmt.Errorf("read-only memory") }
func (readOnly) Reset(...string) error    { return fmt.Errorf("read-only memory") }
