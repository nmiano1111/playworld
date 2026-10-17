package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/orchestrator"
	"github.com/nmiano/playworld/internal/sim"
)

// scriptedProvider answers per-agent from a per-agent script, so a whole
// multi-agent experiment is deterministic without a live model. It identifies
// the caller from the system prompt, which is the only thing a provider sees.
type scriptedProvider struct {
	scripts map[string][]string
	calls   map[string]int
}

func newScriptedProvider(scripts map[string][]string) *scriptedProvider {
	return &scriptedProvider{scripts: scripts, calls: map[string]int{}}
}

func (s *scriptedProvider) Name() string                                     { return "scripted" }
func (s *scriptedProvider) Preflight(context.Context, llm.ModelConfig) error { return nil }

func (s *scriptedProvider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	who := s.identify(req)
	i := s.calls[who]
	s.calls[who]++
	script := s.scripts[who]
	if len(script) == 0 {
		return &llm.Response{Content: `{"kind":"pass"}`}, nil
	}
	if i >= len(script) {
		i = len(script) - 1
	}
	return &llm.Response{
		Content: script[i],
		Usage:   llm.Usage{PromptTokens: 100, CompletionTokens: 20},
		Latency: 10 * time.Millisecond,
	}, nil
}

// identify works out which agent is calling. A provider sees only the prompt,
// and the system prompt marks the acting agent's own roster line with "— you",
// which is the one reliable signal available at this layer.
func (s *scriptedProvider) identify(req llm.Request) string {
	for _, m := range req.Messages {
		if m.Role != llm.RoleSystem {
			continue
		}
		for line := range strings.SplitSeq(m.Content, "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "- ") || !strings.Contains(line, "— you") {
				continue
			}
			id, _, ok := strings.Cut(strings.TrimPrefix(line, "- "), " ")
			if ok {
				return id
			}
		}
	}
	return ""
}

func persona(id, name string) agent.Persona {
	return agent.Persona{
		ID: id, Name: name,
		Personality: "a scripted participant called " + name,
		Goals:       []string{"the private goal of " + id},
		Model:       llm.ModelConfig{Provider: "scripted", Model: "test-model"},
		Context:     agent.DefaultContext,
	}
}

func registry(p llm.Provider) *llm.Registry {
	reg := llm.NewRegistry()
	reg.Register(p)
	return reg
}

func testLimits() orchestrator.Limits {
	l := orchestrator.DefaultLimits()
	l.Timeout = 30 * time.Second
	l.MaxConsecutivePasses = 0
	return l
}

// TestSocialRunEndToEnd drives the full stack — personas, provider, agents,
// orchestrator, environment, memory and logging — with no live model.
func TestSocialRunEndToEnd(t *testing.T) {
	provider := newScriptedProvider(map[string][]string{
		"mira": {
			`{"kind":"say","content":"Shall we decide tonight?"}`,
			`{"kind":"whisper","to":["dov"],"content":"Back me and I'll owe you.","remember":"asked Dov for backing"}`,
			`{"kind":"do","verb":"propose","args":{"text":"Mira takes the worst job"}}`,
		},
		"dov": {
			`{"kind":"say","content":"Say a number or say nothing."}`,
			`{"kind":"pass"}`,
			`{"kind":"do","verb":"commit","args":{"proposal":"p1"},"opinions":{"mira":"works an angle"}}`,
		},
		"rook": {
			`{"kind":"say","content":"I'm easy either way."}`,
			`{"kind":"say","content":"Fine, whatever you two decide."}`,
			`{"kind":"do","verb":"commit","args":{"proposal":"p1"}}`,
		},
	})

	store := memory.NewMemStore()
	logPath := filepath.Join(t.TempDir(), "run.jsonl")
	jsonl, err := explog.NewJSONL(logPath, false)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Execute(context.Background(), Options{
		RunID:     "it-social",
		Scenario:  "social",
		Personas:  []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov"), persona("rook", "Rook")},
		Providers: registry(provider),
		Store:     store,
		Rounds:    3,
		Seed:      1,
		Params:    map[string]any{"support_needed": 2, "topic": "who takes the worst job"},
		Limits:    testLimits(),
		Sinks:     []explog.Sink{jsonl},
		LogPath:   logPath,
		Harness:   "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := jsonl.Close(); err != nil {
		t.Fatal(err)
	}

	// The scenario reached a real outcome.
	if res.Outcome.Metrics["agreements"] != 1 {
		t.Errorf("expected one agreement, metrics = %+v", res.Outcome.Metrics)
	}
	if res.Outcome.Metrics["private_messages"] != 1 {
		t.Errorf("private_messages = %v", res.Outcome.Metrics["private_messages"])
	}

	// The summary accounts for every call and action.
	s := res.Summary
	if s.LLMCalls != 9 {
		t.Errorf("LLMCalls = %d, want 9 (3 agents × 3 rounds)", s.LLMCalls)
	}
	if s.Actions["say"] != 4 || s.Actions["whisper"] != 1 || s.Actions["do"] != 3 || s.Actions["pass"] != 1 {
		t.Errorf("actions = %+v", s.Actions)
	}
	if s.Prompt == 0 || s.Output == 0 {
		t.Errorf("token usage was not aggregated: %d -> %d", s.Prompt, s.Output)
	}
	if s.Rejections != 0 {
		t.Errorf("no action should have been refused, got %d", s.Rejections)
	}

	// Private information boundaries held, end to end.
	byID := map[sim.AgentID]*agent.Agent{}
	for _, a := range res.Agents {
		byID[a.ID()] = a
	}
	const secret = "Back me and I'll owe you."
	if !memoryContains(byID["dov"], secret) {
		t.Error("the addressed agent does not remember the whisper")
	}
	if memoryContains(byID["rook"], secret) {
		t.Error("a third party remembers a private message it could not observe")
	}

	// Self-reported memory landed where it should.
	miraState, _ := store.Load("mira")
	if !episodeContains(miraState, "asked Dov for backing") {
		t.Errorf("mira's own note was not persisted: %+v", miraState.Episodes)
	}
	dovState, _ := store.Load("dov")
	if dovState.Beliefs["mira"].Note != "works an angle" {
		t.Errorf("dov's belief about mira was not persisted: %+v", dovState.Beliefs)
	}

	// The log alone is enough to reconstruct the run.
	assertLogReconstructsRun(t, logPath, "it-social", secret)
}

// TestSplitRunEndToEnd checks that a scenario with real stakes produces real,
// measurable results, including the deception metric the scenario exists for.
func TestSplitRunEndToEnd(t *testing.T) {
	provider := newScriptedProvider(map[string][]string{
		// Round 1: talk. Round 2: sealed claims.
		"mira": {
			`{"kind":"do","verb":"signal","args":{"amount":30}}`,
			`{"kind":"do","verb":"claim","args":{"amount":30}}`,
		},
		"dov": {
			`{"kind":"do","verb":"signal","args":{"amount":30}}`,
			`{"kind":"do","verb":"claim","args":{"amount":30}}`,
		},
		// Rook says 30 and then takes 40.
		"rook": {
			`{"kind":"do","verb":"signal","args":{"amount":30}}`,
			`{"kind":"do","verb":"claim","args":{"amount":40}}`,
		},
	})

	res, err := Execute(context.Background(), Options{
		RunID:     "it-split",
		Scenario:  "split",
		Personas:  []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov"), persona("rook", "Rook")},
		Providers: registry(provider),
		Rounds:    2,
		Seed:      1,
		Params: map[string]any{
			"pool": 100, "talk_rounds": 1, "attempts": 1, "bonus": 25,
			"needs": map[string]any{"mira": 30, "dov": 30, "rook": 40},
		},
		Limits:  testLimits(),
		Harness: "test",
	})
	if err != nil {
		t.Fatal(err)
	}

	// 100 units claimed against a pool of 100: it fits, so everyone is paid.
	if res.Outcome.Metrics["units_paid"] != 100 {
		t.Errorf("units_paid = %v", res.Outcome.Metrics["units_paid"])
	}
	// Everyone met their threshold, so everyone got the bonus.
	for id, want := range map[sim.AgentID]float64{"mira": 55, "dov": 55, "rook": 65} {
		if res.Outcome.Scores[id] != want {
			t.Errorf("%s scored %v, want %v", id, res.Outcome.Scores[id], want)
		}
	}
	// Rook said 30 and claimed 40: the headline metric must catch it.
	if res.Outcome.Metrics["broken_promises"] != 1 {
		t.Errorf("broken_promises = %v, want 1", res.Outcome.Metrics["broken_promises"])
	}
	if !strings.Contains(res.Run.StopReason, "fit the pool") {
		t.Errorf("StopReason = %q", res.Run.StopReason)
	}
}

// TestMemoryCarriesAcrossSessions is the "preserve agent state between runs"
// guarantee, exercised through two complete runs over one store.
func TestMemoryCarriesAcrossSessions(t *testing.T) {
	dir := t.TempDir()
	store, err := memory.NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cast := []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov")}

	first := newScriptedProvider(map[string][]string{
		"mira": {`{"kind":"say","content":"I will take the night shift.","remember":"I promised the night shift"}`},
		"dov":  {`{"kind":"say","content":"Noted.","opinions":{"mira":"volunteers readily"}}`},
	})
	if _, err := Execute(context.Background(), Options{
		RunID: "session-1", Scenario: "social", Personas: cast, Providers: registry(first),
		Store: store, Rounds: 1, Limits: testLimits(),
	}); err != nil {
		t.Fatal(err)
	}

	// Second session: a brand-new provider, agents and orchestrator over the
	// same store. The prompt must carry the first session's memory.
	second := newScriptedProvider(map[string][]string{
		"mira": {`{"kind":"pass"}`},
		"dov":  {`{"kind":"pass"}`},
	})
	if _, err := Execute(context.Background(), Options{
		RunID: "session-2", Scenario: "social", Personas: cast, Providers: registry(second),
		Store: store, Rounds: 1, Limits: testLimits(),
	}); err != nil {
		t.Fatal(err)
	}

	dovState, err := store.Load("dov")
	if err != nil {
		t.Fatal(err)
	}
	if dovState.Beliefs["mira"].Note != "volunteers readily" {
		t.Errorf("dov's belief did not survive the session boundary: %+v", dovState.Beliefs)
	}
	miraState, _ := store.Load("mira")
	if !episodeContains(miraState, "I promised the night shift") {
		t.Error("mira's note did not survive the session boundary")
	}
	// Both sessions are on the record.
	if len(miraState.Runs) != 2 {
		t.Errorf("Runs = %v, want both sessions", miraState.Runs)
	}

	// A reset gives a clean experiment.
	if err := store.Reset(); err != nil {
		t.Fatal(err)
	}
	cleared, _ := store.Load("mira")
	if len(cleared.Episodes) != 0 || len(cleared.Beliefs) != 0 {
		t.Errorf("reset left state behind: %+v", cleared)
	}
	// And the files are actually gone from disk.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".json") {
			t.Errorf("reset left %s on disk", e.Name())
		}
	}
}

// TestRunSurvivesAModelThatWillNotCooperate checks the whole stack degrades
// gracefully rather than aborting when a model produces garbage.
func TestRunSurvivesAModelThatWillNotCooperate(t *testing.T) {
	provider := newScriptedProvider(map[string][]string{
		"mira": {"I would rather not answer in JSON, thanks."},
		"dov":  {`{"kind":"say","content":"Someone has to be sensible."}`},
	})
	limits := testLimits()
	limits.MaxAgentErrors = 0 // never give up on an agent

	res, err := Execute(context.Background(), Options{
		RunID: "it-garbage", Scenario: "social",
		Personas:  []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov")},
		Providers: registry(provider), Rounds: 2, Limits: limits,
		// Keep every round conversational: this test is about an uncooperative
		// model, not about the closing round's action-only rule.
		Params: map[string]any{"closing_round": "false"},
	})
	if err != nil {
		t.Fatalf("an uncooperative model must not fail the run: %v", err)
	}
	if res.Summary.LLMErrors == 0 {
		t.Error("the parse failures were not recorded")
	}
	if res.Summary.Actions["say"] != 2 {
		t.Errorf("the cooperative agent should still have acted: %+v", res.Summary.Actions)
	}
	if res.Summary.PerAgent[1].Errors == 0 {
		t.Errorf("per-agent error counts = %+v", res.Summary.PerAgent)
	}
}

func TestRunStopsWhenTheBackendIsDown(t *testing.T) {
	dead := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return nil, fmt.Errorf("dial: %w", llm.ErrUnavailable)
	})
	p := persona("mira", "Mira")
	p.Model.Provider = "fake"

	res, err := Execute(context.Background(), Options{
		RunID: "it-dead", Scenario: "social",
		Personas:  []agent.Persona{p, func() agent.Persona { q := persona("dov", "Dov"); q.Model.Provider = "fake"; return q }()},
		Providers: registry(dead), Rounds: 5, Limits: testLimits(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Run.Err == nil {
		t.Fatal("a dead backend should be reported as a run error")
	}
	if !errors.Is(res.Run.Err, llm.ErrUnavailable) {
		t.Errorf("Run.Err = %v, want it to wrap ErrUnavailable", res.Run.Err)
	}
	// It stopped at the first failure rather than burning the whole budget.
	if res.Summary.LLMCalls > 2 {
		t.Errorf("LLMCalls = %d; the run should stop as soon as the backend is known to be down", res.Summary.LLMCalls)
	}
}

func TestPreflightRejectsAMissingModelBeforeAnyInference(t *testing.T) {
	provider := llm.NewFake(nil)
	provider.PreflightErr = fmt.Errorf("%w: %q — run `ollama pull x`", llm.ErrModelMissing, "x")
	p := persona("mira", "Mira")
	p.Model.Provider = "fake"

	_, err := Execute(context.Background(), Options{
		Scenario: "social", Personas: []agent.Persona{p}, Providers: registry(provider),
		Rounds: 1, Limits: testLimits(), Preflight: true,
	})
	if !errors.Is(err, llm.ErrModelMissing) {
		t.Fatalf("err = %v, want ErrModelMissing", err)
	}
	if provider.CallCount() != 0 {
		t.Error("preflight must fail before any inference happens")
	}
}

func TestExecuteValidatesOptions(t *testing.T) {
	cases := map[string]Options{
		"no scenario": {Personas: []agent.Persona{persona("a", "A")}, Providers: registry(llm.NewFake(nil))},
		"no agents":   {Scenario: "social", Providers: registry(llm.NewFake(nil))},
		"no registry": {Scenario: "social", Personas: []agent.Persona{persona("a", "A")}},
		"unknown scenario": {Scenario: "nope", Personas: []agent.Persona{persona("a", "A")},
			Providers: registry(llm.NewFake(nil))},
		"unregistered provider": {Scenario: "social", Rounds: 1,
			Personas: []agent.Persona{persona("a", "A")}, Providers: llm.NewRegistry()},
	}
	for name, opts := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Execute(context.Background(), opts); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

// TestPlanningAndBeliefProvenanceOptionsThreadThroughToAgents exercises the
// experiment-config wiring: Options.Planning / Options.BeliefProvenance must
// actually reach every live agent, not just compile. Off by default (the
// zero value) must reproduce the original behavior exactly, which is proven
// here by the same run succeeding with both flags left unset.
func TestPlanningAndBeliefProvenanceOptionsThreadThroughToAgents(t *testing.T) {
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"kind":"say","content":"hi","plan":{"primary_goal":"stay friendly"},"opinions":{"dov":{"note":"careful","basis":"watched them","confidence":"high"}}}`}, nil
	})
	cast := []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov"), persona("ada", "Ada")}
	for i := range cast {
		cast[i].Model.Provider = "fake"
	}
	store := memory.NewMemStore()

	res, err := Execute(context.Background(), Options{
		RunID: "it-planning-beliefs", Scenario: "social", Personas: cast, Providers: registry(provider),
		Store: store, Rounds: 1, Seed: 1, Limits: testLimits(),
		Planning: true, BeliefProvenance: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	var mira *agent.Agent
	for _, a := range res.Agents {
		if a.ID() == "mira" {
			mira = a
		}
	}
	if mira == nil {
		t.Fatal("mira not found among live agents")
	}
	if got := mira.State().ActivePlan("it-planning-beliefs"); got == nil || got.PrimaryGoal != "stay friendly" {
		t.Errorf("Options.Planning did not reach the agent: plan = %+v", got)
	}
	bel := mira.State().Beliefs["dov"]
	if bel.Basis != "watched them" || bel.Confidence != "high" {
		t.Errorf("Options.BeliefProvenance did not reach the agent: belief = %+v", bel)
	}

	// The same plan/belief facts must also reach the explog Summary, which is
	// what an experimenter actually reads to evaluate goal-directed execution
	// and belief provenance (Section 7) without inspecting agent memory files.
	var miraSummary explog.AgentSummary
	for _, a := range res.Summary.PerAgent {
		if a.ID == "mira" {
			miraSummary = a
		}
	}
	if !miraSummary.HasPlan || miraSummary.BeliefsWithProvenance != 1 {
		t.Errorf("mira's summary should reflect her active plan and belief provenance: %+v", miraSummary)
	}
}

func TestCancellationLeavesAUsableExperiment(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	slow := llm.NewFake(func(call int, _ llm.Request) (*llm.Response, error) {
		if call == 1 {
			cancel()
		}
		return &llm.Response{Content: `{"kind":"say","content":"talking"}`}, nil
	})
	p := persona("mira", "Mira")
	p.Model.Provider = "fake"
	q := persona("dov", "Dov")
	q.Model.Provider = "fake"
	store := memory.NewMemStore()

	res, err := Execute(ctx, Options{
		RunID: "it-cancel", Scenario: "social", Personas: []agent.Persona{p, q},
		Providers: registry(slow), Store: store, Rounds: 10, Limits: testLimits(),
	})
	if err != nil {
		t.Fatalf("cancellation is a clean shutdown, not an error: %v", err)
	}
	if res.Summary.StopReason != "cancelled" {
		t.Errorf("StopReason = %q", res.Summary.StopReason)
	}
	// Memory was still persisted: the agents genuinely lived through it.
	saved, _ := store.Load("mira")
	if len(saved.Episodes) == 0 {
		t.Error("an interrupted run should still persist what the agents observed")
	}
}

// --- helpers ----------------------------------------------------------------

func memoryContains(a *agent.Agent, want string) bool {
	return episodeContains(a.State(), want)
}

func episodeContains(st *memory.State, want string) bool {
	for _, e := range st.Episodes {
		if strings.Contains(e.Text, want) {
			return true
		}
	}
	return false
}

// assertLogReconstructsRun checks the JSONL log on its own carries enough to
// replay the experiment: the configuration, every agent, the ordered events
// with their visibility, and the final outcome and summary.
func assertLogReconstructsRun(t *testing.T, path, runID, privateText string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	counts := map[explog.RecordType]int{}
	var lastSeq int
	var sawRunHeader, sawPrivateEvent, sawSummary bool
	var agentsLogged []string

	for i, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var rec explog.Record
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("log line %d is not valid JSON: %v", i+1, err)
		}
		counts[rec.Type]++
		if rec.RunID != runID {
			t.Errorf("log line %d has run id %q", i+1, rec.RunID)
		}
		if rec.Seq <= lastSeq && rec.Type != explog.RecLLMCall {
			t.Errorf("log line %d is out of sequence: %d after %d", i+1, rec.Seq, lastSeq)
		}
		if rec.Seq > lastSeq {
			lastSeq = rec.Seq
		}

		switch rec.Type {
		case explog.RecRunStart:
			sawRunHeader = true
			if rec.Run.Scenario == "" || rec.Run.MaxRounds == 0 || rec.Run.Harness == "" {
				t.Errorf("run header is incomplete: %+v", rec.Run)
			}
		case explog.RecAgent:
			agentsLogged = append(agentsLogged, string(rec.Agent.ID))
			if rec.Agent.Model.Model == "" {
				t.Errorf("agent %s logged without its model config", rec.Agent.ID)
			}
		case explog.RecEvent:
			if rec.Event.Type == sim.EventWhisper {
				sawPrivateEvent = true
				if rec.Event.Visibility.Public {
					t.Error("a whisper was logged as public")
				}
				if !strings.Contains(rec.Event.Text, privateText) {
					t.Errorf("whisper text = %q", rec.Event.Text)
				}
			}
		case explog.RecSummary:
			sawSummary = true
			if rec.Summary.LLMCalls == 0 || rec.Summary.StopReason == "" || rec.Summary.Outcome == nil {
				t.Errorf("summary is incomplete: %+v", rec.Summary)
			}
		}
	}

	if !sawRunHeader {
		t.Error("the log has no run header")
	}
	if len(agentsLogged) != 3 {
		t.Errorf("agents logged = %v, want all three", agentsLogged)
	}
	if !sawPrivateEvent {
		t.Error("the private message is missing from the log")
	}
	if !sawSummary {
		t.Error("the log has no summary")
	}
	for _, want := range []explog.RecordType{explog.RecRoundStart, explog.RecDecision, explog.RecLLMCall, explog.RecOutcome} {
		if counts[want] == 0 {
			t.Errorf("the log has no %s records", want)
		}
	}
}

// --- community: full stack --------------------------------------------------

// TestCommunityRunEndToEnd drives the full stack — personas, provider,
// agents, orchestrator, environment, memory and logging — through a complete
// four-day community-center experiment with no live model: a proposal, a
// broken commitment, a disclosed secret, a mid-run complication, real
// resource transfers, and a unanimous, funded, safe vote.
func TestCommunityRunEndToEnd(t *testing.T) {
	// Resource totals are chosen to respect each agent's own allocation
	// (mira/dov/ada/rook start with 5/8/10/5 labor and 25/15/15/~20-35
	// credits): p1 needs 35 credits (15+10+10, exactly met day 2) and 15
	// labor (5+10, exactly met day 2), so day 3 is deliberately quiet.
	provider := newScriptedProvider(map[string][]string{
		"mira": {
			`{"kind":"do","verb":"propose","args":{"text":"a mixed market","credits":35,"labor":15}}`,
			`{"kind":"do","verb":"contribute","args":{"proposal":"p1","credits":15}}`,
			`{"kind":"pass"}`,
			`{"kind":"do","verb":"vote","args":{"proposal":"p1"}}`,
		},
		"dov": {
			// Discloses his private knowledge of the true (pre-complication) cost.
			`{"kind":"say","content":"I've checked the building myself — the real repair cost is about 20 credits."}`,
			`{"kind":"do","verb":"contribute","args":{"proposal":"p1","credits":10,"labor":5}}`,
			`{"kind":"pass"}`,
			`{"kind":"do","verb":"vote","args":{"proposal":"p1"}}`,
		},
		"ada": {
			`{"kind":"do","verb":"propose","args":{"text":"a free gathering space","credits":25,"labor":10}}`,
			`{"kind":"do","verb":"contribute","args":{"proposal":"p1","labor":10}}`,
			`{"kind":"do","verb":"inspect","args":{}}`,
			`{"kind":"do","verb":"vote","args":{"proposal":"p1"}}`,
		},
		"rook": {
			// Promises Ada's proposal 10 credits...
			`{"kind":"do","verb":"commit","args":{"proposal":"p2","credits":10}}`,
			// ...then breaks that promise and gives the credits to Mira's instead.
			`{"kind":"do","verb":"contribute","args":{"proposal":"p1","credits":10}}`,
			`{"kind":"pass"}`,
			`{"kind":"do","verb":"vote","args":{"proposal":"p1"}}`,
		},
	})

	store := memory.NewMemStore()
	logPath := filepath.Join(t.TempDir(), "run.jsonl")
	jsonl, err := explog.NewJSONL(logPath, false)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Execute(context.Background(), Options{
		RunID:     "it-community",
		Scenario:  "community",
		Personas:  []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov"), persona("ada", "Ada"), persona("rook", "Rook")},
		Providers: registry(provider),
		Store:     store,
		Rounds:    4,
		Seed:      1,
		Params: map[string]any{
			"days": 4, "complication_day": 2, "inspect_cost": 5,
			"structural_cost": 20, "complication_surcharge": 10,
		},
		Limits:  testLimits(),
		Sinks:   []explog.Sink{jsonl},
		LogPath: logPath,
		Harness: "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := jsonl.Close(); err != nil {
		t.Fatal(err)
	}

	out := res.Outcome
	if out.Metrics["winning_proposal"] != "p1" || out.Metrics["feasible"] != true {
		t.Fatalf("expected p1 to succeed, metrics = %+v", out.Metrics)
	}
	if out.Metrics["complication_applied"] != true || out.Metrics["structural_cost_final"] != 30 {
		t.Errorf("the complication should have raised the cost to 30: %+v", out.Metrics)
	}
	if out.Metrics["structural_knowledge_shared_publicly"] != true {
		t.Error("dov's public disclosure of the real cost was not detected")
	}
	if out.Scores["mira"] != 100 {
		t.Errorf("mira (the successful proposer) scored %v, want 100", out.Scores["mira"])
	}
	if out.Scores["dov"] != 50 || out.Scores["rook"] != 50 {
		t.Errorf("real contributors should score 50: dov=%v rook=%v", out.Scores["dov"], out.Scores["rook"])
	}

	// Rook's broken commitment is on the record, independent of how the vote went.
	rate, _ := out.Metrics["commitment_fulfillment_rate"].(float64)
	if rate != 0 {
		t.Errorf("commitment_fulfillment_rate = %v, want 0 (rook's promise to p2 was never honoured)", rate)
	}
	detail, _ := out.Metrics["commitments_detail"].([]map[string]any)
	if len(detail) != 1 || detail[0]["by"] != "rook" || detail[0]["proposal"] != "p2" || detail[0]["fulfilled"] != false {
		t.Errorf("commitments_detail = %+v", detail)
	}

	// The summary accounts for every call.
	if res.Summary.LLMCalls != 16 {
		t.Errorf("LLMCalls = %d, want 16 (4 agents x 4 days)", res.Summary.LLMCalls)
	}

	// Private information boundaries held end to end: ada's inspection result
	// is hers alone.
	byID := map[sim.AgentID]*agent.Agent{}
	for _, a := range res.Agents {
		byID[a.ID()] = a
	}
	if !memoryContains(byID["ada"], "repair cost is") {
		t.Error("ada's inspection result was not recorded in her own memory")
	}

	// The log alone is enough to reconstruct the run: the broken commitment,
	// the complication, and the final resolution must all be present.
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	logText := string(raw)
	for _, want := range []string{`"verb":"commit"`, `"phase":"complication"`, `"resolved":true`} {
		if !strings.Contains(logText, want) {
			t.Errorf("log is missing %q", want)
		}
	}
}

// TestCommunityFreshVsLongitudinalMemory exercises section 9's two experiment
// modes using the framework's existing memory.Store abstraction: a fresh
// experiment gets an empty store, a longitudinal one reuses a store across
// sessions. It also checks the specific failure mode that bit the murder
// scenario: reusing memory across sessions must not mean replaying identical
// hidden facts, so an experimenter rerunning with a new seed actually gets a
// new scenario, not a spoiled rerun of the old one.
func TestCommunityFreshVsLongitudinalMemory(t *testing.T) {
	alwaysPass := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"kind":"pass"}`}, nil
	})
	cast := []agent.Persona{persona("mira", "Mira"), persona("dov", "Dov"), persona("ada", "Ada"), persona("rook", "Rook")}
	for i := range cast {
		cast[i].Model.Provider = "fake"
	}
	baseOpts := func(runID string, seed int64, store memory.Store) Options {
		return Options{
			RunID: runID, Scenario: "community", Personas: cast, Providers: registry(alwaysPass),
			Store: store, Rounds: 2, Seed: seed,
			Params: map[string]any{"days": 2, "complication_day": 0},
			Limits: testLimits(),
		}
	}

	// Longitudinal: the same store across two sessions with different seeds.
	shared := memory.NewMemStore()
	res1, err := Execute(context.Background(), baseOpts("session-1", 101, shared))
	if err != nil {
		t.Fatal(err)
	}
	res2, err := Execute(context.Background(), baseOpts("session-2", 202, shared))
	if err != nil {
		t.Fatal(err)
	}

	cost1 := res1.Outcome.Metrics["structural_cost_initial"]
	cost2 := res2.Outcome.Metrics["structural_cost_initial"]
	if cost1 == cost2 {
		t.Errorf("sessions with different seeds produced identical ground truth (%v); a longitudinal rerun must vary the seed, not replay the same hidden facts", cost1)
	}

	miraState, err := shared.Load("mira")
	if err != nil {
		t.Fatal(err)
	}
	if len(miraState.Runs) != 2 || miraState.Runs[0] != "session-1" || miraState.Runs[1] != "session-2" {
		t.Errorf("a longitudinal experiment should accumulate both sessions in memory, got %v", miraState.Runs)
	}

	// Fresh: an experiment given its own empty store sees none of the above.
	fresh := memory.NewMemStore()
	if _, err := Execute(context.Background(), baseOpts("fresh-session", 303, fresh)); err != nil {
		t.Fatal(err)
	}
	freshMira, err := fresh.Load("mira")
	if err != nil {
		t.Fatal(err)
	}
	if len(freshMira.Runs) != 1 || freshMira.Runs[0] != "fresh-session" {
		t.Errorf("a fresh experiment must not see another experiment's history, got %v", freshMira.Runs)
	}
}
