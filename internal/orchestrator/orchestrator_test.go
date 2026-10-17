package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/sim"
)

// --- test doubles ------------------------------------------------------------

// scriptedActor plays a fixed list of actions, then passes forever. It records
// everything it observes so visibility can be asserted directly.
type scriptedActor struct {
	id       sim.AgentID
	name     string
	script   []sim.Action
	calls    int
	err      error
	observed []sim.Observation
	// turns records the Turn it was handed each time, to check what the
	// orchestrator exposes to an agent.
	turns []sim.Turn
	mu    sync.Mutex
}

func actor(id string, script ...sim.Action) *scriptedActor {
	return &scriptedActor{id: sim.AgentID(id), name: strings.ToUpper(id[:1]) + id[1:], script: script}
}

func (a *scriptedActor) ID() sim.AgentID { return a.id }
func (a *scriptedActor) Name() string    { return a.name }

func (a *scriptedActor) Observe(o sim.Observation) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.observed = append(a.observed, o)
}

func (a *scriptedActor) Decide(ctx context.Context, t sim.Turn) (sim.Action, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.turns = append(a.turns, t)
	if a.err != nil {
		a.calls++
		return sim.Action{}, a.err
	}
	i := a.calls
	a.calls++
	if i < len(a.script) {
		return a.script[i], nil
	}
	return sim.Action{Kind: sim.ActionPass}, nil
}

func (a *scriptedActor) CallCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.calls
}

func (a *scriptedActor) heard() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, 0, len(a.observed))
	for _, o := range a.observed {
		out = append(out, o.Text)
	}
	return out
}

// testEnv is a minimal environment: it echoes speech, enforces one rule, and
// terminates when told.
type testEnv struct {
	participants []sim.AgentID
	// reject causes Validate to refuse any action whose content matches.
	reject string
	// finishAfter ends the scenario after this many applied actions.
	finishAfter int
	applied     int
	rounds      []int
	endRounds   []int
	beginEvents func(round int) []sim.Event
	done        bool
	setupErr    error
}

func (e *testEnv) Name() string        { return "test" }
func (e *testEnv) Description() string { return "a test environment" }

func (e *testEnv) Setup(p []sim.AgentID) ([]sim.Event, error) {
	if e.setupErr != nil {
		return nil, e.setupErr
	}
	e.participants = p
	return []sim.Event{{Type: sim.EventSystem, Text: "the scene is set", Visibility: sim.PublicTo()}}, nil
}

func (e *testEnv) Participants() []sim.AgentID { return e.participants }
func (e *testEnv) Brief(id sim.AgentID) string { return "you are " + string(id) }

func (e *testEnv) Actions(sim.AgentID) []sim.ActionSpec {
	return []sim.ActionSpec{{Kind: sim.ActionSay, Usage: "say", Description: "speak"}}
}

func (e *testEnv) Validate(id sim.AgentID, a sim.Action) error {
	if e.reject != "" && strings.Contains(a.Content, e.reject) {
		return fmt.Errorf("you may not mention %q here", e.reject)
	}
	return nil
}

func (e *testEnv) Apply(id sim.AgentID, a sim.Action) ([]sim.Event, error) {
	e.applied++
	if e.finishAfter > 0 && e.applied >= e.finishAfter {
		e.done = true
	}
	switch a.Kind {
	case sim.ActionSay:
		return []sim.Event{{Type: sim.EventSay, Actor: id, Text: a.Content, Visibility: sim.PublicTo()}}, nil
	case sim.ActionWhisper:
		return []sim.Event{{Type: sim.EventWhisper, Actor: id, Targets: a.To, Text: a.Content,
			Visibility: sim.Only(append([]sim.AgentID{id}, a.To...)...)}}, nil
	default:
		return []sim.Event{{Type: sim.EventPass, Actor: id, Visibility: sim.PublicTo()}}, nil
	}
}

func (e *testEnv) PublicState() map[string]any { return map[string]any{"applied": e.applied} }
func (e *testEnv) PrivateState(id sim.AgentID) map[string]any {
	return map[string]any{"secret_for": string(id)}
}

func (e *testEnv) BeginRound(round int) []sim.Event {
	e.rounds = append(e.rounds, round)
	if e.beginEvents != nil {
		return e.beginEvents(round)
	}
	return nil
}

func (e *testEnv) EndRound(round int) []sim.Event {
	e.endRounds = append(e.endRounds, round)
	return nil
}

func (e *testEnv) Done() (bool, string) {
	if e.done {
		return true, "the test environment said so"
	}
	return false, ""
}

func (e *testEnv) Outcome() sim.Outcome {
	return sim.Outcome{Summary: "test complete", Metrics: map[string]any{"applied": e.applied}}
}

// collector is a Sink that keeps every record for assertions.
type collector struct {
	mu      sync.Mutex
	records []explog.Record
}

func (c *collector) Emit(r explog.Record) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.records = append(c.records, r)
}

func (c *collector) Close() error { return nil }

func (c *collector) ofType(t explog.RecordType) []explog.Record {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []explog.Record
	for _, r := range c.records {
		if r.Type == t {
			out = append(out, r)
		}
	}
	return out
}

func testLimits(rounds int) Limits {
	l := DefaultLimits()
	l.MaxRounds = rounds
	l.Timeout = 10 * time.Second
	return l
}

// --- tests -------------------------------------------------------------------

func TestSchedulingIsDeterministicAndBounded(t *testing.T) {
	a, b, c := actor("mira"), actor("dov"), actor("rook")
	env := &testEnv{}
	col := &collector{}

	orch, err := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{a, b, c},
		Limits: testLimits(3), Sink: col, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	// Every agent passes, so disable the stall guard for this test.
	orch.cfg.Limits.MaxConsecutivePasses = 0

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rounds != 3 {
		t.Errorf("Rounds = %d, want 3", res.Rounds)
	}
	for _, ac := range []*scriptedActor{a, b, c} {
		if ac.CallCount() != 3 {
			t.Errorf("%s acted %d times, want once per round", ac.id, ac.CallCount())
		}
	}
	// Turn order follows the configured order, every round.
	if len(env.rounds) != 3 || env.rounds[0] != 1 || env.rounds[2] != 3 {
		t.Errorf("BeginRound calls = %v", env.rounds)
	}
	if len(env.endRounds) != 3 {
		t.Errorf("EndRound calls = %v", env.endRounds)
	}
	if !strings.Contains(res.StopReason, "round limit") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
	if res.Outcome.Summary != "test complete" {
		t.Errorf("Outcome = %+v", res.Outcome)
	}
}

func TestOrchestratorStopsWhenEnvironmentTerminates(t *testing.T) {
	a := actor("mira", sim.Action{Kind: sim.ActionSay, Content: "one"})
	env := &testEnv{finishAfter: 1}

	orch, _ := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{a, actor("dov")},
		Limits: testLimits(10), Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rounds != 1 {
		t.Errorf("Rounds = %d; a terminating environment should end the run immediately", res.Rounds)
	}
	if !strings.Contains(res.StopReason, "test environment said so") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
	// The second agent never got a turn, because the world ended first.
	if dovCalls := orch.cfg.Agents[1].(*scriptedActor).CallCount(); dovCalls != 0 {
		t.Errorf("dov acted %d times after termination", dovCalls)
	}
}

// TestPrivateMessagesReachOnlyTheirAudience is the central information-boundary
// guarantee: the orchestrator is the only path by which an agent learns
// anything, so a whisper cannot reach a third party.
func TestPrivateMessagesReachOnlyTheirAudience(t *testing.T) {
	mira := actor("mira",
		sim.Action{Kind: sim.ActionWhisper, Content: "rook is lying to you", To: []sim.AgentID{"dov"}},
		sim.Action{Kind: sim.ActionSay, Content: "nice weather"},
	)
	dov, rook := actor("dov"), actor("rook")
	env := &testEnv{}

	orch, _ := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{mira, dov, rook},
		Limits: testLimits(2), Sink: explog.Discard{}})
	orch.cfg.Limits.MaxConsecutivePasses = 0
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}

	const secret = "rook is lying to you"
	if !containsSubstring(mira.heard(), secret) {
		t.Error("the speaker should observe its own whisper")
	}
	if !containsSubstring(dov.heard(), secret) {
		t.Error("the addressed agent did not receive the whisper")
	}
	if containsSubstring(rook.heard(), secret) {
		t.Fatalf("a third party observed a private message: %v", rook.heard())
	}
	// Public speech, by contrast, reaches everyone.
	for _, a := range []*scriptedActor{mira, dov, rook} {
		if !containsSubstring(a.heard(), "nice weather") {
			t.Errorf("%s did not hear public speech: %v", a.id, a.heard())
		}
	}
}

func TestObservationsArePhrasedFromTheObserversPointOfView(t *testing.T) {
	mira := actor("mira", sim.Action{Kind: sim.ActionSay, Content: "hello"})
	dov := actor("dov")
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{mira, dov},
		Limits: testLimits(1), Sink: explog.Discard{}})
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}

	if !containsSubstring(mira.heard(), "you said: hello") {
		t.Errorf("an agent should read its own speech as \"you\": %v", mira.heard())
	}
	if !containsSubstring(dov.heard(), "Mira (mira) said: hello") {
		t.Errorf("others should see the speaker named: %v", dov.heard())
	}
}

func TestPrivateStateIsPerAgent(t *testing.T) {
	mira, dov := actor("mira"), actor("dov")
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{mira, dov},
		Limits: testLimits(1), Sink: explog.Discard{}})
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := mira.turns[0].PrivateState["secret_for"]; got != "mira" {
		t.Errorf("mira's private state = %v", got)
	}
	if got := dov.turns[0].PrivateState["secret_for"]; got != "dov" {
		t.Errorf("dov's private state = %v", got)
	}
	// And each agent gets its own brief and the public state.
	if mira.turns[0].Brief != "you are mira" {
		t.Errorf("brief = %q", mira.turns[0].Brief)
	}
	if mira.turns[0].PublicState["applied"] == nil {
		t.Error("public state was not passed through")
	}
}

func TestRejectedActionIsRetriedOnceThenAbandoned(t *testing.T) {
	// The agent insists on a forbidden action both times.
	stubborn := actor("mira",
		sim.Action{Kind: sim.ActionSay, Content: "the forbidden word"},
		sim.Action{Kind: sim.ActionSay, Content: "the forbidden word again"},
		sim.Action{Kind: sim.ActionSay, Content: "fine, something else"},
	)
	env := &testEnv{reject: "forbidden"}
	col := &collector{}

	orch, _ := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{stubborn},
		Limits: testLimits(1), Sink: col})
	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	if stubborn.CallCount() != 2 {
		t.Errorf("Decide was called %d times; a rejection should buy exactly one retry", stubborn.CallCount())
	}
	if res.Rejections != 2 {
		t.Errorf("Rejections = %d, want 2", res.Rejections)
	}
	// The agent must be told why, privately, so it can adapt.
	if !containsSubstring(stubborn.heard(), "may not mention") {
		t.Errorf("the agent was not told why its action failed: %v", stubborn.heard())
	}
	// And the rejection reaches the log with its reason.
	rejections := col.ofType(explog.RecRejected)
	if len(rejections) != 2 || !strings.Contains(rejections[0].Rejected.Reason, "forbidden") {
		t.Errorf("rejection records = %+v", rejections)
	}
	// Decisions are logged whether or not they were accepted.
	for _, d := range col.ofType(explog.RecDecision) {
		if d.Decision.Accepted {
			t.Errorf("a refused action was logged as accepted: %+v", d.Decision)
		}
	}
}

func TestRejectedActionRecoversOnTheRetry(t *testing.T) {
	a := actor("mira",
		sim.Action{Kind: sim.ActionSay, Content: "the forbidden word"},
		sim.Action{Kind: sim.ActionSay, Content: "something allowed"},
	)
	env := &testEnv{reject: "forbidden"}
	orch, _ := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{a},
		Limits: testLimits(1), Sink: explog.Discard{}})
	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rejections != 1 {
		t.Errorf("Rejections = %d, want 1", res.Rejections)
	}
	if res.Actions["say"] != 1 {
		t.Errorf("the retry should have been applied: %+v", res.Actions)
	}
}

func TestInferenceErrorsDoNotStopTheRun(t *testing.T) {
	broken := actor("broken")
	broken.err = errors.New("the model dribbled")
	healthy := actor("mira", sim.Action{Kind: sim.ActionSay, Content: "carrying on"})
	col := &collector{}

	limits := testLimits(2)
	limits.MaxAgentErrors = 0 // never give up on an agent
	limits.MaxConsecutivePasses = 0
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{broken, healthy},
		Limits: limits, Sink: col})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatalf("a failing agent must not fail the run: %v", err)
	}
	if res.Rounds != 2 {
		t.Errorf("Rounds = %d, want 2", res.Rounds)
	}
	if res.Actions["say"] != 1 {
		t.Errorf("the healthy agent should still have acted: %+v", res.Actions)
	}
	failures := col.ofType(explog.RecError)
	if len(failures) != 2 {
		t.Fatalf("expected one error record per failed turn, got %d", len(failures))
	}
	if failures[0].Failure.Stage != "decide" || !strings.Contains(failures[0].Failure.Err, "dribbled") {
		t.Errorf("failure record = %+v", failures[0].Failure)
	}
}

func TestRepeatedAgentFailureEndsTheRun(t *testing.T) {
	broken := actor("broken")
	broken.err = errors.New("still broken")

	limits := testLimits(10)
	limits.MaxAgentErrors = 2
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{broken},
		Limits: limits, Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Err == nil {
		t.Fatal("a persistently failing agent should end the run with an error in the Result")
	}
	if !strings.Contains(res.StopReason, "failed 2 consecutive times") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
	// But the run still reports an outcome, so the experiment is salvageable.
	if res.Outcome.Summary == "" {
		t.Error("an aborted run should still produce an outcome")
	}
}

func TestUnrecoverableBackendErrorEndsTheRunImmediately(t *testing.T) {
	dead := actor("dead")
	dead.err = sim.Fatal(errors.New("llm backend unavailable"))
	other := actor("mira")

	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{dead, other},
		Limits: testLimits(5), Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Err == nil || !strings.Contains(res.StopReason, "unavailable") {
		t.Errorf("Result = %+v", res)
	}
	if other.CallCount() != 0 {
		t.Error("a dead backend should stop the run before every other agent fails in turn")
	}
}

func TestStallGuardStopsADeadConversation(t *testing.T) {
	limits := testLimits(100)
	limits.MaxConsecutivePasses = 4
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{actor("a"), actor("b")},
		Limits: limits, Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.StopReason, "stalled") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
	if res.Rounds > 3 {
		t.Errorf("a stalled run should stop early, ran %d rounds", res.Rounds)
	}
}

func TestStallStreakResetsWhenSomeoneActs(t *testing.T) {
	// b speaks every turn, so the run should reach its round limit.
	talker := actor("b")
	talker.script = []sim.Action{
		{Kind: sim.ActionSay, Content: "1"}, {Kind: sim.ActionSay, Content: "2"},
		{Kind: sim.ActionSay, Content: "3"}, {Kind: sim.ActionSay, Content: "4"},
	}
	limits := testLimits(4)
	limits.MaxConsecutivePasses = 3
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{actor("a"), talker},
		Limits: limits, Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Rounds != 4 {
		t.Errorf("Rounds = %d, want 4 — one active agent should keep the run alive", res.Rounds)
	}
}

func TestLLMCallBudgetStopsTheRun(t *testing.T) {
	limits := testLimits(100)
	limits.MaxLLMCalls = 5
	limits.MaxConsecutivePasses = 0
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{actor("a"), actor("b")},
		Limits: limits, Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.StopReason, "LLM call budget") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
	if res.LLMCalls > 6 {
		t.Errorf("LLMCalls = %d; the budget should be enforced between turns", res.LLMCalls)
	}
}

func TestActionBudgetStopsTheRun(t *testing.T) {
	a := actor("a")
	a.script = make([]sim.Action, 50)
	for i := range a.script {
		a.script[i] = sim.Action{Kind: sim.ActionSay, Content: "chatter"}
	}
	limits := testLimits(100)
	limits.MaxActions = 4
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{a},
		Limits: limits, Sink: explog.Discard{}})

	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.StopReason, "action budget") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
}

func TestTimeoutStopsTheRun(t *testing.T) {
	slow := &slowActor{id: "slow", delay: 40 * time.Millisecond}
	limits := testLimits(100)
	limits.Timeout = 80 * time.Millisecond
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{slow},
		Limits: limits, Sink: explog.Discard{}})

	start := time.Now()
	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Errorf("run took %s; the timeout was not enforced", elapsed)
	}
	if !strings.Contains(res.StopReason, "timeout") {
		t.Errorf("StopReason = %q", res.StopReason)
	}
}

func TestCancellationStopsCleanlyAndStillReportsAnOutcome(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	// A slow agent keeps the run going long enough for the cancel to land
	// mid-run rather than after the round budget is already spent.
	a := &slowActor{id: "a", delay: 15 * time.Millisecond}

	limits := testLimits(100)
	limits.MaxConsecutivePasses = 0
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{a},
		Limits: limits, Sink: explog.Discard{}})

	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	res, err := orch.Run(ctx)
	if err != nil {
		t.Fatalf("cancellation is a normal outcome, not an error: %v", err)
	}
	if res.StopReason != "cancelled" {
		t.Errorf("StopReason = %q", res.StopReason)
	}
	if res.Outcome.Summary == "" {
		t.Error("a cancelled run should still report what happened")
	}
}

func TestShuffleIsReproducibleForASeed(t *testing.T) {
	orderFor := func(seed int64) []sim.AgentID {
		var order []sim.AgentID
		env := &testEnv{}
		// Record the order agents are asked to decide.
		agents := []sim.Actor{&recordingActor{id: "a", log: &order}, &recordingActor{id: "b", log: &order},
			&recordingActor{id: "c", log: &order}, &recordingActor{id: "d", log: &order}}
		limits := testLimits(3)
		limits.MaxConsecutivePasses = 0
		orch, _ := New(Config{RunID: "run", Env: env, Agents: agents, Limits: limits,
			Sink: explog.Discard{}, Seed: seed, Shuffle: true})
		orch.Run(context.Background())
		return order
	}

	first, second := orderFor(99), orderFor(99)
	if len(first) == 0 {
		t.Fatal("no turns were taken")
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("the same seed produced different turn orders:\n%v\n%v", first, second)
		}
	}
	// A different seed should generally differ; assert it is at least capable
	// of differing rather than asserting a specific permutation.
	other := orderFor(1234)
	same := true
	for i := range first {
		if i < len(other) && first[i] != other[i] {
			same = false
		}
	}
	if same {
		t.Log("note: seeds 99 and 1234 happened to produce the same order")
	}
}

func TestEverythingNeededToReconstructTheRunIsLogged(t *testing.T) {
	a := actor("mira",
		sim.Action{Kind: sim.ActionSay, Content: "public line", Reasoning: "a private rationale"},
		sim.Action{Kind: sim.ActionWhisper, Content: "private line", To: []sim.AgentID{"dov"}},
	)
	col := &collector{}
	limits := testLimits(2)
	limits.MaxConsecutivePasses = 0
	orch, _ := New(Config{RunID: "run-42", Env: &testEnv{}, Agents: []sim.Actor{a, actor("dov")},
		Limits: limits, Sink: col})
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}

	for _, want := range []explog.RecordType{explog.RecRoundStart, explog.RecEvent, explog.RecDecision, explog.RecOutcome} {
		if len(col.ofType(want)) == 0 {
			t.Errorf("no %s records were logged", want)
		}
	}
	// Every record is stamped with the run id, and world events are numbered
	// in strictly increasing order so history is reconstructible. Log-line
	// numbering is explog.Sequencer's job, not the orchestrator's.
	col.mu.Lock()
	for _, r := range col.records {
		if r.RunID != "run-42" {
			t.Errorf("record missing the run id: %+v", r)
		}
	}
	col.mu.Unlock()
	lastEvent := 0
	for _, r := range col.ofType(explog.RecEvent) {
		if r.Event.Seq <= lastEvent {
			t.Errorf("event sequence is not increasing: %d after %d", r.Event.Seq, lastEvent)
		}
		lastEvent = r.Event.Seq
		if r.Event.Time.IsZero() {
			t.Error("an event was logged without a timestamp")
		}
	}

	// Every event carries its visibility, so a reader can tell who saw what.
	var sawPrivate bool
	for _, r := range col.ofType(explog.RecEvent) {
		if r.Event.Type == sim.EventWhisper {
			sawPrivate = true
			if r.Event.Visibility.Public {
				t.Error("a whisper was logged as public")
			}
			if r.Event.Visibility.Describe() != "private:dov,mira" {
				t.Errorf("whisper visibility = %q", r.Event.Visibility.Describe())
			}
		}
	}
	if !sawPrivate {
		t.Error("the private message was not logged at all")
	}
	// Self-reported reasoning is recorded on the decision, kept apart from the
	// event the other agents observed.
	decisions := col.ofType(explog.RecDecision)
	if decisions[0].Decision.Action.Reasoning != "a private rationale" {
		t.Errorf("reasoning was not recorded: %+v", decisions[0].Decision)
	}
	for _, r := range col.ofType(explog.RecEvent) {
		if strings.Contains(r.Event.Text, "a private rationale") {
			t.Error("self-reported reasoning leaked into an observable event")
		}
	}
}

func TestEnvironmentEventsFromBeginRoundAreDelivered(t *testing.T) {
	mira, dov := actor("mira"), actor("dov")
	env := &testEnv{beginEvents: func(round int) []sim.Event {
		return []sim.Event{
			{Type: sim.EventState, Text: fmt.Sprintf("round %d begins", round), Visibility: sim.PublicTo()},
			{Type: sim.EventSystem, Text: "a note for mira only", Visibility: sim.Only("mira")},
		}
	}}
	orch, _ := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{mira, dov},
		Limits: testLimits(1), Sink: explog.Discard{}})
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !containsSubstring(dov.heard(), "round 1 begins") {
		t.Errorf("public environment narration was not delivered: %v", dov.heard())
	}
	if containsSubstring(dov.heard(), "a note for mira only") {
		t.Errorf("a restricted environment event leaked: %v", dov.heard())
	}
	if !containsSubstring(mira.heard(), "a note for mira only") {
		t.Errorf("mira did not receive her own note: %v", mira.heard())
	}
}

func TestNewValidatesConfig(t *testing.T) {
	if _, err := New(Config{RunID: "r", Agents: []sim.Actor{actor("a")}}); err == nil {
		t.Error("an orchestrator with no environment should fail")
	}
	if _, err := New(Config{RunID: "r", Env: &testEnv{}}); err == nil {
		t.Error("an orchestrator with no agents should fail")
	}
	if _, err := New(Config{RunID: "r", Env: &testEnv{}, Agents: []sim.Actor{actor("a"), actor("a")}}); err == nil {
		t.Error("duplicate agent ids should fail")
	}
}

func TestSetupFailureIsFatal(t *testing.T) {
	env := &testEnv{setupErr: errors.New("the room was locked")}
	orch, _ := New(Config{RunID: "r", Env: env, Agents: []sim.Actor{actor("a")},
		Limits: testLimits(1), Sink: explog.Discard{}})
	if _, err := orch.Run(context.Background()); err == nil {
		t.Error("a scenario that cannot be set up should not run")
	}
}

// --- helper doubles ---------------------------------------------------------

type slowActor struct {
	id    sim.AgentID
	delay time.Duration
}

func (s *slowActor) ID() sim.AgentID         { return s.id }
func (s *slowActor) Name() string            { return string(s.id) }
func (s *slowActor) Observe(sim.Observation) {}
func (s *slowActor) Decide(ctx context.Context, _ sim.Turn) (sim.Action, error) {
	select {
	case <-time.After(s.delay):
		return sim.Action{Kind: sim.ActionSay, Content: "slowly"}, nil
	case <-ctx.Done():
		return sim.Action{}, ctx.Err()
	}
}

type recordingActor struct {
	id  sim.AgentID
	log *[]sim.AgentID
}

func (r *recordingActor) ID() sim.AgentID         { return r.id }
func (r *recordingActor) Name() string            { return string(r.id) }
func (r *recordingActor) Observe(sim.Observation) {}
func (r *recordingActor) Decide(context.Context, sim.Turn) (sim.Action, error) {
	*r.log = append(*r.log, r.id)
	return sim.Action{Kind: sim.ActionPass}, nil
}

func containsText(lines []string, want string) bool {
	for _, l := range lines {
		if l == want {
			return true
		}
	}
	return false
}

func containsSubstring(lines []string, want string) bool {
	for _, l := range lines {
		if strings.Contains(l, want) {
			return true
		}
	}
	return false
}

// --- dual-action (MultiDecider) ---------------------------------------------

// multiActor is a sim.MultiDecider test double: each call to DecideMany pops
// the next scripted response off a queue, so a test can script exactly what
// an agent proposes on its first attempt, its retry, and so on.
type multiActor struct {
	id       sim.AgentID
	script   [][]sim.Action // one entry per call
	calls    int
	observed []sim.Observation
}

func newMultiActor(id string, script ...[]sim.Action) *multiActor {
	return &multiActor{id: sim.AgentID(id), script: script}
}

func (m *multiActor) ID() sim.AgentID           { return m.id }
func (m *multiActor) Name() string              { return string(m.id) }
func (m *multiActor) Observe(o sim.Observation) { m.observed = append(m.observed, o) }

func (m *multiActor) Decide(context.Context, sim.Turn) (sim.Action, error) {
	return sim.Action{}, fmt.Errorf("multiActor: Decide should never be called when DecideMany is available")
}

func (m *multiActor) DecideMany(ctx context.Context, t sim.Turn) ([]sim.Action, error) {
	i := m.calls
	m.calls++
	if i >= len(m.script) {
		return nil, nil
	}
	return m.script[i], nil
}

func (m *multiActor) heard() []string {
	out := make([]string, 0, len(m.observed))
	for _, o := range m.observed {
		out = append(out, o.Text)
	}
	return out
}

func TestMultiDeciderAppliesBothSlotsFromOneCall(t *testing.T) {
	mira := newMultiActor("mira", []sim.Action{
		{Kind: sim.ActionSay, Content: "I'll put this forward"},
		{Kind: sim.ActionDo, Verb: "noop"}, // testEnv has no "noop" handling beyond echoing say/whisper/pass; use say's sibling instead
	})
	// testEnv.Apply only special-cases Say/Whisper/default-pass; a "do" action
	// with any verb falls through to its default branch, which returns a pass
	// event — good enough to prove both slots were accepted and applied.
	env := &testEnv{}
	col := &collector{}
	orch, err := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{mira}, Limits: testLimits(1), Sink: col})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}

	if mira.calls != 1 {
		t.Fatalf("DecideMany was called %d times, want exactly 1 for both slots", mira.calls)
	}
	decisions := col.ofType(explog.RecDecision)
	if len(decisions) != 2 {
		t.Fatalf("expected 2 decision records (one per slot), got %d: %+v", len(decisions), decisions)
	}
	var sawComm, sawWorld bool
	for _, d := range decisions {
		if !d.Decision.Accepted {
			t.Errorf("decision was rejected unexpectedly: %+v", d.Decision)
		}
		switch d.Decision.Slot {
		case "communication":
			sawComm = true
		case "world":
			sawWorld = true
		}
	}
	if !sawComm || !sawWorld {
		t.Errorf("expected one decision per slot, sawComm=%v sawWorld=%v", sawComm, sawWorld)
	}
	if !containsSubstring(mira.heard(), "I'll put this forward") {
		t.Error("the communication half of the turn was not delivered")
	}
}

func TestMultiDeciderEmptyResponseIsAnExplicitPass(t *testing.T) {
	mira := newMultiActor("mira", []sim.Action{}) // both slots null
	env := &testEnv{}
	orch, _ := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{mira}, Limits: testLimits(1), Sink: explog.Discard{}})
	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if res.Actions["pass"] != 1 {
		t.Errorf("an empty dual response should synthesize a pass, actions = %+v", res.Actions)
	}
}

func TestMultiDeciderRetryIsScopedToTheRejectedSlotOnly(t *testing.T) {
	env := &testEnv{reject: "forbidden"}
	mira := newMultiActor("mira",
		// Attempt 0: communication is fine, world action is rejected (testEnv
		// rejects by content, regardless of kind).
		[]sim.Action{
			{Kind: sim.ActionSay, Content: "first message"},
			{Kind: sim.ActionDo, Verb: "x", Content: "forbidden content"},
		},
		// Attempt 1 (retry): tries to change the already-accepted
		// communication AND fix the world slot. Only the world slot's fix
		// should take effect.
		[]sim.Action{
			{Kind: sim.ActionSay, Content: "a different message"},
			{Kind: sim.ActionDo, Verb: "x", Content: "a fine action"},
		},
	)
	col := &collector{}
	orch, err := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{mira}, Limits: testLimits(1), Sink: col})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}

	if mira.calls != 2 {
		t.Fatalf("expected exactly one retry call, got %d calls", mira.calls)
	}
	// Exactly one "say" ever reached anyone — the first message, not the
	// retry's attempted replacement.
	if !containsSubstring(mira.heard(), "first message") {
		t.Error("the originally accepted communication should stand")
	}
	if containsSubstring(mira.heard(), "a different message") {
		t.Error("an already-accepted slot must not be re-applied from a retry response")
	}
	// The world action succeeded only on the retry.
	decisions := col.ofType(explog.RecDecision)
	var worldAttempts []int
	for _, d := range decisions {
		if d.Decision.Slot == "world" {
			worldAttempts = append(worldAttempts, d.Decision.Attempt)
		}
	}
	if len(worldAttempts) != 2 || worldAttempts[0] != 0 || worldAttempts[1] != 1 {
		t.Errorf("world slot attempts = %v, want [0, 1]", worldAttempts)
	}
	rejections := col.ofType(explog.RecRejected)
	if len(rejections) != 1 {
		t.Fatalf("expected exactly one rejection (attempt 0's forbidden content), got %d: %+v", len(rejections), rejections)
	}
	if rejections[0].Rejected.Attempt != 0 || !rejections[0].Rejected.Retried {
		t.Errorf("rejection record = %+v", rejections[0].Rejected)
	}
}

func TestMultiDeciderBothSlotsRejectedExhaustsRetryWithoutApplyingEither(t *testing.T) {
	env := &testEnv{reject: "forbidden"}
	mira := newMultiActor("mira",
		[]sim.Action{{Kind: sim.ActionSay, Content: "forbidden one"}, {Kind: sim.ActionDo, Verb: "x", Content: "forbidden two"}},
		[]sim.Action{{Kind: sim.ActionSay, Content: "forbidden three"}, {Kind: sim.ActionDo, Verb: "x", Content: "forbidden four"}},
	)
	col := &collector{}
	limits := testLimits(1)
	orch, err := New(Config{RunID: "run", Env: env, Agents: []sim.Actor{mira}, Limits: limits, Sink: col})
	if err != nil {
		t.Fatal(err)
	}
	res, err := orch.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if mira.calls != 2 {
		t.Fatalf("expected exactly one retry call, got %d", mira.calls)
	}
	if res.Rejections != 4 {
		t.Errorf("Rejections = %d, want 4 (2 slots x 2 attempts)", res.Rejections)
	}
	if res.Actions["say"] != 0 {
		t.Errorf("nothing should have been applied, actions = %+v", res.Actions)
	}
	// The rejection feedback legitimately quotes the forbidden word (the
	// agent needs to know what to avoid); what must never appear is a
	// success-phrased echo of the rejected content having actually happened.
	for _, line := range mira.heard() {
		if strings.Contains(line, "said: forbidden") || strings.Contains(line, "whispered") {
			t.Errorf("a rejected action must never be echoed back as if it happened: %q", line)
		}
	}
}

func TestSingleActionAgentDecisionsCarryAWorldOrCommunicationSlot(t *testing.T) {
	// A plain sim.Actor (not a MultiDecider) still gets its single decision
	// classified into the same slot vocabulary, so logs are consistent
	// whether or not an agent uses the dual protocol.
	a := actor("mira", sim.Action{Kind: sim.ActionSay, Content: "hi"})
	col := &collector{}
	orch, _ := New(Config{RunID: "run", Env: &testEnv{}, Agents: []sim.Actor{a}, Limits: testLimits(1), Sink: col})
	if _, err := orch.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	decisions := col.ofType(explog.RecDecision)
	if len(decisions) != 1 || decisions[0].Decision.Slot != "communication" {
		t.Errorf("decisions = %+v", decisions)
	}
}
