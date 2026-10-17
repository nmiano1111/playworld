package explog

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/sim"
)

func TestParseVerbosity(t *testing.T) {
	for in, want := range map[string]Verbosity{
		"": Normal, "quiet": Quiet, "q": Quiet, "normal": Normal,
		"verbose": Verbose, "v": Verbose, "debug": Debug, "DEBUG": Debug,
	} {
		got, err := ParseVerbosity(in)
		if err != nil || got != want {
			t.Errorf("ParseVerbosity(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	if _, err := ParseVerbosity("loud"); err == nil {
		t.Error("an unknown level should error")
	}
}

func TestJSONLWritesOneObjectPerLineAndFlushesEagerly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "run.jsonl")
	j, err := NewJSONL(path, false)
	if err != nil {
		t.Fatal(err)
	}

	j.Emit(Record{Type: RecRunStart, RunID: "r1", Seq: 1, Time: time.Now(),
		Run: &RunInfo{RunID: "r1", Scenario: "social", MaxRounds: 3}})
	j.Emit(Record{Type: RecEvent, RunID: "r1", Seq: 2, Round: 1,
		Event: &sim.Event{Type: sim.EventSay, Actor: "mira", Text: "hello", Visibility: sim.PublicTo()}})

	// Readable before Close: an interrupted run must still leave a usable log.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines before Close, got %d: %q", len(lines), raw)
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}

	// Every line is independently parseable, which is the point of JSONL.
	sc := bufio.NewScanner(bytes.NewReader(raw))
	n := 0
	for sc.Scan() {
		var rec Record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatalf("line %d is not valid JSON: %v", n+1, err)
		}
		if rec.RunID != "r1" {
			t.Errorf("line %d lost the run id", n+1)
		}
		n++
	}
	if n != 2 {
		t.Errorf("parsed %d lines", n)
	}
	// Closing twice is safe, because defer and an explicit Close both happen.
	if err := j.Close(); err != nil {
		t.Errorf("second Close = %v", err)
	}
}

func TestJSONLOmitsPromptsUnlessVerbose(t *testing.T) {
	call := &Call{
		Agent: "mira", Model: "qwen3:4b", PromptTokens: 100, OutputTokens: 10,
		Raw:    `{"kind":"pass"}`,
		Prompt: []llm.Message{llm.System("a long system prompt"), llm.User("a long user prompt")},
	}

	t.Run("terse log keeps the metrics and drops the text", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "r.jsonl")
		j, _ := NewJSONL(path, false)
		c := *call
		j.Emit(Record{Type: RecLLMCall, Call: &c})
		j.Close()

		raw, _ := os.ReadFile(path)
		if strings.Contains(string(raw), "a long system prompt") {
			t.Error("a non-verbose log should not carry full prompts")
		}
		if strings.Contains(string(raw), `{\"kind\":\"pass\"}`) {
			t.Error("a successful reply's raw text is redundant with the decision record")
		}
		if !strings.Contains(string(raw), `"prompt_tokens":100`) {
			t.Errorf("token usage must survive: %s", raw)
		}
		// Emit must not mutate the caller's Call.
		if len(c.Prompt) != 2 {
			t.Error("Emit mutated the caller's record")
		}
	})

	t.Run("a failed call keeps its raw reply", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "r.jsonl")
		j, _ := NewJSONL(path, false)
		c := *call
		c.Raw = "I am not sure"
		c.Err = "could not parse"
		j.Emit(Record{Type: RecLLMCall, Call: &c})
		j.Close()

		raw, _ := os.ReadFile(path)
		if !strings.Contains(string(raw), "I am not sure") {
			t.Errorf("a failed call must record what the model actually said: %s", raw)
		}
	})

	t.Run("verbose log keeps everything", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "r.jsonl")
		j, _ := NewJSONL(path, true)
		c := *call
		j.Emit(Record{Type: RecLLMCall, Call: &c})
		j.Close()

		raw, _ := os.ReadFile(path)
		if !strings.Contains(string(raw), "a long system prompt") {
			t.Errorf("a verbose log should carry prompts: %s", raw)
		}
	})
}

func TestTallyAggregatesTheRecordStream(t *testing.T) {
	tally := NewTally()
	start := time.Now()

	tally.Emit(Record{Type: RecRunStart, Run: &RunInfo{RunID: "r1", Scenario: "split", Started: start}})
	tally.Emit(Record{Type: RecAgent, Agent: &AgentInfo{ID: "mira", Name: "Mira",
		Model: llm.ModelConfig{Provider: "ollama", Model: "qwen3:4b"}}})
	tally.Emit(Record{Type: RecAgent, Agent: &AgentInfo{ID: "dov", Name: "Dov",
		Model: llm.ModelConfig{Provider: "ollama", Model: "qwen3:8b"}}})
	tally.Emit(Record{Type: RecRoundStart, Round: 1})
	tally.Emit(Record{Type: RecRoundStart, Round: 2})

	tally.Emit(Record{Type: RecLLMCall, Call: &Call{Agent: "mira", PromptTokens: 100, OutputTokens: 20, LatencyMS: 1000}})
	tally.Emit(Record{Type: RecLLMCall, Call: &Call{Agent: "mira", PromptTokens: 110, OutputTokens: 25, LatencyMS: 3000, Attempt: 1}})
	tally.Emit(Record{Type: RecLLMCall, Call: &Call{Agent: "dov", PromptTokens: 90, OutputTokens: 10, LatencyMS: 500, Err: "boom"}})

	tally.Emit(Record{Type: RecDecision, Decision: &Decision{Agent: "mira", Action: sim.Action{Kind: sim.ActionSay}, Accepted: true}})
	tally.Emit(Record{Type: RecDecision, Decision: &Decision{Agent: "mira", Action: sim.Action{Kind: sim.ActionWhisper}, Accepted: true}})
	tally.Emit(Record{Type: RecDecision, Decision: &Decision{Agent: "dov", Action: sim.Action{Kind: sim.ActionDo}, Accepted: true}})
	tally.Emit(Record{Type: RecDecision, Decision: &Decision{Agent: "dov", Action: sim.Action{Kind: sim.ActionPass}, Accepted: true}})
	// A refused decision counts as a rejection, not as an action.
	tally.Emit(Record{Type: RecDecision, Decision: &Decision{Agent: "dov", Action: sim.Action{Kind: sim.ActionSay}, Accepted: false}})
	tally.Emit(Record{Type: RecRejected, Rejected: &Rejection{Agent: "dov", Reason: "not allowed"}})

	tally.Emit(Record{Type: RecOutcome, Outcome: &sim.Outcome{
		Summary: "done", Scores: map[sim.AgentID]float64{"mira": 55, "dov": 30},
	}})
	tally.SetAgentCounts("mira", 2, 3)

	s := tally.Summary("r1", "split", "settled", 90*time.Second, "/tmp/r1.jsonl")

	if s.Rounds != 2 {
		t.Errorf("Rounds = %d", s.Rounds)
	}
	if s.LLMCalls != 3 || s.LLMErrors != 1 || s.Retries != 1 {
		t.Errorf("call accounting = %d calls, %d errors, %d retries", s.LLMCalls, s.LLMErrors, s.Retries)
	}
	if s.Prompt != 300 || s.Output != 55 {
		t.Errorf("tokens = %d -> %d", s.Prompt, s.Output)
	}
	if s.Actions["say"] != 1 || s.Actions["whisper"] != 1 || s.Actions["do"] != 1 || s.Actions["pass"] != 1 {
		t.Errorf("actions = %+v", s.Actions)
	}
	if s.Rejections != 1 {
		t.Errorf("Rejections = %d", s.Rejections)
	}
	if s.Duration != "1m30s" {
		t.Errorf("Duration = %q", s.Duration)
	}
	if s.LogPath != "/tmp/r1.jsonl" || s.StopReason != "settled" {
		t.Errorf("Summary = %+v", s)
	}

	byID := map[sim.AgentID]AgentSummary{}
	for _, a := range s.PerAgent {
		byID[a.ID] = a
	}
	mira := byID["mira"]
	if mira.Calls != 2 || mira.Said != 1 || mira.Whispered != 1 {
		t.Errorf("mira = %+v", mira)
	}
	// Average latency is per agent, not a global mean.
	if mira.AvgLatencyMS != 2000 {
		t.Errorf("mira avg latency = %d, want 2000", mira.AvgLatencyMS)
	}
	if mira.Model != "ollama/qwen3:4b" {
		t.Errorf("mira model = %q", mira.Model)
	}
	if mira.Score == nil || *mira.Score != 55 {
		t.Errorf("mira score = %v", mira.Score)
	}
	if mira.Commitments != 2 || mira.Beliefs != 3 {
		t.Errorf("mira memory counts = %+v", mira)
	}
	dov := byID["dov"]
	if dov.Errors != 1 || dov.Acted != 1 || dov.Passed != 1 {
		t.Errorf("dov = %+v", dov)
	}
	// Agents are reported in a stable order.
	if s.PerAgent[0].ID != "dov" || s.PerAgent[1].ID != "mira" {
		t.Errorf("per-agent order = %v, %v", s.PerAgent[0].ID, s.PerAgent[1].ID)
	}
}

func TestSetPlanAndBeliefCountsFoldsInGoalDirectedExecutionSignals(t *testing.T) {
	tally := NewTally()
	tally.Emit(Record{Type: RecAgent, Agent: &AgentInfo{ID: "mira", Name: "Mira"}})
	tally.Emit(Record{Type: RecAgent, Agent: &AgentInfo{ID: "dov", Name: "Dov"}})

	tally.SetPlanAndBeliefCounts("mira", true, 3, 2)
	// dov never calls this — planning/beliefs disabled for dov — and must
	// read as the zero value, not be missing from the summary.

	s := tally.Summary("r1", "social", "settled", time.Second, "")
	byID := map[sim.AgentID]AgentSummary{}
	for _, a := range s.PerAgent {
		byID[a.ID] = a
	}
	mira := byID["mira"]
	if !mira.HasPlan || mira.PlanRevisions != 3 || mira.BeliefsWithProvenance != 2 {
		t.Errorf("mira = %+v", mira)
	}
	dov := byID["dov"]
	if dov.HasPlan || dov.PlanRevisions != 0 || dov.BeliefsWithProvenance != 0 {
		t.Errorf("dov (never called) should read as the zero value: %+v", dov)
	}
}

func TestConsoleRespectsVerbosity(t *testing.T) {
	render := func(level Verbosity, recs ...Record) string {
		var buf bytes.Buffer
		c := NewConsole(&buf, level)
		for _, r := range recs {
			c.Emit(r)
		}
		return buf.String()
	}

	agentRec := Record{Type: RecAgent, Agent: &AgentInfo{ID: "mira", Name: "Mira",
		Model: llm.ModelConfig{Provider: "ollama", Model: "qwen3:4b"}}}
	say := Record{Type: RecEvent, Event: &sim.Event{Type: sim.EventSay, Actor: "mira", Text: "hello there", Visibility: sim.PublicTo()}}
	whisper := Record{Type: RecEvent, Event: &sim.Event{Type: sim.EventWhisper, Actor: "mira",
		Targets: []sim.AgentID{"dov"}, Text: "psst", Visibility: sim.Only("mira", "dov")}}
	pass := Record{Type: RecEvent, Event: &sim.Event{Type: sim.EventPass, Actor: "mira", Visibility: sim.PublicTo()}}
	call := Record{Type: RecLLMCall, Call: &Call{Agent: "mira", Model: "qwen3:4b", LatencyMS: 1200,
		PromptTokens: 100, OutputTokens: 20, Thinking: "hidden musing",
		Prompt: []llm.Message{llm.System("SYSTEM PROMPT TEXT")}}}
	decision := Record{Type: RecDecision, Decision: &Decision{Agent: "mira",
		Action: sim.Action{Kind: sim.ActionSay, Reasoning: "because I chose to"}, Accepted: true}}

	t.Run("quiet hides the conversation", func(t *testing.T) {
		out := render(Quiet, agentRec, say, whisper, pass)
		if strings.Contains(out, "psst") || strings.Contains(out, "passes") {
			t.Errorf("quiet output is too chatty:\n%s", out)
		}
		// Public speech is the headline and is always shown.
		if !strings.Contains(out, "hello there") {
			t.Errorf("quiet should still show public speech:\n%s", out)
		}
	})

	t.Run("normal shows speech, whispers and passes but not metrics", func(t *testing.T) {
		out := render(Normal, agentRec, say, whisper, pass, call, decision)
		for _, want := range []string{"hello there", "psst", "whisper", "passes"} {
			if !strings.Contains(out, want) {
				t.Errorf("normal output missing %q:\n%s", want, out)
			}
		}
		if strings.Contains(out, "1200ms") {
			t.Errorf("normal output should not show per-call latency:\n%s", out)
		}
		if strings.Contains(out, "because I chose to") {
			t.Errorf("normal output should not show self-reported reasoning:\n%s", out)
		}
	})

	t.Run("verbose adds metrics and self-reported reasoning", func(t *testing.T) {
		out := render(Verbose, agentRec, decision, call)
		if !strings.Contains(out, "1200ms") || !strings.Contains(out, "100→20 tok") {
			t.Errorf("verbose output missing call metrics:\n%s", out)
		}
		if !strings.Contains(out, "because I chose to") {
			t.Errorf("verbose output missing reasoning:\n%s", out)
		}
		if !strings.Contains(out, "hidden musing") {
			t.Errorf("verbose output should surface reported thinking:\n%s", out)
		}
		if strings.Contains(out, "SYSTEM PROMPT TEXT") {
			t.Errorf("verbose should not dump full prompts; that is debug:\n%s", out)
		}
	})

	t.Run("debug dumps prompts", func(t *testing.T) {
		out := render(Debug, agentRec, call)
		if !strings.Contains(out, "SYSTEM PROMPT TEXT") {
			t.Errorf("debug output missing the prompt:\n%s", out)
		}
	})
}

func TestConsoleRendersOutcomeAndSummary(t *testing.T) {
	var buf bytes.Buffer
	c := NewConsole(&buf, Normal)
	c.Emit(Record{Type: RecAgent, Agent: &AgentInfo{ID: "mira", Name: "Mira"}})
	c.Emit(Record{Type: RecOutcome, Outcome: &sim.Outcome{
		Summary: "the claims fit",
		Scores:  map[sim.AgentID]float64{"mira": 55, "dov": 30.5},
		Metrics: map[string]any{"broken_promises": 1, "efficiency": 0.85},
	}})
	c.Emit(Record{Type: RecSummary, Summary: &Summary{
		RunID: "r1", Scenario: "split", Rounds: 4, StopReason: "settled",
		Duration: "1m30s", LLMCalls: 12, Prompt: 1000, Output: 200, Latency: "1m20s",
		Actions: map[string]int{"say": 6, "do": 3}, Rejections: 1,
		PerAgent: []AgentSummary{{ID: "mira", Name: "Mira", Model: "ollama/qwen3:4b", Calls: 6, Said: 3, AvgLatencyMS: 1200}},
		LogPath:  "/tmp/r1.jsonl",
	}})
	out := buf.String()

	for _, want := range []string{"the claims fit", "Mira", "55", "30.50", "broken_promises", "settled", "1m30s", "12 calls", "say=6", "rejected=1", "/tmp/r1.jsonl"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	// Scores are ranked highest first.
	if strings.Index(out, "55") > strings.Index(out, "30.50") {
		t.Errorf("scores should be ordered best first:\n%s", out)
	}
}

func TestConsoleDoesNotRepeatTheOutcomeAsAnEvent(t *testing.T) {
	var buf bytes.Buffer
	c := NewConsole(&buf, Normal)
	c.Emit(Record{Type: RecEvent, Event: &sim.Event{Type: sim.EventOutcome, Text: "everybody busted", Visibility: sim.PublicTo()}})
	if strings.Contains(buf.String(), "everybody busted") {
		t.Errorf("the outcome event should be left to the outcome banner:\n%s", buf.String())
	}
}

func TestConsoleDisablesColourForNonTerminals(t *testing.T) {
	var buf bytes.Buffer
	c := NewConsole(&buf, Normal)
	c.Emit(Record{Type: RecAgent, Agent: &AgentInfo{ID: "mira", Name: "Mira"}})
	c.Emit(Record{Type: RecEvent, Event: &sim.Event{Type: sim.EventSay, Actor: "mira", Text: "hi", Visibility: sim.PublicTo()}})
	if strings.Contains(buf.String(), "\033[") {
		t.Errorf("ANSI codes should not be written to a non-terminal:\n%q", buf.String())
	}
}

func TestMultiFansOutAndClosesAll(t *testing.T) {
	a, b := &countingSink{}, &countingSink{}
	m := Multi{a, b}
	m.Emit(Record{Type: RecEvent})
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if a.emitted != 1 || b.emitted != 1 || !a.closed || !b.closed {
		t.Errorf("a = %+v, b = %+v", a, b)
	}
}

type countingSink struct {
	emitted int
	closed  bool
}

func (c *countingSink) Emit(Record) { c.emitted++ }
func (c *countingSink) Close() error {
	c.closed = true
	return nil
}

func TestFormatDuration(t *testing.T) {
	for in, want := range map[time.Duration]string{
		500 * time.Millisecond:  "500ms",
		1500 * time.Millisecond: "1.5s",
		90 * time.Second:        "1m30s",
		3661 * time.Second:      "61m01s",
	} {
		if got := FormatDuration(in); got != want {
			t.Errorf("FormatDuration(%s) = %q, want %q", in, got, want)
		}
	}
}
