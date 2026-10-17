package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

func testPersona(id, name string) Persona {
	return Persona{
		ID: id, Name: name,
		Personality: "a test subject",
		Goals:       []string{"secret goal of " + id},
		Model:       llm.ModelConfig{Provider: "fake", Model: "test"},
		Context:     DefaultContext,
	}
}

func newTestAgent(t *testing.T, id, name string, p llm.Provider, store memory.Store) *Agent {
	t.Helper()
	a, err := New(Options{Persona: testPersona(id, name), Provider: p, Store: store, RunID: "run-test"})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
	return a
}

func basicTurn() sim.Turn {
	return sim.Turn{
		RunID: "run-test", Scenario: "test", Round: 1, MaxRound: 3,
		Self: "mira", Participants: []sim.AgentID{"mira", "dov"},
		Brief:   "rules go here",
		Actions: []sim.ActionSpec{{Kind: sim.ActionSay, Usage: `{"kind":"say"}`, Description: "speak"}},
	}
}

func TestDecodeActionShapes(t *testing.T) {
	participants := map[string]sim.AgentID{"mira": "mira", "dov": "dov", "Dov": "dov"}
	cases := []struct {
		name    string
		raw     string
		want    sim.Action
		wantErr bool
	}{
		{
			name: "plain say",
			raw:  `{"kind":"say","content":"hello"}`,
			want: sim.Action{Kind: sim.ActionSay, Content: "hello"},
		},
		{
			name: "whisper with a single string recipient",
			raw:  `{"kind":"whisper","to":"dov","content":"psst"}`,
			want: sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"dov"}},
		},
		{
			name: "whisper names resolve case-insensitively",
			raw:  `{"kind":"whisper","to":["DOV"],"content":"psst"}`,
			want: sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"dov"}},
		},
		{
			name: "environment action with args",
			raw:  `{"kind":"do","verb":"claim","args":{"amount":40}}`,
			want: sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": float64(40)}},
		},
		{
			name: "pass",
			raw:  `{"kind":"pass"}`,
			want: sim.Action{Kind: sim.ActionPass},
		},
		{
			name: "synonyms for kind are accepted",
			raw:  `{"kind":"speak","message":"hi"}`,
			want: sim.Action{Kind: sim.ActionSay, Content: "hi"},
		},
		{
			name: "wrapped in a markdown fence",
			raw:  "Sure!\n```json\n{\"kind\":\"say\",\"content\":\"fenced\"}\n```\n",
			want: sim.Action{Kind: sim.ActionSay, Content: "fenced"},
		},
		{
			name: "trailing prose after the object",
			raw:  `{"kind":"say","content":"ok"} — hope that helps`,
			want: sim.Action{Kind: sim.ActionSay, Content: "ok"},
		},
		{
			name: "kind inferred from a verb",
			raw:  `{"verb":"claim","args":{"amount":10}}`,
			want: sim.Action{Kind: sim.ActionDo, Verb: "claim", Args: map[string]any{"amount": float64(10)}},
		},
		{
			name: "optional memory fields are carried through",
			raw:  `{"kind":"say","content":"hi","reasoning":"why","remember":"note","opinions":{"dov":"blunt"}}`,
			want: sim.Action{Kind: sim.ActionSay, Content: "hi", Reasoning: "why", Remember: "note", Opinions: map[string]string{"dov": "blunt"}},
		},
		{name: "whisper with no recipient is refused", raw: `{"kind":"whisper","content":"x"}`, wantErr: true},
		{name: "whisper to an unknown agent is refused", raw: `{"kind":"whisper","to":["ghost"],"content":"x"}`, wantErr: true},
		{name: "not json at all", raw: `I would like to say hello.`, wantErr: true},
		{name: "empty reply", raw: "   ", wantErr: true},
		{name: "unterminated object", raw: `{"kind":"say","content":"oops`, wantErr: true},
		{name: "object with nothing usable", raw: `{"mood":"pensive"}`, wantErr: true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := decodeAction(c.raw, participants, "mira")
			if c.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.Kind != c.want.Kind || got.Content != c.want.Content || got.Verb != c.want.Verb {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
			if len(got.To) != len(c.want.To) {
				t.Fatalf("recipients = %v, want %v", got.To, c.want.To)
			}
			for i := range got.To {
				if got.To[i] != c.want.To[i] {
					t.Errorf("recipients = %v, want %v", got.To, c.want.To)
				}
			}
			if c.want.Remember != "" && got.Remember != c.want.Remember {
				t.Errorf("Remember = %q", got.Remember)
			}
			if c.want.Opinions != nil && got.Opinions["dov"] != c.want.Opinions["dov"] {
				t.Errorf("Opinions = %v", got.Opinions)
			}
		})
	}
}

func TestWhisperToSelfIsDropped(t *testing.T) {
	participants := map[string]sim.AgentID{"mira": "mira", "dov": "dov"}
	// Addressing yourself alone leaves no recipients, which is an error rather
	// than a silent no-op.
	if _, err := decodeAction(`{"kind":"whisper","to":["mira"],"content":"x"}`, participants, "mira"); err == nil {
		t.Error("whispering only to yourself should be refused")
	}
	// Mixed in with a real recipient, the self-address is simply dropped.
	got, err := decodeAction(`{"kind":"whisper","to":["mira","dov"],"content":"x"}`, participants, "mira")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.To) != 1 || got.To[0] != "dov" {
		t.Errorf("recipients = %v, want [dov]", got.To)
	}
}

// TestAgentsAreIsolated is the core guarantee: agents sharing one provider
// must not share identity, prompt context or memory.
func TestAgentsAreIsolated(t *testing.T) {
	provider := llm.NewFake(func(call int, req llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"kind":"say","content":"hi"}`}, nil
	})
	store := memory.NewMemStore()

	mira := newTestAgent(t, "mira", "Mira", provider, store)
	dov := newTestAgent(t, "dov", "Dov", provider, store)

	// Each observes something different.
	mira.Observe(sim.Observation{Round: 1, Type: sim.EventWhisper, Private: true, Text: "a secret for Mira only"})
	dov.Observe(sim.Observation{Round: 1, Type: sim.EventSay, Text: "something public"})

	ctx := context.Background()
	if _, err := mira.Decide(ctx, basicTurn()); err != nil {
		t.Fatal(err)
	}
	dovTurn := basicTurn()
	dovTurn.Self = "dov"
	if _, err := dov.Decide(ctx, dovTurn); err != nil {
		t.Fatal(err)
	}

	calls := provider.Calls()
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
	miraPrompt := promptText(calls[0])
	dovPrompt := promptText(calls[1])

	if !strings.Contains(miraPrompt, "a secret for Mira only") {
		t.Error("Mira's prompt is missing her own observation")
	}
	if strings.Contains(dovPrompt, "a secret for Mira only") {
		t.Error("Dov's prompt contains an observation only Mira saw")
	}
	if !strings.Contains(miraPrompt, "secret goal of mira") {
		t.Error("Mira's prompt is missing her own goals")
	}
	if strings.Contains(dovPrompt, "secret goal of mira") {
		t.Error("Dov's prompt leaked Mira's private goals")
	}
	if !strings.Contains(miraPrompt, "You are Mira") || !strings.Contains(dovPrompt, "You are Dov") {
		t.Error("agents did not get their own identity")
	}

	// Memory is separate on disk too.
	if err := mira.Persist(); err != nil {
		t.Fatal(err)
	}
	if err := dov.Persist(); err != nil {
		t.Fatal(err)
	}
	stored, err := store.Load("dov")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range stored.Episodes {
		if strings.Contains(e.Text, "a secret for Mira only") {
			t.Error("Dov's persisted memory contains Mira's private observation")
		}
	}
}

func promptText(req llm.Request) string {
	var b strings.Builder
	for _, m := range req.Messages {
		b.WriteString(string(m.Role))
		b.WriteString(": ")
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	return b.String()
}

func TestDecideRetriesOnUnparseableReplyThenSucceeds(t *testing.T) {
	provider := llm.NewFake(func(call int, req llm.Request) (*llm.Response, error) {
		if call == 0 {
			return &llm.Response{Content: "I'm not sure what to do."}, nil
		}
		return &llm.Response{Content: `{"kind":"pass"}`}, nil
	})
	a := newTestAgent(t, "mira", "Mira", provider, nil)

	act, err := a.Decide(context.Background(), basicTurn())
	if err != nil {
		t.Fatalf("expected the retry to recover, got %v", err)
	}
	if act.Kind != sim.ActionPass {
		t.Errorf("action = %+v", act)
	}
	if provider.CallCount() != 2 {
		t.Errorf("CallCount = %d, want 2", provider.CallCount())
	}
	// The retry must tell the model what was wrong with its own reply.
	second := provider.Calls()[1]
	if len(second.Messages) < 4 {
		t.Fatalf("retry prompt had %d messages, expected the bad reply plus a complaint", len(second.Messages))
	}
	if !strings.Contains(promptText(second), "could not be used") {
		t.Error("retry prompt does not explain the problem")
	}
}

func TestDecideGivesUpAfterBoundedRetries(t *testing.T) {
	provider := llm.NewFake(func(call int, req llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: "still prose"}, nil
	})
	a := newTestAgent(t, "mira", "Mira", provider, nil)

	if _, err := a.Decide(context.Background(), basicTurn()); err == nil {
		t.Fatal("expected an error after exhausting retries")
	}
	if got := provider.CallCount(); got != maxDecodeRetries+1 {
		t.Errorf("CallCount = %d, want %d — retries must be bounded", got, maxDecodeRetries+1)
	}
}

func TestDecideSurfacesBackendFailureAsUnrecoverable(t *testing.T) {
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return nil, fmt.Errorf("dial tcp: %w", llm.ErrUnavailable)
	})
	a := newTestAgent(t, "mira", "Mira", provider, nil)

	_, err := a.Decide(context.Background(), basicTurn())
	if err == nil {
		t.Fatal("expected an error")
	}
	var unrec *sim.Unrecoverable
	if !errors.As(err, &unrec) {
		t.Fatalf("a dead backend must be Unrecoverable so the run stops, got %T: %v", err, err)
	}
	// And it must not have burned a retry on a backend that is simply down.
	if provider.CallCount() != 1 {
		t.Errorf("CallCount = %d, want 1", provider.CallCount())
	}
}

func TestDecideRetriesTransientErrors(t *testing.T) {
	provider := llm.NewFake(func(call int, _ llm.Request) (*llm.Response, error) {
		if call == 0 {
			return nil, fmt.Errorf("ollama: http 500: overloaded")
		}
		return &llm.Response{Content: `{"kind":"pass"}`}, nil
	})
	a := newTestAgent(t, "mira", "Mira", provider, nil)

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatalf("a transient error should be retried, got %v", err)
	}
}

func TestDecideHonoursContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	a := newTestAgent(t, "mira", "Mira", llm.ScriptedFake(`{"kind":"pass"}`), nil)

	if _, err := a.Decide(ctx, basicTurn()); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestObserveBoundsShortTermContext(t *testing.T) {
	p := testPersona("mira", "Mira")
	p.Context.RecentObservations = 3
	p.Context.MaxEpisodes = 100
	provider := llm.ScriptedFake(`{"kind":"pass"}`)
	a, err := New(Options{Persona: p, Provider: provider, RunID: "run-test"})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira"}, nil)

	for i := 0; i < 10; i++ {
		a.Observe(sim.Observation{Round: 1, Text: fmt.Sprintf("line-%d", i)})
	}
	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	prompt := promptText(provider.Calls()[0])
	if strings.Contains(prompt, "line-0") {
		t.Error("prompt contains an observation that should have aged out of the bounded context")
	}
	if !strings.Contains(prompt, "line-9") {
		t.Error("prompt is missing the newest observation")
	}
	// Episodic memory keeps everything, even what the prompt dropped.
	if len(a.State().Episodes) != 10 {
		t.Errorf("episodes = %d, want 10", len(a.State().Episodes))
	}
}

func TestSelfReportedMemoryIsPersisted(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.ScriptedFake(`{"kind":"say","content":"hi","remember":"Rook lied about 33","opinions":{"Dov":"reliable"}}`)
	a := newTestAgent(t, "mira", "Mira", provider, store)

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	// Opinions keyed by display name are resolved to the agent id.
	if got.Beliefs["dov"].Note != "reliable" {
		t.Errorf("beliefs = %+v", got.Beliefs)
	}
	found := false
	for _, e := range got.Episodes {
		if e.Kind == memory.KindNote && e.Text == "Rook lied about 33" {
			found = true
		}
	}
	if !found {
		t.Errorf("the agent's own note was not persisted: %+v", got.Episodes)
	}
}

func TestPersistedMemoryIsRecalledNextRun(t *testing.T) {
	store := memory.NewMemStore()
	first, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: llm.ScriptedFake(`{"kind":"pass"}`), Store: store, RunID: "run-1"})
	if err != nil {
		t.Fatal(err)
	}
	first.Bind([]sim.AgentID{"mira", "dov"}, nil)
	first.Observe(sim.Observation{Round: 1, Text: "Rook claimed 45 after promising 33"})
	first.Commit("never trust Rook's stated number")
	if err := first.Persist(); err != nil {
		t.Fatal(err)
	}

	provider := llm.ScriptedFake(`{"kind":"pass"}`)
	second, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: provider, Store: store, RunID: "run-2"})
	if err != nil {
		t.Fatal(err)
	}
	second.Bind([]sim.AgentID{"mira", "dov"}, nil)
	if _, err := second.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}

	prompt := promptText(provider.Calls()[0])
	if !strings.Contains(prompt, "Rook claimed 45 after promising 33") {
		t.Error("a new session did not recall the previous run's episode")
	}
	if !strings.Contains(prompt, "never trust Rook's stated number") {
		t.Error("a new session did not recall its standing commitment")
	}
}

func TestCallCountImplementsMetered(t *testing.T) {
	a := newTestAgent(t, "mira", "Mira", llm.ScriptedFake(`{"kind":"pass"}`), nil)
	var metered sim.Metered = a
	if metered.CallCount() != 0 {
		t.Fatalf("CallCount = %d before any decision", metered.CallCount())
	}
	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if metered.CallCount() != 1 {
		t.Errorf("CallCount = %d after one decision", metered.CallCount())
	}
}

func TestTraceHookSeparatesThinkingFromAction(t *testing.T) {
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Content:  `{"kind":"say","content":"out loud"}`,
			Thinking: "privately I distrust everyone",
			Usage:    llm.Usage{PromptTokens: 100, CompletionTokens: 20},
		}, nil
	})
	a := newTestAgent(t, "mira", "Mira", provider, nil)

	var traces []Trace
	a.Hook = func(tr Trace) { traces = append(traces, tr) }

	act, err := a.Decide(context.Background(), basicTurn())
	if err != nil {
		t.Fatal(err)
	}
	if len(traces) != 1 {
		t.Fatalf("traces = %d, want 1", len(traces))
	}
	if traces[0].Thinking != "privately I distrust everyone" {
		t.Errorf("thinking was not reported: %q", traces[0].Thinking)
	}
	if strings.Contains(act.Content, "distrust") {
		t.Error("reasoning text leaked into the action")
	}
	if traces[0].Usage.PromptTokens != 100 || a.Stats().PromptTokens != 100 {
		t.Errorf("token usage not recorded: trace=%+v stats=%+v", traces[0].Usage, a.Stats())
	}
}

// --- dual-action mode --------------------------------------------------------

func newDualTestAgent(t *testing.T, id, name string, p llm.Provider) *Agent {
	t.Helper()
	a, err := New(Options{Persona: testPersona(id, name), Provider: p, RunID: "run-test", DualAction: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
	return a
}

func TestDecodeDualActionShapes(t *testing.T) {
	participants := map[string]sim.AgentID{"mira": "mira", "dov": "dov"}

	cases := []struct {
		name      string
		raw       string
		wantComm  *sim.Action
		wantWorld *sim.Action
		wantErr   bool
	}{
		{
			name:      "both present",
			raw:       `{"communication":{"type":"say","message":"hello"},"world_action":{"type":"propose","args":{"text":"x"}}}`,
			wantComm:  &sim.Action{Kind: sim.ActionSay, Content: "hello"},
			wantWorld: &sim.Action{Kind: sim.ActionDo, Verb: "propose"},
		},
		{
			name:     "communication only, world null",
			raw:      `{"communication":{"type":"say","message":"hi"},"world_action":null}`,
			wantComm: &sim.Action{Kind: sim.ActionSay, Content: "hi"},
		},
		{
			name:      "world only, communication null",
			raw:       `{"communication":null,"world_action":{"type":"inspect"}}`,
			wantWorld: &sim.Action{Kind: sim.ActionDo, Verb: "inspect"},
		},
		{
			name: "both null means pass",
			raw:  `{"communication":null,"world_action":null}`,
		},
		{
			name: "both absent means pass",
			raw:  `{}`,
		},
		{
			// An explicit {"type":"pass"} decodes to a real Pass action (not
			// nil), so it is logged as a deliberate choice rather than
			// silently omitted — distinct from leaving world_action out or
			// null entirely, which decodes to no action at all.
			name:      "explicit world_action pass decodes to an explicit Pass",
			raw:       `{"world_action":{"type":"pass"}}`,
			wantWorld: &sim.Action{Kind: sim.ActionPass},
		},
		{
			name:     "whisper by recipient",
			raw:      `{"communication":{"type":"whisper","recipient":"dov","message":"psst"}}`,
			wantComm: &sim.Action{Kind: sim.ActionWhisper, Content: "psst", To: []sim.AgentID{"dov"}},
		},
		{
			name:      "flat world_action params merge into args",
			raw:       `{"world_action":{"type":"propose","title":"x","credits":45,"labor":5}}`,
			wantWorld: &sim.Action{Kind: sim.ActionDo, Verb: "propose"},
		},
		{name: "empty say message is refused", raw: `{"communication":{"type":"say","message":""}}`, wantErr: true},
		{name: "whisper with no recipient is refused", raw: `{"communication":{"type":"whisper","message":"x"}}`, wantErr: true},
		{name: "whisper to unknown agent is refused", raw: `{"communication":{"type":"whisper","recipient":"ghost","message":"x"}}`, wantErr: true},
		{name: "unknown communication type", raw: `{"communication":{"type":"shout","message":"x"}}`, wantErr: true},
		{name: "not json", raw: `I would like to say hello`, wantErr: true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := decodeDualAction(c.raw, participants, "mira")
			if c.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if (got.Communication == nil) != (c.wantComm == nil) {
				t.Fatalf("Communication = %+v, want %+v", got.Communication, c.wantComm)
			}
			if c.wantComm != nil {
				if got.Communication.Kind != c.wantComm.Kind || got.Communication.Content != c.wantComm.Content {
					t.Errorf("Communication = %+v, want %+v", got.Communication, c.wantComm)
				}
				if len(c.wantComm.To) > 0 && (len(got.Communication.To) != 1 || got.Communication.To[0] != c.wantComm.To[0]) {
					t.Errorf("Communication.To = %v, want %v", got.Communication.To, c.wantComm.To)
				}
			}
			if (got.World == nil) != (c.wantWorld == nil) {
				t.Fatalf("World = %+v, want %+v", got.World, c.wantWorld)
			}
			if c.wantWorld != nil && (got.World.Kind != c.wantWorld.Kind || got.World.Verb != c.wantWorld.Verb) {
				t.Errorf("World = %+v, want %+v", got.World, c.wantWorld)
			}
		})
	}
}

func TestDecodeDualActionMergesFlatAndNestedArgs(t *testing.T) {
	participants := map[string]sim.AgentID{"mira": "mira"}
	got, err := decodeDualAction(`{"world_action":{"type":"propose","args":{"text":"from args"},"credits":45,"labor":5}}`, participants, "mira")
	if err != nil {
		t.Fatal(err)
	}
	if got.World.Args["text"] != "from args" {
		t.Errorf("nested args.text lost: %+v", got.World.Args)
	}
	if credits, ok := got.World.ArgFloat("credits"); !ok || credits != 45 {
		t.Errorf("flat credits not merged into args: %+v (ok=%v)", got.World.Args, ok)
	}
	if labor, ok := got.World.ArgFloat("labor"); !ok || labor != 5 {
		t.Errorf("flat labor not merged into args: %+v (ok=%v)", got.World.Args, ok)
	}
}

// TestDecodeWorldSlotRecoversFromTheLiteralDoConfusion is the regression test
// for the "do do" bug: a model primed on the single-action protocol's
// {"kind":"do","verb":"propose",...} sometimes copies the literal word "do"
// into the dual protocol's "type" field while still correctly putting the
// real verb in "verb". "do" is never itself a legal verb, so Verb should win.
func TestDecodeWorldSlotRecoversFromTheLiteralDoConfusion(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"type do, verb propose", `{"type":"do","verb":"propose","args":{"text":"x"}}`, "propose"},
		{"type action, verb inspect", `{"type":"action","verb":"inspect"}`, "inspect"},
		{"type world_action, verb vote", `{"type":"world_action","verb":"vote","args":{"proposal":"p1"}}`, "vote"},
		{"a real verb in type is left alone", `{"type":"propose","args":{"text":"x"}}`, "propose"},
		{"do with no verb present stays do (nothing better to recover)", `{"type":"do"}`, "do"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := decodeWorldSlot(json.RawMessage(c.raw))
			if err != nil {
				t.Fatal(err)
			}
			if got == nil || got.Verb != c.want {
				t.Errorf("decodeWorldSlot(%s) = %+v, want verb %q", c.raw, got, c.want)
			}
		})
	}
}

func TestActionExampleDoesNotLeakTheSingleActionEnvelopeIntoDualMode(t *testing.T) {
	spec := sim.ActionSpec{
		Kind: sim.ActionDo, Verb: "propose", Params: []string{"text", "credits", "labor"},
		Usage: `{"kind":"do","verb":"propose","args":{"text":"...","credits":<int>,"labor":<int>}}`,
	}
	dual := actionExample(spec, true)
	if strings.Contains(dual, `"kind"`) || strings.Contains(dual, `"do"`) {
		t.Errorf("dual-mode example still shows the single-action envelope: %q", dual)
	}
	if !strings.Contains(dual, `"type":"propose"`) {
		t.Errorf("dual-mode example should use the verb as \"type\": %q", dual)
	}
	// Single-action mode is completely unaffected.
	if got := actionExample(spec, false); got != spec.Usage {
		t.Errorf("single-action mode changed: %q, want unchanged %q", got, spec.Usage)
	}
}

func TestActionExampleFallsBackToUsageArgsWhenParamsAreAbsent(t *testing.T) {
	spec := sim.ActionSpec{
		Kind: sim.ActionDo, Verb: "claim",
		Usage: `{"kind":"do","verb":"claim","args":{"amount":<int>}}`,
	}
	dual := actionExample(spec, true)
	if !strings.Contains(dual, `"type":"claim"`) || !strings.Contains(dual, `"amount"`) {
		t.Errorf("dual-mode example should still carry the arg names via Usage fallback: %q", dual)
	}
	if strings.Contains(dual, `"kind"`) {
		t.Errorf("fallback must not leak the single-action envelope either: %q", dual)
	}
}

func TestDualProtocolExplicitlyMarksCommunicationUnavailable(t *testing.T) {
	voteOnly := []sim.ActionSpec{
		{Kind: sim.ActionDo, Verb: "vote", Usage: `{"kind":"do","verb":"vote"}`},
		{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`},
	}
	if anyCommunicationOffered(voteOnly) {
		t.Fatal("test setup: vote-only actions should not offer communication")
	}
	turn := basicTurn()
	turn.Actions = voteOnly
	user := buildUser(testPersona("mira", "Mira"), turn, memory.NewState("mira", "Mira"), nil, "r1", true, false)
	if !strings.Contains(user, "communication") || !strings.Contains(user, "must be null") {
		t.Errorf("dual-mode prompt should explicitly say communication is unavailable:\n%s", user)
	}
}

func TestDualActionAgentCanCommunicateAndActInOneCall(t *testing.T) {
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"communication":{"type":"say","message":"I'll put forward a proposal."},"world_action":{"type":"propose","args":{"text":"a market","credits":45,"labor":5}}}`}, nil
	})
	a := newDualTestAgent(t, "mira", "Mira", provider)

	actions, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 2 {
		t.Fatalf("expected 2 actions, got %d: %+v", len(actions), actions)
	}
	if actions[0].Kind != sim.ActionSay {
		t.Errorf("first action = %+v, want say", actions[0])
	}
	if actions[1].Kind != sim.ActionDo || actions[1].Verb != "propose" {
		t.Errorf("second action = %+v, want do propose", actions[1])
	}
	if got := provider.CallCount(); got != 1 {
		t.Errorf("CallCount = %d, want exactly 1 inference call for both slots", got)
	}
}

func TestDualActionAgentCanDoNothing(t *testing.T) {
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"communication":null,"world_action":null}`}, nil
	})
	a := newDualTestAgent(t, "mira", "Mira", provider)

	actions, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 0 {
		t.Errorf("expected no actions, got %+v", actions)
	}
}

func TestDualActionAgentCanActWithoutSpeaking(t *testing.T) {
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"world_action":{"type":"inspect"}}`}, nil
	})
	a := newDualTestAgent(t, "mira", "Mira", provider)

	actions, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 1 || actions[0].Verb != "inspect" {
		t.Errorf("actions = %+v, want just inspect", actions)
	}
}

func TestNonDualAgentDecideManyDelegatesToDecide(t *testing.T) {
	// A plain agent (DualAction: false) must behave exactly as a single-action
	// agent through DecideMany too, since the orchestrator always calls
	// DecideMany when it's available.
	provider := llm.ScriptedFake(`{"kind":"say","content":"hi"}`)
	a := newTestAgent(t, "mira", "Mira", provider, nil)

	actions, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 1 || actions[0].Kind != sim.ActionSay || actions[0].Content != "hi" {
		t.Errorf("actions = %+v", actions)
	}
	// And the dual-mode protocol text must never have been used for this agent.
	prompt := promptText(provider.Calls()[0])
	if strings.Contains(prompt, "world_action") {
		t.Error("a non-dual agent's prompt should not mention the dual protocol")
	}
}

func TestDualActionRetriesOnUnparseableReply(t *testing.T) {
	provider := llm.NewFake(func(call int, _ llm.Request) (*llm.Response, error) {
		if call == 0 {
			return &llm.Response{Content: "not json at all"}, nil
		}
		return &llm.Response{Content: `{"world_action":{"type":"pass"}}`}, nil
	})
	a := newDualTestAgent(t, "mira", "Mira", provider)

	actions, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil {
		t.Fatalf("expected the retry to recover, got %v", err)
	}
	if len(actions) != 1 || actions[0].Kind != sim.ActionPass {
		t.Errorf("actions = %+v", actions)
	}
	if provider.CallCount() != 2 {
		t.Errorf("CallCount = %d, want 2", provider.CallCount())
	}
}

func TestDualActionSelfReportedMemoryAppliesOnce(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"communication":{"type":"say","message":"hi"},"world_action":{"type":"inspect"},"remember":"paid for an inspection","opinions":{"dov":"careful"}}`}, nil
	})
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: provider, Store: store, RunID: "run-test", DualAction: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})

	if _, err := a.DecideMany(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	found := false
	for _, e := range got.Episodes {
		if e.Text == "paid for an inspection" {
			found = true
		}
	}
	if !found {
		t.Error("the self-reported note was not persisted")
	}
	if got.Beliefs["dov"].Note != "careful" {
		t.Errorf("beliefs = %+v", got.Beliefs)
	}
}

// --- persistent planning -----------------------------------------------------

func newPlanningTestAgent(t *testing.T, id, name string, p llm.Provider, store memory.Store, runID string) *Agent {
	t.Helper()
	a, err := New(Options{Persona: testPersona(id, name), Provider: p, Store: store, RunID: runID, Planning: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})
	return a
}

func TestPlanningDisabledByDefaultNeverTouchesAPlan(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.ScriptedFake(`{"kind":"say","content":"hi","plan":{"primary_goal":"should be ignored"}}`)
	a := newTestAgent(t, "mira", "Mira", provider, store) // Planning: false (default)

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	if got.ActivePlan("run-test") != nil {
		t.Error("a plan field must be ignored entirely when planning is disabled")
	}
	prompt := promptText(provider.Calls()[0])
	if strings.Contains(prompt, `"plan"`) {
		t.Error("the protocol should not mention planning when it is disabled")
	}
}

func TestPlanningEnabledPersistsAndRendersThePlan(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.NewFake(func(call int, req llm.Request) (*llm.Response, error) {
		if call == 0 {
			return &llm.Response{Content: `{"kind":"say","content":"let's get this approved","plan":{"primary_goal":"get p1 approved","next_steps":["talk to dov"]}}`}, nil
		}
		// Second call: the plan from the first call should now be visible
		// in the prompt.
		return &llm.Response{Content: `{"kind":"pass"}`}, nil
	})
	a := newPlanningTestAgent(t, "mira", "Mira", provider, store, "run-test")

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if got := a.State().ActivePlan("run-test"); got == nil || got.PrimaryGoal != "get p1 approved" {
		t.Fatalf("plan not applied in-memory: %+v", got)
	}

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	secondPrompt := promptText(provider.Calls()[1])
	if !strings.Contains(secondPrompt, "get p1 approved") || !strings.Contains(secondPrompt, "talk to dov") {
		t.Errorf("the second turn's prompt should show the plan written on the first turn:\n%s", secondPrompt)
	}

	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	reloaded, _ := store.Load("mira")
	if got := reloaded.ActivePlan("run-test"); got == nil || got.PrimaryGoal != "get p1 approved" {
		t.Errorf("plan did not survive Persist(): %+v", got)
	}
}

func TestPlanOmittedLeavesExistingPlanUnchanged(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.NewFake(func(call int, _ llm.Request) (*llm.Response, error) {
		if call == 0 {
			return &llm.Response{Content: `{"kind":"say","content":"x","plan":{"primary_goal":"original"}}`}, nil
		}
		// No "plan" field at all this time.
		return &llm.Response{Content: `{"kind":"say","content":"y"}`}, nil
	})
	a := newPlanningTestAgent(t, "mira", "Mira", provider, store, "run-test")

	a.Decide(context.Background(), basicTurn())
	a.Decide(context.Background(), basicTurn())

	got := a.State().ActivePlan("run-test")
	if got == nil || got.PrimaryGoal != "original" {
		t.Errorf("omitting \"plan\" should leave the existing one in place, got %+v", got)
	}
}

func TestPlanExplicitNullAbandonsIt(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.NewFake(func(call int, _ llm.Request) (*llm.Response, error) {
		if call == 0 {
			return &llm.Response{Content: `{"kind":"say","content":"x","plan":{"primary_goal":"original"}}`}, nil
		}
		return &llm.Response{Content: `{"kind":"say","content":"y","plan":null}`}, nil
	})
	a := newPlanningTestAgent(t, "mira", "Mira", provider, store, "run-test")

	a.Decide(context.Background(), basicTurn())
	if a.State().ActivePlan("run-test") == nil {
		t.Fatal("setup: plan should exist after the first turn")
	}
	a.Decide(context.Background(), basicTurn())
	if got := a.State().ActivePlan("run-test"); got != nil {
		t.Errorf("explicit plan:null should abandon the plan, got %+v", got)
	}
}

func TestPlanDoesNotSurviveARetryOfTheSameTurn(t *testing.T) {
	// A decode failure does not adopt whatever "plan" happened to be in the
	// unparseable reply — the plan update only lands alongside a reply whose
	// action(s) actually decoded.
	store := memory.NewMemStore()
	provider := llm.NewFake(func(call int, _ llm.Request) (*llm.Response, error) {
		if call == 0 {
			return &llm.Response{Content: `not json, but if it were: "plan":{"primary_goal":"should not stick"}`}, nil
		}
		return &llm.Response{Content: `{"kind":"pass"}`}, nil
	})
	a := newPlanningTestAgent(t, "mira", "Mira", provider, store, "run-test")

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if got := a.State().ActivePlan("run-test"); got != nil {
		t.Errorf("a plan mentioned in an unparseable reply must not be adopted: %+v", got)
	}
}

func TestPlanIsScopedToItsOwnRunAtTheAgentLevel(t *testing.T) {
	store := memory.NewMemStore()
	first := newPlanningTestAgent(t, "mira", "Mira", llm.ScriptedFake(
		`{"kind":"say","content":"x","plan":{"primary_goal":"goal for session one"}}`,
	), store, "session-1")
	if _, err := first.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := first.Persist(); err != nil {
		t.Fatal(err)
	}

	// A brand new agent instance, same underlying store, same persona, but a
	// DIFFERENT run — resuming memory does not mean resuming the plan.
	secondProvider := llm.ScriptedFake(`{"kind":"pass"}`)
	second := newPlanningTestAgent(t, "mira", "Mira", secondProvider, store, "session-2")
	if got := second.State().ActivePlan("session-2"); got != nil {
		t.Errorf("a fresh scenario instance must not inherit a plan from an unrelated one: %+v", got)
	}
	// The prompt itself must not mention session-1's plan either.
	if _, err := second.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if prompt := promptText(secondProvider.Calls()[0]); strings.Contains(prompt, "goal for session one") {
		t.Errorf("session-2's prompt must not mention session-1's plan:\n%s", prompt)
	}
}

func TestDualActionPlanningWorksTogether(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{
			Content: `{"communication":{"type":"say","message":"on it"},"world_action":{"type":"inspect"},` +
				`"plan":{"primary_goal":"learn the real cost","next_steps":["inspect","then propose"]}}`,
		}, nil
	})
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: provider, Store: store, RunID: "run-test", DualAction: true, Planning: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})

	actions, err := a.DecideMany(context.Background(), basicTurn())
	if err != nil {
		t.Fatal(err)
	}
	if len(actions) != 2 {
		t.Fatalf("expected both a communication and a world action, got %+v", actions)
	}
	if got := a.State().ActivePlan("run-test"); got == nil || got.PrimaryGoal != "learn the real cost" {
		t.Errorf("plan should be captured alongside dual-mode actions in the same call: %+v", got)
	}
	if provider.CallCount() != 1 {
		t.Errorf("CallCount = %d, want exactly 1 — plan, communication and world action all came from one call", provider.CallCount())
	}
}

// --- belief provenance --------------------------------------------------

func TestBeliefProvenanceDisabledIgnoresBasisAndConfidence(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.ScriptedFake(`{"kind":"say","content":"hi","opinions":{"dov":{"note":"careful","basis":"watched them do it","confidence":"high"}}}`)
	a := newTestAgent(t, "mira", "Mira", provider, store) // BeliefProvenance: false (default)
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	bel := got.Beliefs["dov"]
	if bel.Note != "careful" {
		t.Errorf("the note itself should still be recorded, got %+v", bel)
	}
	if bel.Basis != "" || bel.Confidence != "" {
		t.Errorf("provenance must be ignored entirely when the feature is disabled, got %+v", bel)
	}
	prompt := promptText(provider.Calls()[0])
	if strings.Contains(prompt, "basis") {
		t.Error("the protocol should not mention belief provenance when it is disabled")
	}
}

func TestBeliefProvenanceEnabledRecordsBasisAndConfidence(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.ScriptedFake(`{"kind":"say","content":"hi","opinions":{"dov":{"note":"careful","basis":"watched them do it","confidence":"high"}}}`)
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: provider, Store: store, RunID: "run-test", BeliefProvenance: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	bel := got.Beliefs["dov"]
	if bel.Note != "careful" || bel.Basis != "watched them do it" || bel.Confidence != "high" {
		t.Errorf("belief = %+v", bel)
	}
	prompt := promptText(provider.Calls()[0])
	if !strings.Contains(prompt, "basis") {
		t.Error("the protocol should describe the richer opinions shape when provenance is enabled")
	}
}

func TestBeliefProvenancePlainStringStillWorksWhenEnabled(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.ScriptedFake(`{"kind":"say","content":"hi","opinions":{"dov":"careful"}}`)
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: provider, Store: store, RunID: "run-test", BeliefProvenance: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})

	if _, err := a.Decide(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	bel := got.Beliefs["dov"]
	if bel.Note != "careful" || bel.Basis != "" || bel.Confidence != "" {
		t.Errorf("a plain-string opinion should still work with no provenance attached: %+v", bel)
	}
}

func TestDualActionBeliefProvenanceAppliesOnce(t *testing.T) {
	store := memory.NewMemStore()
	provider := llm.NewFake(func(int, llm.Request) (*llm.Response, error) {
		return &llm.Response{Content: `{"communication":{"type":"say","message":"hi"},"world_action":{"type":"inspect"},` +
			`"opinions":{"dov":{"note":"careful","basis":"they told me","confidence":"low"}}}`}, nil
	})
	a, err := New(Options{Persona: testPersona("mira", "Mira"), Provider: provider, Store: store, RunID: "run-test", DualAction: true, BeliefProvenance: true})
	if err != nil {
		t.Fatal(err)
	}
	a.Bind([]sim.AgentID{"mira", "dov"}, map[sim.AgentID]string{"mira": "Mira", "dov": "Dov"})

	if _, err := a.DecideMany(context.Background(), basicTurn()); err != nil {
		t.Fatal(err)
	}
	if err := a.Persist(); err != nil {
		t.Fatal(err)
	}
	got, _ := store.Load("mira")
	bel := got.Beliefs["dov"]
	if bel.Note != "careful" || bel.Basis != "they told me" || bel.Confidence != "low" {
		t.Errorf("belief = %+v", bel)
	}
}
