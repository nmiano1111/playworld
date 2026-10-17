package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func f64(v float64) *float64 { return &v }
func i(v int) *int           { return &v }
func b(v bool) *bool         { return &v }

func TestModelConfigWithDefaults(t *testing.T) {
	defaults := ModelConfig{
		Provider: "ollama", Model: "qwen3:4b", Timeout: time.Minute,
		Sampling: Sampling{Temperature: f64(0.8), TopP: f64(0.9), MaxTokens: i(400)},
		Think:    b(false),
	}
	// An agent that overrides only temperature keeps every other default.
	agent := ModelConfig{Sampling: Sampling{Temperature: f64(0.2)}}
	got := agent.WithDefaults(defaults)

	if got.Provider != "ollama" || got.Model != "qwen3:4b" || got.Timeout != time.Minute {
		t.Errorf("defaults not applied: %+v", got)
	}
	if *got.Sampling.Temperature != 0.2 {
		t.Errorf("override lost: %v", *got.Sampling.Temperature)
	}
	if got.Sampling.TopP == nil || *got.Sampling.TopP != 0.9 {
		t.Errorf("top_p not inherited: %+v", got.Sampling)
	}
	if got.Think == nil || *got.Think {
		t.Errorf("think not inherited: %+v", got.Think)
	}
	// A deliberate zero must survive, which is why these fields are pointers.
	zero := ModelConfig{Sampling: Sampling{Temperature: f64(0)}}.WithDefaults(defaults)
	if *zero.Sampling.Temperature != 0 {
		t.Errorf("an explicit zero temperature was clobbered by the default")
	}
}

func TestModelConfigString(t *testing.T) {
	if got := (ModelConfig{Provider: "ollama", Model: "qwen3:8b"}).String(); got != "ollama/qwen3:8b" {
		t.Errorf("String = %q", got)
	}
}

func TestFatalClassification(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		{ErrUnavailable, true},
		{ErrModelMissing, true},
		{errors.New("wrapped: " + ErrUnavailable.Error()), false},
		{fmtWrap(ErrUnavailable), true},
		{ErrBadResponse, false},
		{errors.New("http 500"), false},
		{nil, false},
	}
	for _, c := range cases {
		if got := Fatal(c.err); got != c.want {
			t.Errorf("Fatal(%v) = %v, want %v", c.err, got, c.want)
		}
	}
}

func fmtWrap(err error) error { return errors.Join(errors.New("context"), err) }

func TestRegistry(t *testing.T) {
	reg := NewRegistry()
	fake := NewFake(nil)
	reg.Register(fake)

	got, err := reg.Get("fake")
	if err != nil || got != fake {
		t.Fatalf("Get = %v, %v", got, err)
	}
	if _, err := reg.Get("anthropic"); err == nil {
		t.Error("an unregistered provider must be an error, not a nil provider")
	}
	if names := reg.Names(); len(names) != 1 || names[0] != "fake" {
		t.Errorf("Names = %v", names)
	}
}

func TestUsageTotal(t *testing.T) {
	if got := (Usage{PromptTokens: 10, CompletionTokens: 5}).Total(); got != 15 {
		t.Errorf("Total = %d", got)
	}
}

func TestFakeRecordsCallsAndSynthesisesUsage(t *testing.T) {
	fake := ScriptedFake(`{"kind":"say","content":"one"}`, `{"kind":"pass"}`)
	ctx := context.Background()
	req := Request{Model: ModelConfig{Model: "test"}, Messages: []Message{System("sys"), User("hello")}}

	first, err := fake.Chat(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if first.Content != `{"kind":"say","content":"one"}` {
		t.Errorf("Content = %q", first.Content)
	}
	if first.Usage.PromptTokens == 0 {
		t.Error("fake should synthesise usage so metric paths are exercised")
	}
	second, _ := fake.Chat(ctx, req)
	if second.Content != `{"kind":"pass"}` {
		t.Errorf("second Content = %q", second.Content)
	}
	// The script repeats its final entry rather than running out.
	third, _ := fake.Chat(ctx, req)
	if third.Content != `{"kind":"pass"}` {
		t.Errorf("third Content = %q", third.Content)
	}
	if fake.CallCount() != 3 || len(fake.Calls()) != 3 {
		t.Errorf("CallCount = %d", fake.CallCount())
	}
}

func TestFakeHonoursCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewFake(nil).Chat(ctx, Request{}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v", err)
	}
}

func TestFakePropagatesResponderError(t *testing.T) {
	boom := errors.New("boom")
	fake := NewFake(func(int, Request) (*Response, error) { return nil, boom })
	if _, err := fake.Chat(context.Background(), Request{}); !errors.Is(err, boom) {
		t.Errorf("err = %v", err)
	}
	fake = NewFake(func(int, Request) (*Response, error) { return nil, nil })
	if _, err := fake.Chat(context.Background(), Request{}); !errors.Is(err, ErrBadResponse) {
		t.Errorf("a nil response must be ErrBadResponse, got %v", err)
	}
}
