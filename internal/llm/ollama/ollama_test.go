package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/llm"
)

func testConfig(endpoint string) llm.ModelConfig {
	temp := 0.3
	maxTok := 128
	seed := 7
	return llm.ModelConfig{
		Provider: "ollama", Model: "qwen3:4b", Endpoint: endpoint,
		Timeout:  5 * time.Second,
		Sampling: llm.Sampling{Temperature: &temp, MaxTokens: &maxTok, Seed: &seed},
	}
}

func TestChatHappyPath(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path = %q", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &gotBody)
		json.NewEncoder(w).Encode(map[string]any{
			"model":             "qwen3:4b",
			"message":           map[string]any{"role": "assistant", "content": `{"kind":"pass"}`},
			"done":              true,
			"done_reason":       "stop",
			"prompt_eval_count": 120,
			"eval_count":        9,
		})
	}))
	defer srv.Close()

	p := New(WithEndpoint(srv.URL))
	resp, err := p.Chat(context.Background(), llm.Request{
		Model:    testConfig(srv.URL),
		Messages: []llm.Message{llm.System("you are a test"), llm.User("go")},
		Format:   llm.FormatJSON,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != `{"kind":"pass"}` {
		t.Errorf("Content = %q", resp.Content)
	}
	if resp.Usage.PromptTokens != 120 || resp.Usage.CompletionTokens != 9 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
	if resp.Latency <= 0 {
		t.Error("latency should be measured")
	}
	if resp.StopReason != "stop" {
		t.Errorf("StopReason = %q", resp.StopReason)
	}

	// Sampling parameters must be translated into Ollama's option names.
	opts, _ := gotBody["options"].(map[string]any)
	if opts["temperature"] != 0.3 || opts["num_predict"] != float64(128) || opts["seed"] != float64(7) {
		t.Errorf("options = %+v", opts)
	}
	if gotBody["stream"] != false {
		t.Errorf("stream = %v, want false", gotBody["stream"])
	}
	if gotBody["format"] != "json" {
		t.Errorf("format = %v, want json", gotBody["format"])
	}
	// Thinking must be explicitly off: Qwen3 defaults it on, and the harness
	// wants a compact action, not a monologue it pays for and discards.
	if gotBody["think"] != false {
		t.Errorf("think = %v, want false", gotBody["think"])
	}
	if gotBody["keep_alive"] == "" {
		t.Error("keep_alive should be set so many agents can share one loaded model")
	}
}

func TestChatSeparatesInlineThinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"content": "<think>let me consider everything</think>\n{\"kind\":\"pass\"}",
			},
			"done": true,
		})
	}))
	defer srv.Close()

	resp, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
		Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != `{"kind":"pass"}` {
		t.Errorf("Content = %q — reasoning must not contaminate the action", resp.Content)
	}
	if resp.Thinking != "let me consider everything" {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
}

func TestChatPrefersReportedThinkingField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": `{"kind":"pass"}`, "thinking": "reported separately"},
			"done":    true,
		})
	}))
	defer srv.Close()
	resp, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
		Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Thinking != "reported separately" {
		t.Errorf("Thinking = %q", resp.Thinking)
	}
}

func TestChatErrorHandling(t *testing.T) {
	t.Run("missing model is classified and actionable", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]string{"error": `model "qwen3:4b" not found`})
		}))
		defer srv.Close()

		_, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
			Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
		})
		if !errors.Is(err, llm.ErrModelMissing) {
			t.Fatalf("err = %v, want ErrModelMissing", err)
		}
		if !strings.Contains(err.Error(), "ollama pull qwen3:4b") {
			t.Errorf("error should tell the user how to fix it, got %v", err)
		}
	})

	t.Run("server error is retryable, not fatal", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, "overloaded")
		}))
		defer srv.Close()

		_, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
			Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
		})
		if err == nil {
			t.Fatal("expected an error")
		}
		if llm.Fatal(err) {
			t.Errorf("an http 500 should not be fatal: %v", err)
		}
	})

	t.Run("unreachable daemon is fatal and says how to start it", func(t *testing.T) {
		// Port 1 is reliably closed.
		_, err := New(WithEndpoint("http://127.0.0.1:1")).Chat(context.Background(), llm.Request{
			Model:    llm.ModelConfig{Model: "qwen3:4b", Endpoint: "http://127.0.0.1:1", Timeout: 2 * time.Second},
			Messages: []llm.Message{llm.User("go")},
		})
		if !errors.Is(err, llm.ErrUnavailable) {
			t.Fatalf("err = %v, want ErrUnavailable", err)
		}
		if !strings.Contains(err.Error(), "ollama serve") {
			t.Errorf("error should say how to start Ollama, got %v", err)
		}
	})

	t.Run("malformed body is a bad response", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, "not json at all")
		}))
		defer srv.Close()
		_, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
			Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
		})
		if !errors.Is(err, llm.ErrBadResponse) {
			t.Errorf("err = %v, want ErrBadResponse", err)
		}
	})

	t.Run("error reported in a 200 body is surfaced", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			json.NewEncoder(w).Encode(map[string]string{"error": "out of memory"})
		}))
		defer srv.Close()
		_, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
			Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
		})
		if err == nil || !strings.Contains(err.Error(), "out of memory") {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("no model configured", func(t *testing.T) {
		_, err := New().Chat(context.Background(), llm.Request{Messages: []llm.Message{llm.User("go")}})
		if err == nil {
			t.Error("expected an error when no model is set")
		}
	})
}

func TestChatHonoursContextCancellation(t *testing.T) {
	// The handler is slow but finite, so closing the server never blocks.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		json.NewEncoder(w).Encode(map[string]any{"message": map[string]any{"content": "{}"}, "done": true})
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	_, err := New(WithEndpoint(srv.URL)).Chat(ctx, llm.Request{
		Model: testConfig(srv.URL), Messages: []llm.Message{llm.User("go")},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

// TestChatTimesOutPerRequest checks that ModelConfig.Timeout bounds a single
// generation, which is what keeps one wedged call from stalling a whole run.
func TestChatTimesOutPerRequest(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.Timeout = 50 * time.Millisecond
	_, err := New(WithEndpoint(srv.URL)).Chat(context.Background(), llm.Request{
		Model: cfg, Messages: []llm.Message{llm.User("go")},
	})
	if err == nil {
		t.Fatal("expected a timeout")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("error should say it timed out and suggest a fix, got %v", err)
	}
}

func TestPreflight(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{{"name": "qwen3:4b"}, {"name": "llama3:latest"}},
		})
	}))
	defer srv.Close()

	p := New(WithEndpoint(srv.URL))
	if err := p.Preflight(context.Background(), testConfig(srv.URL)); err != nil {
		t.Errorf("installed model should pass preflight: %v", err)
	}
	// A ":latest" tag matches the bare name, as the Ollama CLI treats them.
	if err := p.Preflight(context.Background(), llm.ModelConfig{Model: "llama3", Endpoint: srv.URL}); err != nil {
		t.Errorf("llama3 should match llama3:latest: %v", err)
	}

	err := p.Preflight(context.Background(), llm.ModelConfig{Model: "qwen3:32b", Endpoint: srv.URL})
	if !errors.Is(err, llm.ErrModelMissing) {
		t.Fatalf("err = %v, want ErrModelMissing", err)
	}
	// The message must list what *is* installed, so the fix is obvious.
	if !strings.Contains(err.Error(), "qwen3:4b") || !strings.Contains(err.Error(), "ollama pull qwen3:32b") {
		t.Errorf("unhelpful error: %v", err)
	}
}

func TestPreflightUnreachable(t *testing.T) {
	err := New(WithEndpoint("http://127.0.0.1:1")).Preflight(context.Background(), llm.ModelConfig{Model: "x"})
	if !errors.Is(err, llm.ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
	if !strings.Contains(err.Error(), "brew install ollama") {
		t.Errorf("error should include install guidance, got %v", err)
	}
}

func TestListCachesResults(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		json.NewEncoder(w).Encode(map[string]any{"models": []map[string]any{{"name": "b"}, {"name": "a"}}})
	}))
	defer srv.Close()

	p := New(WithEndpoint(srv.URL))
	got, err := p.List(context.Background(), llm.ModelConfig{})
	if err != nil {
		t.Fatal(err)
	}
	// Sorted, so `doctor` output is stable.
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("List = %v", got)
	}
	if _, err := p.List(context.Background(), llm.ModelConfig{}); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("/api/tags was fetched %d times; the result should be cached", calls)
	}
}

func TestProviderSatisfiesInterface(t *testing.T) {
	var _ llm.Provider = New()
	if New().Name() != "ollama" {
		t.Error("Name should be ollama")
	}
}
