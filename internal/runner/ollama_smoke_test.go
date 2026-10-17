package runner

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/llm/ollama"
	"github.com/nmiano/playworld/internal/orchestrator"
)

// TestOllamaSmoke drives a real local model end to end. It is skipped unless
// PLAYWORLD_OLLAMA_TEST is set, so the ordinary `go test ./...` never needs
// Ollama running:
//
//	PLAYWORLD_OLLAMA_TEST=1 go test ./internal/runner -run Ollama -v
//	PLAYWORLD_OLLAMA_MODEL=qwen3:8b PLAYWORLD_OLLAMA_TEST=1 go test ./internal/runner -run Ollama -v
func TestOllamaSmoke(t *testing.T) {
	if os.Getenv("PLAYWORLD_OLLAMA_TEST") == "" {
		t.Skip("set PLAYWORLD_OLLAMA_TEST=1 to run the live Ollama smoke test")
	}

	model := os.Getenv("PLAYWORLD_OLLAMA_MODEL")
	if model == "" {
		model = "qwen3:4b"
	}

	provider := ollama.New(ollama.WithEndpoint(os.Getenv("OLLAMA_HOST")))
	cfg := llm.ModelConfig{Provider: "ollama", Model: model, Timeout: 3 * time.Minute}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Preflight first: a missing model should fail with a fixable message
	// rather than halfway through a scenario.
	if err := provider.Preflight(ctx, cfg); err != nil {
		t.Fatalf("preflight failed — is Ollama running and %q pulled?\n%v", model, err)
	}

	reg := llm.NewRegistry()
	reg.Register(provider)

	temp := 0.8
	maxTok := 300
	cast := make([]agent.Persona, 0, 2)
	for _, p := range []struct{ id, name, personality string }{
		{"ada", "Ada", "Patient and cooperative. Looks for the split that leaves nobody badly off."},
		{"rook", "Rook", "Competitive and sharp-elbowed. Wants the largest share and will talk a good game to get it."},
	} {
		cast = append(cast, agent.Persona{
			ID: p.id, Name: p.name, Personality: p.personality,
			Goals: []string{"end up with as much as you can without busting the pool"},
			Model: llm.ModelConfig{
				Provider: "ollama", Model: model, Timeout: 3 * time.Minute,
				Sampling: llm.Sampling{Temperature: &temp, MaxTokens: &maxTok},
			},
			Context: agent.DefaultContext,
		})
	}

	limits := orchestrator.DefaultLimits()
	limits.Timeout = 9 * time.Minute
	limits.MaxConsecutivePasses = 0

	res, err := Execute(ctx, Options{
		RunID: "ollama-smoke", Scenario: "split", Personas: cast, Providers: reg,
		Rounds: 2, Seed: 1,
		Params: map[string]any{
			"pool": 100, "talk_rounds": 1, "attempts": 1, "bonus": 20,
			"needs": map[string]any{"ada": 40, "rook": 60},
		},
		Limits:    limits,
		Sinks:     []explog.Sink{explog.NewConsole(os.Stdout, explog.Verbose)},
		Harness:   "smoke-test",
		Preflight: true,
	})
	if err != nil {
		t.Fatalf("live run failed: %v", err)
	}
	if res.Run.Err != nil {
		t.Fatalf("live run reported a fatal error: %v", res.Run.Err)
	}

	// We cannot assert what a model will say, only that the harness got real
	// work out of it: inference happened, tokens were counted, and the agents
	// produced actions the environment accepted.
	if res.Summary.LLMCalls == 0 {
		t.Fatal("no inference calls were made")
	}
	if res.Summary.Prompt == 0 || res.Summary.Output == 0 {
		t.Errorf("Ollama reported no token usage: %d -> %d", res.Summary.Prompt, res.Summary.Output)
	}
	total := 0
	for _, n := range res.Summary.Actions {
		total += n
	}
	if total == 0 {
		t.Errorf("the model produced no usable action in %d call(s); raw replies are in the log", res.Summary.LLMCalls)
	}
	if res.Outcome.Summary == "" {
		t.Error("the scenario produced no outcome")
	}
	for _, a := range res.Summary.PerAgent {
		if a.AvgLatencyMS <= 0 {
			t.Errorf("no latency recorded for %s", a.ID)
		}
		if !strings.Contains(a.Model, model) {
			t.Errorf("%s ran on %q, want %q", a.ID, a.Model, model)
		}
	}
	t.Logf("live run: %d calls, %d→%d tokens, actions %v, ended %q",
		res.Summary.LLMCalls, res.Summary.Prompt, res.Summary.Output,
		res.Summary.Actions, res.Summary.StopReason)
}
