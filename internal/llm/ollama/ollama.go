// Package ollama implements the llm.Provider interface against a local Ollama
// daemon. It speaks the /api/chat endpoint directly over net/http so the
// harness needs no third-party client library.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/nmiano/playworld/internal/llm"
)

// DefaultEndpoint is where `ollama serve` listens by default.
const DefaultEndpoint = "http://127.0.0.1:11434"

// DefaultTimeout bounds a single generation. Local 4B-8B models on Apple
// Silicon answer a few hundred tokens well inside this.
const DefaultTimeout = 3 * time.Minute

// Provider talks to an Ollama daemon.
type Provider struct {
	endpoint string
	client   *http.Client

	mu     sync.Mutex
	models []string // cached result of /api/tags
}

// Option configures a Provider.
type Option func(*Provider)

// WithEndpoint overrides the daemon base URL.
func WithEndpoint(url string) Option {
	return func(p *Provider) {
		if url != "" {
			p.endpoint = strings.TrimRight(url, "/")
		}
	}
}

// WithHTTPClient supplies a custom client (used by tests against httptest).
func WithHTTPClient(c *http.Client) Option {
	return func(p *Provider) {
		if c != nil {
			p.client = c
		}
	}
}

// New builds a Provider.
func New(opts ...Option) *Provider {
	p := &Provider{
		endpoint: DefaultEndpoint,
		// No client-level timeout: per-request deadlines come from the context
		// so that a slow generation can be bounded per call.
		client: &http.Client{},
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *Provider) Name() string { return "ollama" }

func (p *Provider) base(cfg llm.ModelConfig) string {
	if cfg.Endpoint != "" {
		return strings.TrimRight(cfg.Endpoint, "/")
	}
	return p.endpoint
}

// chatRequest mirrors the subset of Ollama's /api/chat payload we use.
type chatRequest struct {
	Model    string          `json:"model"`
	Messages []chatMessage   `json:"messages"`
	Stream   bool            `json:"stream"`
	Format   json.RawMessage `json:"format,omitempty"`
	Options  map[string]any  `json:"options,omitempty"`
	Think    *bool           `json:"think,omitempty"`
	// KeepAlive keeps the model resident between agent turns, which is what
	// lets many agents share one loaded model cheaply.
	KeepAlive string `json:"keep_alive,omitempty"`
}

type chatMessage struct {
	Role     string `json:"role"`
	Content  string `json:"content"`
	Thinking string `json:"thinking,omitempty"`
}

type chatResponse struct {
	Model           string      `json:"model"`
	Message         chatMessage `json:"message"`
	Done            bool        `json:"done"`
	DoneReason      string      `json:"done_reason"`
	PromptEvalCount int         `json:"prompt_eval_count"`
	EvalCount       int         `json:"eval_count"`
	Error           string      `json:"error"`
}

// Chat performs a single non-streaming completion.
func (p *Provider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	if req.Model.Model == "" {
		return nil, fmt.Errorf("ollama: no model specified")
	}
	timeout := req.Model.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	msgs := make([]chatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, chatMessage{Role: string(m.Role), Content: m.Content})
	}

	body := chatRequest{
		Model:     req.Model.Model,
		Messages:  msgs,
		Stream:    false,
		Options:   options(req.Model.Sampling),
		KeepAlive: "10m",
	}
	// Qwen3 and friends default to thinking mode; the harness wants a compact
	// structured action, so disable it unless explicitly asked for.
	think := false
	if req.Model.Think != nil {
		think = *req.Model.Think
	}
	body.Think = &think

	switch req.Format {
	case llm.FormatJSON:
		if req.JSONSchema != nil {
			raw, err := json.Marshal(req.JSONSchema)
			if err != nil {
				return nil, fmt.Errorf("ollama: marshal json schema: %w", err)
			}
			body.Format = raw
		} else {
			body.Format = json.RawMessage(`"json"`)
		}
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ollama: marshal request: %w", err)
	}

	url := p.base(req.Model) + "/api/chat"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("ollama: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := p.client.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, fmt.Errorf("ollama: request to %s timed out after %s (model %q may be loading; try a smaller model or raise the timeout)", url, timeout, req.Model.Model)
		}
		if isConnRefused(err) {
			return nil, fmt.Errorf("%w: cannot reach Ollama at %s — start it with `ollama serve` (or open the Ollama app): %v", llm.ErrUnavailable, p.base(req.Model), err)
		}
		return nil, fmt.Errorf("ollama: request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("ollama: read response: %w", err)
	}
	latency := time.Since(start)

	if resp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(string(raw))
		var errResp struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(raw, &errResp) == nil && errResp.Error != "" {
			msg = errResp.Error
		}
		if resp.StatusCode == http.StatusNotFound || strings.Contains(strings.ToLower(msg), "not found") {
			return nil, fmt.Errorf("%w: %q is not installed — run `ollama pull %s` (%s)", llm.ErrModelMissing, req.Model.Model, req.Model.Model, msg)
		}
		return nil, fmt.Errorf("ollama: http %d: %s", resp.StatusCode, msg)
	}

	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("%w: %v (body: %.200s)", llm.ErrBadResponse, err, raw)
	}
	if out.Error != "" {
		return nil, fmt.Errorf("ollama: %s", out.Error)
	}

	content, inlineThinking := splitThinking(out.Message.Content)
	thinking := out.Message.Thinking
	if thinking == "" {
		thinking = inlineThinking
	}

	return &llm.Response{
		Content:    content,
		Thinking:   thinking,
		Usage:      llm.Usage{PromptTokens: out.PromptEvalCount, CompletionTokens: out.EvalCount},
		Latency:    latency,
		Model:      out.Model,
		StopReason: out.DoneReason,
	}, nil
}

// options translates sampling parameters into Ollama's option names.
func options(s llm.Sampling) map[string]any {
	o := map[string]any{}
	if s.Temperature != nil {
		o["temperature"] = *s.Temperature
	}
	if s.TopP != nil {
		o["top_p"] = *s.TopP
	}
	if s.TopK != nil {
		o["top_k"] = *s.TopK
	}
	if s.MaxTokens != nil {
		o["num_predict"] = *s.MaxTokens
	}
	if s.Seed != nil {
		o["seed"] = *s.Seed
	}
	if len(s.Stop) > 0 {
		o["stop"] = s.Stop
	}
	if s.ContextWindow != nil {
		o["num_ctx"] = *s.ContextWindow
	}
	if len(o) == 0 {
		return nil
	}
	return o
}

var thinkTag = regexp.MustCompile(`(?s)<think>(.*?)</think>\s*`)

// splitThinking removes any <think>…</think> block some models emit inline and
// returns it separately, so reasoning never contaminates the parsed action.
func splitThinking(s string) (content, thinking string) {
	m := thinkTag.FindStringSubmatch(s)
	if m == nil {
		return strings.TrimSpace(s), ""
	}
	return strings.TrimSpace(thinkTag.ReplaceAllString(s, "")), strings.TrimSpace(m[1])
}

// Preflight checks the daemon is up and the model is installed, returning an
// error the user can act on directly.
func (p *Provider) Preflight(ctx context.Context, cfg llm.ModelConfig) error {
	models, err := p.List(ctx, cfg)
	if err != nil {
		return err
	}
	if cfg.Model == "" {
		return nil
	}
	for _, m := range models {
		if m == cfg.Model || strings.TrimSuffix(m, ":latest") == cfg.Model {
			return nil
		}
	}
	hint := "none installed"
	if len(models) > 0 {
		hint = "installed: " + strings.Join(models, ", ")
	}
	return fmt.Errorf("%w: %q — run `ollama pull %s` (%s)", llm.ErrModelMissing, cfg.Model, cfg.Model, hint)
}

// List returns the installed model tags, caching the result for the process.
func (p *Provider) List(ctx context.Context, cfg llm.ModelConfig) ([]string, error) {
	p.mu.Lock()
	cached := p.models
	p.mu.Unlock()
	if cached != nil {
		return cached, nil
	}

	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	url := p.base(cfg) + "/api/tags"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: cannot reach Ollama at %s — install it (`brew install ollama`) and start it (`ollama serve`): %v", llm.ErrUnavailable, p.base(cfg), err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%w: Ollama at %s returned http %d from /api/tags", llm.ErrUnavailable, p.base(cfg), resp.StatusCode)
	}
	var out struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("%w: decoding /api/tags: %v", llm.ErrBadResponse, err)
	}
	names := make([]string, 0, len(out.Models))
	for _, m := range out.Models {
		names = append(names, m.Name)
	}
	sort.Strings(names)

	p.mu.Lock()
	p.models = names
	p.mu.Unlock()
	return names, nil
}

func isConnRefused(err error) bool {
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	return strings.Contains(err.Error(), "connection refused")
}
