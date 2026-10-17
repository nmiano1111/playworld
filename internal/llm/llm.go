// Package llm is the model-agnostic inference boundary. Nothing above this
// package knows which backend is in use; nothing in this package knows about
// agents, environments or scenarios.
package llm

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Role identifies the author of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn of a conversation handed to a provider.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

func System(content string) Message    { return Message{Role: RoleSystem, Content: content} }
func User(content string) Message      { return Message{Role: RoleUser, Content: content} }
func Assistant(content string) Message { return Message{Role: RoleAssistant, Content: content} }

// Sampling holds the generation parameters a provider should honour where it
// can. Pointer fields distinguish "unset" from "deliberately zero" so that
// per-agent config can override defaults without clobbering them.
type Sampling struct {
	Temperature *float64 `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	TopP        *float64 `yaml:"top_p,omitempty" json:"top_p,omitempty"`
	TopK        *int     `yaml:"top_k,omitempty" json:"top_k,omitempty"`
	// MaxTokens caps the response length (Ollama: num_predict).
	MaxTokens *int `yaml:"max_tokens,omitempty" json:"max_tokens,omitempty"`
	// Seed makes a backend's sampling reproducible where supported. LLM output
	// is still not guaranteed deterministic, but this removes one source of
	// variance between runs.
	Seed *int `yaml:"seed,omitempty" json:"seed,omitempty"`
	// Stop sequences, where the backend supports them.
	Stop []string `yaml:"stop,omitempty" json:"stop,omitempty"`
	// ContextWindow sets the backend's context length where it is
	// configurable (Ollama: num_ctx). Unset leaves the backend default, which
	// can be too small for long prompts and silently truncates them.
	ContextWindow *int `yaml:"num_ctx,omitempty" json:"num_ctx,omitempty"`
}

// ModelConfig is the per-agent description of which model to call and how.
// It is plain data so it can live in a YAML config file.
type ModelConfig struct {
	// Provider selects a registered provider ("ollama", "fake").
	Provider string `yaml:"provider" json:"provider"`
	// Model is the backend's model name, e.g. "qwen3:4b".
	Model string `yaml:"model" json:"model"`
	// Endpoint overrides the provider's default base URL.
	Endpoint string `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`
	// Timeout bounds a single request. Zero means the provider default.
	Timeout time.Duration `yaml:"timeout,omitempty" json:"timeout,omitempty"`
	// Sampling parameters.
	Sampling Sampling `yaml:",inline" json:"sampling,omitempty"`
	// Think controls reasoning-model "thinking" mode where the backend exposes
	// a switch (Qwen3 via Ollama does). Defaults to off: we want actions, not
	// long internal monologues we would have to pay for and then discard.
	Think *bool `yaml:"think,omitempty" json:"think,omitempty"`
}

// WithDefaults returns c with any unset field filled from d.
func (c ModelConfig) WithDefaults(d ModelConfig) ModelConfig {
	out := c
	if out.Provider == "" {
		out.Provider = d.Provider
	}
	if out.Model == "" {
		out.Model = d.Model
	}
	if out.Endpoint == "" {
		out.Endpoint = d.Endpoint
	}
	if out.Timeout == 0 {
		out.Timeout = d.Timeout
	}
	if out.Think == nil {
		out.Think = d.Think
	}
	s, ds := &out.Sampling, d.Sampling
	if s.Temperature == nil {
		s.Temperature = ds.Temperature
	}
	if s.TopP == nil {
		s.TopP = ds.TopP
	}
	if s.TopK == nil {
		s.TopK = ds.TopK
	}
	if s.MaxTokens == nil {
		s.MaxTokens = ds.MaxTokens
	}
	if s.Seed == nil {
		s.Seed = ds.Seed
	}
	if len(s.Stop) == 0 {
		s.Stop = ds.Stop
	}
	if s.ContextWindow == nil {
		s.ContextWindow = ds.ContextWindow
	}
	return out
}

// String renders the config compactly for logs.
func (c ModelConfig) String() string {
	return fmt.Sprintf("%s/%s", c.Provider, c.Model)
}

// ResponseFormat requests a particular output shape. Providers that cannot
// enforce it should ignore it; callers must still parse defensively.
type ResponseFormat string

const (
	// FormatText is free-form output.
	FormatText ResponseFormat = ""
	// FormatJSON asks the backend to emit a single JSON object.
	FormatJSON ResponseFormat = "json"
)

// Request is one chat completion call.
type Request struct {
	Model    ModelConfig
	Messages []Message
	// Format requests structured output where the provider supports it.
	Format ResponseFormat
	// JSONSchema optionally constrains FormatJSON output. Providers without
	// schema support fall back to plain JSON mode.
	JSONSchema map[string]any
}

// Usage reports token accounting, where the backend provides it. Zero values
// mean "not reported" rather than "zero tokens".
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

func (u Usage) Total() int { return u.PromptTokens + u.CompletionTokens }

// Response is a completed chat call plus the observability the harness records.
type Response struct {
	Content string `json:"content"`
	// Thinking holds any separately-reported reasoning text the backend chose
	// to expose. It is recorded as commentary only and never fed back as fact.
	Thinking string `json:"thinking,omitempty"`
	Usage    Usage  `json:"usage"`
	// Latency is the wall-clock duration of the request.
	Latency time.Duration `json:"latency"`
	// Model is the backend-reported model name, which may differ from the
	// requested one (aliases, tags).
	Model string `json:"model,omitempty"`
	// StopReason is the backend's termination reason where available.
	StopReason string `json:"stop_reason,omitempty"`
}

// Provider is the one interface a backend must satisfy.
type Provider interface {
	// Name identifies the provider in config and logs.
	Name() string
	// Chat performs a single completion.
	Chat(ctx context.Context, req Request) (*Response, error)
	// Preflight verifies the backend is reachable and the model is usable. It
	// should return an actionable error; callers may skip it.
	Preflight(ctx context.Context, model ModelConfig) error
}

// Error classification. The orchestrator needs to tell "this one call failed,
// carry on" from "the backend is unusable, stop".
var (
	// ErrUnavailable means the backend could not be reached at all.
	ErrUnavailable = errors.New("llm backend unavailable")
	// ErrModelMissing means the backend is up but the model is not installed.
	ErrModelMissing = errors.New("model not available")
	// ErrBadResponse means the backend replied with something unusable.
	ErrBadResponse = errors.New("malformed response from llm backend")
)

// Fatal reports whether an error means further calls are pointless.
func Fatal(err error) bool {
	return errors.Is(err, ErrUnavailable) || errors.Is(err, ErrModelMissing)
}

// Registry maps provider names to constructors so that agent configs can
// select a backend by string without the agent package importing it.
type Registry struct {
	providers map[string]Provider
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{providers: map[string]Provider{}}
}

// Register adds a provider under its own name, replacing any previous entry.
func (r *Registry) Register(p Provider) { r.providers[p.Name()] = p }

// Get looks up a provider by name.
func (r *Registry) Get(name string) (Provider, error) {
	p, ok := r.providers[name]
	if !ok {
		return nil, fmt.Errorf("no llm provider registered as %q", name)
	}
	return p, nil
}

// Names lists the registered providers.
func (r *Registry) Names() []string {
	out := make([]string, 0, len(r.providers))
	for n := range r.providers {
		out = append(out, n)
	}
	return out
}
