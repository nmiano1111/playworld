package llm

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Fake is a deterministic in-process provider for tests and dry runs. It
// records every request it receives and answers from a scripted responder, so
// the whole harness can be exercised without Ollama running.
type Fake struct {
	// Respond produces the reply for a request. The call index starts at 0.
	// If nil, Fake returns a pass action.
	Respond func(call int, req Request) (*Response, error)
	// PreflightErr is returned by Preflight when non-nil.
	PreflightErr error
	// Latency is reported on each response.
	Latency time.Duration

	mu    sync.Mutex
	calls []Request
}

// NewFake builds a Fake with a responder.
func NewFake(respond func(call int, req Request) (*Response, error)) *Fake {
	return &Fake{Respond: respond}
}

// ScriptedFake replies with the given contents in order, repeating the last
// entry once exhausted. Useful for driving a scenario through known moves.
func ScriptedFake(contents ...string) *Fake {
	return NewFake(func(call int, _ Request) (*Response, error) {
		if len(contents) == 0 {
			return &Response{Content: `{"kind":"pass"}`}, nil
		}
		i := call
		if i >= len(contents) {
			i = len(contents) - 1
		}
		return &Response{Content: contents[i]}, nil
	})
}

func (f *Fake) Name() string { return "fake" }

func (f *Fake) Preflight(_ context.Context, _ ModelConfig) error { return f.PreflightErr }

func (f *Fake) Chat(ctx context.Context, req Request) (*Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.mu.Lock()
	call := len(f.calls)
	f.calls = append(f.calls, req)
	f.mu.Unlock()

	var resp *Response
	if f.Respond == nil {
		resp = &Response{Content: `{"kind":"pass"}`}
	} else {
		var err error
		if resp, err = f.Respond(call, req); err != nil {
			return nil, err
		}
	}
	if resp == nil {
		return nil, fmt.Errorf("%w: fake responder returned nil", ErrBadResponse)
	}
	if resp.Latency == 0 {
		resp.Latency = f.Latency
	}
	if resp.Model == "" {
		resp.Model = req.Model.Model
	}
	// Report plausible usage so metrics paths are exercised in tests.
	if resp.Usage.Total() == 0 {
		for _, m := range req.Messages {
			resp.Usage.PromptTokens += len(m.Content) / 4
		}
		resp.Usage.CompletionTokens = len(resp.Content) / 4
	}
	return resp, nil
}

// Calls returns a copy of the requests received so far.
func (f *Fake) Calls() []Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Request(nil), f.calls...)
}

// CallCount reports how many requests were received.
func (f *Fake) CallCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}
