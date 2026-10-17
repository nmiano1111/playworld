package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/world"
)

// recorder wraps a provider and keeps every request, per agent.
type recorder struct {
	*scriptedProvider
	mu   sync.Mutex
	reqs map[string][]llm.Request
}

func (r *recorder) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	r.mu.Lock()
	r.reqs[r.identify(req)] = append(r.reqs[r.identify(req)], req)
	r.mu.Unlock()
	return r.scriptedProvider.Chat(ctx, req)
}

func newWorldFixture(t *testing.T) (world.Store, []agent.Persona) {
	t.Helper()
	store := world.Store{Root: t.TempDir()}
	personas := []agent.Persona{persona("mira", "Mira"), persona("rook", "Rook")}
	w, err := world.New("town", 1, world.DefaultSettings(), []sim.AgentID{"mira", "rook"},
		map[sim.AgentID]string{"mira": "Mira", "rook": "Rook"})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Save(w); err != nil {
		t.Fatal(err)
	}
	return store, personas
}

func readActivations(t *testing.T, path string) []explog.Activation {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []explog.Activation
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var rec explog.Record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Activation != nil {
			out = append(out, *rec.Activation)
		}
	}
	return out
}

// TestWorldSessionsResumeTheSameWorld runs two sessions over one world in
// what are, as far as the code is concerned, two separate processes: nothing
// survives between them except what is on disk.
func TestWorldSessionsResumeTheSameWorld(t *testing.T) {
	store, personas := newWorldFixture(t)
	scripts := map[string][]string{
		"mira": {
			`{"communication":null,"world_action":{"type":"go","to":"market"},
			  "plan":{"primary_goal":"open a stall"},
			  "intentions":[{"when":"day 2 09:00","note":"approach Rook about financing"}]}`,
			`{"communication":{"type":"say","message":"Rook, got a minute?"},"world_action":null}`,
			`{"communication":null,"world_action":null}`,
		},
		"rook": {
			`{"communication":null,"world_action":{"type":"go","args":{"to":"market"}}}`,
			`{"communication":null,"world_action":null}`,
		},
	}

	run := func(span time.Duration) (*WorldResult, *recorder) {
		rec := &recorder{scriptedProvider: newScriptedProvider(scripts), reqs: map[string][]llm.Request{}}
		res, err := ExecuteWorld(context.Background(), WorldOptions{
			Store: store, Name: "town", Personas: personas, Providers: registry(rec), Span: span,
			Limits: world.Limits{Timeout: 30 * time.Second},
		})
		if err != nil {
			t.Fatal(err)
		}
		return res, rec
	}

	first, _ := run(24 * time.Hour)
	if got := world.Stamp(first.World.Now); got != "Day 2 07:00" {
		t.Fatalf("after session 1 the clock is %s", got)
	}
	saved, err := store.Load("town")
	if err != nil {
		t.Fatal(err)
	}
	if saved.Residents["mira"].Location != "market" || len(saved.Sessions) != 1 {
		t.Fatalf("session 1 was not saved: mira at %q, %d sessions", saved.Residents["mira"].Location, len(saved.Sessions))
	}
	if len(saved.Intentions) != 1 || saved.Intentions[0].Status != "pending" {
		t.Fatalf("mira's intention should be pending across the save: %+v", saved.Intentions)
	}

	second, rec := run(24 * time.Hour)
	if got := world.Stamp(second.World.Now); got != "Day 3 07:00" {
		t.Fatalf("session 2 should continue from Day 2: clock %s", got)
	}
	if second.Session.Number != 2 {
		t.Errorf("session number = %d", second.Session.Number)
	}
	fired := false
	for _, a := range readActivations(t, second.Session.LogPath) {
		if a.Agent == "mira" && a.Stamp == "Day 2 09:00" && strings.Contains(strings.Join(a.Reasons, ";"), "approach Rook about financing") {
			fired = true
		}
	}
	if !fired {
		t.Error("the wakeup mira scheduled in session 1 did not fire in session 2")
	}

	// Mira's first prompt in session 2 still carries her own past: Day 1's
	// events, and the plan she wrote then.
	reqs := rec.reqs["mira"]
	if len(reqs) == 0 {
		t.Fatal("mira was never activated in session 2")
	}
	prompt := reqs[0].Messages[len(reqs[0].Messages)-1].Content
	if !strings.Contains(prompt, "[Day 1 07:00] ✔ You set off for the market.") {
		t.Errorf("resumed prompt lost the transcript:\n%s", prompt)
	}
	if !strings.Contains(prompt, "Primary goal: open a stall") {
		t.Errorf("resumed prompt lost the plan:\n%s", prompt)
	}
}

func TestWorldSessionRefusesAMismatchedCast(t *testing.T) {
	store, _ := newWorldFixture(t)
	_, err := ExecuteWorld(context.Background(), WorldOptions{
		Store: store, Name: "town", Personas: []agent.Persona{persona("mira", "Mira")},
		Providers: registry(newScriptedProvider(nil)), Span: time.Hour,
	})
	if err == nil || !strings.Contains(err.Error(), "rook") {
		t.Fatalf("want an error naming the missing resident, got %v", err)
	}
}

func TestWorldMemoryIsIsolatedBetweenWorldsAndAgents(t *testing.T) {
	store := world.Store{Root: t.TempDir()}
	for _, name := range []string{"alpha", "beta"} {
		w, err := world.New(name, 1, world.DefaultSettings(), []sim.AgentID{"mira", "rook", "ada"},
			map[sim.AgentID]string{"mira": "Mira", "rook": "Rook", "ada": "Ada"})
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Save(w); err != nil {
			t.Fatal(err)
		}
	}
	scripts := map[string][]string{
		"rook": {`{"communication":{"type":"whisper","recipient":"mira","message":"the secret is in alpha"},"world_action":null}`, `{"kind":"pass"}`},
	}
	personas := []agent.Persona{persona("mira", "Mira"), persona("rook", "Rook"), persona("ada", "Ada")}
	if _, err := ExecuteWorld(context.Background(), WorldOptions{Store: store, Name: "alpha", Personas: personas,
		Providers: registry(newScriptedProvider(scripts)), Span: 2 * time.Hour}); err != nil {
		t.Fatal(err)
	}
	if _, err := ExecuteWorld(context.Background(), WorldOptions{Store: store, Name: "beta", Personas: personas,
		Providers: registry(newScriptedProvider(nil)), Span: 2 * time.Hour}); err != nil {
		t.Fatal(err)
	}
	read := func(world, id string) string {
		raw, err := os.ReadFile(store.MemoryDir(world) + "/" + id + ".json")
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	if !strings.Contains(read("alpha", "mira"), "the secret is in alpha") {
		t.Fatal("mira should remember the note in alpha")
	}
	if strings.Contains(read("beta", "mira"), "the secret") {
		t.Error("memory from world alpha leaked into world beta")
	}
	if strings.Contains(read("alpha", "ada"), "the secret") {
		t.Error("ada's memory contains a private note she was never shown")
	}
}

func TestWorldRunsOnTheBuiltInFakeProviderAndLogsPlans(t *testing.T) {
	store, personas := newWorldFixture(t)
	for i := range personas {
		personas[i].Model.Provider = "fake"
	}
	fake := llm.ScriptedFake(`{"communication":null,"world_action":{"type":"go","to":"square"},"plan":{"primary_goal":"see who is around","next_steps":["go to the square"]},"remember":"started the day"}`,
		`{"communication":null,"world_action":null}`)
	res, err := ExecuteWorld(context.Background(), WorldOptions{Store: store, Name: "town", Personas: personas,
		Providers: registry(fake), Span: 6 * time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	if res.Run.LLMCalls == 0 || res.Summary.Prompt == 0 {
		t.Fatalf("calls=%d prompt tokens=%d", res.Run.LLMCalls, res.Summary.Prompt)
	}
	raw, err := os.ReadFile(res.Session.LogPath)
	if err != nil {
		t.Fatal(err)
	}
	log := string(raw)
	for _, want := range []string{`"type":"plan"`, `"primary_goal":"see who is around"`, `"type":"memory"`, `"type":"scheduled"`, `"sim_stamp":"Day 1 07:00"`} {
		if !strings.Contains(log, want) {
			t.Errorf("session log lacks %s", want)
		}
	}
	plans := 0
	for _, a := range res.Summary.PerAgent {
		plans += a.PlanUpdates
	}
	if plans == 0 {
		t.Error("summary should count plan updates")
	}
}
