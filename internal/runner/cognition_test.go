package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/world"
)

func readViews(t *testing.T, path string) []explog.View {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []explog.View
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var rec explog.Record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.View != nil {
			out = append(out, *rec.View)
		}
	}
	return out
}

// TestGroundedSessionsIntrospectAndResume runs a grounded world for two
// sessions with the fake model: the agent's view is logged separately from
// the experimenter's privileged snapshot, a speech-in-the-wrong-slot reply is
// repaired and counted, token usage lands in the session metrics, and a
// session-only legacy override is never written back.
func TestGroundedSessionsIntrospectAndResume(t *testing.T) {
	store, personas := newWorldFixture(t)
	scripts := map[string][]string{
		"mira": {`{"communication":null,"world_action":{"type":"say","message":"Anyone around?"}}`},
		"rook": {`{"communication":null,"world_action":{"type":"go","to":"square"}}`},
	}
	res, err := ExecuteWorld(context.Background(), WorldOptions{
		Store: store, Name: "town", Personas: personas, Providers: registry(newScriptedProvider(scripts)), Span: 2 * time.Hour,
		Limits: world.Limits{Timeout: 30 * time.Second}, Introspect: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	m := res.Session.Metrics
	// The scripted model repeats its last reply, so mira keeps trying.
	if m["slot_repairs"] == 0 || m["slot_repairs"] != res.Run.PerAgent["mira"]["slot_repairs"] {
		t.Errorf("mira's say in the world slot should be repaired and counted: %v", m)
	}
	if m["rejected_nobody_here"] == 0 {
		t.Errorf("…and then refused by the world, since she is alone: %v", m)
	}
	if m["prompt_tokens"] == 0 || m["sim_minutes"] <= 0 {
		t.Errorf("token and time metrics: %v", m)
	}
	views := readViews(t, res.Session.LogPath)
	if len(views) == 0 {
		t.Fatal("introspection should log views")
	}
	for _, v := range views {
		if !strings.Contains(v.Situation, "YOU (known only to you)") {
			t.Errorf("view is not the agent's grounded situation: %q", v.Situation)
		}
		if strings.Contains(v.Situation, "rook@") || !strings.Contains(v.Privileged, "rook@") {
			t.Error("the privileged snapshot must be kept apart from what the agent saw")
		}
	}

	// Second session under a legacy override: it runs legacy, but the world
	// on disk stays grounded.
	res2, err := ExecuteWorld(context.Background(), WorldOptions{
		Store: store, Name: "town", Personas: personas, Providers: registry(newScriptedProvider(nil)), Span: time.Hour,
		Limits: world.Limits{Timeout: 30 * time.Second}, Introspect: true, Cognition: world.CognitionLegacy,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range readViews(t, res2.Session.LogPath) {
		if strings.Contains(v.Situation, "YOU (known only to you)") {
			t.Error("the override should run the legacy observation")
		}
	}
	back, err := store.Load("town")
	if err != nil {
		t.Fatal(err)
	}
	if !back.Settings.Grounded() || len(back.Sessions) != 2 || !back.Now.Equal(res2.Run.To) {
		t.Errorf("after two sessions: grounded=%v sessions=%d now=%s", back.Settings.Grounded(), len(back.Sessions), world.Stamp(back.Now))
	}
}

// TestPreviewWorldIsReadOnly checks that previewing a prompt changes
// nothing on disk.
func TestPreviewWorldIsReadOnly(t *testing.T) {
	store, personas := newWorldFixture(t)
	if _, err := ExecuteWorld(context.Background(), WorldOptions{Store: store, Name: "town", Personas: personas,
		Providers: registry(newScriptedProvider(nil)), Span: time.Hour}); err != nil {
		t.Fatal(err)
	}
	snapshot := func() map[string]string {
		out := map[string]string{}
		for _, p := range []string{store.Dir("town") + "/world.json", store.MemoryDir("town") + "/mira.json"} {
			raw, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			out[p] = string(raw)
		}
		return out
	}
	before := snapshot()
	for _, mode := range []string{"", world.CognitionLegacy} {
		msgs, err := PreviewWorld(store, "town", personas, sim.AgentID("mira"), mode)
		if err != nil {
			t.Fatal(err)
		}
		if len(msgs) != 2 || !strings.Contains(msgs[1].Content, "--- CURRENT SITUATION ---") {
			t.Errorf("preview (%q) = %+v", mode, msgs)
		}
		if mode == "" && !strings.Contains(msgs[1].Content, "YOU (known only to you)") {
			t.Error("the default preview uses the world's own (grounded) mode")
		}
	}
	after := snapshot()
	for k, v := range before {
		if after[k] != v {
			t.Errorf("%s changed during a preview", k)
		}
	}
}
