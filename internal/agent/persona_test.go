package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeRoster(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "agents.yaml")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadRosterAppliesDefaultsWithoutClobberingOverrides(t *testing.T) {
	path := writeRoster(t, `
defaults:
  model:
    provider: ollama
    model: qwen3:4b
    temperature: 0.8
    max_tokens: 400
  context:
    recent_observations: 10
agents:
  - id: mira
    personality: inherits everything
  - id: dov
    name: Dov
    personality: overrides the model
    model:
      model: qwen3:8b
      temperature: 0.2
    context:
      recent_observations: 40
`)
	roster, err := LoadRoster(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(roster.Agents) != 2 {
		t.Fatalf("agents = %d", len(roster.Agents))
	}

	mira, dov := roster.Agents[0], roster.Agents[1]
	if mira.Model.Model != "qwen3:4b" || mira.Model.Provider != "ollama" {
		t.Errorf("mira did not inherit the default model: %+v", mira.Model)
	}
	if mira.Name != "Mira" {
		t.Errorf("a missing name should be derived from the id, got %q", mira.Name)
	}
	if mira.Context.RecentObservations != 10 {
		t.Errorf("mira context = %+v", mira.Context)
	}
	// An agent's own model wins, but unset fields still inherit — this is what
	// lets one agent run a bigger model without restating the whole config.
	if dov.Model.Model != "qwen3:8b" {
		t.Errorf("dov model = %q", dov.Model.Model)
	}
	if dov.Model.Provider != "ollama" {
		t.Errorf("dov did not inherit the default provider: %q", dov.Model.Provider)
	}
	if dov.Model.Sampling.Temperature == nil || *dov.Model.Sampling.Temperature != 0.2 {
		t.Errorf("dov temperature not overridden: %+v", dov.Model.Sampling)
	}
	if dov.Model.Sampling.MaxTokens == nil || *dov.Model.Sampling.MaxTokens != 400 {
		t.Errorf("dov did not inherit max_tokens: %+v", dov.Model.Sampling)
	}
	if dov.Context.RecentObservations != 40 {
		t.Errorf("dov context = %+v", dov.Context)
	}
}

func TestLoadRosterRejectsBadConfig(t *testing.T) {
	cases := map[string]string{
		"no agents":      "defaults:\n  model:\n    provider: ollama\n",
		"missing id":     "agents:\n  - personality: nameless\n",
		"duplicate id":   "agents:\n  - id: a\n    personality: one\n  - id: a\n    personality: two\n",
		"no personality": "agents:\n  - id: a\n",
		"unknown field":  "agents:\n  - id: a\n    personality: x\n    favourite_colour: blue\n",
		"malformed yaml": "agents: [{",
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadRoster(writeRoster(t, body)); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

func TestLoadRosterMissingFileMentionsPath(t *testing.T) {
	_, err := LoadRoster("/definitely/not/here.yaml")
	if err == nil || !strings.Contains(err.Error(), "not/here.yaml") {
		t.Errorf("error should name the missing file, got %v", err)
	}
}

func TestSelectAndTake(t *testing.T) {
	path := writeRoster(t, `
agents:
  - id: a
    personality: x
  - id: b
    personality: y
  - id: c
    personality: z
`)
	roster, err := LoadRoster(path)
	if err != nil {
		t.Fatal(err)
	}

	// Select preserves the requested order, not the file order.
	got, err := roster.Select([]string{"c", "a"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].ID != "c" || got[1].ID != "a" {
		t.Errorf("Select = %v", ids(got))
	}

	if _, err := roster.Select([]string{"nobody"}); err == nil {
		t.Error("selecting an unknown id should fail")
	} else if !strings.Contains(err.Error(), "a, b, c") {
		t.Errorf("error should list the available ids, got %v", err)
	}

	if n := len(roster.Take(2)); n != 2 {
		t.Errorf("Take(2) = %d", n)
	}
	if n := len(roster.Take(99)); n != 3 {
		t.Errorf("Take(99) = %d, want all 3", n)
	}
	if n := len(roster.Take(0)); n != 3 {
		t.Errorf("Take(0) = %d, want all 3", n)
	}
}

func TestIdentityBlockIncludesPersonalityAndGoals(t *testing.T) {
	p := Persona{
		ID: "rook", Name: "Rook", Role: "a trader",
		Personality: "competitive", SpeechStyle: "fast",
		Goals:  []string{"win"},
		Traits: map[string]string{"candour": "low", "warmth": "medium"},
	}
	got := p.identityBlock()
	for _, want := range []string{"You are Rook, a trader", "competitive", "fast", "win", "candour=low, warmth=medium"} {
		if !strings.Contains(got, want) {
			t.Errorf("identity block missing %q:\n%s", want, got)
		}
	}
	// Goals must be framed as private, or an agent will recite them aloud.
	if !strings.Contains(got, "private goals") {
		t.Error("goals are not marked private")
	}
}

func ids(ps []Persona) []string {
	out := make([]string, 0, len(ps))
	for _, p := range ps {
		out = append(out, p.ID)
	}
	return out
}
