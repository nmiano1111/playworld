// Package agent turns a persona plus a model into an autonomous participant.
//
// Configuration (who the agent is) is kept strictly separate from runtime state
// (what it has seen and decided). A Persona is immutable, declarative data
// loaded from YAML; an Agent is the live entity built from it.
package agent

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/sim"
)

// Persona is the declarative identity of an agent. Everything that shapes
// behaviour lives here rather than in the orchestrator, so personalities can be
// changed without touching Go code.
type Persona struct {
	ID   string `yaml:"id"`
	Name string `yaml:"name"`
	// Role is a short situational label, e.g. "a retired smuggler".
	Role string `yaml:"role,omitempty"`
	// Personality is free prose describing disposition and temperament.
	Personality string `yaml:"personality"`
	// SpeechStyle shapes how the agent talks.
	SpeechStyle string `yaml:"speech_style,omitempty"`
	// Goals are the agent's standing motivations. They are private: the
	// harness never reveals them to other agents.
	Goals []string `yaml:"goals,omitempty"`
	// Traits are arbitrary key/value dispositions rendered into the prompt,
	// e.g. {trust: "low", risk_appetite: "high"}.
	Traits map[string]string `yaml:"traits,omitempty"`
	// Model is this agent's own LLM configuration. Unset fields inherit from
	// the file's defaults, which is what lets five agents share one loaded
	// model while still being independently configurable.
	Model llm.ModelConfig `yaml:"model,omitempty"`
	// Context bounds how much history this agent carries into a prompt.
	Context ContextPolicy `yaml:"context,omitempty"`
}

// ContextPolicy bounds prompt size. This is the seam where summarisation or
// relevance-based retrieval would later plug in.
type ContextPolicy struct {
	// RecentObservations is how many observed lines of the current run are
	// included verbatim.
	RecentObservations int `yaml:"recent_observations,omitempty"`
	// RecallEpisodes is how many episodes from *previous* runs are recalled.
	RecallEpisodes int `yaml:"recall_episodes,omitempty"`
	// MaxEpisodes caps persisted episodic memory per agent.
	MaxEpisodes int `yaml:"max_episodes,omitempty"`
}

// DefaultContext is used when a config leaves the policy unset.
var DefaultContext = ContextPolicy{RecentObservations: 24, RecallEpisodes: 8, MaxEpisodes: 500}

func (c ContextPolicy) withDefaults(d ContextPolicy) ContextPolicy {
	if c.RecentObservations == 0 {
		c.RecentObservations = d.RecentObservations
	}
	if c.RecallEpisodes == 0 {
		c.RecallEpisodes = d.RecallEpisodes
	}
	if c.MaxEpisodes == 0 {
		c.MaxEpisodes = d.MaxEpisodes
	}
	return c
}

// Defaults are the file-level fallbacks applied to every persona.
type Defaults struct {
	Model   llm.ModelConfig `yaml:"model,omitempty"`
	Context ContextPolicy   `yaml:"context,omitempty"`
}

// Roster is the contents of an agents YAML file.
type Roster struct {
	Defaults Defaults  `yaml:"defaults"`
	Agents   []Persona `yaml:"agents"`
}

// LoadRoster reads and validates a roster file, applying defaults to each
// persona so callers receive fully-resolved configuration.
func LoadRoster(path string) (*Roster, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("agents config: %w", err)
	}
	var r Roster
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&r); err != nil {
		return nil, fmt.Errorf("agents config %s: %w", path, err)
	}
	if len(r.Agents) == 0 {
		return nil, fmt.Errorf("agents config %s: no agents defined", path)
	}

	baseCtx := r.Defaults.Context.withDefaults(DefaultContext)
	seen := map[string]bool{}
	for i := range r.Agents {
		p := &r.Agents[i]
		if p.ID == "" {
			return nil, fmt.Errorf("agents config %s: agent #%d has no id", path, i+1)
		}
		if seen[p.ID] {
			return nil, fmt.Errorf("agents config %s: duplicate agent id %q", path, p.ID)
		}
		seen[p.ID] = true
		if p.Name == "" {
			p.Name = strings.ToUpper(p.ID[:1]) + p.ID[1:]
		}
		if strings.TrimSpace(p.Personality) == "" {
			return nil, fmt.Errorf("agents config %s: agent %q has no personality", path, p.ID)
		}
		p.Model = p.Model.WithDefaults(r.Defaults.Model)
		p.Context = p.Context.withDefaults(baseCtx)
	}
	return &r, nil
}

// Select returns personas by id, preserving the requested order.
func (r *Roster) Select(ids []string) ([]Persona, error) {
	byID := map[string]Persona{}
	for _, p := range r.Agents {
		byID[p.ID] = p
	}
	out := make([]Persona, 0, len(ids))
	for _, id := range ids {
		p, ok := byID[id]
		if !ok {
			return nil, fmt.Errorf("no agent %q in roster (have: %s)", id, strings.Join(r.IDs(), ", "))
		}
		out = append(out, p)
	}
	return out, nil
}

// Take returns the first n personas, or all of them if n is out of range.
func (r *Roster) Take(n int) []Persona {
	if n <= 0 || n > len(r.Agents) {
		n = len(r.Agents)
	}
	return append([]Persona(nil), r.Agents[:n]...)
}

// IDs lists the roster's agent ids.
func (r *Roster) IDs() []string {
	out := make([]string, 0, len(r.Agents))
	for _, p := range r.Agents {
		out = append(out, p.ID)
	}
	return out
}

// AgentID returns the persona's typed identifier.
func (p Persona) AgentID() sim.AgentID { return sim.AgentID(p.ID) }

// identityBlock renders the persona into the stable part of the system prompt.
func (p Persona) identityBlock() string {
	var b strings.Builder
	fmt.Fprintf(&b, "You are %s", p.Name)
	if p.Role != "" {
		fmt.Fprintf(&b, ", %s", p.Role)
	}
	b.WriteString(".\n")
	fmt.Fprintf(&b, "\nPersonality: %s\n", strings.TrimSpace(p.Personality))
	if p.SpeechStyle != "" {
		fmt.Fprintf(&b, "Voice: %s\n", strings.TrimSpace(p.SpeechStyle))
	}
	if len(p.Traits) > 0 {
		keys := make([]string, 0, len(p.Traits))
		for k := range p.Traits {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s=%s", k, p.Traits[k]))
		}
		fmt.Fprintf(&b, "Dispositions: %s\n", strings.Join(parts, ", "))
	}
	if len(p.Goals) > 0 {
		b.WriteString("\nYour private goals (never state them outright unless you judge it useful; nobody else can see them):\n")
		for _, g := range p.Goals {
			fmt.Fprintf(&b, "  - %s\n", g)
		}
	}
	return b.String()
}
