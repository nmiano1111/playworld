// Package scenario holds the simulation environments and a small registry so
// the CLI can list and build them by name. Environments own all world state and
// are the only component allowed to decide what each agent may know.
package scenario

import (
	"fmt"
	"sort"
	"strings"

	"github.com/nmiano/playworld/internal/sim"
)

// Spec is everything an environment needs to be constructed.
type Spec struct {
	Participants []sim.AgentID
	// Names maps ids to display names, for readable narration.
	Names map[sim.AgentID]string
	// Rounds is the orchestrator's round budget, so an environment can lay out
	// its phases to fit.
	Rounds int
	// Seed makes any environment randomness reproducible.
	Seed int64
	// Params are scenario-specific settings from YAML or the CLI.
	Params map[string]any
}

// Factory builds an environment from a spec.
type Factory func(Spec) (sim.Environment, error)

// ParamDoc documents one tunable parameter for `playworld list`.
type ParamDoc struct {
	Name    string
	Default string
	Doc     string
}

// Info describes a registered scenario.
type Info struct {
	Name          string
	Title         string
	Description   string
	MinAgents     int
	MaxAgents     int
	DefaultRounds int
	Params        []ParamDoc

	factory Factory
}

var registry = map[string]Info{}

// Register adds a scenario. Called from package init functions.
func Register(info Info, f Factory) {
	info.factory = f
	registry[info.Name] = info
}

// List returns all registered scenarios, ordered by name.
func List() []Info {
	out := make([]Info, 0, len(registry))
	for _, i := range registry {
		out = append(out, i)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Lookup finds a scenario's metadata.
func Lookup(name string) (Info, error) {
	i, ok := registry[name]
	if !ok {
		return Info{}, fmt.Errorf("unknown scenario %q (available: %s)", name, strings.Join(Names(), ", "))
	}
	return i, nil
}

// Names lists registered scenario names.
func Names() []string {
	out := make([]string, 0, len(registry))
	for n := range registry {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Build constructs a scenario, validating the participant count first.
func Build(name string, spec Spec) (sim.Environment, error) {
	info, err := Lookup(name)
	if err != nil {
		return nil, err
	}
	n := len(spec.Participants)
	if info.MinAgents > 0 && n < info.MinAgents {
		return nil, fmt.Errorf("scenario %q needs at least %d agents, got %d", name, info.MinAgents, n)
	}
	if info.MaxAgents > 0 && n > info.MaxAgents {
		return nil, fmt.Errorf("scenario %q supports at most %d agents, got %d", name, info.MaxAgents, n)
	}
	return info.factory(spec)
}

// --- parameter helpers -------------------------------------------------------
// Params arrive from YAML (any) or CLI flags (strings), so these coerce
// leniently and report a clear error rather than silently defaulting.

func paramInt(p map[string]any, key string, def int) (int, error) {
	v, ok := p[key]
	if !ok || v == nil {
		return def, nil
	}
	switch n := v.(type) {
	case int:
		return n, nil
	case int64:
		return int(n), nil
	case float64:
		return int(n), nil
	case string:
		var i int
		if _, err := fmt.Sscanf(strings.TrimSpace(n), "%d", &i); err == nil {
			return i, nil
		}
	}
	return 0, fmt.Errorf("param %q: want an integer, got %v", key, v)
}

// paramBool reads a boolean-ish param, tolerating the string forms YAML and
// -set both produce ("true"/"false", "1"/"0").
func paramBool(p map[string]any, key string, def bool) (bool, error) {
	v, ok := p[key]
	if !ok || v == nil {
		return def, nil
	}
	switch b := v.(type) {
	case bool:
		return b, nil
	case string:
		switch strings.ToLower(strings.TrimSpace(b)) {
		case "true", "1", "yes", "on":
			return true, nil
		case "false", "0", "no", "off":
			return false, nil
		}
	}
	return false, fmt.Errorf("param %q: want a boolean, got %v", key, v)
}

func paramString(p map[string]any, key, def string) string {
	v, ok := p[key]
	if !ok || v == nil {
		return def
	}
	if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// paramIntMap reads a map of agent id to integer, used for per-agent settings
// such as private targets.
func paramIntMap(p map[string]any, key string) (map[sim.AgentID]int, error) {
	v, ok := p[key]
	if !ok || v == nil {
		return nil, nil
	}
	raw, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("param %q: want a mapping of agent id to integer", key)
	}
	out := map[sim.AgentID]int{}
	for k, val := range raw {
		n, err := paramInt(map[string]any{k: val}, k, 0)
		if err != nil {
			return nil, fmt.Errorf("param %q: %w", key, err)
		}
		out[sim.AgentID(k)] = n
	}
	return out, nil
}

// paramStringMap reads a map of agent id to string, used for per-agent
// overrides such as assigned clues.
func paramStringMap(p map[string]any, key string) (map[sim.AgentID]string, error) {
	v, ok := p[key]
	if !ok || v == nil {
		return nil, nil
	}
	raw, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("param %q: want a mapping of agent id to string", key)
	}
	out := map[sim.AgentID]string{}
	for k, val := range raw {
		s, ok := val.(string)
		if !ok {
			return nil, fmt.Errorf("param %q: value for %q must be a string", key, k)
		}
		out[sim.AgentID(k)] = s
	}
	return out, nil
}

// displayName renders an agent for narration: "Mira (mira)".
func displayName(names map[sim.AgentID]string, id sim.AgentID) string {
	if n := names[id]; n != "" && n != string(id) {
		return fmt.Sprintf("%s (%s)", n, id)
	}
	return string(id)
}
