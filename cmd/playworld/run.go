package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/llm/ollama"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/orchestrator"
	"github.com/nmiano/playworld/internal/runner"
	"github.com/nmiano/playworld/internal/scenario"
)

// defaults for paths, chosen so a fresh clone works with no setup.
const (
	defaultRosterPath = "configs/agents.yaml"
	defaultStateDir   = ".playworld"
)

// runFlags are the knobs `playworld run` exposes.
type runFlags struct {
	roster     string
	n          int
	only       string
	model      string
	provider   string
	rounds     int
	seed       int64
	logLevel   string
	params     paramList
	config     string
	memoryDir  string
	logsDir    string
	fresh      bool
	noMemory   bool
	timeout    time.Duration
	maxCalls   int
	shuffle    bool
	dualAction bool
	planning   bool
	beliefs    bool
	skipPre    bool
}

// paramList collects repeatable -set k=v flags.
type paramList []string

func (p *paramList) String() string { return strings.Join(*p, ",") }
func (p *paramList) Set(v string) error {
	if !strings.Contains(v, "=") {
		return fmt.Errorf("want key=value, got %q", v)
	}
	*p = append(*p, v)
	return nil
}

// scenarioConfig is the optional per-scenario YAML file.
type scenarioConfig struct {
	Scenario string         `yaml:"scenario,omitempty"`
	Rounds   int            `yaml:"rounds,omitempty"`
	Seed     int64          `yaml:"seed,omitempty"`
	Agents   []string       `yaml:"agents,omitempty"`
	Params   map[string]any `yaml:"params,omitempty"`
}

func cmdRun(ctx context.Context, args []string) error {
	// The scenario name is positional and comes first. Go's flag package stops
	// parsing at the first non-flag argument, so pull the name off before
	// handing the rest to the FlagSet.
	var name string
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		name, args = args[0], args[1:]
	}

	var f runFlags
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.StringVar(&f.roster, "agents", defaultRosterPath, "agent roster YAML")
	fs.IntVar(&f.n, "n", 0, "number of agents from the roster")
	fs.StringVar(&f.only, "only", "", "comma-separated agent ids to use")
	fs.StringVar(&f.model, "model", "", "override every agent's model")
	fs.StringVar(&f.provider, "provider", "", "override every agent's provider")
	fs.IntVar(&f.rounds, "rounds", 0, "maximum simulation rounds")
	fs.Int64Var(&f.seed, "seed", 1, "random seed")
	fs.StringVar(&f.logLevel, "log", "normal", "quiet | normal | verbose | debug")
	fs.Var(&f.params, "set", "scenario parameter key=value (repeatable)")
	fs.StringVar(&f.config, "config", "", "scenario settings YAML")
	fs.StringVar(&f.memoryDir, "memory", "", "agent memory directory")
	fs.StringVar(&f.logsDir, "logs", "", "experiment log directory")
	fs.BoolVar(&f.fresh, "fresh", false, "ignore persisted memory for this run")
	fs.BoolVar(&f.noMemory, "no-memory", false, "neither read nor write persisted memory")
	fs.DurationVar(&f.timeout, "timeout", 60*time.Minute, "wall-clock limit for the run")
	fs.IntVar(&f.maxCalls, "max-calls", 0, "cap on total LLM calls (0 = scenario default)")
	fs.BoolVar(&f.shuffle, "shuffle", false, "randomise turn order each round")
	fs.BoolVar(&f.dualAction, "dual-actions", false, "let every agent independently choose a communication and a world action in one decision")
	fs.BoolVar(&f.planning, "plan", false, "let every agent maintain a private, persistent, self-authored plan between turns")
	fs.BoolVar(&f.beliefs, "beliefs", false, "let every agent attach a basis and confidence to its opinions of others")
	fs.BoolVar(&f.skipPre, "skip-preflight", false, "skip the backend readiness check")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if name == "" {
		if fs.NArg() < 1 {
			return fmt.Errorf("which scenario? available: %s", strings.Join(scenario.Names(), ", "))
		}
		name = fs.Arg(0)
	}

	if name == "world" {
		return fmt.Errorf("world is an open-ended, persistent mode rather than a round-based scenario: use `playworld world new <name>` then `playworld world run <name>`")
	}
	info, err := scenario.Lookup(name)
	if err != nil {
		return err
	}

	// Scenario YAML first, then flags, so an explicit flag always wins.
	cfg := scenarioConfig{Params: map[string]any{}}
	if f.config != "" {
		if cfg, err = loadScenarioConfig(f.config); err != nil {
			return err
		}
	}
	if cfg.Params == nil {
		cfg.Params = map[string]any{}
	}
	for _, kv := range f.params {
		k, v, _ := strings.Cut(kv, "=")
		cfg.Params[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	rounds := firstPositive(f.rounds, cfg.Rounds, info.DefaultRounds)
	seed := f.seed
	if !flagSet(fs, "seed") && cfg.Seed != 0 {
		seed = cfg.Seed
	}

	roster, err := agent.LoadRoster(f.roster)
	if err != nil {
		return err
	}

	// Choose the cast: explicit ids, then the scenario file, then the first n.
	var personas []agent.Persona
	switch {
	case f.only != "":
		ids := splitCSV(f.only)
		if personas, err = roster.Select(ids); err != nil {
			return err
		}
	case len(cfg.Agents) > 0:
		if personas, err = roster.Select(cfg.Agents); err != nil {
			return err
		}
	default:
		n := f.n
		if n <= 0 {
			n = max(info.MinAgents, 3)
		}
		if n > len(roster.Agents) {
			return fmt.Errorf("asked for %d agents but %s defines only %d", n, f.roster, len(roster.Agents))
		}
		personas = roster.Take(n)
	}
	if f.model != "" || f.provider != "" {
		for i := range personas {
			if f.model != "" {
				personas[i].Model.Model = f.model
			}
			if f.provider != "" {
				personas[i].Model.Provider = f.provider
			}
		}
	}

	stateDir := defaultStateDir
	memDir := f.memoryDir
	if memDir == "" {
		memDir = filepath.Join(stateDir, "memory")
	}
	logsDir := f.logsDir
	if logsDir == "" {
		logsDir = filepath.Join(stateDir, "logs")
	}

	level, err := explog.ParseVerbosity(f.logLevel)
	if err != nil {
		return err
	}

	// Providers. Both are cheap to construct; a run that uses only the fake
	// provider never touches Ollama.
	reg := llm.NewRegistry()
	reg.Register(ollama.New())
	reg.Register(llm.NewFake(nil))

	// Memory.
	var store memory.Store
	if !f.noMemory {
		fileStore, err := memory.NewFileStore(memDir)
		if err != nil {
			return err
		}
		if f.fresh {
			ids := make([]string, 0, len(personas))
			for _, p := range personas {
				ids = append(ids, p.ID)
			}
			if err := fileStore.Reset(ids...); err != nil {
				return err
			}
		}
		store = fileStore
	}

	runID := newRunID(name)
	jsonlPath := filepath.Join(logsDir, runID+".jsonl")
	jsonl, err := explog.NewJSONL(jsonlPath, level >= explog.Debug)
	if err != nil {
		return err
	}
	defer jsonl.Close()

	limits := orchestrator.DefaultLimits()
	limits.Timeout = f.timeout
	limits.MaxLLMCalls = f.maxCalls

	result, err := runner.Execute(ctx, runner.Options{
		RunID:            runID,
		Scenario:         name,
		Personas:         personas,
		Providers:        reg,
		Store:            store,
		Rounds:           rounds,
		Seed:             seed,
		Params:           cfg.Params,
		Limits:           limits,
		Shuffle:          f.shuffle,
		DualAction:       f.dualAction,
		Planning:         f.planning,
		BeliefProvenance: f.beliefs,
		Sinks:            []explog.Sink{explog.NewConsole(os.Stdout, level), jsonl},
		LogPath:          jsonlPath,
		MemoryDir:        memDir,
		Harness:          version,
		Preflight:        !f.skipPre,
	})
	if err != nil {
		return err
	}

	if err := jsonl.Close(); err != nil {
		return err
	}
	// A run cut short by a dead backend or a repeatedly failing agent still
	// produced a log and a summary; surface the cause as the exit status.
	return result.Run.Err
}

func loadScenarioConfig(path string) (scenarioConfig, error) {
	var cfg scenarioConfig
	raw, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("scenario config: %w", err)
	}
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return cfg, fmt.Errorf("scenario config %s: %w", path, err)
	}
	return cfg, nil
}

func newRunID(scenarioName string) string {
	return fmt.Sprintf("%s-%s", scenarioName, time.Now().Format("20060102-150405"))
}

func flagSet(fs *flag.FlagSet, name string) bool {
	found := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			found = true
		}
	})
	return found
}

func splitCSV(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func firstPositive(vals ...int) int {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 1
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func oneLine(s string, n int) string {
	return truncate(strings.Join(strings.Fields(s), " "), n)
}
