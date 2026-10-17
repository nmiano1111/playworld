package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/llm/ollama"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/runner"
	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/world"
)

const (
	defaultWorldRoster = "configs/world-agents.yaml"
	defaultWorldConfig = "configs/world.yaml"
)

var defaultWorldRoot = filepath.Join(defaultStateDir, "worlds")

func cmdWorld(ctx context.Context, args []string) error {
	if len(args) == 0 {
		worldUsage()
		return nil
	}
	switch args[0] {
	case "new", "create":
		return cmdWorldNew(args[1:])
	case "run":
		return cmdWorldRun(ctx, args[1:], false)
	case "resume", "continue":
		return cmdWorldRun(ctx, args[1:], true)
	case "inspect", "show":
		return cmdWorldInspect(args[1:])
	case "list", "ls":
		return cmdWorldList(args[1:])
	case "delete", "rm":
		return cmdWorldDelete(args[1:])
	case "compare":
		return cmdWorldCompare(args[1:])
	case "log", "events":
		return cmdWorldLog(args[1:])
	case "help", "-h", "--help":
		worldUsage()
		return nil
	}
	worldUsage()
	return fmt.Errorf("unknown world command %q", args[0])
}

func worldUsage() {
	fmt.Print(`playworld world — a persistent, open-ended, event-driven town

usage:
  playworld world new [name] [flags]       create a fresh world (refuses to overwrite without -force)
  playworld world run [name] [flags]       run one session: continues the world, or creates it if it
                                           does not exist yet (-fresh replaces an existing one)
  playworld world resume [name] [flags]    like run, but only ever continues an existing world
  playworld world log [name] [flags]       chronological record from the session logs
  playworld world inspect <name> [flags]   show the saved world state (and optionally one agent's memory)
  playworld world list                     list saved worlds
  playworld world compare <a> <b> ...      metrics of several worlds side by side
  playworld world delete <name> -yes       delete a world, its agents' memory and its logs

The name defaults to "town"; -id <name> may be used instead of the positional name.

new flags (also accepted by run -fresh):
  -agents <path>    persona roster (default configs/world-agents.yaml)
  -config <path>    world settings YAML (default configs/world.yaml if present)
  -only <ids>       comma-separated residents; default: the config's "agents", else the first -n
  -n <int>          how many residents from the roster (default 4)
  -seed <int>       recorded with the world
  -cognition <mode> grounded (default for new worlds) or legacy (the original observation and
                    prompt, for before/after comparisons); on an existing world, a session-only override
  -force            replace an existing world of the same name, memory included

run flags:
  -fresh            create the world first (replacing one of the same name, memory included)
  -for <dur>        simulated time to advance, e.g. 1d, 36h, 2d12h (default 1d)
  -days <n>         shorthand for -for <n>d
  -max-calls <n>    cap on LLM calls this session (default 100)
  -max-activations <n>  cap on agent activations this session
  -max-events <n>   cap on events delivered this session
  -timeout <dur>    wall-clock limit (default 120m)
  -log <level>      quiet | normal | verbose | debug
  -model, -provider override every resident's model/provider for this session
  -agents <path>    roster (on a resumed world: reload personas from it instead of the world's own)
  -dual-actions, -plan, -beliefs   override the world's stored protocol options (e.g. -plan=false)
  -introspect       log every activation's agent view, prompt and reply
  -skip-preflight   do not check the backend first

log flags:
  -session <n>      only this session
  -agent <id>       only records involving this resident (plus public events)
  -introspect       include each activation's agent view, recalled memory, reply and what it chose
  -activation <n>   only activation #n of the session (with -session) and what followed it

inspect flags:
  -events <n>       recent events to show (default 30)
  -agent <id>       also show that resident's memory: plan, impressions, recent episodes
  -json             print the raw world file
  -prompt           with -agent: the exact prompt that resident would get if it decided now
                    (read-only; -cognition legacy|grounded previews under another mode)

compare flags (playworld world compare <world> <world> ...):
  objective per-world metrics side by side: calls, tokens, refusals by reason, repeats,
  intentions superseded, projects and transfers — for before/after experiments

examples:
  playworld world run -max-calls 80             # create "town" (first time) and run a day
  playworld world resume -max-calls 80          # continue it
  playworld world log -agent mira
  playworld world new town -n 4
  playworld world run town -for 2d
  playworld world inspect town -agent mira
  playworld world run town -for 3d          # continues from Day 3
`)
}

// splitName pulls a leading positional name off args (Go's flag package stops
// at the first non-flag), returning it and the remaining flags.
func splitName(args []string) (string, []string) {
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		return args[0], args[1:]
	}
	return "", args
}

func worldStore(root string) world.Store { return world.Store{Root: root} }

// createFlags are the flags that shape a new world, shared by `world new`
// and `world run -fresh`.
type createFlags struct {
	roster, config, only *string
	n                    *int
	seed                 *int64
	cognition            *string
}

func addCreateFlags(fs *flag.FlagSet) createFlags {
	return createFlags{
		roster:    fs.String("agents", defaultWorldRoster, "persona roster YAML"),
		config:    fs.String("config", "", "world settings YAML (default configs/world.yaml if present)"),
		only:      fs.String("only", "", "comma-separated resident ids"),
		n:         fs.Int("n", 4, "number of residents from the roster"),
		seed:      fs.Int64("seed", 1, "seed recorded with the world"),
		cognition: fs.String("cognition", "", `"grounded" or "legacy"; stored with a new world, a session-only override for an existing one`),
	}
}

// worldName resolves the world to act on: the positional name, else -id,
// else the default "town".
func worldName(positional string, fs *flag.FlagSet, id string) string {
	switch {
	case positional != "":
		return positional
	case id != "":
		return id
	case fs.NArg() > 0:
		return fs.Arg(0)
	}
	return defaultWorldName
}

const defaultWorldName = "town"

func cmdWorldNew(args []string) error {
	name, args := splitName(args)
	fs := flag.NewFlagSet("world new", flag.ContinueOnError)
	cf := addCreateFlags(fs)
	id := fs.String("id", "", "world name (alternative to the positional name)")
	force := fs.Bool("force", false, "replace an existing world")
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	name = worldName(name, fs, *id)
	store := worldStore(*root)
	if store.Exists(name) && !*force {
		return fmt.Errorf("world %q already exists in %s; resume it with `playworld world run %s`, or pass -force to start over", name, *root, name)
	}
	if err := createWorld(store, name, cf, fs); err != nil {
		return err
	}
	fmt.Printf("\nnext: playworld world run %s -for 1d\n", name)
	return nil
}

// createWorld builds a fresh world (replacing any existing one of the same
// name, memory included) and saves it.
func createWorld(store world.Store, name string, cf createFlags, fs *flag.FlagSet) error {
	if !world.ValidName(name) {
		return fmt.Errorf("give the world a name of letters, digits, - or _ (e.g. `playworld world new town`)")
	}
	settings := world.DefaultSettings()
	cfgPath := *cf.config
	if cfgPath == "" {
		if _, err := os.Stat(defaultWorldConfig); err == nil {
			cfgPath = defaultWorldConfig
		}
	}
	seed := *cf.seed
	var castFromConfig []string
	if cfgPath != "" {
		var err error
		if settings, err = world.LoadSettings(cfgPath); err != nil {
			return err
		}
		cast, cfgSeed, err := world.ConfigCast(cfgPath)
		if err != nil {
			return err
		}
		castFromConfig = cast
		if !flagSet(fs, "seed") && cfgSeed != 0 {
			seed = cfgSeed
		}
	}

	if *cf.cognition != "" {
		settings.Cognition = *cf.cognition
		if err := settings.Validate(); err != nil {
			return err
		}
	}
	roster, err := agent.LoadRoster(*cf.roster)
	if err != nil {
		return err
	}
	var personas []agent.Persona
	switch {
	case *cf.only != "":
		personas, err = roster.Select(splitCSV(*cf.only))
	case len(castFromConfig) > 0 && !flagSet(fs, "n"):
		personas, err = roster.Select(castFromConfig)
	default:
		if *cf.n > len(roster.Agents) {
			return fmt.Errorf("asked for %d residents but %s defines only %d", *cf.n, *cf.roster, len(roster.Agents))
		}
		personas = roster.Take(*cf.n)
	}
	if err != nil {
		return err
	}
	cast := make([]sim.AgentID, 0, len(personas))
	names := map[sim.AgentID]string{}
	for _, p := range personas {
		cast = append(cast, p.AgentID())
		names[p.AgentID()] = p.Name
	}

	w, err := world.New(name, seed, settings, cast, names)
	if err != nil {
		return err
	}
	w.RosterPath = *cf.roster
	if store.Exists(name) {
		if err := store.Delete(name); err != nil {
			return err
		}
	}
	// A fresh world means fresh minds: clear any memory left in its directory.
	mem, err := memory.NewFileStore(store.MemoryDir(name))
	if err != nil {
		return err
	}
	if err := mem.Reset(); err != nil {
		return err
	}
	if err := store.Save(w); err != nil {
		return err
	}
	fmt.Printf("created world %q in %s\n", name, store.Dir(name))
	fmt.Printf("residents  %s\n", strings.Join(idStrings(cast), ", "))
	fmt.Printf("clock      %s\n", world.Clock(w.Now))
	cog := w.Settings.Cognition
	if cog == "" {
		cog = world.CognitionLegacy
	}
	fmt.Printf("cognition  %s\n", cog)
	return nil
}

// cmdWorldRun runs one session. `run` creates the world first if it does not
// exist yet (and replaces it with -fresh); `resume` only ever continues an
// existing world, so a mistyped name is an error rather than a new world.
func cmdWorldRun(ctx context.Context, args []string, resume bool) error {
	name, args := splitName(args)
	fs := flag.NewFlagSet("world run", flag.ContinueOnError)
	cf := addCreateFlags(fs)
	id := fs.String("id", "", "world name (alternative to the positional name)")
	fresh := fs.Bool("fresh", false, "create the world first, replacing any existing one of that name")
	maxEvents := fs.Int("max-events", 0, "cap on events delivered this session")
	span := fs.String("for", "1d", "simulated time to advance")
	days := fs.Float64("days", 0, "simulated days to advance")
	maxCalls := fs.Int("max-calls", 0, "cap on LLM calls")
	maxActs := fs.Int("max-activations", 0, "cap on agent activations")
	timeout := fs.Duration("timeout", 120*time.Minute, "wall-clock limit")
	logLevel := fs.String("log", "normal", "quiet | normal | verbose | debug")
	model := fs.String("model", "", "override every resident's model")
	provider := fs.String("provider", "", "override every resident's provider")
	dual := fs.Bool("dual-actions", true, "communication and world action decided independently")
	plan := fs.Bool("plan", true, "persistent self-authored plans")
	beliefs := fs.Bool("beliefs", false, "basis/confidence on opinions")
	skipPre := fs.Bool("skip-preflight", false, "skip the backend readiness check")
	introspect := fs.Bool("introspect", false, "log each activation's agent view, prompt and reply (see `world log -introspect`)")
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	name = worldName(name, fs, *id)
	length, err := world.ParseSpan(*span)
	if err != nil {
		return err
	}
	if *days > 0 {
		length = time.Duration(*days * float64(24*time.Hour))
	}

	store := worldStore(*root)
	switch {
	case resume && *fresh:
		return fmt.Errorf("resume continues an existing world; use `playworld world run %s -fresh` to start over", name)
	case resume && !store.Exists(name):
		return noWorld(store, name)
	case *fresh || !store.Exists(name):
		if err := createWorld(store, name, cf, fs); err != nil {
			return err
		}
		fmt.Println()
		*cf.cognition = "" // stored with the world; not an override
	}
	w, err := store.Load(name)
	if err != nil {
		return err
	}
	// -agents on a resumed world reloads personas from another roster; by
	// default they come from the one the world was created with.
	path := w.RosterPath
	if flagSet(fs, "agents") {
		path = *cf.roster
	}
	roster, err := agent.LoadRoster(path)
	if err != nil {
		return err
	}
	personas, err := roster.Select(idStrings(w.Order))
	if err != nil {
		return fmt.Errorf("world %s: %w", name, err)
	}
	for i := range personas {
		if *model != "" {
			personas[i].Model.Model = *model
		}
		if *provider != "" {
			personas[i].Model.Provider = *provider
		}
	}

	level, err := explog.ParseVerbosity(*logLevel)
	if err != nil {
		return err
	}
	reg := llm.NewRegistry()
	reg.Register(ollama.New())
	reg.Register(llm.NewFake(nil))

	opt := func(name string, v *bool) *bool {
		if flagSet(fs, name) {
			return v
		}
		return nil
	}
	result, err := runner.ExecuteWorld(ctx, runner.WorldOptions{
		Store: store, Name: name, Personas: personas, Providers: reg, Span: length,
		Limits:     world.Limits{MaxLLMCalls: *maxCalls, MaxActivations: *maxActs, Timeout: *timeout, MaxEvents: *maxEvents},
		DualAction: opt("dual-actions", dual), Planning: opt("plan", plan), Beliefs: opt("beliefs", beliefs),
		Cognition: *cf.cognition, Introspect: *introspect,
		Sinks:      []explog.Sink{explog.NewConsole(os.Stdout, level)},
		VerboseLog: level >= explog.Debug, Harness: version, Preflight: !*skipPre,
	})
	if err != nil {
		return err
	}
	fmt.Printf("\nworld     %s saved at %s — resume with `playworld world run %s`\n",
		name, world.Clock(result.World.Now), name)
	return result.Run.Err
}

func cmdWorldInspect(args []string) error {
	name, args := splitName(args)
	fs := flag.NewFlagSet("world inspect", flag.ContinueOnError)
	id := fs.String("id", "", "world name (alternative to the positional name)")
	events := fs.Int("events", 30, "recent events to show")
	who := fs.String("agent", "", "also show this resident's memory")
	asJSON := fs.Bool("json", false, "print the raw world file")
	prompt := fs.Bool("prompt", false, "with -agent: print the exact prompt that resident would be sent if it decided now (read-only)")
	cognition := fs.String("cognition", "", "with -prompt: preview under this cognition mode instead of the world's own")
	roster := fs.String("agents", "", "with -prompt: roster to read personas from (default: the world's own)")
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	name = worldName(name, fs, *id)
	store := worldStore(*root)
	if !store.Exists(name) {
		return noWorld(store, name)
	}
	w, err := store.Load(name)
	if err != nil {
		return err
	}
	if *prompt {
		if *who == "" {
			return fmt.Errorf("-prompt needs -agent <id>")
		}
		path := w.RosterPath
		if *roster != "" {
			path = *roster
		}
		r, err := agent.LoadRoster(path)
		if err != nil {
			return err
		}
		personas, err := r.Select(idStrings(w.Order))
		if err != nil {
			return err
		}
		msgs, err := runner.PreviewWorld(store, name, personas, sim.AgentID(*who), *cognition)
		if err != nil {
			return err
		}
		for _, m := range msgs {
			fmt.Printf("════ %s (%d chars, ~%d tokens) ════\n%s\n\n", strings.ToUpper(string(m.Role)), len(m.Content), len(m.Content)/4, m.Content)
		}
		return nil
	}
	if *asJSON {
		raw, err := os.ReadFile(filepath.Join(store.Dir(name), "world.json"))
		if err != nil {
			return err
		}
		_, err = os.Stdout.Write(raw)
		return err
	}
	fmt.Print(w.Describe(*events))
	mem, err := memory.NewFileStore(store.MemoryDir(name))
	if err != nil {
		return err
	}
	fmt.Println("\nplans (self-authored, private to each resident)")
	for _, rid := range w.Order {
		st, err := mem.Load(string(rid))
		if err != nil {
			return err
		}
		p := st.ActivePlan(runner.WorldRunID(name))
		if p == nil {
			fmt.Printf("  %-8s (none)\n", rid)
			continue
		}
		fmt.Printf("  %-8s goal: %s  [revised %d times, %s]\n", rid, p.PrimaryGoal, p.Revisions, p.Updated.Format("15:04:05 wall"))
		for _, step := range p.NextSteps {
			fmt.Printf("           next: %s\n", step)
		}
		for _, c := range p.Commitments {
			fmt.Printf("           owes/owed: %s\n", c)
		}
		for _, o := range p.Obstacles {
			fmt.Printf("           obstacle: %s\n", o)
		}
	}
	if *who == "" {
		return nil
	}
	st, err := mem.Load(*who)
	if err != nil {
		return err
	}
	fmt.Printf("\n── memory of %s ──\n", *who)
	if p := st.ActivePlan(runner.WorldRunID(name)); p != nil {
		raw, _ := json.MarshalIndent(p, "  ", "  ")
		fmt.Printf("plan (%d revisions)\n  %s\n", p.Revisions, raw)
	} else {
		fmt.Println("plan      (none)")
	}
	if bs := st.SortedBeliefs(); len(bs) > 0 {
		fmt.Println("impressions")
		for _, b := range bs {
			fmt.Printf("  %-8s %s\n", b.Subject, b.Note)
		}
	}
	eps := st.Recent(*events)
	fmt.Printf("recent memory (%d of %d episodes)\n", len(eps), len(st.Episodes))
	for _, e := range eps {
		stamp := e.Stamp
		if stamp == "" {
			stamp = "-"
		}
		fmt.Printf("  %-12s %-11s %s\n", stamp, e.Kind, oneLine(e.Text, 140))
	}
	return nil
}

func cmdWorldList(args []string) error {
	fs := flag.NewFlagSet("world list", flag.ContinueOnError)
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	store := worldStore(*root)
	names, err := store.List()
	if err != nil {
		return err
	}
	if len(names) == 0 {
		fmt.Printf("no worlds in %s yet — create one with `playworld world new town`\n", *root)
		return nil
	}
	for _, n := range names {
		w, err := store.Load(n)
		if err != nil {
			fmt.Printf("  %-16s (unreadable: %v)\n", n, err)
			continue
		}
		fmt.Printf("  %-16s %-26s %d resident(s) · %d session(s)\n", n, world.Clock(w.Now), len(w.Order), len(w.Sessions))
	}
	return nil
}

func cmdWorldDelete(args []string) error {
	name, args := splitName(args)
	fs := flag.NewFlagSet("world delete", flag.ContinueOnError)
	yes := fs.Bool("yes", false, "confirm deletion")
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	store := worldStore(*root)
	if !store.Exists(name) {
		return fmt.Errorf("no world named %q in %s", name, *root)
	}
	if !*yes {
		return fmt.Errorf("this deletes %s and everything in it; rerun with -yes to confirm", store.Dir(name))
	}
	if err := store.Delete(name); err != nil {
		return err
	}
	fmt.Printf("deleted world %q\n", name)
	return nil
}

// cmdWorldLog prints a chronological record of one or all sessions from
// their JSONL logs: activations and why, events, refusals, scheduling and
// plan changes. It reads the logs, not the world file, so it shows exactly
// what was recorded at the time.
func cmdWorldLog(args []string) error {
	name, args := splitName(args)
	fs := flag.NewFlagSet("world log", flag.ContinueOnError)
	id := fs.String("id", "", "world name (alternative to the positional name)")
	session := fs.Int("session", 0, "only this session number (default: all)")
	who := fs.String("agent", "", "only records involving this resident")
	introspect := fs.Bool("introspect", false, "also print each activation's agent view and reply (sessions run with -introspect)")
	activation := fs.Int("activation", 0, "only this activation (its number in the session, as `wakes #N` shows) and what followed it")
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	name = worldName(name, fs, *id)
	store := worldStore(*root)
	if !store.Exists(name) {
		return noWorld(store, name)
	}
	w, err := store.Load(name)
	if err != nil {
		return err
	}
	showViews = *introspect
	onlyActivation = *activation
	if len(w.Sessions) == 0 {
		fmt.Printf("world %q has not run yet; start it with `playworld world run %s`\n", name, name)
	}
	for _, s := range w.Sessions {
		if *session != 0 && s.Number != *session {
			continue
		}
		fmt.Printf("── session %d · %s → %s · %s\n", s.Number, world.Stamp(s.SimFrom), world.Stamp(s.SimTo), s.StopReason)
		if err := printSessionLog(s.LogPath, sim.AgentID(*who)); err != nil {
			return err
		}
	}
	return nil
}

type promptSection struct {
	title string
	lines []string
}

// recalledSections pulls, from a logged prompt, the parts that come from the
// agent's memory rather than from the world: its own notes, its plan, its
// impressions, and the older events it was reminded of. The situation and
// the transcript are shown elsewhere.
func recalledSections(msgs []llm.Message) []promptSection {
	var user string
	for _, m := range msgs {
		if m.Role == llm.RoleUser {
			user = m.Content
			break
		}
	}
	keep := map[string]bool{
		"NOTES YOU WROTE TO YOURSELF": true, "YOUR PLAN": true, "WHAT YOU REMEMBER FROM BEFORE": true,
		"EARLIER, YOU SAW THESE HAPPEN (older than what follows)": true,
	}
	var out []promptSection
	var cur *promptSection
	for _, l := range strings.Split(user, "\n") {
		if strings.HasPrefix(l, "--- ") && strings.HasSuffix(l, " ---") {
			title := strings.TrimSuffix(strings.TrimPrefix(l, "--- "), " ---")
			cur = nil
			if keep[title] {
				out = append(out, promptSection{title: strings.ToLower(title)})
				cur = &out[len(out)-1]
			}
			continue
		}
		if cur != nil && strings.TrimSpace(l) != "" {
			cur.lines = append(cur.lines, strings.TrimSpace(l))
		}
	}
	return out
}

// showViews makes printSessionLog include introspection records;
// onlyActivation, when set, limits it to one activation and what followed.
var (
	showViews      bool
	onlyActivation int
)

func oneLineCLI(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func printSessionLog(path string, who sim.AgentID) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 4<<20), 4<<20)
	involves := func(ids ...sim.AgentID) bool {
		if who == "" {
			return true
		}
		for _, x := range ids {
			if x == who {
				return true
			}
		}
		return false
	}
	lastStamp := ""
	current := 0
	for sc.Scan() {
		var r explog.Record
		if json.Unmarshal(sc.Bytes(), &r) != nil {
			continue
		}
		if r.SimStamp != "" {
			lastStamp = r.SimStamp
		}
		if r.Activation != nil {
			current = r.Activation.Index
		}
		if onlyActivation > 0 && current != onlyActivation {
			continue
		}
		ts := "[" + lastStamp + "]"
		switch {
		case r.Activation != nil && involves(r.Activation.Agent):
			fmt.Printf("%s %s wakes #%d — %s\n", ts, r.Activation.Agent, r.Activation.Index, strings.Join(r.Activation.Reasons, "; "))
		case r.Event != nil:
			e := r.Event
			if !involves(append([]sim.AgentID{e.Actor}, append(e.Targets, e.Visibility.Audience...)...)...) && !e.Visibility.Public {
				continue
			}
			who := ""
			if e.Actor != "" {
				who = string(e.Actor) + " "
			}
			fmt.Printf("%s   %-8s %s%s  {%s}\n", ts, e.Type, who, e.Text, e.Visibility.Describe())
		case r.Rejected != nil && involves(r.Rejected.Agent):
			code := ""
			if r.Rejected.Code != "" {
				code = " [" + r.Rejected.Code + "]"
			}
			fmt.Printf("%s   refused  %s%s: %s\n", ts, r.Rejected.Agent, code, r.Rejected.Reason)
		case r.View != nil && showViews && involves(r.View.Agent):
			fmt.Printf("%s   ┌ what %s saw (agent view) ─────\n", ts, r.View.Agent)
			for _, l := range strings.Split(strings.TrimRight(r.View.Situation, "\n"), "\n") {
				fmt.Printf("%s   │ %s\n", ts, l)
			}
			fmt.Printf("%s   │ offered: %s\n", ts, strings.Join(r.View.Actions, ", "))
			fmt.Printf("%s   └ privileged (experimenter only, not shown to %s): %s\n", ts, r.View.Agent, r.View.Privileged)
		case r.Call != nil && showViews && involves(r.Call.Agent) && r.Call.Raw != "":
			if r.Call.Attempt == 0 {
				for _, sec := range recalledSections(r.Call.Prompt) {
					fmt.Printf("%s   ┌ %s recalled: %s\n", ts, r.Call.Agent, sec.title)
					for _, l := range sec.lines {
						fmt.Printf("%s   │ %s\n", ts, oneLineCLI(l, 220))
					}
				}
			}
			fmt.Printf("%s   reply    %s: %s\n", ts, r.Call.Agent, oneLineCLI(r.Call.Raw, 600))
		case r.Scheduled != nil && involves(r.Scheduled.Agent):
			switch {
			case r.Scheduled.Damped:
				fmt.Printf("%s   damped   %s not woken: %s\n", ts, r.Scheduled.Agent, r.Scheduled.Reason)
			case !r.Scheduled.Coalesced:
				fmt.Printf("%s   schedule %s at %s: %s\n", ts, r.Scheduled.Agent, r.Scheduled.Stamp, r.Scheduled.Reason)
			}
		case r.Plan != nil && involves(r.Plan.Agent):
			if r.Plan.Plan == nil {
				fmt.Printf("%s   plan     %s abandoned its plan\n", ts, r.Plan.Agent)
			} else {
				p := r.Plan.Plan
				held := ""
				if p.GoalStamp != "" && p.Turns > 1 {
					held = fmt.Sprintf(" [goal held since %s, %d decisions]", p.GoalStamp, p.Turns)
				}
				fmt.Printf("%s   plan     %s: %s%s (next: %s)\n", ts, r.Plan.Agent, p.PrimaryGoal, held, strings.Join(p.NextSteps, "; "))
				if showViews && len(p.Obstacles) > 0 {
					fmt.Printf("%s            obstacles: %s\n", ts, strings.Join(p.Obstacles, "; "))
				}
			}
		case r.Decision != nil && showViews && involves(r.Decision.Agent):
			d := r.Decision
			verdict := "accepted"
			if !d.Accepted {
				verdict = "refused"
			}
			raw, _ := json.Marshal(d.Action)
			fmt.Printf("%s   chose    %s [%s] %s → %s\n", ts, d.Agent, d.Slot, oneLineCLI(string(raw), 300), verdict)
		case r.Memory != nil && involves(r.Memory.Agent):
			fmt.Printf("%s   %-8s %s %s %s\n", ts, r.Memory.Kind, r.Memory.Agent, r.Memory.Subject, r.Memory.Text)
		}
	}
	return sc.Err()
}

// noWorld explains a missing world and lists the ones that do exist.
func noWorld(store world.Store, name string) error {
	have, _ := store.List()
	hint := "no worlds exist there yet"
	if len(have) > 0 {
		hint = "existing worlds: " + strings.Join(have, ", ")
	}
	return fmt.Errorf("no world named %q in %s (%s); create it with `playworld world run %s`", name, store.Root, hint, name)
}

func idStrings(ids []sim.AgentID) []string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		out = append(out, string(id))
	}
	return out
}
