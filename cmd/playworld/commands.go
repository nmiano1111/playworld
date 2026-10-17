package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/agent"
	"github.com/nmiano/playworld/internal/llm"
	"github.com/nmiano/playworld/internal/llm/ollama"
	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/scenario"
)

// cmdList prints the registered scenarios and, if a roster is readable, the
// configured agents.
func cmdList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	rosterPath := fs.String("agents", defaultRosterPath, "agent roster YAML")
	if err := fs.Parse(args); err != nil {
		return err
	}

	fmt.Println("scenarios:")
	for _, info := range scenario.List() {
		fmt.Printf("\n  %-10s %s\n", info.Name, info.Title)
		fmt.Printf("             %s\n", wrap(info.Description, 68, "             "))
		fmt.Printf("             agents %d-%d · default %d rounds\n", info.MinAgents, info.MaxAgents, info.DefaultRounds)
		if len(info.Params) > 0 {
			fmt.Printf("             parameters (-set key=value):\n")
			for _, p := range info.Params {
				fmt.Printf("               %-16s %-28s %s\n", p.Name, "default "+p.Default, p.Doc)
			}
		}
	}

	fmt.Printf("\n  %-10s %s\n", "world", "Open-ended persistent town (event-driven, not round-based)")
	fmt.Printf("             %s\n", wrap("No shared objective and no scripted ending: residents decide what to do, whom to see and when. Run with `playworld world new <name>` and `playworld world run <name> -for 2d`; see `playworld world help`.", 68, "             "))

	roster, err := agent.LoadRoster(*rosterPath)
	if err != nil {
		fmt.Printf("\nagents: (%v)\n", err)
		return nil
	}
	fmt.Printf("\nagents in %s:\n", *rosterPath)
	for _, p := range roster.Agents {
		fmt.Printf("\n  %-10s %-14s %s\n", p.ID, p.Name, p.Model.String())
		if p.Role != "" {
			fmt.Printf("             %s\n", p.Role)
		}
		fmt.Printf("             %s\n", wrap(oneLine(p.Personality, 400), 68, "             "))
		for _, g := range p.Goals {
			fmt.Printf("             goal: %s\n", g)
		}
	}
	return nil
}

// cmdReset deletes persisted agent memory so an experiment starts clean.
func cmdReset(args []string) error {
	fs := flag.NewFlagSet("reset", flag.ContinueOnError)
	memDir := fs.String("memory", filepath.Join(defaultStateDir, "memory"), "agent memory directory")
	logs := fs.Bool("logs", false, "also delete experiment logs")
	logsDir := fs.String("logs-dir", filepath.Join(defaultStateDir, "logs"), "experiment log directory")
	if err := fs.Parse(args); err != nil {
		return err
	}

	store, err := memory.NewFileStore(*memDir)
	if err != nil {
		return err
	}
	existing, err := store.List()
	if err != nil {
		return err
	}
	targets := fs.Args()
	if err := store.Reset(targets...); err != nil {
		return err
	}
	if len(targets) == 0 {
		fmt.Printf("cleared memory for %d agent(s) in %s\n", len(existing), *memDir)
	} else {
		fmt.Printf("cleared memory for %s in %s\n", strings.Join(targets, ", "), *memDir)
	}

	if *logs {
		entries, err := os.ReadDir(*logsDir)
		if err != nil && !os.IsNotExist(err) {
			return err
		}
		n := 0
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
				continue
			}
			if err := os.Remove(filepath.Join(*logsDir, e.Name())); err != nil {
				return err
			}
			n++
		}
		fmt.Printf("deleted %d experiment log(s) in %s\n", n, *logsDir)
	}
	return nil
}

// cmdDoctor verifies the local inference backend and the roster, so a failed
// first run has an obvious cause.
func cmdDoctor(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	rosterPath := fs.String("agents", defaultRosterPath, "agent roster YAML")
	endpoint := fs.String("endpoint", "", "Ollama endpoint (default "+ollama.DefaultEndpoint+")")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ok := true
	oll := ollama.New(ollama.WithEndpoint(*endpoint))
	cfg := llm.ModelConfig{Provider: "ollama", Endpoint: *endpoint, Timeout: 10 * time.Second}

	installed, err := oll.List(ctx, cfg)
	if err != nil {
		ok = false
		fmt.Printf("ollama      FAIL  %v\n", err)
	} else {
		fmt.Printf("ollama      ok    %d model(s) installed: %s\n", len(installed), strings.Join(installed, ", "))
	}

	roster, err := agent.LoadRoster(*rosterPath)
	if err != nil {
		fmt.Printf("roster      FAIL  %v\n", err)
		return fmt.Errorf("configuration is not usable yet")
	}
	fmt.Printf("roster      ok    %d agent(s) in %s\n", len(roster.Agents), *rosterPath)

	// Check each distinct model the roster asks for.
	checked := map[string]bool{}
	for _, p := range roster.Agents {
		key := p.Model.Provider + "/" + p.Model.Model
		if checked[key] || p.Model.Provider != "ollama" {
			continue
		}
		checked[key] = true
		mc := p.Model
		if *endpoint != "" {
			mc.Endpoint = *endpoint
		}
		if err := oll.Preflight(ctx, mc); err != nil {
			ok = false
			fmt.Printf("model       FAIL  %v\n", err)
			continue
		}
		fmt.Printf("model       ok    %s is installed\n", p.Model.Model)
	}

	if !ok {
		return fmt.Errorf("some checks failed; see above")
	}
	fmt.Println("\nready: try `playworld run social -n 3`")
	return nil
}

// wrap reflows text to width, indenting continuation lines.
func wrap(s string, width int, indent string) string {
	words := strings.Fields(s)
	var lines []string
	cur := ""
	for _, w := range words {
		if cur == "" {
			cur = w
			continue
		}
		if len(cur)+1+len(w) > width {
			lines = append(lines, cur)
			cur = w
			continue
		}
		cur += " " + w
	}
	if cur != "" {
		lines = append(lines, cur)
	}
	return strings.Join(lines, "\n"+indent)
}
