// Command playworld runs multi-agent LLM experiments locally.
//
//	playworld list                     # show scenarios and the agent roster
//	playworld run social               # run a scenario
//	playworld reset                    # clear persisted agent memory
//	playworld doctor                   # check the local inference backend
//	playworld world new town           # create a persistent, open-ended world
//	playworld world run town -for 2d   # advance it; run again later to resume
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

// version is reported in experiment logs so a log says which harness wrote it.
const version = "0.1.0"

func main() {
	// Ctrl-C cancels the run context; the orchestrator stops at the next clean
	// turn boundary, writes the outcome and flushes the log.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, os.Args[1:]); err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "\ninterrupted")
			os.Exit(130)
		}
		fmt.Fprintf(os.Stderr, "\nplayworld: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		usage()
		return nil
	}
	switch args[0] {
	case "list", "ls":
		return cmdList(args[1:])
	case "run":
		return cmdRun(ctx, args[1:])
	case "reset":
		return cmdReset(args[1:])
	case "doctor":
		return cmdDoctor(ctx, args[1:])
	case "world":
		return cmdWorld(ctx, args[1:])
	case "help", "-h", "--help":
		usage()
		return nil
	case "version", "--version":
		fmt.Println("playworld", version)
		return nil
	default:
		usage()
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func usage() {
	fmt.Print(`playworld — a local multi-agent LLM experimentation harness

usage:
  playworld list                        list scenarios and the configured agents
  playworld run <scenario> [flags]      run a scenario
  playworld reset [agent-id...]         delete persisted agent memory
  playworld doctor                      check that the inference backend is ready
  playworld world <command> [flags]     persistent open-ended world mode (see: playworld world help)

run flags:
  -agents <path>     agent roster YAML (default configs/agents.yaml)
  -n <int>           how many agents from the roster to use (default: scenario minimum)
  -only <ids>        comma-separated agent ids to use instead of -n
  -model <name>      override every agent's model, e.g. qwen3:8b
  -provider <name>   override every agent's provider (ollama, fake)
  -rounds <int>      maximum simulation rounds
  -seed <int>        random seed for scheduling and scenario setup
  -log <level>       quiet | normal | verbose | debug (default normal)
  -set k=v           scenario parameter; repeatable (e.g. -set pool=120)
  -config <path>     scenario settings YAML
  -memory <dir>      where agent memory lives (default .playworld/memory)
  -logs <dir>        where JSONL experiment logs are written (default .playworld/logs)
  -fresh             ignore and overwrite persisted memory for this run
  -no-memory         do not read or write persisted memory at all
  -timeout <dur>     wall-clock limit for the whole run (default 60m)
  -max-calls <int>   cap on total LLM calls
  -shuffle           randomise turn order each round using the seed
  -dual-actions      let every agent choose a communication and a world action independently, in one decision
  -skip-preflight    do not check the backend before starting

examples:
  playworld run social -n 4
  playworld run split -n 3 -set pool=120 -set attempts=3 -log verbose
  playworld run social -model qwen3:8b -rounds 8 -fresh
`)
}
