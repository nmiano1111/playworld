package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/nmiano/playworld/internal/explog"
	"github.com/nmiano/playworld/internal/world"
)

// cmdWorldCompare prints objective metrics for several worlds side by side —
// the before/after view for experiments that run the same configuration
// under different cognition modes. Every number is a count the world or the
// harness recorded; nothing here scores behaviour.
func cmdWorldCompare(args []string) error {
	fs := flag.NewFlagSet("world compare", flag.ContinueOnError)
	root := fs.String("root", defaultWorldRoot, "directory holding worlds")
	// Allow flags after the names too.
	var names []string
	for len(args) > 0 {
		if err := fs.Parse(args); err != nil {
			return err
		}
		if fs.NArg() == 0 {
			break
		}
		names = append(names, fs.Arg(0))
		args = fs.Args()[1:]
	}
	if len(names) == 0 {
		return fmt.Errorf("usage: playworld world compare <world> <world> ... (see `playworld world list`)")
	}
	store := worldStore(*root)
	type metrics = map[string]float64
	cols := make([]metrics, len(names))
	modes := make([]string, len(names))
	codes := map[string]bool{}
	for i, name := range names {
		if !store.Exists(name) {
			return noWorld(store, name)
		}
		w, err := store.Load(name)
		if err != nil {
			return err
		}
		modes[i] = w.Settings.Cognition
		if modes[i] == "" {
			modes[i] = world.CognitionLegacy
		}
		m := worldMetrics(w)
		for k := range m {
			if strings.HasPrefix(k, "rejected_") && k != "rejected_per_activation" {
				codes[k] = true
			}
		}
		cols[i] = m
	}

	type row struct{ label, key string }
	rows := []row{
		{"sessions", "sessions"}, {"simulated days", "sim_days"},
		{"activations", "activations"}, {"llm calls", "llm_calls"}, {"llm calls / sim day", "calls_per_day"},
		{"prompt tokens / sim day", "prompt_per_day"}, {"output tokens / sim day", "output_per_day"},
		{"prompt tokens / call", "prompt_per_call"},
		{"", ""},
		{"refused actions", "rejected"}, {"refused / activation", "rejected_per_activation"},
	}
	var codeKeys []string
	for k := range codes {
		codeKeys = append(codeKeys, k)
	}
	sort.Strings(codeKeys)
	for _, k := range codeKeys {
		rows = append(rows, row{"  " + strings.TrimPrefix(k, "rejected_"), k})
	}
	rows = append(rows,
		row{"slot repairs (say as world action)", "slot_repairs"},
		row{"  refused though not offered", "rejected_unoffered"},
		row{"  refused though offered", "rejected_offered"},
		row{"  same refusal reason again (24h)", "repeat_refusals"},
		row{"after a setback: same kind of move", "after_setback_same"},
		row{"after a setback: a different move", "after_setback_changed"},
		row{"", ""},
		row{"speech", "speech"}, row{"private notes", "messages"},
		row{"repeated remarks (same listeners, 12h)", "repeated_speech"},
		row{"repeated offers (same terms, 24h)", "repeated_offers"},
		row{"triggers suppressed as repeats", "repeat_triggers_suppressed"},
		row{"reactions damped (reply chain)", "reactions_damped"},
		row{"triggers coalesced", "triggers_coalesced"},
		row{"", ""},
		row{"intentions set", "intentions_set"}, row{"intentions superseded", "intentions_superseded"},
		row{"intentions cancelled", "intentions_cancelled"}, row{"intentions come due", "intentions_due"},
		row{"plan updates", "plan_updates"},
		row{"goal changes", "goal_changes"}, row{"plans dropped", "plans_dropped"},
		row{"plan steps marked done", "plan_steps_done"},
		row{"plans shown related changes", "plan_change_notes"},
		row{"plans shown a lasting obstacle", "reconsider_notes"},
		row{"intentions due on a closed subject", "stale_intentions_due"},
		row{"trips with a stated purpose", "trips_with_purpose"},
		row{"", ""},
		row{"projects started", "projects_started"}, row{"projects completed", "projects_completed"},
		row{"projects abandoned", "projects_abandoned"}, row{"contributions", "contributions"},
		row{"offers made", "offers"}, row{"offers declined", "offers_declined"}, row{"trades", "trades"}, row{"gifts", "gifts"},
		row{"offers re-made on declined terms", "reoffers_after_decline"},
		row{"offers citing a finished project", "offers_citing_closed_project"},
		row{"consequential changes", "consequential"}, row{"llm calls / consequential change", "calls_per_change"},
	)

	width := 40
	fmt.Printf("%-*s", width, "")
	for _, n := range names {
		fmt.Printf(" %14s", n)
	}
	fmt.Printf("\n%-*s", width, "cognition")
	for _, m := range modes {
		fmt.Printf(" %14s", m)
	}
	fmt.Println()
	for _, r := range rows {
		if r.key == "" {
			fmt.Println()
			continue
		}
		fmt.Printf("%-*s", width, r.label)
		for _, c := range cols {
			v, ok := c[r.key]
			// Sessions record only non-zero counters, so a missing counter
			// is a zero — except the counters added with grounded
			// cognition, in sessions that predate them.
			unknown := (newCounters[r.key] && c["partial"] > 0) || (v2Counters[r.key] && c["partial2"] > 0)
			switch {
			case !ok && unknown:
				fmt.Printf(" %14s", "?")
			case !ok:
				fmt.Printf(" %14d", 0)
			case v == float64(int64(v)):
				fmt.Printf(" %14d", int64(v))
			default:
				fmt.Printf(" %14.2f", v)
			}
		}
		fmt.Println()
	}
	fmt.Println("\nconsequential changes = trades + gifts + contributions + projects started, completed or abandoned.")
	fmt.Println("? means some of the world's sessions predate that counter. Refusal codes for such sessions are")
	fmt.Println("recovered from the logged refusal messages.")
	return nil
}

// newCounters were introduced together with cognition modes; sessions run
// before then did not record them.
var newCounters = map[string]bool{
	"slot_repairs": true, "repeated_speech": true, "repeated_offers": true, "repeat_triggers_suppressed": true,
	"intentions_superseded": true, "projects_abandoned": true,
}

// v2Counters were introduced with decision-coherence tracking.
var v2Counters = map[string]bool{
	"rejected_unoffered": true, "rejected_offered": true, "repeat_refusals": true,
	"after_setback_same": true, "after_setback_changed": true,
	"goal_changes": true, "plans_dropped": true, "plan_steps_done": true, "plan_change_notes": true, "reconsider_notes": true,
	"stale_intentions_due": true, "trips_with_purpose": true, "reoffers_after_decline": true, "offers_citing_closed_project": true,
}

// worldMetrics sums a world's session counters and derives per-day rates.
// Token counts and plan updates come from the session logs when a session's
// own counters predate them.
func worldMetrics(w *world.World) map[string]float64 {
	m := map[string]float64{"sessions": float64(len(w.Sessions))}
	minutes := 0.0
	for _, s := range w.Sessions {
		minutes += s.SimTo.Sub(s.SimFrom).Minutes()
		m["activations"] += float64(s.Activations)
		m["llm_calls"] += float64(s.LLMCalls)
		for k, v := range s.Metrics {
			switch k {
			case "llm_calls", "activations", "sim_minutes", "events":
				continue
			}
			m[k] += float64(v)
		}
		scan := scanLog(s.LogPath)
		if _, ok := s.Metrics["counters_v2"]; ok {
			m["instrumented2"]++
		}
		if _, ok := s.Metrics["sim_minutes"]; ok {
			m["instrumented"]++
		} else {
			// A session from before these counters: recover what the log
			// can tell — tokens, and refusal codes from their wording.
			m["prompt_tokens"] += float64(scan.prompt)
			m["output_tokens"] += float64(scan.output)
			for code, n := range scan.refusals {
				m["rejected_"+code] += float64(n)
			}
		}
		m["plan_updates"] += float64(scan.plans)
	}
	if m["instrumented"] < m["sessions"] {
		m["partial"] = 1
	}
	if m["instrumented2"] < m["sessions"] {
		m["partial2"] = 1
	}
	m["consequential"] = m["trades"] + m["gifts"] + m["contributions"] + m["projects_started"] + m["projects_completed"] + m["projects_abandoned"]
	if days := minutes / (24 * 60); days > 0 {
		m["sim_days"] = round2(days)
		m["calls_per_day"] = round2(m["llm_calls"] / days)
		m["prompt_per_day"] = float64(int64(m["prompt_tokens"] / days))
		m["output_per_day"] = float64(int64(m["output_tokens"] / days))
	}
	if m["llm_calls"] > 0 {
		m["prompt_per_call"] = float64(int64(m["prompt_tokens"] / m["llm_calls"]))
	}
	if m["activations"] > 0 {
		m["rejected_per_activation"] = round2(m["rejected"] / m["activations"])
	}
	if m["consequential"] > 0 {
		m["calls_per_change"] = round2(m["llm_calls"] / m["consequential"])
	}
	return m
}

func round2(f float64) float64 { return float64(int64(f*100+0.5)) / 100 }

type logScan struct {
	prompt, output, plans int
	refusals              map[string]int
}

// scanLog totals token usage, plan updates and refusals by code from a
// session log.
func scanLog(path string) logScan {
	out := logScan{refusals: map[string]int{}}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 4<<20), 4<<20)
	for sc.Scan() {
		var r explog.Record
		if json.Unmarshal(sc.Bytes(), &r) != nil {
			continue
		}
		switch {
		case r.Call != nil:
			out.prompt += r.Call.PromptTokens
			out.output += r.Call.OutputTokens
		case r.Plan != nil:
			out.plans++
		case r.Rejected != nil:
			code := r.Rejected.Code
			if code == "" {
				code = world.ClassifyRefusal(r.Rejected.Reason)
			}
			out.refusals[code]++
		}
	}
	return out
}
