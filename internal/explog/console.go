package explog

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/nmiano/playworld/internal/sim"
)

// Verbosity controls how much the terminal renderer prints.
type Verbosity int

const (
	// Quiet prints only round banners, outcomes and the summary.
	Quiet Verbosity = iota
	// Normal prints the conversation, actions and rejections. The default.
	Normal
	// Verbose adds per-call latency/token lines and self-reported reasoning.
	Verbose
	// Debug adds full prompts and raw replies.
	Debug
)

// ParseVerbosity maps a CLI string to a level.
func ParseVerbosity(s string) (Verbosity, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "quiet", "q":
		return Quiet, nil
	case "", "normal", "n", "info":
		return Normal, nil
	case "verbose", "v":
		return Verbose, nil
	case "debug", "d":
		return Debug, nil
	}
	return Normal, fmt.Errorf("unknown log level %q (want quiet, normal, verbose or debug)", s)
}

// Console renders records as readable live terminal output.
type Console struct {
	mu    sync.Mutex
	w     io.Writer
	level Verbosity
	color bool

	names  map[sim.AgentID]string
	colors map[sim.AgentID]string
	next   int
	// world is set by a world-mode run_start: events then carry their
	// simulated time, and scheduling/plan records are shown.
	world bool
}

// agentPalette is cycled through so each agent keeps a stable colour.
var agentPalette = []string{"\033[36m", "\033[35m", "\033[32m", "\033[33m", "\033[34m", "\033[91m", "\033[96m", "\033[95m"}

const (
	cReset = "\033[0m"
	cDim   = "\033[2m"
	cBold  = "\033[1m"
	cRed   = "\033[31m"
	cYell  = "\033[33m"
)

// NewConsole builds a renderer. Colour is enabled when writing to a terminal
// and NO_COLOR is unset.
func NewConsole(w io.Writer, level Verbosity) *Console {
	if w == nil {
		w = os.Stdout
	}
	color := os.Getenv("NO_COLOR") == ""
	if f, ok := w.(*os.File); ok {
		if fi, err := f.Stat(); err == nil && fi.Mode()&os.ModeCharDevice == 0 {
			color = false
		}
	} else {
		color = false
	}
	return &Console{w: w, level: level, color: color, names: map[sim.AgentID]string{}, colors: map[sim.AgentID]string{}}
}

func (c *Console) paint(s, code string) string {
	if !c.color || code == "" {
		return s
	}
	return code + s + cReset
}

func (c *Console) agentColor(id sim.AgentID) string {
	if col, ok := c.colors[id]; ok {
		return col
	}
	col := agentPalette[c.next%len(agentPalette)]
	c.next++
	c.colors[id] = col
	return col
}

func (c *Console) label(id sim.AgentID) string {
	name := c.names[id]
	if name == "" {
		name = string(id)
	}
	return c.paint(name, c.agentColor(id))
}

func (c *Console) printf(format string, args ...any) {
	fmt.Fprintf(c.w, format, args...)
}

// Emit renders one record.
func (c *Console) Emit(r Record) {
	c.mu.Lock()
	defer c.mu.Unlock()

	switch r.Type {
	case RecRunStart:
		if r.Run == nil {
			return
		}
		c.printf("\n%s\n", c.paint("── playworld ─────────────────────────────────────────────", cBold))
		if w := r.Run.World; w != nil {
			c.world = true
			c.printf("world     %s · session %d\n", c.paint(w.World, cBold), w.Session)
			c.printf("window    %s → %s\n", w.From, w.To)
			c.printf("session   %s\n", r.Run.RunID)
			c.printf("limits    %d activations · %d llm calls", r.Run.MaxRounds, r.Run.MaxCalls)
		} else {
			c.printf("scenario  %s\n", c.paint(r.Run.Scenario, cBold))
			c.printf("run id    %s\n", r.Run.RunID)
			c.printf("limits    %d rounds · %d llm calls · %d actions",
				r.Run.MaxRounds, r.Run.MaxCalls, r.Run.MaxActions)
		}
		if r.Run.Timeout != "" {
			c.printf(" · %s", r.Run.Timeout)
		}
		c.printf("\nseed      %d\n", r.Run.Seed)

	case RecAgent:
		if r.Agent == nil {
			return
		}
		c.names[r.Agent.ID] = r.Agent.Name
		line := fmt.Sprintf("agent     %-10s %s", r.Agent.ID, c.label(r.Agent.ID))
		c.printf("%s %s", line, c.paint(r.Agent.Model.String(), cDim))
		if r.Agent.Role != "" {
			c.printf(" %s", c.paint("— "+r.Agent.Role, cDim))
		}
		if r.Agent.Episodes > 0 {
			c.printf(" %s", c.paint(fmt.Sprintf("[recalls %d episodes]", r.Agent.Episodes), cDim))
		}
		c.printf("\n")

	case RecRoundStart:
		c.printf("\n%s\n", c.paint(fmt.Sprintf("── round %d ───────────────────────────────────────────────", r.Round), cBold))

	case RecActivation:
		if r.Activation == nil {
			return
		}
		why := strings.Join(r.Activation.Reasons, "; ")
		c.printf("\n%s %s wakes %s\n", c.paint("["+r.Activation.Stamp+"]", cBold), c.label(r.Activation.Agent),
			c.paint("— "+oneLine(why, 140), cDim))

	case RecScheduled:
		s := r.Scheduled
		if s == nil || c.level < Normal {
			return
		}
		switch {
		case s.Damped:
			if c.level >= Verbose {
				c.printf("  %s\n", c.paint(fmt.Sprintf("↳ %s not woken (reply chain at its limit): %s", s.Agent, s.Reason), cDim))
			}
		case s.Coalesced:
			if c.level >= Verbose {
				c.printf("  %s\n", c.paint(fmt.Sprintf("↳ %s: added to the activation already pending at %s", s.Agent, clockOf(s.Stamp)), cDim))
			}
		case s.Kind != "idle" && s.Kind != "activity":
			c.printf("  %s\n", c.paint(fmt.Sprintf("↳ %s will decide at %s (%s)", c.names[s.Agent], clockOf(s.Stamp), s.Reason), cDim))
		}

	case RecPlan:
		p := r.Plan
		if p == nil || c.level < Normal {
			return
		}
		if p.Plan == nil {
			c.printf("  %s %s\n", c.label(p.Agent), c.paint("abandoned its plan", cDim))
			return
		}
		line := p.Plan.PrimaryGoal
		if len(p.Plan.NextSteps) > 0 {
			line += " — next: " + p.Plan.NextSteps[0]
		}
		c.printf("  %s %s\n", c.paint("plan", cDim), c.paint(oneLine(line, 160), cDim))

	case RecMemory:
		m := r.Memory
		if m == nil || c.level < Verbose {
			return
		}
		if m.Kind == "opinion" {
			c.printf("  %s\n", c.paint(fmt.Sprintf("%s now thinks of %s: %s", m.Agent, m.Subject, oneLine(m.Text, 140)), cDim))
		} else {
			c.printf("  %s\n", c.paint(fmt.Sprintf("%s noted: %s", m.Agent, oneLine(m.Text, 140)), cDim))
		}

	case RecEvent:
		c.renderEvent(r)

	case RecDecision:
		if c.level >= Verbose && r.Decision != nil && r.Decision.Action.Reasoning != "" {
			c.printf("   %s\n", c.paint("("+r.Decision.Action.Reasoning+")", cDim))
		}

	case RecLLMCall:
		if r.Call == nil {
			return
		}
		if r.Call.Err != "" && c.level >= Normal {
			c.printf("   %s %s\n", c.paint("llm error", cRed), r.Call.Err)
		}
		if c.level >= Verbose {
			c.printf("   %s\n", c.paint(fmt.Sprintf("%s · %s · %dms · %d→%d tok",
				r.Call.Agent, r.Call.Model, r.Call.LatencyMS, r.Call.PromptTokens, r.Call.OutputTokens), cDim))
			if r.Call.Thinking != "" {
				c.printf("   %s\n", c.paint("thinking: "+oneLine(r.Call.Thinking, 300), cDim))
			}
		}
		if c.level >= Debug {
			for _, m := range r.Call.Prompt {
				c.printf("%s\n", c.paint(fmt.Sprintf("   ┌─ %s ─┐\n%s", m.Role, indent(m.Content, "   │ ")), cDim))
			}
			if r.Call.Raw != "" {
				c.printf("%s\n", c.paint(indent("raw: "+r.Call.Raw, "   "), cDim))
			}
		}

	case RecRejected:
		if r.Rejected == nil || c.level < Normal {
			return
		}
		c.printf("   %s %s tried %s — %s\n", c.paint("refused", cYell), c.label(r.Rejected.Agent),
			describeAction(r.Rejected.Action), r.Rejected.Reason)

	case RecError:
		if r.Failure == nil {
			return
		}
		who := ""
		if r.Failure.Agent != "" {
			who = " " + string(r.Failure.Agent)
		}
		c.printf("   %s%s (%s) %s\n", c.paint("error", cRed), who, r.Failure.Stage, r.Failure.Err)

	case RecOutcome:
		if r.Outcome == nil {
			return
		}
		c.printf("\n%s\n", c.paint("── outcome ───────────────────────────────────────────────", cBold))
		if r.Outcome.Summary != "" {
			c.printf("%s\n", r.Outcome.Summary)
		}
		if len(r.Outcome.Scores) > 0 {
			ids := make([]sim.AgentID, 0, len(r.Outcome.Scores))
			for id := range r.Outcome.Scores {
				ids = append(ids, id)
			}
			sort.Slice(ids, func(i, j int) bool {
				if r.Outcome.Scores[ids[i]] != r.Outcome.Scores[ids[j]] {
					return r.Outcome.Scores[ids[i]] > r.Outcome.Scores[ids[j]]
				}
				return ids[i] < ids[j]
			})
			c.printf("\nscores:\n")
			for _, id := range ids {
				c.printf("  %-22s %s\n", c.label(id), formatNum(r.Outcome.Scores[id]))
			}
		}
		if len(r.Outcome.Metrics) > 0 {
			c.printf("\nmetrics:\n")
			var nested []string
			for _, k := range SortedKeys(r.Outcome.Metrics) {
				if _, ok := r.Outcome.Metrics[k].(map[sim.AgentID]map[string]int); ok {
					nested = append(nested, k)
					continue
				}
				c.printf("  %-22s %v\n", k, r.Outcome.Metrics[k])
			}
			for _, k := range nested {
				c.renderPerAgent(k, r.Outcome.Metrics[k].(map[sim.AgentID]map[string]int))
			}
		}

	case RecSummary:
		c.renderSummary(r.Summary)
	}
}

func (c *Console) renderEvent(r Record) {
	e := r.Event
	if e == nil {
		return
	}
	if c.world {
		c.renderWorldEvent(r)
		return
	}
	switch e.Type {
	case sim.EventSay:
		c.printf("  %s: %s\n", c.label(e.Actor), e.Text)
	case sim.EventWhisper:
		if c.level < Normal {
			return
		}
		c.printf("  %s %s %s: %s\n", c.label(e.Actor),
			c.paint("→whisper→", cDim), c.paint(joinIDs(e.Targets), cDim), c.paint(e.Text, cDim))
	case sim.EventAction:
		c.printf("  %s %s %s\n", c.label(e.Actor), c.paint("»", cBold), e.Text)
	case sim.EventPass:
		if c.level < Normal {
			return
		}
		c.printf("  %s %s\n", c.label(e.Actor), c.paint("… passes", cDim))
	case sim.EventState, sim.EventSystem:
		c.printf("  %s\n", c.paint("· "+e.Text, cDim))
	case sim.EventOutcome:
		// Rendered by the outcome banner instead, to avoid saying it twice.
	case sim.EventRejected:
		// The rejection is already reported with its reason.
	default:
		c.printf("  %s\n", e.Text)
	}
}

// renderWorldEvent prints one event of a world session on its own
// timestamped line, as a chronological record:
//
//	[Day 1 09:10] Mira » set off for the market.
//	[Day 1 09:20] Mira → Rook (private): "Would you be interested…"
func (c *Console) renderWorldEvent(r Record) {
	e := r.Event
	ts := c.paint("["+r.SimStamp+"]", cDim)
	switch e.Type {
	case sim.EventSay:
		c.printf("%s %s %s %q\n", ts, c.label(e.Actor), c.paint("(aloud; heard by "+joinIDs(e.Visibility.Audience)+"):", cDim), e.Text)
	case sim.EventWhisper:
		names := make([]string, 0, len(e.Targets))
		for _, t := range e.Targets {
			names = append(names, c.label(t))
		}
		c.printf("%s %s → %s %s %q\n", ts, c.label(e.Actor), strings.Join(names, ", "), c.paint("(private):", cDim), e.Text)
	case sim.EventAction:
		c.printf("%s %s %s %s\n", ts, c.label(e.Actor), c.paint("»", cBold), e.Text)
	case sim.EventPass:
		if c.level >= Normal {
			c.printf("%s %s %s\n", ts, c.label(e.Actor), c.paint("… does nothing for now", cDim))
		}
	case sim.EventState, sim.EventSystem:
		who := ""
		if e.Actor != "" && !e.Visibility.Public {
			who = string(e.Actor) + ": "
		} else if !e.Visibility.Public && len(e.Visibility.Audience) > 0 {
			who = joinIDs(e.Visibility.Audience) + ": "
		}
		c.printf("%s %s\n", ts, c.paint("· "+who+e.Text, cDim))
	case sim.EventOutcome, sim.EventRejected:
	default:
		c.printf("%s %s\n", ts, e.Text)
	}
}

// renderPerAgent prints a per-agent counter breakdown, one agent per line.
func (c *Console) renderPerAgent(title string, m map[sim.AgentID]map[string]int) {
	ids := make([]string, 0, len(m))
	for id := range m {
		ids = append(ids, string(id))
	}
	sort.Strings(ids)
	c.printf("\n%s:\n", title)
	for _, id := range ids {
		counts := m[sim.AgentID(id)]
		parts := make([]string, 0, len(counts))
		for _, k := range SortedKeys(counts) {
			parts = append(parts, fmt.Sprintf("%s=%d", k, counts[k]))
		}
		c.printf("  %-8s %s\n", id, strings.Join(parts, " "))
	}
}

// clockOf trims a "Day 2 09:35" stamp to "09:35" for compact lines.
func clockOf(stamp string) string {
	if i := strings.LastIndex(stamp, " "); i >= 0 {
		return stamp[i+1:]
	}
	return stamp
}

func (c *Console) renderSummary(s *Summary) {
	if s == nil {
		return
	}
	c.printf("\n%s\n", c.paint("── summary ───────────────────────────────────────────────", cBold))
	unit := "round(s)"
	if s.Scenario == "world" {
		unit = "activation(s)"
	}
	c.printf("ended     %s after %d %s in %s\n", c.paint(s.StopReason, cBold), s.Rounds, unit, s.Duration)
	c.printf("inference %d calls · %d errors · %d retries · %d→%d tokens · %s in model\n",
		s.LLMCalls, s.LLMErrors, s.Retries, s.Prompt, s.Output, s.Latency)
	if len(s.Actions) > 0 {
		parts := make([]string, 0, len(s.Actions))
		for _, k := range SortedKeys(s.Actions) {
			parts = append(parts, fmt.Sprintf("%s=%d", k, s.Actions[k]))
		}
		c.printf("actions   %s", strings.Join(parts, " · "))
		if s.Rejections > 0 {
			c.printf(" · %s", c.paint(fmt.Sprintf("rejected=%d", s.Rejections), cYell))
		}
		c.printf("\n")
	}
	if len(s.PerAgent) > 0 && s.Scenario == "world" {
		c.printf("\n%-8s %5s %5s %7s %4s %5s %4s %5s %5s %8s %9s\n",
			"agent", "calls", "say", "private", "act", "pass", "err", "plans", "notes", "opinions", "avg ms")
		for _, a := range s.PerAgent {
			c.printf("%-8s %5d %5d %7d %4d %5d %4d %5d %5d %8d %9d\n",
				a.ID, a.Calls, a.Said, a.Whispered, a.Acted, a.Passed, a.Errors, a.PlanUpdates, a.NotesWritten, a.OpinionUpdates, a.AvgLatencyMS)
		}
	} else if len(s.PerAgent) > 0 {
		c.printf("\n%-12s %-16s %5s %5s %6s %5s %5s %4s %9s\n",
			"agent", "model", "calls", "say", "whisper", "act", "pass", "err", "avg ms")
		for _, a := range s.PerAgent {
			c.printf("%-12s %-16s %5d %5d %6d %5d %5d %4d %9d\n",
				a.ID, truncate(a.Model, 16), a.Calls, a.Said, a.Whispered, a.Acted, a.Passed, a.Errors, a.AvgLatencyMS)
		}
	}
	if s.LogPath != "" {
		c.printf("\nlog       %s\n", s.LogPath)
	}
}

func (c *Console) Close() error { return nil }

func joinIDs(ids []sim.AgentID) string {
	parts := make([]string, 0, len(ids))
	for _, id := range ids {
		parts = append(parts, string(id))
	}
	return strings.Join(parts, ",")
}

func describeAction(a sim.Action) string {
	switch a.Kind {
	case sim.ActionDo:
		if len(a.Args) > 0 {
			return fmt.Sprintf("do %s %v", a.Verb, a.Args)
		}
		return "do " + a.Verb
	case sim.ActionWhisper:
		return "whisper to " + joinIDs(a.To)
	default:
		return string(a.Kind)
	}
}

func formatNum(f float64) string {
	if f == float64(int64(f)) {
		return fmt.Sprintf("%d", int64(f))
	}
	return fmt.Sprintf("%.2f", f)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}

func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	return truncate(s, max)
}

func indent(s, prefix string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, l := range lines {
		lines[i] = prefix + l
	}
	return strings.Join(lines, "\n")
}

// FormatDuration renders a duration compactly for summaries.
func FormatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}
