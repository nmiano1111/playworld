package agent

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/textsim"
)

// idRE matches object ids (o12, p3) in a line.
var idRE = regexp.MustCompile(`\b[a-z]\d+\b`)

// shownLine is one transcript line as rendered: possibly standing for
// several observations that said the same thing.
type shownLine struct {
	stamp string
	text  string
}

// compactTranscript folds repeats in the agent's short-term context. A line
// that says the same thing as an earlier line from the same source — the
// same words said again, the same offer made again — is shown once, at its
// latest occurrence, with how many times it occurred and since when. Nothing
// is dropped silently: the count and the first time stay visible, and lines
// that differ in any number (an amount, a price, a time) are never merged.
// earlier and fresh keep the caller's split between what the agent already
// saw and what arrived since its last turn.
func compactTranscript(earlier, fresh []sim.Observation) ([]shownLine, []shownLine) {
	all := append(append([]sim.Observation(nil), earlier...), fresh...)
	type group struct {
		source  string
		print   textsim.Print
		masked  string
		members []int
	}
	var groups []*group
	owner := make([]*group, len(all))
	for i, o := range all {
		source, body := splitSource(o.Text)
		print := textsim.Of(body)
		threshold := textsim.Strict
		if source != "" && source != "✔" {
			threshold = textsim.Speech
		}
		// A witnessed event only folds with one that reads identically but
		// for its ids (the same offer made again): word sets alone would
		// fold "o10 from Mira to Ada expired" into "o12 from Ada to Mira".
		masked := ""
		if source == "✔" {
			masked = idRE.ReplaceAllString(body, "#")
		}
		var match *group
		for g := len(groups) - 1; g >= 0; g-- {
			c := groups[g]
			if c.source == source && c.masked == masked && textsim.Same(c.print, print, threshold) {
				match = c
				break
			}
		}
		if match == nil {
			match = &group{source: source, print: print, masked: masked}
			groups = append(groups, match)
		}
		match.members = append(match.members, i)
		owner[i] = match
	}
	var outEarlier, outFresh []shownLine
	for i, o := range all {
		g := owner[i]
		if g.members[len(g.members)-1] != i {
			continue // shown at its latest occurrence
		}
		text := o.Text
		if n := len(g.members); n > 1 {
			first := all[g.members[0]]
			text += " (" + strconv.Itoa(n) + " times"
			if first.Stamp != "" {
				text += ", first " + first.Stamp
			}
			text += ")"
		}
		stamp := o.Stamp
		if stamp == "" {
			stamp = "r" + strconv.Itoa(o.Round)
		}
		l := shownLine{stamp: stamp, text: text}
		if i >= len(earlier) {
			outFresh = append(outFresh, l)
		} else {
			outEarlier = append(outEarlier, l)
		}
	}
	return outEarlier, outFresh
}

// splitSource separates who a line is from ("Dov (dov) said", "you
// whispered to Ada (ada)") from what it says. Lines that are not speech keep
// their verified mark as their source, so a claim and a witnessed event are
// never folded together.
func splitSource(text string) (source, body string) {
	if i := strings.Index(text, ": "); i > 0 && i < 80 {
		head := text[:i]
		if strings.HasSuffix(head, " said") || strings.Contains(head, " whispered") || strings.HasSuffix(head, " posted") {
			return head, text[i+2:]
		}
	}
	if strings.HasPrefix(text, "✔") {
		return "✔", text
	}
	return "", text
}

// speechInWorldSlot recognises a say or whisper sent as the world action —
// {"type":"say","message":"..."} in "world_action" — and returns it as the
// communication it was meant to be. Anything else is left alone.
func speechInWorldSlot(a sim.Action, participants map[string]sim.AgentID, self sim.AgentID) (sim.Action, bool) {
	if a.Kind != sim.ActionDo {
		return sim.Action{}, false
	}
	content := ""
	for _, k := range []string{"message", "content", "text", "say"} {
		if v, ok := a.Args[k].(string); ok && strings.TrimSpace(v) != "" {
			content = strings.TrimSpace(v)
			break
		}
	}
	if content == "" {
		return sim.Action{}, false
	}
	switch a.Verb {
	case "say", "speak", "talk":
		return sim.Action{Kind: sim.ActionSay, Content: content}, true
	case "whisper":
		var to []sim.AgentID
		for _, k := range []string{"recipient", "to", "target"} {
			switch v := a.Args[k].(type) {
			case string:
				to = append(to, sim.AgentID(v))
			case []any:
				for _, x := range v {
					if s, ok := x.(string); ok {
						to = append(to, sim.AgentID(s))
					}
				}
			}
		}
		resolved, err := resolveWhisperTargets(to, participants, self)
		if err != nil {
			return sim.Action{}, false
		}
		return sim.Action{Kind: sim.ActionWhisper, Content: content, To: resolved}, true
	}
	return sim.Action{}, false
}
