package agent

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/textsim"
)

// This file keeps a long-lived agent's plan connected to what actually
// happens: how long it has held a goal and each obstacle, which consequential
// events since the plan was written concern things the plan mentions, and
// when an obstacle has outlasted several decisions. It only ever shows the
// agent facts about its own notes and observations; what to do about them —
// keep going, wait, try another way, drop the goal — stays the model's call.

// sameGoal is how alike two goal statements must be (word overlap) to count
// as the agent still pursuing the same thing in new words.
const sameGoal = 0.5

// sameObstacle is the same for an obstacle carried from one plan to the next.
const sameObstacle = 0.5

// stuckTurns is how many decisions a goal and its oldest obstacle must both
// have lasted before the plan notes that the obstacle has persisted.
const (
	stuckGoalTurns     = 4
	stuckObstacleTurns = 3
)

// carryPlan fills next's goal and obstacle ages from prev (the plan it
// replaces, or nil), as of this decision.
func carryPlan(prev, next *memory.Plan, now time.Time, stamp string) (goalChanged bool) {
	if prev != nil && prev.PrimaryGoal != "" && textsim.Overlap(textsim.Of(prev.PrimaryGoal), textsim.Of(next.PrimaryGoal)) >= sameGoal {
		next.GoalSince, next.GoalStamp, next.Turns = prev.GoalSince, prev.GoalStamp, prev.Turns+1
		if next.GoalStamp == "" {
			next.GoalSince, next.GoalStamp = prev.At, prev.Stamp
		}
	} else {
		next.GoalSince, next.GoalStamp, next.Turns = now, stamp, 1
		goalChanged = prev != nil
	}
	next.ObstacleSince = make([]string, len(next.Obstacles))
	next.ObstacleTurns = make([]int, len(next.Obstacles))
	for i, ob := range next.Obstacles {
		next.ObstacleSince[i], next.ObstacleTurns[i] = stamp, 1
		if prev == nil {
			continue
		}
		best, at := 0.0, -1
		for j, old := range prev.Obstacles {
			if o := textsim.Overlap(textsim.Of(old), textsim.Of(ob)); o >= sameObstacle && o > best {
				best, at = o, j
			}
		}
		if at >= 0 {
			if at < len(prev.ObstacleSince) && prev.ObstacleSince[at] != "" {
				next.ObstacleSince[i] = prev.ObstacleSince[at]
			} else if prev.Stamp != "" {
				next.ObstacleSince[i] = prev.Stamp
			}
			turns := 1
			if at < len(prev.ObstacleTurns) {
				turns = prev.ObstacleTurns[at]
			}
			next.ObstacleTurns[i] = turns + 1
		}
	}
	return goalChanged
}

// tickPlan counts one more decision taken under an unchanged plan.
func tickPlan(p *memory.Plan) {
	if p == nil {
		return
	}
	p.Turns++
	for len(p.ObstacleTurns) < len(p.Obstacles) {
		p.ObstacleTurns = append(p.ObstacleTurns, 1)
	}
	for i := range p.Obstacles {
		p.ObstacleTurns[i]++
	}
}

// newlyDone counts entries of next.Done that prev did not already have.
func newlyDone(prev, next *memory.Plan) int {
	had := map[string]bool{}
	if prev != nil {
		for _, d := range prev.Done {
			had[d] = true
		}
	}
	n := 0
	for _, d := range next.Done {
		if !had[d] {
			n++
		}
	}
	return n
}

// planStuck reports whether the plan has held the same goal, and the same
// obstacle, through several decisions.
func planStuck(p *memory.Plan) bool {
	if p == nil || p.Turns < stuckGoalTurns {
		return false
	}
	for _, n := range p.ObstacleTurns {
		if n >= stuckObstacleTurns {
			return true
		}
	}
	return false
}

// quotedRE finds quoted names in event text ("Community Garden Bed").
var quotedRE = regexp.MustCompile(`"([^"]{3,60})"`)

// planText is everything the agent wrote in its plan, for matching.
func planText(p *memory.Plan) string {
	parts := []string{p.PrimaryGoal, p.CurrentStrategy}
	parts = append(parts, p.NextSteps...)
	parts = append(parts, p.Obstacles...)
	parts = append(parts, p.Commitments...)
	return strings.Join(parts, "\n")
}

// planChanges picks, from what the agent has seen, the consequential events
// since the plan was written that concern something the plan mentions: the
// same object id (an offer or project), or a named project. It is a
// pointer back at evidence the agent already saw, not a verdict on the plan.
func planChanges(p *memory.Plan, seen []sim.Observation, n int) []sim.Observation {
	if p == nil || p.At.IsZero() {
		return nil
	}
	text := planText(p)
	refs := map[string]bool{}
	for _, r := range textsim.Refs(text) {
		refs[r] = true
	}
	var out []sim.Observation
	for _, o := range seen {
		if !o.Time.After(p.At) || o.Importance < 2 || !strings.HasPrefix(o.Text, "✔") {
			continue
		}
		hit := false
		for _, r := range textsim.Refs(o.Text) {
			hit = hit || refs[r]
		}
		for _, m := range quotedRE.FindAllStringSubmatch(o.Text, -1) {
			hit = hit || textsim.Mentions(text, m[1])
		}
		if hit {
			out = append(out, o)
		}
	}
	if len(out) > n {
		out = out[len(out)-n:]
	}
	return out
}

// renderPlanTracking is the part of the plan section that connects the plan
// to time and to what happened: goal and obstacle ages, related changes, and
// a plain note when an obstacle has outlasted several decisions.
func renderPlanTracking(p *memory.Plan, seen []sim.Observation) string {
	var b strings.Builder
	if p.GoalStamp != "" && p.Turns > 1 {
		fmt.Fprintf(&b, "You have held this goal since %s, through %d of your decisions.\n", p.GoalStamp, p.Turns)
	}
	var lasting []string
	for i, ob := range p.Obstacles {
		if i < len(p.ObstacleTurns) && p.ObstacleTurns[i] >= 2 {
			since := ""
			if i < len(p.ObstacleSince) && p.ObstacleSince[i] != "" {
				since = " since " + p.ObstacleSince[i]
			}
			lasting = append(lasting, fmt.Sprintf("%q — on your plan%s, through %d decisions", oneLine(ob, 120), since, p.ObstacleTurns[i]))
		}
	}
	if len(lasting) > 0 {
		b.WriteString("Obstacles that have lasted:\n")
		for _, l := range lasting {
			fmt.Fprintf(&b, "  - %s\n", l)
		}
	}
	if changes := planChanges(p, seen, 4); len(changes) > 0 {
		b.WriteString("Since you wrote it, these happened to things it mentions:\n")
		for _, o := range changes {
			fmt.Fprintf(&b, "  [%s] %s\n", o.Stamp, oneLine(o.Text, 200))
		}
		b.WriteString("Steps that rested on them may now be done, pointless, or still worth doing; that is yours to judge.\n")
	}
	if planStuck(p) {
		b.WriteString("What stands in your way has not moved through several of your decisions. Whether to keep at it the same way, wait, come at it some other way, or put it aside is up to you.\n")
	}
	return b.String()
}
