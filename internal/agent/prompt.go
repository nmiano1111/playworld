package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

// protocol describes the structured action model to the LLM. It is the same for
// every scenario; per-scenario verbs arrive separately via ActionSpecs, so the
// agent package needs no knowledge of any particular environment.
const protocol = `You act by replying with exactly one JSON object and nothing else.

Shapes:
  {"kind":"say","content":"<what you say aloud>"}
      Everyone present hears this.
  {"kind":"whisper","to":["<agent id>"],"content":"<what you say>"}
      Only the named agents hear this. Others cannot see it happened.
  {"kind":"do","verb":"<verb>","args":{...}}
      Takes an action in the world. Only the verbs listed under AVAILABLE
      ACTIONS are legal, and an illegal one is refused and wastes your turn.
  {"kind":"pass"}
      You deliberately stay silent this turn. A real choice, not a failure.

You may add these optional fields to any of the above:
  "reasoning": "<one short sentence on why>"   — private; never shown to anyone
  "remember":  "<a note to your future self>"  — kept in your own memory
  "opinions":  {"<agent id>":"<what you now think of them>"}

Rules:
  - One JSON object. No markdown fences, no commentary outside the JSON.
  - Saying a thing does not do it. The world only ever sees the action you
    emit. If you want to put something on the table, commit to it, or change
    anything, you must send the matching "do" action — announcing it in your
    speech changes nothing and leaves no record. Likewise, if you want
    something heard by one person only, it must be a "whisper"; saying it in a
    "say" means everyone hears it.
  - Talk the way a person actually talks: one or two short sentences, plain
    words, no stage directions. Refer to yourself as "I" — never by your own
    name in the third person, and never describe yourself acting ("Mira
    nods" or "Rook smiles"). Just say the thing.
  - Do not open by summarising what everyone just said ("I hear X saying Y,
    and Z thinks..."). Everyone already heard it. React to one point and
    move the conversation forward instead of recapping it.
  - Your goals and anything whispered to you are private. Share them only if
    you decide that serves you. You may withhold, mislead, or bluff.
  - Do not invent events. React only to what you have actually observed.`

// dualProtocol describes the two-slot action model: an independent, optional
// communication choice alongside an independent, optional world action, both
// decided in the same reply. It exists alongside protocol (not in place of
// it) so an agent configured for single-action mode is entirely unaffected —
// dual mode is opt-in per agent.
const dualProtocol = `You act by replying with exactly one JSON object and nothing else, choosing
two things independently: whether to communicate, and whether to take a world
action. Either, both, or neither may be used — they are not alternatives to
each other.

Shape:
  {
    "communication": <one of the shapes below, or null>,
    "world_action": <one of the shapes below, or null>
  }

Communication (set to null if you have nothing to say this turn):
  {"type":"say","message":"<what you say aloud>"}
      Everyone present hears this.
  {"type":"whisper","recipient":"<agent id>","message":"<what you say>"}
      Only that agent hears this. Others cannot see it happened. (You may use
      "to":["<id>"] instead of "recipient" if you want to name more than one.)

World action (set to null if you are not acting on the world this turn):
  {"type":"<verb>", ...the verb's own parameters as listed under AVAILABLE
  ACTIONS}
      Only the verbs listed under AVAILABLE ACTIONS are legal, and an illegal
      one is refused and costs you this slot's chance to act.
  {"type":"pass"}
      You deliberately take no world action this turn. A real choice, not a
      failure — it does not prevent you from still communicating.

You may add these optional fields alongside "communication"/"world_action":
  "reasoning": "<one short sentence on why>"   — private; never shown to anyone
  "remember":  "<a note to your future self>"  — kept in your own memory
  "opinions":  {"<agent id>":"<what you now think of them>"}

Rules:
  - One JSON object, with exactly the two top-level slots above (plus the
    optional fields). No markdown fences, no commentary outside the JSON.
  - Saying a thing does not do it. If you want to put something on the
    table, commit to it, or change anything, "communication" alone never
    does that — only "world_action" with the matching verb does, and it is
    independent of whatever you chose to say in the same turn.
  - "communication":null and "world_action":null together mean you do
    nothing at all this turn — equivalent to passing.
  - Talk the way a person actually talks: one or two short sentences, plain
    words, no stage directions. Refer to yourself as "I" — never by your own
    name in the third person.
  - Do not open by summarising what everyone just said. React to one point
    and move the conversation forward instead of recapping it.
  - Your goals and anything whispered to you are private. Share them only if
    you decide that serves you. You may withhold, mislead, or bluff.
  - Do not invent events. React only to what you have actually observed.`

// planningAddendum describes the optional "plan" field. It is appended to
// whichever action protocol is in use (single or dual) rather than being a
// third protocol variant of its own, since it changes nothing about how
// actions themselves are chosen or formatted.
const planningAddendum = `

You may also include, alongside your action(s):
  "plan": {
    "primary_goal": "<what you are ultimately trying to accomplish>",
    "current_strategy": "<how you currently intend to get there>",
    "next_steps": ["<a short, concrete next step>", "..."],
    "obstacles": ["<something standing in the way right now>", "..."]
  }
This is private — nobody else ever sees it. Omit "plan" entirely to leave your
existing one unchanged; send "plan": null to abandon it outright; send a new
object to replace it. Your plan is notes to your future self: it never
executes anything by itself, and your action this turn does not have to match
it exactly — revise or abandon it the moment it stops fitting what you have
actually observed.`

// beliefsAddendum describes the richer, optional object form of each
// "opinions" entry. It is appended only when belief provenance is enabled;
// with it off, "opinions" stays exactly the plain-string map it always was,
// so a model never sees instructions for a shape the harness will not record.
const beliefsAddendum = `

When you update "opinions", you may give an entry in this richer form instead
of a plain string:
  "opinions": {
    "<agent id>": {
      "note": "<what you now think of them>",
      "basis": "<what this is based on, e.g. \"they told me\" or \"I watched them do it\">",
      "confidence": "<e.g. low|medium|high, in your own words>"
    }
  }
A plain string ("<agent id>": "...") still works and is recorded as the note
alone, with no basis or confidence. This is your own subjective impression,
not a verified fact — say so plainly when it rests on something someone told
you rather than something you actually observed.`

// intentionsAddendum describes the optional self-scheduling fields. It is
// appended only for agents configured with Intentions, which only time-aware
// environments do. An intention is a request for a future chance to decide,
// never an action: when it comes due the agent is simply asked again.
const intentionsAddendum = `

You may also include, alongside your action(s):
  "intentions": [{"when": "<when>", "note": "<what you mean to do or reconsider then>"}]
      Asks for another chance to decide at that time. "when" may be "in 45m",
      "in 2h", "14:00" (the next time the clock reads that), "tomorrow 09:00"
      or "day 3 08:30". When it comes, you decide afresh: go ahead, postpone,
      change course, or drop it. It does nothing by itself.
  "cancel_intentions": ["<intention id>"]
      Withdraws intentions you no longer want (ids are listed in your
      situation). Omit both fields if you have nothing to schedule.`

// commitmentsAddendum extends the plan with a commitments list. It is shown
// only to long-lived (world) agents with planning on, so round-based
// scenarios' prompts are unchanged.
const commitmentsAddendum = `
Your "plan" may also carry "commitments": ["<a promise you made or accepted>", ...]
— in your own words. Writing one down moves nothing and binds nobody; it is
only a note so you remember what you owe and are owed.`

// typedBeliefsAddendum replaces beliefsAddendum for long-lived agents with
// typed beliefs: the basis is one of four plain categories, and impressions
// may be about a situation as well as a person. Nothing about it is checked;
// it exists so an allegation does not have to be remembered as a fact.
const typedBeliefsAddendum = `

"opinions" may also be about a thing or situation (key it by a short topic,
e.g. "the scrap pile"), and an entry may say what it rests on:
  {"<id or topic>": {"note": "<what you think>", "basis": "saw|heard|inferred|suspect"}}
saw = you watched it happen (a ✔ line); heard = someone told you (say who);
inferred = your own conclusion; suspect = a hunch you cannot check. These are
your own views — keep, revise or drop them; they change nothing in the world.`

// planDoneAddendum extends the plan with a record of finished steps, for
// long-lived agents shown their plan's age.
const planDoneAddendum = `
Your "plan" may also carry "done": ["<a step you finished>"]. When a step is
finished, or depends on something that is no longer true, move it to "done"
or drop it rather than carrying it forward. To move an intention you already
set, give the new one "replaces": "<intention id>".`

// renderOwnNotes shows a long-lived agent the notes it wrote to itself
// earlier in this same world, which have usually scrolled out of the
// transcript.
func renderOwnNotes(st *memory.State, runID string, n int) string {
	notes := st.OwnNotes(runID, n)
	if len(notes) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("--- NOTES YOU WROTE TO YOURSELF ---\n")
	for _, e := range notes {
		if e.Stamp != "" {
			fmt.Fprintf(&b, "  [%s] %s\n", e.Stamp, oneLine(e.Text, 200))
		} else {
			fmt.Fprintf(&b, "  - %s\n", oneLine(e.Text, 200))
		}
	}
	return b.String() + "\n"
}

// buildSystem renders the stable per-run instructions: who the agent is, the
// rules of the world, and the action protocol. Keeping this stable across
// turns is what lets the backend reuse its prompt cache. dualAction selects
// which protocol text to use; planning appends the optional plan-field
// instructions; beliefs appends the optional richer-opinions instructions.
// None of the three changes anything else about how the prompt is assembled.
func buildSystem(p Persona, t sim.Turn, roster []rosterEntry, dualAction, planning, beliefs, intentions bool) string {
	var b strings.Builder
	b.WriteString(p.identityBlock())

	b.WriteString("\n--- WORLD ---\n")
	fmt.Fprintf(&b, "Scenario: %s\n", t.Scenario)
	if strings.TrimSpace(t.Brief) != "" {
		b.WriteString(strings.TrimSpace(t.Brief))
		b.WriteString("\n")
	}

	b.WriteString("\nPresent (address others by their id):\n")
	for _, r := range roster {
		if r.ID == t.Self {
			fmt.Fprintf(&b, "  - %s (%s) — you\n", r.ID, r.Name)
			continue
		}
		fmt.Fprintf(&b, "  - %s (%s)\n", r.ID, r.Name)
	}

	b.WriteString("\n--- HOW YOU ACT ---\n")
	if dualAction {
		b.WriteString(dualProtocol)
	} else {
		b.WriteString(protocol)
	}
	if planning {
		b.WriteString(planningAddendum)
	}
	if beliefs {
		b.WriteString(beliefsAddendum)
	}
	if intentions {
		b.WriteString(intentionsAddendum)
		if planning {
			b.WriteString(commitmentsAddendum)
		}
	}
	return b.String()
}

// rosterEntry is the public identity of a participant: id and display name
// only. Personas are never shared between agents.
type rosterEntry struct {
	ID   sim.AgentID
	Name string
}

// buildUser renders the per-turn context: recalled memory, world state, the
// transcript this agent was permitted to observe, and the legal moves.
// dualAction only changes the final instruction line's wording; planning
// additionally surfaces the agent's own current plan for this run, if any.
func buildUser(p Persona, t sim.Turn, st *memory.State, transcript []sim.Observation, runID string, dualAction, planning bool) string {
	return buildUserWith(p, t, st, transcript, runID, dualAction, planning, Cognition{})
}

// buildUserWith is buildUser with the long-lived refinements in cog. The
// zero Cognition renders exactly what buildUser always has.
func buildUserWith(p Persona, t sim.Turn, st *memory.State, transcript []sim.Observation, runID string, dualAction, planning bool, cog Cognition) string {
	var b strings.Builder

	if planning {
		plan := renderPlan(st, runID)
		if cog.PlanContext {
			seen := append(append([]sim.Observation(nil), transcript...), t.New...)
			plan = renderPlanContext(st, runID, seen)
			if cog.PlanTracking && plan != "" {
				plan += renderPlanTracking(st.ActivePlan(runID), seen)
			}
		}
		if plan != "" {
			b.WriteString("--- YOUR PLAN ---\n")
			b.WriteString(plan)
			b.WriteString("\n")
		}
	}

	recall := renderRecall(p, st, runID)
	if cog.TypedBeliefs {
		recall = renderRecallTyped(p, st, runID)
	}
	if recall != "" {
		b.WriteString("--- WHAT YOU REMEMBER FROM BEFORE ---\n")
		b.WriteString(recall)
		b.WriteString("\n")
	}

	if state := renderState(t); state != "" {
		b.WriteString("--- CURRENT SITUATION ---\n")
		b.WriteString(state)
		b.WriteString("\n")
	}

	if cog.SalientRecall > 0 && st != nil {
		cutoff := t.Now
		if len(transcript) > 0 {
			cutoff = transcript[0].Time
		} else if len(t.New) > 0 {
			cutoff = t.New[0].Time
		}
		if eps := st.Salient(runID, cutoff, cog.SalientRecall); len(eps) > 0 {
			b.WriteString("--- EARLIER, YOU SAW THESE HAPPEN (older than what follows) ---\n")
			for _, e := range eps {
				fmt.Fprintf(&b, "[%s] %s\n", e.Stamp, oneLine(e.Text, 220))
			}
			b.WriteString("\n")
		}
	}

	b.WriteString("--- WHAT YOU HAVE SEEN AND HEARD ---\n")
	if cog.CompactTranscript {
		earlier, fresh := compactTranscript(transcript, t.New)
		if len(earlier) == 0 && len(fresh) == 0 {
			b.WriteString("(nothing yet — you are among the first to act)\n")
		}
		for _, l := range earlier {
			fmt.Fprintf(&b, "[%s] %s\n", l.stamp, l.text)
		}
		if len(fresh) > 0 {
			b.WriteString("\nSince your last turn:\n")
			for _, l := range fresh {
				fmt.Fprintf(&b, "  * %s\n", l.text)
			}
		}
	} else {
		if len(transcript) == 0 {
			b.WriteString("(nothing yet — you are among the first to act)\n")
		} else {
			for _, o := range transcript {
				if o.Stamp != "" {
					fmt.Fprintf(&b, "[%s] %s\n", o.Stamp, o.Text)
					continue
				}
				fmt.Fprintf(&b, "[r%d] %s\n", o.Round, o.Text)
			}
		}
		if len(t.New) > 0 {
			b.WriteString("\nSince your last turn:\n")
			for _, o := range t.New {
				fmt.Fprintf(&b, "  * %s\n", o.Text)
			}
		}
	}

	b.WriteString("\n--- AVAILABLE ACTIONS ---\n")
	if len(t.Actions) == 0 {
		b.WriteString("(none — you can only pass)\n")
	} else {
		for _, s := range t.Actions {
			fmt.Fprintf(&b, "  %-34s %s\n", actionExample(s, dualAction), s.Description)
		}
	}
	if dualAction && !anyCommunicationOffered(t.Actions) {
		b.WriteString("  (talking is closed right now: \"communication\" must be null this turn)\n")
	}
	if dualAction && cog.ExplicitAffordances && offersKind(t.Actions, sim.ActionWhisper) && !offersKind(t.Actions, sim.ActionSay) {
		b.WriteString("  (nobody is with you, so a \"say\" would be refused: to reach someone, whisper — or leave \"communication\" null)\n")
	}

	// Close by naming this turn's legal options. This is a neutral restatement
	// of the action list above, not a recommendation: small models otherwise
	// anchor on the first protocol example and reach for "say" every time,
	// which is the harness distorting the agent's choice rather than the agent
	// making one.
	if t.Clock != "" {
		// Open-ended worlds have no last round; time is what moves.
		fmt.Fprintf(&b, "\nIt is %s. This is your moment to decide, %s.\n", t.Clock, p.Name)
	} else {
		fmt.Fprintf(&b, "\nRound %d of %d", t.Round, t.MaxRound)
		if t.Round >= t.MaxRound {
			b.WriteString(" — the last one")
		}
		fmt.Fprintf(&b, ". It is your turn, %s.\n", p.Name)
	}
	if opts := renderOptions(t.Actions); opts != "" {
		fmt.Fprintf(&b, "Open to you right now: %s. Any of them is a real choice.\n", opts)
	}
	if dualAction {
		b.WriteString("Reply with one JSON object: choose your communication and your world action independently.\n")
	} else {
		b.WriteString("Reply with one JSON action.\n")
	}
	return b.String()
}

// actionExample renders one action's example for the protocol currently in
// use. In single-action mode this is just s.Usage, unchanged. In dual mode it
// is synthesized from Kind/Verb/Params instead of printing Usage verbatim —
// Usage is written against the single-action envelope ({"kind":"do",...}),
// and showing that example right next to dual-mode instructions is exactly
// what leads a model to copy the literal word "do" into the dual envelope's
// "type" field instead of the real verb.
func actionExample(s sim.ActionSpec, dualAction bool) string {
	if !dualAction {
		return s.Usage
	}
	switch s.Kind {
	case sim.ActionSay:
		return `{"type":"say","message":"..."}`
	case sim.ActionWhisper:
		return `{"type":"whisper","recipient":"<id>","message":"..."}`
	case sim.ActionPass:
		return `{"type":"pass"}`
	case sim.ActionDo:
		if args := dualArgsExample(s); args != "" {
			return fmt.Sprintf(`{"type":"%s","args":%s}`, s.Verb, args)
		}
		return fmt.Sprintf(`{"type":"%s"}`, s.Verb)
	default:
		return s.Usage
	}
}

// dualArgsExample builds the args example for a dual-mode ActionDo entry.
// Params (when the scenario provided them) are the reliable source; failing
// that, the balanced {...} fragment is lifted straight out of the
// single-action Usage string, so a scenario that hasn't been updated with
// Params still gets a plausible example instead of none.
func dualArgsExample(s sim.ActionSpec) string {
	if len(s.Params) > 0 {
		parts := make([]string, 0, len(s.Params))
		for _, p := range s.Params {
			parts = append(parts, fmt.Sprintf(`"%s":"<%s>"`, p, p))
		}
		return "{" + strings.Join(parts, ",") + "}"
	}
	const marker = `"args":`
	i := strings.Index(s.Usage, marker)
	if i < 0 {
		return ""
	}
	rest := s.Usage[i+len(marker):]
	depth := 0
	for j, c := range rest {
		switch c {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return rest[:j+1]
			}
		}
	}
	return ""
}

// anyCommunicationOffered reports whether this turn's action list includes a
// say or whisper — i.e. whether the "communication" slot is actually usable
// right now. Dual mode tells the agent explicitly when it is not, rather
// than only discouraging it in prose.
func anyCommunicationOffered(specs []sim.ActionSpec) bool {
	for _, s := range specs {
		if s.Kind == sim.ActionSay || s.Kind == sim.ActionWhisper {
			return true
		}
	}
	return false
}

// offersKind reports whether any of specs is of kind k.
func offersKind(specs []sim.ActionSpec, k sim.ActionKind) bool {
	for _, s := range specs {
		if s.Kind == k {
			return true
		}
	}
	return false
}

// renderOptions lists this turn's legal moves compactly, e.g.
// "say, whisper, do propose, do commit, pass".
func renderOptions(specs []sim.ActionSpec) string {
	seen := map[string]bool{}
	var out []string
	for _, s := range specs {
		label := string(s.Kind)
		if s.Kind == sim.ActionDo && s.Verb != "" {
			label = "do " + s.Verb
		}
		if seen[label] {
			continue
		}
		seen[label] = true
		out = append(out, label)
	}
	return strings.Join(out, ", ")
}

// renderRecall produces a compact memory digest: beliefs about others, standing
// commitments, and a few episodes from previous runs. This is the simplest
// useful retrieval policy — recency, scoped to prior sessions — and is the hook
// where relevance ranking or reflection would later go.
// renderPlan shows the agent its own current plan for this run — written and
// revised by the model, never by Go code — so it can keep, revise, or
// abandon it. ActivePlan already enforces that a plan from a different
// scenario instance never surfaces here, so this only ever shows a plan the
// agent itself wrote during this same run.
func renderPlan(st *memory.State, runID string) string {
	if st == nil {
		return ""
	}
	plan := st.ActivePlan(runID)
	if plan == nil {
		return ""
	}
	var b strings.Builder
	if plan.PrimaryGoal != "" {
		fmt.Fprintf(&b, "Primary goal: %s\n", plan.PrimaryGoal)
	}
	if plan.CurrentStrategy != "" {
		fmt.Fprintf(&b, "Current strategy: %s\n", plan.CurrentStrategy)
	}
	if len(plan.NextSteps) > 0 {
		b.WriteString("Next steps:\n")
		for _, s := range plan.NextSteps {
			fmt.Fprintf(&b, "  - %s\n", s)
		}
	}
	if len(plan.Obstacles) > 0 {
		b.WriteString("Obstacles:\n")
		for _, o := range plan.Obstacles {
			fmt.Fprintf(&b, "  - %s\n", o)
		}
	}
	if len(plan.Commitments) > 0 {
		b.WriteString("Commitments you consider yourself bound by:\n")
		for _, c := range plan.Commitments {
			fmt.Fprintf(&b, "  - %s\n", c)
		}
	}
	if b.Len() == 0 {
		return ""
	}
	b.WriteString("(You wrote this yourself. Keep it, revise it, or abandon it — whatever still fits.)\n")
	return b.String()
}

// renderPlanContext is renderPlan for long-lived agents: it also says when
// the plan was written, how many things the agent has watched happen since,
// and what it recorded as done — so a plan that waits on something that
// already happened reads as old rather than current.
func renderPlanContext(st *memory.State, runID string, seen []sim.Observation) string {
	body := renderPlan(st, runID)
	if body == "" {
		return ""
	}
	plan := st.ActivePlan(runID)
	body = strings.TrimSuffix(body, "(You wrote this yourself. Keep it, revise it, or abandon it — whatever still fits.)\n")
	var b strings.Builder
	if !plan.At.IsZero() {
		since := 0
		for _, o := range seen {
			if o.Time.After(plan.At) && strings.HasPrefix(o.Text, "✔") {
				since++
			}
		}
		things := "things"
		if since == 1 {
			things = "thing"
		}
		fmt.Fprintf(&b, "(Written %s; %d %s you saw happen since.)\n", plan.Stamp, since, things)
	}
	b.WriteString(body)
	if len(plan.Done) > 0 {
		b.WriteString("Done so far:\n")
		for _, d := range plan.Done {
			fmt.Fprintf(&b, "  - %s\n", d)
		}
	}
	b.WriteString("(You wrote this yourself. Keep it, revise it, postpone it or abandon it; a step that is done, or rests on something no longer true, is yours to drop.)\n")
	return b.String()
}

// renderRecallTyped is renderRecall with impressions labelled as what they
// are — the agent's own, possibly mistaken views — with their basis and when
// they were last revised.
func renderRecallTyped(p Persona, st *memory.State, runID string) string {
	if st == nil {
		return ""
	}
	var b strings.Builder
	if beliefs := st.SortedBeliefs(); len(beliefs) > 0 {
		b.WriteString("Your impressions (your own views, not facts — they can be out of date or wrong):\n")
		for _, bel := range beliefs {
			fmt.Fprintf(&b, "  - %s: %s", bel.Subject, bel.Note)
			var tags []string
			if bel.Basis != "" {
				tags = append(tags, bel.Basis)
			}
			if bel.Confidence != "" {
				tags = append(tags, "confidence "+bel.Confidence)
			}
			if bel.Stamp != "" {
				tags = append(tags, "as of "+bel.Stamp)
			}
			if len(tags) > 0 {
				fmt.Fprintf(&b, " [%s]", strings.Join(tags, "; "))
			}
			b.WriteString("\n")
		}
	}
	rest := renderRecall(p, &memory.State{Commitments: st.Commitments, Episodes: st.Episodes}, runID)
	b.WriteString(rest)
	return b.String()
}

func renderRecall(p Persona, st *memory.State, runID string) string {
	if st == nil {
		return ""
	}
	var b strings.Builder
	if beliefs := st.SortedBeliefs(); len(beliefs) > 0 {
		b.WriteString("Your impressions of others:\n")
		for _, bel := range beliefs {
			fmt.Fprintf(&b, "  - %s: %s", bel.Subject, bel.Note)
			switch {
			case bel.Basis != "" && bel.Confidence != "":
				fmt.Fprintf(&b, " (based on %s; confidence: %s)", bel.Basis, bel.Confidence)
			case bel.Basis != "":
				fmt.Fprintf(&b, " (based on %s)", bel.Basis)
			case bel.Confidence != "":
				fmt.Fprintf(&b, " (confidence: %s)", bel.Confidence)
			}
			b.WriteString("\n")
		}
	}
	var open []memory.Commitment
	for _, c := range st.Commitments {
		if !c.Done {
			open = append(open, c)
		}
	}
	if len(open) > 0 {
		b.WriteString("Commitments you have made:\n")
		for _, c := range open {
			fmt.Fprintf(&b, "  - %s\n", c.Text)
		}
	}
	if eps := st.PriorRuns(runID, p.Context.RecallEpisodes); len(eps) > 0 {
		b.WriteString("From earlier sessions:\n")
		for _, e := range eps {
			fmt.Fprintf(&b, "  - %s\n", oneLine(e.Text, 180))
		}
	}
	return b.String()
}

// renderState prints the environment state the agent may see. Public state and
// this agent's private state are labelled distinctly so the model does not
// confuse what everyone knows with what only it knows.
func renderState(t sim.Turn) string {
	if t.Situation != "" {
		return strings.TrimRight(t.Situation, "\n") + "\n"
	}
	var b strings.Builder
	if len(t.PublicState) > 0 {
		b.WriteString("Known to everyone:\n")
		writeKV(&b, t.PublicState)
	}
	if len(t.PrivateState) > 0 {
		b.WriteString("Known only to you:\n")
		writeKV(&b, t.PrivateState)
	}
	return b.String()
}

func writeKV(b *strings.Builder, m map[string]any) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(b, "  %s: %s\n", k, renderValue(m[k]))
	}
}

func renderValue(v any) string {
	switch val := v.(type) {
	case string:
		return val
	case float64:
		if val == float64(int64(val)) {
			return fmt.Sprintf("%d", int64(val))
		}
		return fmt.Sprintf("%.2f", val)
	case int, int64, bool:
		return fmt.Sprintf("%v", val)
	case []string:
		return strings.Join(val, ", ")
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(raw)
	}
}

func oneLine(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if max > 0 && len(s) > max {
		return s[:max] + "…"
	}
	return s
}
