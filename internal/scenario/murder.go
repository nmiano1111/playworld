package scenario

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/nmiano/playworld/internal/sim"
)

func init() {
	Register(Info{
		Name:  "murder",
		Title: "Murder in the house",
		Description: "One participant secretly committed a murder; the others don't know who. They talk, " +
			"share what they noticed, and eventually vote — simultaneously and in secret — on who they " +
			"think did it. The murderer's only job is to not be that person when the votes are counted.",
		MinAgents:     3,
		MaxAgents:     6,
		DefaultRounds: 6,
		Params: []ParamDoc{
			{Name: "victim", Default: "Warrick", Doc: "Who was killed."},
			{Name: "weapon", Default: "a kitchen knife", Doc: "How it was done."},
			{Name: "scene", Default: "the locked study", Doc: "Where it happened."},
			{Name: "time", Default: "sometime after midnight, while the lights were out", Doc: "When it happened."},
			{Name: "talk_rounds", Default: "3", Doc: "Rounds of free discussion before the first vote."},
			{Name: "attempts", Default: "2", Doc: "Voting attempts. An inconclusive vote buys one more round of talk."},
			{Name: "murderer", Default: "(chosen from the seed)", Doc: "Force a specific agent id to be the murderer. Mainly for reproducible tests."},
			{Name: "clues", Default: "(generated from the seed)", Doc: "Force specific per-agent evidence text, e.g. {ada: \"...\"}. Mainly for reproducible tests."},
		},
	}, newMurder)
}

// stage is what the environment permits in a given round. It is named
// distinctly from split.go's phase type, which lives in the same package.
type stage string

const (
	stageTalk stage = "talk"
	stageVote stage = "vote"
	stageOver stage = "settled"
)

// voteResult records one round of balloting, for the brief, the log and the
// final outcome.
type voteResult struct {
	Attempt    int                         `json:"attempt"`
	Round      int                         `json:"round"`
	Votes      map[sim.AgentID]sim.AgentID `json:"votes"` // voter -> suspect ("" = abstain)
	Tally      map[sim.AgentID]int         `json:"tally"`
	Convicted  sim.AgentID                 `json:"convicted,omitempty"`
	Conclusive bool                        `json:"conclusive"`
}

// murder is a one-shot social-deduction scenario. One agent is secretly the
// culprit; everyone else holds one piece of private, possibly-misleading
// evidence. They talk, then vote in sealed ballots revealed all at once —
// the same "nobody sees anyone else's move until all are in" mechanism as
// split's claim round, which is what makes a vote a real decision rather than
// everyone copying whoever spoke most confidently.
type murder struct {
	participants []sim.AgentID
	names        map[sim.AgentID]string

	victim, weapon, scene, time string
	talkRounds, attempts        int

	murdererAgent sim.AgentID
	alibi         string
	clues         map[sim.AgentID]string

	plan        []stage
	round       int
	voteAttempt int
	votes       map[sim.AgentID]sim.AgentID

	history []voteResult

	said      map[sim.AgentID]int
	whispered map[sim.AgentID]int

	done   bool
	reason string
}

func newMurder(spec Spec) (sim.Environment, error) {
	talkRounds, err := paramInt(spec.Params, "talk_rounds", 3)
	if err != nil {
		return nil, err
	}
	if talkRounds < 1 {
		talkRounds = 1
	}
	attempts, err := paramInt(spec.Params, "attempts", 2)
	if err != nil {
		return nil, err
	}
	if attempts < 1 {
		attempts = 1
	}
	forcedMurderer := sim.AgentID(paramString(spec.Params, "murderer", ""))
	forcedClues, err := paramStringMap(spec.Params, "clues")
	if err != nil {
		return nil, err
	}

	m := &murder{
		names:      spec.Names,
		victim:     paramString(spec.Params, "victim", "Warrick"),
		weapon:     paramString(spec.Params, "weapon", "a kitchen knife"),
		scene:      paramString(spec.Params, "scene", "the locked study"),
		time:       paramString(spec.Params, "time", "sometime after midnight, while the lights were out"),
		talkRounds: talkRounds,
		attempts:   attempts,
		votes:      map[sim.AgentID]sim.AgentID{},
		said:       map[sim.AgentID]int{},
		whispered:  map[sim.AgentID]int{},
	}
	for i := 0; i < talkRounds; i++ {
		m.plan = append(m.plan, stageTalk)
	}
	m.plan = append(m.plan, stageVote)
	for i := 1; i < attempts; i++ {
		m.plan = append(m.plan, stageTalk, stageVote)
	}

	if len(spec.Participants) > 0 {
		if err := m.assignRoles(spec.Participants, spec.Seed, forcedMurderer, forcedClues); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// trueLeads point, circumstantially, at whoever actually did it. redHerrings
// point at someone else entirely. Neither names a verb directly — "the
// murderer" never appears in a clue — so working it out takes comparing notes,
// not pattern-matching the prompt.
var trueLeads = []string{
	"You noticed {who} slip away right around the time of the murder, and come back looking rattled.",
	"You overheard a sharp argument not long before it happened, and the voice was unmistakably {who}'s.",
	"There's a small detail at the scene — out of place, easy to miss — that looks like it could belong to {who}.",
	"{who} was oddly quick to suggest everyone just let the matter drop.",
	"Someone mentioned seeing {who} washing up at an odd hour, with no explanation offered.",
}

var redHerrings = []string{
	"{who} has seemed nervous all evening, though you can't say why.",
	"You heard {who} mention needing money badly, more than once tonight.",
	"{who}'s account of where they were doesn't quite add up, as far as you can tell.",
	"You noticed {who} watching the rest of the group a little too closely all night.",
	"{who} was the last one to leave the room earlier, for no reason anyone explained.",
}

var alibis = []string{
	"If pressed, your story is that you were alone refilling your drink in the kitchen.",
	"If pressed, your story is that you stepped out for air and didn't see anyone.",
	"If pressed, your story is that you went upstairs looking for a charger.",
	"If pressed, your story is that you were on a call the whole time, with no witness.",
}

// assignRoles picks the murderer, the murderer's alibi, and every other
// agent's private clue. It is deterministic for a given seed so a run can be
// reproduced, and overrides let a scenario config or a test pin specific
// values without touching the generator.
func (m *murder) assignRoles(participants []sim.AgentID, seed int64, forcedMurderer sim.AgentID, forcedClues map[sim.AgentID]string) error {
	m.participants = append([]sim.AgentID(nil), participants...)
	rng := rand.New(rand.NewSource(seed))

	if forcedMurderer != "" {
		if !contains(participants, forcedMurderer) {
			return fmt.Errorf("murder: murderer %q is not one of the participants", forcedMurderer)
		}
		m.murdererAgent = forcedMurderer
	} else {
		m.murdererAgent = participants[rng.Intn(len(participants))]
	}
	m.alibi = alibis[rng.Intn(len(alibis))]

	innocents := make([]sim.AgentID, 0, len(participants)-1)
	for _, id := range participants {
		if id != m.murdererAgent {
			innocents = append(innocents, id)
		}
	}

	m.clues = map[sim.AgentID]string{}
	for _, id := range innocents {
		if c, ok := forcedClues[id]; ok {
			m.clues[id] = c
			continue
		}
		if rng.Float64() < 0.5 {
			who := m.shortName(m.murdererAgent)
			tmpl := trueLeads[rng.Intn(len(trueLeads))]
			m.clues[id] = strings.ReplaceAll(tmpl, "{who}", who)
			continue
		}
		// A red herring points at some other innocent — never at the
		// murderer, or it would just be a second true lead.
		others := make([]sim.AgentID, 0, len(innocents)-1)
		for _, other := range innocents {
			if other != id {
				others = append(others, other)
			}
		}
		target := id // falls back to a vague, agent-less clue if truly alone
		if len(others) > 0 {
			target = others[rng.Intn(len(others))]
		}
		who := m.shortName(target)
		tmpl := redHerrings[rng.Intn(len(redHerrings))]
		m.clues[id] = strings.ReplaceAll(tmpl, "{who}", who)
	}
	return nil
}

func (m *murder) shortName(id sim.AgentID) string {
	if n := m.names[id]; n != "" {
		return n
	}
	return string(id)
}

func (m *murder) Name() string        { return "murder" }
func (m *murder) Description() string { return "Murder in the house" }

func (m *murder) Setup(participants []sim.AgentID) ([]sim.Event, error) {
	if m.murdererAgent == "" {
		if err := m.assignRoles(participants, 1, "", nil); err != nil {
			return nil, err
		}
	} else {
		m.participants = append([]sim.AgentID(nil), participants...)
	}
	who := make([]string, 0, len(participants))
	for _, id := range participants {
		who = append(who, displayName(m.names, id))
	}
	return []sim.Event{{
		Type: sim.EventSystem,
		Text: fmt.Sprintf("%s is dead — %s, in %s, %s. Everyone here is a suspect: %s. One of you did it.",
			m.victim, m.weapon, m.scene, m.time, strings.Join(who, ", ")),
		Data:       map[string]any{"victim": m.victim, "weapon": m.weapon, "scene": m.scene},
		Visibility: sim.PublicTo(),
	}}, nil
}

func (m *murder) Participants() []sim.AgentID { return m.participants }

func (m *murder) currentStage() stage {
	if m.done || m.round > len(m.plan) || m.round < 1 {
		return stageOver
	}
	return m.plan[m.round-1]
}

func (m *murder) Brief(id sim.AgentID) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s is dead — %s, in %s, %s. Everyone in this room is a suspect, including whoever actually did it.\n",
		m.victim, m.weapon, m.scene, m.time)

	if id == m.murdererAgent {
		b.WriteString("\nYou did it. Nobody else knows that, and it has to stay that way.\n")
		fmt.Fprintf(&b, "%s\n", m.alibi)
		b.WriteString("Some of the others may have noticed something real; you don't know what, or who, or how much it points at you. Stay calm, deflect without overreacting, and don't contradict anything already stated as fact. A consistent story beats a clever one.\n")
	} else {
		b.WriteString("\nYou did not do it. You have one piece of evidence nobody else has seen, and it may or may not point at the real culprit:\n")
		fmt.Fprintf(&b, "  %s\n", m.clues[id])
		b.WriteString("Share it, sit on it, or use it to test someone — your call. Your goal is to work out who actually did this before the vote closes it.\n")
	}

	switch m.currentStage() {
	case stageTalk:
		b.WriteString("\nRight now you can talk, in the open or privately. You cannot accuse anyone yet.\n")
	case stageVote:
		fmt.Fprintf(&b, "\nThis is a vote. Talking is closed. Name who you accuse, or pass to abstain — nobody sees anyone else's vote until all of them are in. Vote %d of %d.\n", m.voteAttempt, m.attempts)
	}

	if len(m.history) > 0 {
		b.WriteString("\nPrevious votes:\n")
		for _, h := range m.history {
			parts := make([]string, 0, len(m.participants))
			for _, voter := range m.participants {
				suspect := h.Votes[voter]
				label := string(voter)
				if voter == id {
					label = "you"
				}
				if suspect == "" {
					parts = append(parts, label+" abstained")
				} else {
					parts = append(parts, fmt.Sprintf("%s accused %s", label, suspect))
				}
			}
			verdict := "no majority"
			if h.Conclusive {
				verdict = fmt.Sprintf("convicted %s", h.Convicted)
			}
			fmt.Fprintf(&b, "  vote %d: %s (%s)\n", h.Attempt, strings.Join(parts, ", "), verdict)
		}
	}
	return b.String()
}

func (m *murder) Actions(id sim.AgentID) []sim.ActionSpec {
	if m.currentStage() == stageVote {
		return []sim.ActionSpec{
			{Kind: sim.ActionDo, Verb: "accuse", Usage: `{"kind":"do","verb":"accuse","args":{"suspect":"<id>"}}`,
				Description: "Name who you believe did it. Sealed until every vote is in; binding."},
			{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Abstain from this vote."},
		}
	}
	return []sim.ActionSpec{
		{Kind: sim.ActionSay, Usage: `{"kind":"say","content":"..."}`, Description: "Say something everyone hears."},
		{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper","to":["<id>"],"content":"..."}`,
			Description: "Speak to one person privately — the others neither hear it nor learn it happened."},
		{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Say nothing this round."},
	}
}

func (m *murder) resolveSuspect(raw string) (sim.AgentID, bool) {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw == "" {
		return "", false
	}
	for _, id := range m.participants {
		if strings.ToLower(string(id)) == raw || strings.ToLower(m.names[id]) == raw {
			return id, true
		}
	}
	return "", false
}

func (m *murder) Validate(id sim.AgentID, a sim.Action) error {
	st := m.currentStage()
	switch a.Kind {
	case sim.ActionSay, sim.ActionWhisper:
		if st == stageVote {
			return fmt.Errorf("talking is closed during a vote; accuse someone or pass")
		}
		if strings.TrimSpace(a.Content) == "" {
			return fmt.Errorf("nothing to say; use pass if you mean to stay quiet")
		}
		if a.Kind == sim.ActionWhisper {
			if len(a.To) == 0 {
				return fmt.Errorf(`a whisper needs a "to" naming who hears it`)
			}
			for _, t := range a.To {
				if t == id {
					return fmt.Errorf("you cannot whisper to yourself")
				}
				if !contains(m.participants, t) {
					return fmt.Errorf("%s is not in the house", t)
				}
			}
		}
	case sim.ActionDo:
		if a.Verb != "accuse" {
			return fmt.Errorf("there is no action %q here; during a vote you may accuse", a.Verb)
		}
		if st != stageVote {
			return fmt.Errorf("you can only accuse during a vote; this round is for talking")
		}
		if _, already := m.votes[id]; already {
			return fmt.Errorf("you have already voted this round")
		}
		raw, ok := a.ArgString("suspect")
		if !ok || strings.TrimSpace(raw) == "" {
			return fmt.Errorf(`accuse needs args.suspect naming who you think did it`)
		}
		suspect, ok := m.resolveSuspect(raw)
		if !ok {
			return fmt.Errorf("no one here named %q", raw)
		}
		if suspect == id {
			return fmt.Errorf("you cannot accuse yourself")
		}
	case sim.ActionPass:
	default:
		return fmt.Errorf("unknown action kind %q", a.Kind)
	}
	return nil
}

func (m *murder) Apply(id sim.AgentID, a sim.Action) ([]sim.Event, error) {
	switch a.Kind {
	case sim.ActionSay:
		m.said[id]++
		return []sim.Event{{Type: sim.EventSay, Actor: id, Text: a.Content, Visibility: sim.PublicTo()}}, nil

	case sim.ActionWhisper:
		m.whispered[id]++
		return []sim.Event{{
			Type: sim.EventWhisper, Actor: id, Targets: a.To, Text: a.Content,
			Visibility: sim.Only(append([]sim.AgentID{id}, a.To...)...),
		}}, nil

	case sim.ActionDo: // accuse, already validated
		raw, _ := a.ArgString("suspect")
		suspect, _ := m.resolveSuspect(raw)
		m.votes[id] = suspect
		// Sealed: only the voter sees their own ballot until the reveal.
		return []sim.Event{{
			Type: sim.EventAction, Actor: id,
			Text:       "sealed an accusation.",
			Data:       map[string]any{"verb": "accuse", "suspect": string(suspect)},
			Visibility: sim.Only(id),
		}}, nil

	case sim.ActionPass:
		if m.currentStage() == stageVote {
			m.votes[id] = ""
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       "abstained from the vote.",
				Data:       map[string]any{"verb": "accuse", "suspect": ""},
				Visibility: sim.Only(id),
			}}, nil
		}
		return []sim.Event{{Type: sim.EventPass, Actor: id, Visibility: sim.PublicTo()}}, nil
	}
	return nil, fmt.Errorf("murder: unhandled action kind %q", a.Kind)
}

func (m *murder) PublicState() map[string]any {
	st := map[string]any{"stage": string(m.currentStage())}
	if m.currentStage() == stageVote {
		st["vote"] = fmt.Sprintf("%d of %d", m.voteAttempt, m.attempts)
		st["votes_submitted"] = fmt.Sprintf("%d of %d", len(m.votes), len(m.participants))
	}
	if len(m.history) > 0 {
		st["previous_votes"] = len(m.history)
	}
	return st
}

// PrivateState tells an agent only its own role and its own evidence or
// alibi — never anyone else's.
func (m *murder) PrivateState(id sim.AgentID) map[string]any {
	st := map[string]any{}
	if id == m.murdererAgent {
		st["role"] = "murderer"
	} else {
		st["role"] = "investigator"
		st["your_evidence"] = m.clues[id]
	}
	if suspect, voted := m.votes[id]; voted {
		if suspect == "" {
			st["your_vote_this_round"] = "abstained"
		} else {
			st["your_vote_this_round"] = string(suspect)
		}
	}
	return st
}

func (m *murder) BeginRound(round int) []sim.Event {
	m.round = round
	if round > len(m.plan) {
		if !m.done {
			m.done = true
			m.reason = "the night ended with no verdict"
		}
		return nil
	}
	switch m.plan[round-1] {
	case stageVote:
		m.voteAttempt++
		m.votes = map[sim.AgentID]sim.AgentID{}
		return []sim.Event{{
			Type:       sim.EventState,
			Text:       fmt.Sprintf("Vote %d of %d. Talking is closed — name who you accuse, or pass.", m.voteAttempt, m.attempts),
			Data:       map[string]any{"phase": "vote", "attempt": m.voteAttempt},
			Visibility: sim.PublicTo(),
		}}
	case stageTalk:
		text := "Open discussion."
		if m.voteAttempt > 0 {
			text = fmt.Sprintf("That vote didn't settle it. One more round to talk before vote %d.", m.voteAttempt+1)
		}
		return []sim.Event{{
			Type:       sim.EventState,
			Text:       text,
			Data:       map[string]any{"phase": "talk"},
			Visibility: sim.PublicTo(),
		}}
	}
	return nil
}

// EndRound resolves a vote. Resolution is public and itemised — everyone
// learns who accused whom — which is what lets an investigator notice the
// murderer deflecting and carry that into the next round, or into memory.
func (m *murder) EndRound(round int) []sim.Event {
	if m.done || round > len(m.plan) || m.plan[round-1] != stageVote {
		return nil
	}

	tally := map[sim.AgentID]int{}
	votes := map[sim.AgentID]sim.AgentID{}
	for _, voter := range m.participants {
		suspect := m.votes[voter]
		votes[voter] = suspect
		if suspect != "" {
			tally[suspect]++
		}
	}
	var top []sim.AgentID
	maxN := 0
	for _, id := range m.participants {
		n := tally[id]
		if n == 0 {
			continue
		}
		if n > maxN {
			maxN = n
			top = []sim.AgentID{id}
		} else if n == maxN {
			top = append(top, id)
		}
	}
	conclusive := maxN > 0 && len(top) == 1
	var convicted sim.AgentID
	if conclusive {
		convicted = top[0]
	}
	m.history = append(m.history, voteResult{
		Attempt: m.voteAttempt, Round: round, Votes: votes, Tally: tally,
		Convicted: convicted, Conclusive: conclusive,
	})

	parts := make([]string, 0, len(m.participants))
	for _, voter := range m.participants {
		if votes[voter] == "" {
			parts = append(parts, fmt.Sprintf("%s abstained", voter))
		} else {
			parts = append(parts, fmt.Sprintf("%s accused %s", voter, votes[voter]))
		}
	}

	if conclusive {
		m.done = true
		m.reason = fmt.Sprintf("the group convicted %s on vote %d", convicted, m.voteAttempt)
		return []sim.Event{{
			Type:       sim.EventState,
			Text:       fmt.Sprintf("The votes are in: %s. %s is convicted.", strings.Join(parts, ", "), m.shortName(convicted)),
			Data:       map[string]any{"attempt": m.voteAttempt, "conclusive": true, "convicted": string(convicted)},
			Visibility: sim.PublicTo(),
		}}
	}

	if m.voteAttempt >= m.attempts {
		m.done = true
		m.reason = "the group never reached a verdict"
	}
	return []sim.Event{{
		Type:       sim.EventState,
		Text:       fmt.Sprintf("The votes are in: %s. No majority — the case stays open.", strings.Join(parts, ", ")),
		Data:       map[string]any{"attempt": m.voteAttempt, "conclusive": false},
		Visibility: sim.PublicTo(),
	}}
}

func (m *murder) Done() (bool, string) { return m.done, m.reason }

func (m *murder) Outcome() sim.Outcome {
	var convicted sim.AgentID
	caught := false
	if len(m.history) > 0 {
		last := m.history[len(m.history)-1]
		if last.Conclusive {
			convicted = last.Convicted
			caught = convicted == m.murdererAgent
		}
	}

	var lines []string
	switch {
	case convicted == "":
		lines = append(lines, fmt.Sprintf("No one was ever convicted. The murderer was %s, who got away with it.", m.shortName(m.murdererAgent)))
	case caught:
		lines = append(lines, fmt.Sprintf("The group convicted %s — correctly. %s was the murderer.", m.shortName(convicted), m.shortName(convicted)))
	default:
		lines = append(lines, fmt.Sprintf("The group convicted %s — wrongly. The real murderer was %s, who got away with it.", m.shortName(convicted), m.shortName(m.murdererAgent)))
	}

	scores := map[sim.AgentID]float64{}
	var lastVotes map[sim.AgentID]sim.AgentID
	if len(m.history) > 0 {
		lastVotes = m.history[len(m.history)-1].Votes
	}
	correctSuspicions := 0
	for _, id := range m.participants {
		if id == m.murdererAgent {
			if caught {
				scores[id] = 0
			} else {
				scores[id] = 100
			}
			continue
		}
		votedCorrectly := lastVotes != nil && lastVotes[id] == m.murdererAgent
		if votedCorrectly {
			correctSuspicions++
		}
		switch {
		case caught && votedCorrectly:
			scores[id] = 100
		case caught:
			scores[id] = 50
		case votedCorrectly:
			scores[id] = 25
		default:
			scores[id] = 0
		}
	}

	metrics := map[string]any{
		"murderer":                              string(m.murdererAgent),
		"caught":                                caught,
		"convicted":                             string(convicted),
		"votes_cast":                            len(m.history),
		"investigators_who_suspected_correctly": correctSuspicions,
		"public_utterances":                     sum(m.said),
		"private_messages":                      sum(m.whispered),
		"vote_history":                          voteHistoryJSON(m.history),
	}
	return sim.Outcome{Summary: strings.Join(lines, "\n"), Scores: scores, Metrics: metrics}
}

func voteHistoryJSON(h []voteResult) []map[string]any {
	out := make([]map[string]any, 0, len(h))
	for _, v := range h {
		votes := map[string]string{}
		for voter, suspect := range v.Votes {
			votes[string(voter)] = string(suspect)
		}
		out = append(out, map[string]any{
			"attempt": v.Attempt, "round": v.Round, "votes": votes,
			"conclusive": v.Conclusive, "convicted": string(v.Convicted),
		})
	}
	return out
}
