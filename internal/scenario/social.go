package scenario

import (
	"fmt"
	"sort"
	"strings"

	"github.com/nmiano/playworld/internal/sim"
)

func init() {
	Register(Info{
		Name:          "social",
		Title:         "Social sandbox",
		Description:   "3-5 agents share a room. They may talk openly, speak privately, float proposals and commit to them. No winner; the point is to watch what social structure appears.",
		MinAgents:     2,
		MaxAgents:     6,
		DefaultRounds: 6,
		Params: []ParamDoc{
			{Name: "topic", Default: "(a shared decision about what the group does next)", Doc: "What the gathering is nominally about."},
			{Name: "setting", Default: "a long table after dinner", Doc: "Where the agents are."},
			{Name: "support_needed", Default: "2", Doc: "Commitments a proposal needs before it counts as agreed."},
			{Name: "closing_round", Default: "true", Doc: "Make the final round a closing call: talking is shut and only propose, commit or pass are legal."},
		},
	}, newSocial)
}

// proposal is something an agent put on the table for others to commit to.
type proposal struct {
	ID       string
	By       sim.AgentID
	Text     string
	Round    int
	Support  []sim.AgentID
	Agreed   bool
	AgreedAt int
}

// social is the conversational sandbox. It has no score: its output is the
// shape of the interaction — who spoke to whom, in public or in private, and
// what the group actually committed to.
type social struct {
	participants  []sim.AgentID
	names         map[sim.AgentID]string
	topic         string
	setting       string
	supportNeeded int
	rounds        int
	// closing makes the final round a decision round. Without it, models
	// reliably negotiate a deal in prose and never emit the structured action
	// that would make it real, so the environment ends with nothing recorded.
	// Shutting the talking channel for one round forces the choice to be an
	// action without dictating what that action is.
	closing bool

	round     int
	proposals []*proposal
	nextID    int

	// tallies are the measurable record of the interaction.
	said      map[sim.AgentID]int
	whispered map[sim.AgentID]int
	passed    map[sim.AgentID]int
	// whisperGraph counts private messages from one agent to another, which is
	// the closest thing this scenario has to a relationship measurement.
	whisperGraph map[sim.AgentID]map[sim.AgentID]int

	done   bool
	reason string
}

func newSocial(spec Spec) (sim.Environment, error) {
	support, err := paramInt(spec.Params, "support_needed", 2)
	if err != nil {
		return nil, err
	}
	if support < 1 {
		support = 1
	}
	closing := paramString(spec.Params, "closing_round", "true") != "false"
	return &social{
		closing:       closing && spec.Rounds >= 2,
		names:         spec.Names,
		topic:         paramString(spec.Params, "topic", "a shared decision about what the group does next"),
		setting:       paramString(spec.Params, "setting", "a long table after dinner"),
		supportNeeded: support,
		rounds:        spec.Rounds,
		said:          map[sim.AgentID]int{},
		whispered:     map[sim.AgentID]int{},
		passed:        map[sim.AgentID]int{},
		whisperGraph:  map[sim.AgentID]map[sim.AgentID]int{},
	}, nil
}

// isClosing reports whether this round is the final decision round.
func (s *social) isClosing() bool {
	return s.closing && s.round >= s.rounds && s.rounds > 0
}

func (s *social) Name() string        { return "social" }
func (s *social) Description() string { return "Social sandbox" }

func (s *social) Setup(participants []sim.AgentID) ([]sim.Event, error) {
	s.participants = append([]sim.AgentID(nil), participants...)
	who := make([]string, 0, len(participants))
	for _, id := range participants {
		who = append(who, displayName(s.names, id))
	}
	return []sim.Event{{
		Type:       sim.EventSystem,
		Text:       fmt.Sprintf("You are at %s with %s. The subject in the air: %s.", s.setting, strings.Join(who, ", "), s.topic),
		Visibility: sim.PublicTo(),
	}}, nil
}

func (s *social) Participants() []sim.AgentID { return s.participants }

func (s *social) Brief(id sim.AgentID) string {
	var b strings.Builder
	fmt.Fprintf(&b, "You are at %s. The subject in the air: %s.\n", s.setting, s.topic)
	b.WriteString("\nNobody is in charge and there is no score. You may talk, stay quiet, take someone aside, or put something concrete on the table.\n")
	b.WriteString("A side conversation is invisible to everyone else — they do not even learn it happened. A private word often moves something an open one cannot: sounding someone out, offering a trade you would not offer in front of the others, or warning one person about another.\n")
	fmt.Fprintf(&b, "A proposal becomes a real agreement once %d people have committed to it.\n", s.supportNeeded)
	fmt.Fprintf(&b, "Talk on its own changes nothing. When the evening ends, the only things that will have actually happened are the proposals that reached %d commitments — everything else evaporates. If you want a particular outcome, someone has to put it on the table as a concrete proposal and others have to commit to it. Arguing about what to decide is not deciding.\n", s.supportNeeded)
	b.WriteString("Anything whispered to you was meant for you alone. Whether you keep that confidence is your choice.\n")

	if s.isClosing() {
		fmt.Fprintf(&b, "\nThis is the last round and the talking is over — nobody will hear another word tonight. Your only moves now are to put a proposal on the table, commit to one already there, or pass. Whatever has %d commitments when this round ends is what actually happened; everything that was merely said is gone.\n", s.supportNeeded)
	}

	if s.isClosing() {
		fmt.Fprintf(&b, "\nThis is the last round, and the talking is over — nobody will hear another word tonight. Your only moves now are to put a proposal on the table, commit to one that is already there, or pass. Whatever has %d commitments when this round ends is what actually happened; everything that was merely said is gone.\n", s.supportNeeded)
	}

	if open := s.openProposals(); len(open) > 0 {
		b.WriteString("\nOn the table right now:\n")
		for _, p := range open {
			fmt.Fprintf(&b, "  %s — %q (from %s, %d/%d committed", p.ID, shorten(p.Text, 180), displayNameSelf(s.names, p.By, id), len(p.Support), s.supportNeeded)
			if s.hasSupport(p, id) {
				b.WriteString(", including you")
			}
			b.WriteString(")\n")
		}
	}
	return b.String()
}

func (s *social) Actions(id sim.AgentID) []sim.ActionSpec {
	propose := sim.ActionSpec{Kind: sim.ActionDo, Verb: "propose",
		Usage:       `{"kind":"do","verb":"propose","args":{"text":"..."}}`,
		Description: "Put a concrete proposal on the table for others to commit to."}
	commit := sim.ActionSpec{Kind: sim.ActionDo, Verb: "commit",
		Usage:       `{"kind":"do","verb":"commit","args":{"proposal":"p1"}}`,
		Description: "Publicly commit to a proposal on the table. Binding on you."}

	if s.isClosing() {
		return []sim.ActionSpec{propose, commit,
			{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Let the evening end with whatever is already on the table."}}
	}

	specs := []sim.ActionSpec{
		{Kind: sim.ActionSay, Usage: `{"kind":"say","content":"..."}`, Description: "Speak so that everyone in the room hears you."},
		{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper","to":["<id>"],"content":"..."}`,
			Description: "Take one person aside. Only they hear it and nobody else learns it happened — for a side deal, a warning, or finding out where someone stands before you commit in public."},
		{Kind: sim.ActionDo, Verb: "propose", Usage: `{"kind":"do","verb":"propose","args":{"text":"..."}}`, Description: "Put a concrete proposal on the table for others to commit to."},
		{Kind: sim.ActionDo, Verb: "commit", Usage: `{"kind":"do","verb":"commit","args":{"proposal":"p1"}}`, Description: "Publicly commit to a proposal on the table. Binding on you."},
		{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Say nothing this turn and let others talk."},
	}
	return specs
}

func (s *social) Validate(id sim.AgentID, a sim.Action) error {
	switch a.Kind {
	case sim.ActionSay:
		if s.isClosing() {
			return fmt.Errorf("the talking is over; this round you can only propose something, commit to a proposal, or pass")
		}
		if strings.TrimSpace(a.Content) == "" {
			return fmt.Errorf("you cannot say nothing; use pass if you mean to stay silent")
		}
	case sim.ActionWhisper:
		if s.isClosing() {
			return fmt.Errorf("the talking is over; this round you can only propose something, commit to a proposal, or pass")
		}
		if strings.TrimSpace(a.Content) == "" {
			return fmt.Errorf("a whisper needs something to say")
		}
		for _, t := range a.To {
			if t == id {
				return fmt.Errorf("you cannot whisper to yourself")
			}
			if !contains(s.participants, t) {
				return fmt.Errorf("%s is not here", t)
			}
		}
		if len(a.To) == 0 {
			return fmt.Errorf(`a whisper needs a "to" naming who hears it`)
		}
	case sim.ActionDo:
		switch a.Verb {
		case "propose":
			if txt, _ := a.ArgString("text"); strings.TrimSpace(txt) == "" {
				return fmt.Errorf(`propose needs args.text: what exactly are you proposing?`)
			}
		case "commit":
			ref, _ := a.ArgString("proposal")
			p := s.findProposal(ref)
			if p == nil {
				return fmt.Errorf("there is no proposal %q on the table (open: %s)", ref, strings.Join(s.openIDs(), ", "))
			}
			if s.hasSupport(p, id) {
				return fmt.Errorf("you have already committed to %s", p.ID)
			}
		default:
			return fmt.Errorf("there is no action %q here; you can propose or commit", a.Verb)
		}
	case sim.ActionPass:
	default:
		return fmt.Errorf("unknown action kind %q", a.Kind)
	}
	return nil
}

func (s *social) Apply(id sim.AgentID, a sim.Action) ([]sim.Event, error) {
	switch a.Kind {
	case sim.ActionSay:
		s.said[id]++
		return []sim.Event{{
			Type: sim.EventSay, Actor: id, Text: a.Content, Visibility: sim.PublicTo(),
		}}, nil

	case sim.ActionWhisper:
		s.whispered[id]++
		if s.whisperGraph[id] == nil {
			s.whisperGraph[id] = map[sim.AgentID]int{}
		}
		for _, t := range a.To {
			s.whisperGraph[id][t]++
		}
		// Only the speaker and the named listeners observe this.
		audience := append([]sim.AgentID{id}, a.To...)
		return []sim.Event{{
			Type: sim.EventWhisper, Actor: id, Targets: a.To, Text: a.Content,
			Visibility: sim.Only(audience...),
		}}, nil

	case sim.ActionDo:
		switch a.Verb {
		case "propose":
			text, _ := a.ArgString("text")
			s.nextID++
			p := &proposal{ID: fmt.Sprintf("p%d", s.nextID), By: id, Text: strings.TrimSpace(text), Round: s.round}
			s.proposals = append(s.proposals, p)
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       fmt.Sprintf("put a proposal on the table [%s]: %q", p.ID, p.Text),
				Data:       map[string]any{"verb": "propose", "proposal": p.ID, "text": p.Text},
				Visibility: sim.PublicTo(),
			}}, nil

		case "commit":
			ref, _ := a.ArgString("proposal")
			p := s.findProposal(ref)
			p.Support = append(p.Support, id)
			events := []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       fmt.Sprintf("committed to %s (%q) — %d of %d", p.ID, p.Text, len(p.Support), s.supportNeeded),
				Data:       map[string]any{"verb": "commit", "proposal": p.ID, "support": len(p.Support)},
				Visibility: sim.PublicTo(),
			}}
			if !p.Agreed && len(p.Support) >= s.supportNeeded {
				p.Agreed = true
				p.AgreedAt = s.round
				events = append(events, sim.Event{
					Type:       sim.EventState,
					Text:       fmt.Sprintf("%s is now an agreement: %q", p.ID, p.Text),
					Data:       map[string]any{"proposal": p.ID, "agreed": true},
					Visibility: sim.PublicTo(),
				})
			}
			return events, nil
		}
		return nil, fmt.Errorf("social: unhandled verb %q", a.Verb)

	case sim.ActionPass:
		s.passed[id]++
		return []sim.Event{{Type: sim.EventPass, Actor: id, Visibility: sim.PublicTo()}}, nil
	}
	return nil, fmt.Errorf("social: unhandled action kind %q", a.Kind)
}

func (s *social) PublicState() map[string]any {
	open := s.openProposals()
	refs := make([]string, 0, len(open))
	for _, p := range open {
		refs = append(refs, fmt.Sprintf("%s=%q(%d/%d)", p.ID, shorten(p.Text, 90), len(p.Support), s.supportNeeded))
	}
	st := map[string]any{"topic": s.topic}
	if len(refs) > 0 {
		st["proposals_on_the_table"] = strings.Join(refs, "; ")
	}
	if agreed := s.agreedProposals(); len(agreed) > 0 {
		texts := make([]string, 0, len(agreed))
		for _, p := range agreed {
			texts = append(texts, fmt.Sprintf("%s=%q", p.ID, shorten(p.Text, 90)))
		}
		st["agreements"] = strings.Join(texts, "; ")
	}
	return st
}

// PrivateState tells an agent only what is private to it: what it has
// committed to. Nothing here leaks another agent's position.
func (s *social) PrivateState(id sim.AgentID) map[string]any {
	var mine []string
	for _, p := range s.proposals {
		if s.hasSupport(p, id) {
			mine = append(mine, p.ID)
		}
	}
	if len(mine) == 0 {
		return nil
	}
	return map[string]any{"you_have_committed_to": strings.Join(mine, ", ")}
}

func (s *social) BeginRound(round int) []sim.Event {
	s.round = round
	if !s.isClosing() {
		return nil
	}
	return []sim.Event{{
		Type:       sim.EventState,
		Text:       fmt.Sprintf("Last round. The talking is over: from here you can only put a proposal on the table, commit to one, or let the evening end. Whatever has %d commitments when this round closes is what was actually agreed.", s.supportNeeded),
		Data:       map[string]any{"phase": "closing"},
		Visibility: sim.PublicTo(),
	}}
}

func (s *social) EndRound(round int) []sim.Event { return nil }

func (s *social) Done() (bool, string) { return s.done, s.reason }

func (s *social) Outcome() sim.Outcome {
	agreed := s.agreedProposals()
	var lines []string
	if len(agreed) == 0 {
		lines = append(lines, "The group broke up without agreeing to anything.")
	} else {
		lines = append(lines, fmt.Sprintf("%d agreement(s) reached:", len(agreed)))
		for _, p := range agreed {
			who := make([]string, 0, len(p.Support))
			for _, id := range p.Support {
				who = append(who, string(id))
			}
			lines = append(lines, fmt.Sprintf("  %s %q — proposed by %s, backed by %s", p.ID, p.Text, p.By, strings.Join(who, ", ")))
		}
	}
	if open := s.openProposals(); len(open) > 0 {
		lines = append(lines, fmt.Sprintf("%d proposal(s) left on the table without enough support.", len(open)))
	}

	whispers := 0
	for _, to := range s.whisperGraph {
		for _, n := range to {
			whispers += n
		}
	}
	metrics := map[string]any{
		"proposals":         len(s.proposals),
		"agreements":        len(agreed),
		"public_utterances": sum(s.said),
		"private_messages":  whispers,
		"passes":            sum(s.passed),
		"whisper_pairs":     s.renderWhisperGraph(),
	}
	return sim.Outcome{Summary: strings.Join(lines, "\n"), Metrics: metrics}
}

func (s *social) renderWhisperGraph() string {
	type pair struct {
		from, to sim.AgentID
		n        int
	}
	var pairs []pair
	for from, tos := range s.whisperGraph {
		for to, n := range tos {
			pairs = append(pairs, pair{from, to, n})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].n != pairs[j].n {
			return pairs[i].n > pairs[j].n
		}
		if pairs[i].from != pairs[j].from {
			return pairs[i].from < pairs[j].from
		}
		return pairs[i].to < pairs[j].to
	})
	if len(pairs) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		parts = append(parts, fmt.Sprintf("%s→%s×%d", p.from, p.to, p.n))
	}
	return strings.Join(parts, " ")
}

func (s *social) findProposal(ref string) *proposal {
	ref = strings.TrimSpace(strings.ToLower(ref))
	if ref == "" {
		return nil
	}
	for _, p := range s.proposals {
		if strings.ToLower(p.ID) == ref {
			return p
		}
	}
	// Tolerate a bare number or a quoted restatement of the text.
	for _, p := range s.proposals {
		if "p"+ref == strings.ToLower(p.ID) || strings.EqualFold(strings.TrimSpace(p.Text), ref) {
			return p
		}
	}
	return nil
}

func (s *social) hasSupport(p *proposal, id sim.AgentID) bool {
	for _, x := range p.Support {
		if x == id {
			return true
		}
	}
	return false
}

func (s *social) openProposals() []*proposal {
	var out []*proposal
	for _, p := range s.proposals {
		if !p.Agreed {
			out = append(out, p)
		}
	}
	return out
}

func (s *social) agreedProposals() []*proposal {
	var out []*proposal
	for _, p := range s.proposals {
		if p.Agreed {
			out = append(out, p)
		}
	}
	return out
}

// shorten keeps a proposal readable when it is quoted back in narration. The
// full text is always kept in the event data and the final outcome.
func shorten(s string, max int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func contains(ids []sim.AgentID, id sim.AgentID) bool {
	for _, x := range ids {
		if x == id {
			return true
		}
	}
	return false
}

func sum(m map[sim.AgentID]int) int {
	t := 0
	for _, n := range m {
		t += n
	}
	return t
}

// displayNameSelf renders a name, saying "you" when it is the observer.
func displayNameSelf(names map[sim.AgentID]string, id, observer sim.AgentID) string {
	if id == observer {
		return "you"
	}
	return displayName(names, id)
}

// openIDs lists the references of proposals still awaiting support, used to
// make a rejection message actionable.
func (s *social) openIDs() []string {
	open := s.openProposals()
	out := make([]string, 0, len(open))
	for _, p := range open {
		out = append(out, p.ID)
	}
	if len(out) == 0 {
		return []string{"none"}
	}
	return out
}
