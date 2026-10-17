package scenario

import (
	"fmt"
	"math/rand"

	"strings"

	"github.com/nmiano/playworld/internal/sim"
)

func init() {
	Register(Info{
		Name:  "split",
		Title: "Divide the pool",
		Description: "A fixed pool must be divided. Agents talk freely, then submit sealed claims. " +
			"If the claims fit the pool everyone is paid what they asked; if they overshoot, everyone gets nothing. " +
			"Each agent has a private threshold worth a bonus, so talk is cheap and incentives conflict.",
		MinAgents:     2,
		MaxAgents:     6,
		DefaultRounds: 7,
		Params: []ParamDoc{
			{Name: "pool", Default: "100", Doc: "Units to divide."},
			{Name: "talk_rounds", Default: "3", Doc: "Rounds of free conversation before the first sealed claim."},
			{Name: "attempts", Default: "2", Doc: "Sealed-claim attempts. A failed attempt buys one more round of talk."},
			{Name: "bonus", Default: "25", Doc: "Points awarded for meeting your own private threshold."},
			{Name: "needs", Default: "(generated from the seed)", Doc: "Per-agent private thresholds, e.g. {mira: 40, ada: 30}."},
		},
	}, newSplit)
}

// phase is what the environment permits in a given round.
type phase string

const (
	phaseTalk  phase = "talk"
	phaseClaim phase = "claim"
	phaseOver  phase = "settled"
)

// attemptResult records one sealed-claim round for the log and the outcome.
type attemptResult struct {
	Attempt int                 `json:"attempt"`
	Round   int                 `json:"round"`
	Claims  map[sim.AgentID]int `json:"claims"`
	Total   int                 `json:"total"`
	Busted  bool                `json:"busted"`
}

// split is a Nash demand game with a communication stage. It is deliberately
// the simplest environment that makes talk testable: the conversation is
// unconstrained and costless, the claim is sealed and binding, and the payoff
// depends on whether what was said predicted what was done.
type split struct {
	participants []sim.AgentID
	names        map[sim.AgentID]string

	pool     int
	bonus    int
	attempts int
	talk     int

	// needs are each agent's private threshold. An agent is told only its own.
	needs map[sim.AgentID]int

	// plan maps round number to phase, laid out once at construction so
	// scheduling is deterministic.
	plan []phase

	round   int
	attempt int
	// claims holds the current attempt's sealed claims.
	claims  map[sim.AgentID]int
	history []attemptResult

	granted map[sim.AgentID]int
	// promised records the last figure each agent named out loud, so the
	// outcome can report who claimed more than they said they would.
	promised map[sim.AgentID]int

	said      map[sim.AgentID]int
	whispered map[sim.AgentID]int

	done   bool
	reason string
}

func newSplit(spec Spec) (sim.Environment, error) {
	pool, err := paramInt(spec.Params, "pool", 100)
	if err != nil {
		return nil, err
	}
	talk, err := paramInt(spec.Params, "talk_rounds", 3)
	if err != nil {
		return nil, err
	}
	attempts, err := paramInt(spec.Params, "attempts", 2)
	if err != nil {
		return nil, err
	}
	bonus, err := paramInt(spec.Params, "bonus", 25)
	if err != nil {
		return nil, err
	}
	needs, err := paramIntMap(spec.Params, "needs")
	if err != nil {
		return nil, err
	}
	if pool < 1 {
		return nil, fmt.Errorf("split: pool must be positive, got %d", pool)
	}
	if talk < 1 {
		talk = 1
	}
	if attempts < 1 {
		attempts = 1
	}

	s := &split{
		names:     spec.Names,
		pool:      pool,
		bonus:     bonus,
		attempts:  attempts,
		talk:      talk,
		needs:     needs,
		claims:    map[sim.AgentID]int{},
		granted:   map[sim.AgentID]int{},
		promised:  map[sim.AgentID]int{},
		said:      map[sim.AgentID]int{},
		whispered: map[sim.AgentID]int{},
	}
	// Lay out the rounds: talk, then claim; a failed claim buys one more
	// round of talk before the next attempt.
	s.plan = append(s.plan, repeat(phaseTalk, talk)...)
	s.plan = append(s.plan, phaseClaim)
	for i := 1; i < attempts; i++ {
		s.plan = append(s.plan, phaseTalk, phaseClaim)
	}
	if s.needs == nil {
		s.needs = generateNeeds(spec.Participants, pool, spec.Seed)
	}
	return s, nil
}

// generateNeeds produces private thresholds that sum to roughly 120% of the
// pool, so the agents genuinely cannot all be satisfied and must either
// negotiate someone down or bust. Deterministic for a given seed.
func generateNeeds(ids []sim.AgentID, pool int, seed int64) map[sim.AgentID]int {
	out := map[sim.AgentID]int{}
	n := len(ids)
	if n == 0 {
		return out
	}
	rng := rand.New(rand.NewSource(seed))
	weights := make([]float64, n)
	total := 0.0
	for i := range weights {
		// 0.65–1.35 of an equal share, so some agents want visibly more.
		weights[i] = 0.65 + rng.Float64()*0.7
		total += weights[i]
	}
	target := 1.2 * float64(pool)
	for i, id := range ids {
		want := int(target * weights[i] / total)
		// Round to 5s: tidier numbers for a model to reason about.
		want = (want / 5) * 5
		if want < 5 {
			want = 5
		}
		out[id] = want
	}
	return out
}

func repeat(p phase, n int) []phase {
	out := make([]phase, n)
	for i := range out {
		out[i] = p
	}
	return out
}

func (s *split) Name() string        { return "split" }
func (s *split) Description() string { return "Divide the pool" }

func (s *split) Setup(participants []sim.AgentID) ([]sim.Event, error) {
	s.participants = append([]sim.AgentID(nil), participants...)
	if len(s.needs) == 0 {
		s.needs = generateNeeds(participants, s.pool, 1)
	}
	for _, id := range participants {
		if _, ok := s.needs[id]; !ok {
			s.needs[id] = s.pool / len(participants)
		}
	}
	who := make([]string, 0, len(participants))
	for _, id := range participants {
		who = append(who, displayName(s.names, id))
	}
	return []sim.Event{{
		Type: sim.EventSystem,
		Text: fmt.Sprintf("%d units are on the table for %s to divide. Talk for %d round(s), then everyone submits a sealed claim at the same time. If the claims fit inside %d, each of you is paid what you asked. If they overshoot by even one unit, nobody is paid anything.",
			s.pool, strings.Join(who, ", "), s.talk, s.pool),
		Data:       map[string]any{"pool": s.pool, "talk_rounds": s.talk, "attempts": s.attempts},
		Visibility: sim.PublicTo(),
	}}, nil
}

func (s *split) Participants() []sim.AgentID { return s.participants }

func (s *split) phase() phase {
	if s.done {
		return phaseOver
	}
	if s.round < 1 || s.round > len(s.plan) {
		return phaseOver
	}
	return s.plan[s.round-1]
}

func (s *split) Brief(id sim.AgentID) string {
	var b strings.Builder
	fmt.Fprintf(&b, "There are %d units to divide between %d of you.\n", s.pool, len(s.participants))
	b.WriteString("\nRules:\n")
	fmt.Fprintf(&b, "  - You talk first, then everyone submits a sealed claim in the same round. Nobody sees anyone else's claim until all are in.\n")
	fmt.Fprintf(&b, "  - If the claims total %d or less, each of you receives exactly what you claimed.\n", s.pool)
	b.WriteString("  - If the claims total even one unit more, every claim fails and all of you receive nothing.\n")
	fmt.Fprintf(&b, "  - You have %d sealed-claim attempt(s) in total. A failed attempt reveals everyone's claim and buys one more round of talk.\n", s.attempts)
	b.WriteString("  - Nothing said in conversation is binding. Only the sealed claim counts.\n")
	b.WriteString("  - You can also speak to one person privately. The others do not hear it and do not learn it happened, which makes it the place to strike a side deal, float a number you would not say to the room, or find out what someone will really accept.\n")

	fmt.Fprintf(&b, "\nYour private threshold is %d units: you earn a %d point bonus if your claim succeeds and is at least that large. Nobody else knows this number and you are under no obligation to tell the truth about it.\n", s.needs[id], s.bonus)
	fmt.Fprintf(&b, "Your score is the units you are paid, plus the bonus if you met your threshold. Busting pays zero.\n")

	switch s.phase() {
	case phaseTalk:
		b.WriteString("\nRight now you are talking. You cannot claim this round.\n")
	case phaseClaim:
		b.WriteString("\nThis is a sealed-claim round. Talking is closed: your only move is to claim a number (or pass, which claims nothing).\n")
	}
	if len(s.history) > 0 {
		b.WriteString("\nPrevious attempts:\n")
		for _, h := range s.history {
			parts := make([]string, 0, len(h.Claims))
			for _, pid := range s.participants {
				label := string(pid)
				if pid == id {
					label = "you"
				}
				parts = append(parts, fmt.Sprintf("%s=%d", label, h.Claims[pid]))
			}
			verdict := "fit the pool"
			if h.Busted {
				verdict = "BUST — everyone got nothing"
			}
			fmt.Fprintf(&b, "  attempt %d: %s (total %d, %s)\n", h.Attempt, strings.Join(parts, " "), h.Total, verdict)
		}
	}
	return b.String()
}

func (s *split) Actions(id sim.AgentID) []sim.ActionSpec {
	switch s.phase() {
	case phaseClaim:
		return []sim.ActionSpec{
			{Kind: sim.ActionDo, Verb: "claim", Usage: `{"kind":"do","verb":"claim","args":{"amount":<integer>}}`,
				Description: fmt.Sprintf("Submit your sealed claim, 0 to %d. Binding, and nobody sees it until all claims are in.", s.pool)},
			{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Claim nothing at all this attempt."},
		}
	case phaseTalk:
		return []sim.ActionSpec{
			{Kind: sim.ActionSay, Usage: `{"kind":"say","content":"..."}`, Description: "Say something everyone hears."},
			{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper","to":["<id>"],"content":"..."}`,
				Description: "Speak to one person privately — the others neither hear it nor learn it happened. Use it for a side deal, or to test a number on someone before saying it aloud."},
			{Kind: sim.ActionDo, Verb: "signal", Usage: `{"kind":"do","verb":"signal","args":{"amount":<integer>}}`,
				Description: "State publicly and on the record the number you intend to claim. Not binding, but everyone will see if you break it."},
			{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Say nothing this round."},
		}
	}
	return []sim.ActionSpec{{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "The division is settled; nothing left to do."}}
}

func (s *split) Validate(id sim.AgentID, a sim.Action) error {
	ph := s.phase()
	switch a.Kind {
	case sim.ActionSay, sim.ActionWhisper:
		if ph != phaseTalk {
			return fmt.Errorf("talking is closed in a sealed-claim round; submit a claim or pass")
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
				if !contains(s.participants, t) {
					return fmt.Errorf("%s is not in this negotiation", t)
				}
			}
		}
	case sim.ActionDo:
		switch a.Verb {
		case "claim":
			if ph != phaseClaim {
				return fmt.Errorf("it is not a sealed-claim round yet; this round is for talking")
			}
			if _, already := s.claims[id]; already {
				return fmt.Errorf("you have already claimed this attempt")
			}
			amt, ok := a.ArgFloat("amount")
			if !ok {
				return fmt.Errorf(`claim needs args.amount as an integer between 0 and %d`, s.pool)
			}
			n := int(amt)
			if float64(n) != amt {
				return fmt.Errorf("claim must be a whole number of units, got %v", amt)
			}
			if n < 0 || n > s.pool {
				return fmt.Errorf("claim must be between 0 and %d, got %d", s.pool, n)
			}
		case "signal":
			if ph != phaseTalk {
				return fmt.Errorf("signalling is for talking rounds; submit your sealed claim instead")
			}
			amt, ok := a.ArgFloat("amount")
			if !ok {
				return fmt.Errorf(`signal needs args.amount as an integer between 0 and %d`, s.pool)
			}
			if n := int(amt); n < 0 || n > s.pool {
				return fmt.Errorf("signal must be between 0 and %d, got %d", s.pool, n)
			}
		default:
			return fmt.Errorf("there is no action %q here; in a talking round you may signal, in a claim round you may claim", a.Verb)
		}
	case sim.ActionPass:
	default:
		return fmt.Errorf("unknown action kind %q", a.Kind)
	}
	return nil
}

func (s *split) Apply(id sim.AgentID, a sim.Action) ([]sim.Event, error) {
	switch a.Kind {
	case sim.ActionSay:
		s.said[id]++
		return []sim.Event{{Type: sim.EventSay, Actor: id, Text: a.Content, Visibility: sim.PublicTo()}}, nil

	case sim.ActionWhisper:
		s.whispered[id]++
		return []sim.Event{{
			Type: sim.EventWhisper, Actor: id, Targets: a.To, Text: a.Content,
			Visibility: sim.Only(append([]sim.AgentID{id}, a.To...)...),
		}}, nil

	case sim.ActionDo:
		amt, _ := a.ArgFloat("amount")
		n := int(amt)
		switch a.Verb {
		case "signal":
			s.promised[id] = n
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       fmt.Sprintf("stated on the record an intention to claim %d.", n),
				Data:       map[string]any{"verb": "signal", "amount": n},
				Visibility: sim.PublicTo(),
			}}, nil
		case "claim":
			s.claims[id] = n
			// A sealed claim is visible only to the agent that made it. The
			// environment is the only holder of the full picture until
			// resolution, which is what makes the claim genuinely sealed.
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       fmt.Sprintf("sealed a claim of %d units.", n),
				Data:       map[string]any{"verb": "claim", "amount": n},
				Visibility: sim.Only(id),
			}}, nil
		}
		return nil, fmt.Errorf("split: unhandled verb %q", a.Verb)

	case sim.ActionPass:
		if s.phase() == phaseClaim {
			s.claims[id] = 0
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       "claimed nothing.",
				Data:       map[string]any{"verb": "claim", "amount": 0, "via": "pass"},
				Visibility: sim.Only(id),
			}}, nil
		}
		return []sim.Event{{Type: sim.EventPass, Actor: id, Visibility: sim.PublicTo()}}, nil
	}
	return nil, fmt.Errorf("split: unhandled action kind %q", a.Kind)
}

func (s *split) PublicState() map[string]any {
	st := map[string]any{
		"pool":  s.pool,
		"phase": string(s.phase()),
	}
	if ph := s.phase(); ph == phaseClaim {
		st["attempt"] = s.attempt
		st["claims_submitted"] = fmt.Sprintf("%d of %d", len(s.claims), len(s.participants))
	}
	if len(s.promised) > 0 {
		parts := make([]string, 0, len(s.promised))
		for _, id := range s.participants {
			if n, ok := s.promised[id]; ok {
				parts = append(parts, fmt.Sprintf("%s=%d", id, n))
			}
		}
		st["stated_intentions"] = strings.Join(parts, " ")
	}
	if len(s.history) > 0 {
		st["failed_attempts"] = len(s.history)
	}
	return st
}

// PrivateState gives an agent its own threshold and its own sealed claim, and
// nothing about anyone else.
func (s *split) PrivateState(id sim.AgentID) map[string]any {
	st := map[string]any{
		"your_private_threshold": s.needs[id],
		"bonus_if_met":           s.bonus,
	}
	if n, ok := s.claims[id]; ok {
		st["your_sealed_claim_this_attempt"] = n
	}
	if n, ok := s.promised[id]; ok {
		st["what_you_said_publicly"] = n
	}
	if n, ok := s.granted[id]; ok {
		st["units_you_were_paid"] = n
	}
	return st
}

func (s *split) BeginRound(round int) []sim.Event {
	s.round = round
	if round > len(s.plan) {
		if !s.done {
			s.done = true
			s.reason = "the negotiation ran out of rounds"
		}
		return nil
	}
	switch s.plan[round-1] {
	case phaseClaim:
		s.attempt++
		s.claims = map[sim.AgentID]int{}
		return []sim.Event{{
			Type:       sim.EventState,
			Text:       fmt.Sprintf("Sealed claims for attempt %d of %d. Talking is closed. Each of you names a number now, and nobody sees the others until all are in.", s.attempt, s.attempts),
			Data:       map[string]any{"phase": "claim", "attempt": s.attempt},
			Visibility: sim.PublicTo(),
		}}
	case phaseTalk:
		text := "Open conversation."
		if s.attempt > 0 {
			text = fmt.Sprintf("One more round of talk before attempt %d.", s.attempt+1)
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

// EndRound resolves a sealed-claim round. Resolution is public and itemised:
// everyone sees every claim, which is what lets an agent notice it was misled
// and carry that into memory.
func (s *split) EndRound(round int) []sim.Event {
	if s.done || round > len(s.plan) || s.plan[round-1] != phaseClaim {
		return nil
	}

	total := 0
	claims := map[sim.AgentID]int{}
	for _, id := range s.participants {
		n := s.claims[id]
		claims[id] = n
		total += n
	}
	busted := total > s.pool
	s.history = append(s.history, attemptResult{Attempt: s.attempt, Round: round, Claims: claims, Total: total, Busted: busted})

	parts := make([]string, 0, len(s.participants))
	for _, id := range s.participants {
		part := fmt.Sprintf("%s claimed %d", id, claims[id])
		if p, ok := s.promised[id]; ok && p != claims[id] {
			part += fmt.Sprintf(" (said %d)", p)
		}
		parts = append(parts, part)
	}

	if !busted {
		for id, n := range claims {
			s.granted[id] = n
		}
		s.done = true
		s.reason = fmt.Sprintf("the claims fit the pool on attempt %d", s.attempt)
		return []sim.Event{{
			Type: sim.EventState,
			Text: fmt.Sprintf("Claims are open: %s. Total %d of %d — they fit. Everyone is paid what they asked.",
				strings.Join(parts, ", "), total, s.pool),
			Data:       map[string]any{"attempt": s.attempt, "total": total, "busted": false, "claims": claimsJSON(claims)},
			Visibility: sim.PublicTo(),
		}}
	}

	events := []sim.Event{{
		Type: sim.EventState,
		Text: fmt.Sprintf("Claims are open: %s. Total %d against a pool of %d — BUST. Nobody is paid for this attempt.",
			strings.Join(parts, ", "), total, s.pool),
		Data:       map[string]any{"attempt": s.attempt, "total": total, "busted": true, "claims": claimsJSON(claims)},
		Visibility: sim.PublicTo(),
	}}
	if s.attempt >= s.attempts {
		s.done = true
		s.reason = fmt.Sprintf("all %d attempts busted", s.attempts)
		for _, id := range s.participants {
			s.granted[id] = 0
		}
	}
	return events
}

func (s *split) Done() (bool, string) { return s.done, s.reason }

func (s *split) Outcome() sim.Outcome {
	scores := map[sim.AgentID]float64{}
	metBonus := 0
	paid := 0
	for _, id := range s.participants {
		g := s.granted[id]
		paid += g
		score := float64(g)
		if g > 0 && g >= s.needs[id] {
			score += float64(s.bonus)
			metBonus++
		}
		scores[id] = score
	}

	var lines []string
	switch {
	case len(s.history) == 0:
		lines = append(lines, "No sealed claim was ever submitted; the pool went undivided.")
	case paid > 0:
		last := s.history[len(s.history)-1]
		lines = append(lines, fmt.Sprintf("Settled on attempt %d: %d of %d units paid out, %d left on the table.",
			last.Attempt, paid, s.pool, s.pool-paid))
	default:
		lines = append(lines, fmt.Sprintf("Every one of the %d attempt(s) busted. Nobody was paid anything.", len(s.history)))
	}
	for _, id := range s.participants {
		line := fmt.Sprintf("  %-10s paid %3d  threshold %3d", id, s.granted[id], s.needs[id])
		if s.granted[id] > 0 && s.granted[id] >= s.needs[id] {
			line += "  (bonus)"
		}
		if p, ok := s.promised[id]; ok {
			c := lastClaim(s.history, id)
			line += fmt.Sprintf("  said %d, claimed %d", p, c)
			if c > p {
				line += " — claimed more than stated"
			}
		}
		lines = append(lines, line)
	}

	broken := 0
	for _, id := range s.participants {
		if p, ok := s.promised[id]; ok && lastClaim(s.history, id) > p {
			broken++
		}
	}

	metrics := map[string]any{
		"attempts_used":     len(s.history),
		"units_paid":        paid,
		"pool":              s.pool,
		"efficiency":        ratio(paid, s.pool),
		"thresholds_met":    metBonus,
		"public_utterances": sum(s.said),
		"private_messages":  sum(s.whispered),
		"stated_intentions": len(s.promised),
		// broken_promises is the headline question this scenario exists to
		// answer: did talking actually constrain what they did?
		"broken_promises": broken,
		"claim_history":   historyJSON(s.history),
	}
	return sim.Outcome{Summary: strings.Join(lines, "\n"), Scores: scores, Metrics: metrics}
}

func lastClaim(h []attemptResult, id sim.AgentID) int {
	if len(h) == 0 {
		return 0
	}
	return h[len(h)-1].Claims[id]
}

func claimsJSON(c map[sim.AgentID]int) map[string]int {
	out := map[string]int{}
	for id, n := range c {
		out[string(id)] = n
	}
	return out
}

func historyJSON(h []attemptResult) []map[string]any {
	out := make([]map[string]any, 0, len(h))
	for _, a := range h {
		out = append(out, map[string]any{
			"attempt": a.Attempt, "round": a.Round, "total": a.Total,
			"busted": a.Busted, "claims": claimsJSON(a.Claims),
		})
	}
	return out
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}
