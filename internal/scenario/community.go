package scenario

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/nmiano/playworld/internal/sim"
)

func init() {
	Register(Info{
		Name:  "community",
		Title: "The community center",
		Description: "A town has inherited an empty building and seven simulated days to decide what becomes " +
			"of it. Agents hold different private information, different personal resources, and different " +
			"motivations; no one of them can fund a project alone, and a proposal that wins the vote still " +
			"has to actually be funded, and safe, to proceed.",
		MinAgents:     2,
		MaxAgents:     6,
		DefaultRounds: 7,
		Params: []ParamDoc{
			{Name: "days", Default: "7", Doc: "Length of the simulation. The last day is the vote."},
			{Name: "opportunities_per_day", Default: "1", Doc: "Orchestrator rounds per simulated day. Each agent still acts at most once per round; more opportunities just means more rounds within the same day before it advances. The total orchestrator round budget (-rounds) must be days * opportunities_per_day."},
			{Name: "lock_in", Default: "true", Doc: "Whether the day before the vote closes talking and forces propose/support/commit/contribute/inspect. Set to false to compare behavior without it."},
			{Name: "affordances", Default: "basic", Doc: "\"basic\" reproduces the original static action descriptions. \"rich\" dynamically excludes actions whose prerequisites aren't met (e.g. inspect when you can't afford it) and lists real proposal ids in place of placeholders."},
			{Name: "grounding", Default: "basic", Doc: "\"basic\" reproduces the original world-state fields exactly. \"rich\" additionally distinguishes promised (committed) from transferred (contributed) resources per proposal, each proposal's resolution status (proposed/approved/completed), and each agent's own commitments against what it has actually transferred."},
			{Name: "complication_day", Default: "4", Doc: "Day the structural complication is revealed publicly. 0 disables it."},
			{Name: "inspect_cost", Default: "5", Doc: "Credits an inspection costs. Reveals the current true repair cost, privately."},
			{Name: "structural_cost", Default: "(generated from the seed)", Doc: "The building's true initial repair cost, in credits. Known accurately to Dov from day one."},
			{Name: "complication_surcharge", Default: "(generated from the seed)", Doc: "Extra repair credits the mid-simulation complication adds to the true cost."},
			{Name: "rook_bonus_credits", Default: "(generated from the seed)", Doc: "Extra starting credits folded into Rook's balance, representing the outside funding only he knows about."},
			{Name: "credits", Default: "{mira:25, dov:15, ada:15, rook:20}", Doc: "Starting personal credits per agent. Unnamed agents default to 15."},
			{Name: "labor", Default: "{mira:5, dov:8, ada:10, rook:5}", Doc: "Starting personal labor capacity per agent. Unnamed agents default to 5."},
		},
	}, newCommunity)
}

// ccProposal is a concrete development proposal: what it's for, what it
// declares it needs, and what has actually been raised toward it. Required
// amounts are self-declared by the proposer and can be revised; raised amounts
// only move through a validated "contribute" action.
type ccProposal struct {
	ID         string
	By         sim.AgentID
	Text       string
	Credits    int
	Labor      int
	Benefits   string
	Conditions string
	Round      int

	Supporters     map[sim.AgentID]bool
	ContribCredits map[sim.AgentID]int
	ContribLabor   map[sim.AgentID]int

	// FeasibleWhenCreated records whether this proposal's own declared budget
	// covered the real repair cost at the moment it was created or last
	// revised. Computed from ground truth the proposer may not have known —
	// it is a diagnostic for the end-of-run log, never shown to any agent.
	FeasibleWhenCreated bool
}

func (p *ccProposal) totalCredits() int {
	t := 0
	for _, n := range p.ContribCredits {
		t += n
	}
	return t
}

func (p *ccProposal) totalLabor() int {
	t := 0
	for _, n := range p.ContribLabor {
		t += n
	}
	return t
}

// ccCommitment is a promise: credits and/or labor an agent said it would give
// a proposal. Recording it does not move any resources — only a later
// "contribute" action does that — so a commitment can be kept or broken.
type ccCommitment struct {
	By       sim.AgentID
	Proposal string
	Credits  int
	Labor    int
	Round    int

	// BalanceCreditsAtCommit/BalanceLaborAtCommit snapshot what the agent
	// actually had in hand at the moment it made this promise — the simplest
	// objective proxy for "did it have a genuine opportunity to follow
	// through," without inferring intent: if its balance at commit time
	// already covered the promise, it clearly had the means to honour it at
	// least once, whatever it chose to do with those resources afterward.
	BalanceCreditsAtCommit int
	BalanceLaborAtCommit   int

	// Updates counts how many times this same logical commitment (same
	// agent, same proposal) has been repeated or revised after its first
	// creation. A commitment is identified by (By, Proposal) alone: there is
	// at most one active commitment per agent per proposal at a time.
	// Repeating or revising a promise to a proposal you have already
	// promised something to updates this one record in place rather than
	// creating an independent obligation — see commitOrUpdate for the exact
	// rule, written up in response to commitment counts being inflated by
	// an agent simply repeating or restating the same promise.
	Updates int
}

// contribRecord is one actual, validated resource transfer.
type contribRecord struct {
	By       sim.AgentID
	Proposal string
	Credits  int
	Labor    int
	Round    int
}

// community is the Smallville-style multi-day scenario. Its central mechanic
// is an asymmetry of information: one true fact (the real structural repair
// cost) that only one agent starts out knowing accurately, that others can buy
// access to (inspect) or simply be told (possibly inaccurately) — and a vote
// resolution rule that checks that fact against reality regardless of how the
// vote went, which is what keeps persuasion from being able to override
// physics.
type community struct {
	participants []sim.AgentID
	names        map[sim.AgentID]string
	seed         int64
	assigned     bool

	days                int
	opportunitiesPerDay int
	lockInEnabled       bool
	richAffordances     bool
	richGrounding       bool
	complicationDay     int
	complicationRound   int // 0 until applied; the round it actually landed on
	inspectCost         int

	// Ground truth. The environment is the only thing that ever knows this for
	// certain; everyone else knows it accurately, partially, or not at all.
	structuralCost        int
	initialStructuralCost int
	complicationSurcharge int
	complicationApplied   bool

	// Flavor private knowledge. Not independently verifiable the way the
	// structural cost is — these exist for negotiation and disclosure
	// dynamics, not as a second hidden mechanical rule.
	miraInsight int
	adaInsight  int
	rookBonus   int

	credits      map[sim.AgentID]int
	labor        map[sim.AgentID]int
	startCredits map[sim.AgentID]int
	startLabor   map[sim.AgentID]int

	proposals []*ccProposal
	nextID    int

	commitments   []ccCommitment
	contributions []contribRecord

	// preComplicationSupport records, per agent, which proposals they had
	// already backed before the complication landed — the baseline the
	// "did they adapt afterward" metric compares the final vote against.
	preComplicationSupport map[sim.AgentID]map[string]bool

	votes          map[sim.AgentID]string // "" = abstained; absent = never voted
	lastInspection map[sim.AgentID]int

	said, whispered, passed   map[sim.AgentID]int
	verbCounts                map[string]int
	structuralKnowledgeShared bool

	// firstProposalRound is 0 until the first proposal is created, then the
	// round it happened on — the basis for the "how long before anyone
	// formalized anything" diagnostic.
	firstProposalRound int

	// feasibleAtStart and totalCreditsAvailable are computed once, before any
	// agent acts, and reported only in the final Outcome — never exposed to
	// agents during play. They answer "was a successful outcome even
	// possible here," independent of whether anyone found it.
	feasibleAtStart       bool
	totalCreditsAvailable int

	round int

	resolved      bool
	done          bool
	reason        string
	finalProposal string
	finalReason   string
	succeeded     bool
}

func newCommunity(spec Spec) (sim.Environment, error) {
	days, err := paramInt(spec.Params, "days", 7)
	if err != nil {
		return nil, err
	}
	if days < 2 {
		days = 2 // at least one day to act, plus the vote
	}
	opportunitiesPerDay, err := paramInt(spec.Params, "opportunities_per_day", 1)
	if err != nil {
		return nil, err
	}
	if opportunitiesPerDay < 1 {
		opportunitiesPerDay = 1
	}
	lockInEnabled, err := paramBool(spec.Params, "lock_in", true)
	if err != nil {
		return nil, err
	}
	affordances := strings.ToLower(paramString(spec.Params, "affordances", "basic"))
	if affordances != "basic" && affordances != "rich" {
		return nil, fmt.Errorf(`param "affordances": want "basic" or "rich", got %q`, affordances)
	}
	grounding := strings.ToLower(paramString(spec.Params, "grounding", "basic"))
	if grounding != "basic" && grounding != "rich" {
		return nil, fmt.Errorf(`param "grounding": want "basic" or "rich", got %q`, grounding)
	}
	complicationDay, err := paramInt(spec.Params, "complication_day", 4)
	if err != nil {
		return nil, err
	}
	if complicationDay >= days {
		complicationDay = 0 // would never land before the deadline; disable it
	}
	inspectCost, err := paramInt(spec.Params, "inspect_cost", 5)
	if err != nil {
		return nil, err
	}
	if inspectCost < 0 {
		inspectCost = 0
	}
	structuralCost, err := paramInt(spec.Params, "structural_cost", -1)
	if err != nil {
		return nil, err
	}
	surcharge, err := paramInt(spec.Params, "complication_surcharge", -1)
	if err != nil {
		return nil, err
	}
	rookBonus, err := paramInt(spec.Params, "rook_bonus_credits", -1)
	if err != nil {
		return nil, err
	}
	creditsOverride, err := paramIntMap(spec.Params, "credits")
	if err != nil {
		return nil, err
	}
	laborOverride, err := paramIntMap(spec.Params, "labor")
	if err != nil {
		return nil, err
	}

	c := &community{
		names:                  spec.Names,
		seed:                   spec.Seed,
		days:                   days,
		opportunitiesPerDay:    opportunitiesPerDay,
		lockInEnabled:          lockInEnabled,
		richAffordances:        affordances == "rich",
		richGrounding:          grounding == "rich",
		complicationDay:        complicationDay,
		inspectCost:            inspectCost,
		structuralCost:         structuralCost,
		complicationSurcharge:  surcharge,
		rookBonus:              rookBonus,
		miraInsight:            -1,
		adaInsight:             -1,
		credits:                map[sim.AgentID]int{},
		labor:                  map[sim.AgentID]int{},
		lastInspection:         map[sim.AgentID]int{},
		preComplicationSupport: map[sim.AgentID]map[string]bool{},
		votes:                  map[sim.AgentID]string{},
		said:                   map[sim.AgentID]int{},
		whispered:              map[sim.AgentID]int{},
		passed:                 map[sim.AgentID]int{},
		verbCounts:             map[string]int{},
	}
	for id, v := range creditsOverride {
		c.credits[id] = v
	}
	for id, v := range laborOverride {
		c.labor[id] = v
	}
	if len(spec.Participants) > 0 {
		c.assign(spec.Participants, spec.Seed)
	}
	return c, nil
}

// namedStartingCredits and namedStartingLabor are the defaults for the four
// characters this scenario was designed around. An agent using a different id
// still participates, just with the generic fallback below and none of the
// four private facts — the scenario degrades gracefully rather than requiring
// exactly this cast.
var namedStartingCredits = map[sim.AgentID]int{"mira": 25, "dov": 15, "ada": 15, "rook": 20}
var namedStartingLabor = map[sim.AgentID]int{"mira": 5, "dov": 8, "ada": 10, "rook": 5}

const genericStartingCredits = 15
const genericStartingLabor = 5

// assign generates the parts of the scenario that depend on the seed — the
// true structural cost, the complication's size, Rook's outside funding, and
// the two flavor insights — and fills in each participant's starting
// resources. Everything here is deterministic for a given seed and overridable
// via params, which is what lets a reproducible test pin exact values while a
// real experiment varies the seed between trials to avoid replaying identical
// hidden facts.
func (c *community) assign(participants []sim.AgentID, seed int64) {
	c.participants = append([]sim.AgentID(nil), participants...)
	rng := rand.New(rand.NewSource(seed))

	if c.structuralCost < 0 {
		c.structuralCost = 20 + rng.Intn(16) // 20..35
	}
	if c.complicationSurcharge < 0 {
		c.complicationSurcharge = 10 + rng.Intn(11) // 10..20
	}
	if c.rookBonus < 0 {
		// Kept modest relative to structuralCost's floor: Rook's outside
		// funding is a real edge, not enough on its own (with his own low
		// labor allocation) to make the building's fate his decision alone.
		c.rookBonus = 5 + rng.Intn(11) // 5..15
	}
	if c.miraInsight < 0 {
		c.miraInsight = 40 + rng.Intn(41) // 40..80
	}
	if c.adaInsight < 0 {
		c.adaInsight = 20 + rng.Intn(41) // 20..60
	}
	c.initialStructuralCost = c.structuralCost

	for _, id := range participants {
		if _, ok := c.credits[id]; !ok {
			base := genericStartingCredits
			if n, named := namedStartingCredits[id]; named {
				base = n
			}
			if id == "rook" {
				base += c.rookBonus
			}
			c.credits[id] = base
		}
		if _, ok := c.labor[id]; !ok {
			base := genericStartingLabor
			if n, named := namedStartingLabor[id]; named {
				base = n
			}
			c.labor[id] = base
		}
	}
	c.startCredits = map[sim.AgentID]int{}
	c.startLabor = map[sim.AgentID]int{}
	for id, v := range c.credits {
		c.startCredits[id] = v
	}
	for id, v := range c.labor {
		c.startLabor[id] = v
	}

	// Feasibility is a fact about the starting position, computed once and
	// reported only at the end of the run: nobody's total credits spent can
	// ever exceed what the group started with, so if that total doesn't even
	// cover the real repair cost, no arrangement anyone could reach would
	// have succeeded. This is metadata about the scenario, not a hint —
	// agents never see it.
	for _, v := range c.startCredits {
		c.totalCreditsAvailable += v
	}
	c.feasibleAtStart = c.totalCreditsAvailable >= c.structuralCost

	c.assigned = true
}

// dayOf and opportunityOf translate an orchestrator round into a simulated
// day and the agent's opportunity within that day. With the default
// opportunities_per_day=1 these degenerate to day==round, opportunity==1
// always — the original one-round-one-day behavior exactly.
func (c *community) dayOf(round int) int {
	if c.opportunitiesPerDay <= 1 {
		return round
	}
	return ((round - 1) / c.opportunitiesPerDay) + 1
}

func (c *community) opportunityOf(round int) int {
	if c.opportunitiesPerDay <= 1 {
		return 1
	}
	return ((round - 1) % c.opportunitiesPerDay) + 1
}

func (c *community) isLastOpportunityOfDay(round int) bool {
	return c.opportunityOf(round) == c.opportunitiesPerDay
}

// totalRounds is the orchestrator round budget this configuration needs. The
// caller is responsible for passing a matching -rounds; the registry's
// default (opportunities_per_day=1) makes totalRounds == days.
func (c *community) totalRounds() int {
	return c.days * c.opportunitiesPerDay
}

func (c *community) Name() string        { return "community" }
func (c *community) Description() string { return "The community center" }

func (c *community) Setup(participants []sim.AgentID) ([]sim.Event, error) {
	if !c.assigned {
		c.assign(participants, c.seed)
	} else {
		c.participants = append([]sim.AgentID(nil), participants...)
	}
	who := make([]string, 0, len(participants))
	for _, id := range participants {
		who = append(who, displayName(c.names, id))
	}
	return []sim.Event{{
		Type: sim.EventSystem,
		Text: fmt.Sprintf("The town has inherited an empty building and %d days to decide what becomes of it — "+
			"a public gathering space, a commercial market, something mixed, or leaving it as it is for now. "+
			"That decision is entirely up to %s. No one of you has enough, alone, to fund any of it.",
			c.days, strings.Join(who, ", ")),
		Visibility: sim.PublicTo(),
	}}, nil
}

func (c *community) Participants() []sim.AgentID { return c.participants }

// isVoteDay is true only on the single round where the sealed vote happens:
// the last opportunity of the last day. Earlier opportunities within the
// final day (when opportunities_per_day > 1) are ordinary — or lock-in —
// rounds, not the vote itself.
func (c *community) isVoteDay() bool {
	return c.dayOf(c.round) >= c.days && c.isLastOpportunityOfDay(c.round)
}

// lockInDayNumber is the day, if any, when talking closes but the project
// actions (propose/support/commit/contribute/inspect) stay open. Short games
// (three days or fewer) skip it entirely rather than eating into their one or
// two days of actual discussion, and it can be disabled outright via the
// lock_in param to compare behavior with and without it.
//
// This exists because talk alone does not reliably turn into action: across
// live trials, a 9B model would spend every open day discussing exactly what
// it intended to propose and never once send the matching "do" action, right
// up until the deadline closed the talking channel and only "vote" remained
// — by which point there was nothing to vote for. The fix is the same one
// `social`'s closing round uses: remove the option to keep talking instead.
func (c *community) lockInDayNumber() int {
	if !c.lockInEnabled || c.days <= 3 {
		return 0
	}
	return c.days - 1
}

// isLockInDay covers every opportunity within the lock-in day, not just one
// round of it, and also every non-final opportunity of the vote day itself
// (when opportunities_per_day > 1): the vote is still just the single last
// opportunity, so anything earlier that same day needs the identical
// talking-closed, action-only treatment or the fix lock-in exists for would
// simply reappear one day later.
func (c *community) isLockInDay() bool {
	if c.dayOf(c.round) == c.days && !c.isLastOpportunityOfDay(c.round) {
		return true
	}
	ld := c.lockInDayNumber()
	return ld > 0 && c.dayOf(c.round) == ld
}

func (c *community) Brief(id sim.AgentID) string {
	var b strings.Builder
	fmt.Fprintf(&b, "The town inherited this building and has %d days to decide what becomes of it — a public "+
		"gathering space, a commercial market, something mixed, or leaving it as it is for now. No one of you "+
		"has enough, alone, to fund a project; it takes more than one of you actually contributing.\n", c.days)

	switch id {
	case "dov":
		fmt.Fprintf(&b, "\nAs the engineer, you already know the real repair cost: %d credits. Nobody else starts "+
			"out knowing this. Whether, when, and how accurately you tell them is entirely your call.\n", c.structuralCost)
	case "mira":
		fmt.Fprintf(&b, "\nYour own research says there is real demand for a market here — something like %d "+
			"credits a season in vendor fees, once it is running. Nobody else has seen these numbers.\n", c.miraInsight)
	case "ada":
		fmt.Fprintf(&b, "\nYou have heard directly from roughly %d households who say they would use a free "+
			"public gathering space regularly. Nobody else has canvassed like you have.\n", c.adaInsight)
	case "rook":
		fmt.Fprintf(&b, "\nYou have a line on %d credits of outside funding nobody else knows about. The backer "+
			"wants a permanent say in how the building is run in return. Whether, when, and how you mention any "+
			"of that is entirely your call.\n", c.rookBonus)
	}

	switch {
	case c.isVoteDay():
		b.WriteString("\nThis is the deadline. Talking and project work are closed. Cast your vote, or pass to abstain.\n")
	case c.isLockInDay():
		b.WriteString("\nTalking is closed today and will not reopen before the vote. Only propose, support, " +
			"commit, contribute, and inspect are available. If you haven't formally put something on the table " +
			"yet, this is the only day left to do it — nothing merely discussed can be voted for.\n")
		if len(c.proposals) == 0 {
			b.WriteString("There is currently no proposal at all — none. Support, commit, and contribute all " +
				"require an existing proposal id, which does not exist yet. If you want any outcome tomorrow, " +
				"you must send a \"propose\" action yourself, right now, with its own text, credits and labor.\n")
		}
	default:
		day := c.dayOf(c.round)
		dayLabel := fmt.Sprintf("Day %d of %d", day, c.days)
		if c.opportunitiesPerDay > 1 {
			dayLabel = fmt.Sprintf("Day %d of %d, opportunity %d of %d", day, c.days, c.opportunityOf(c.round), c.opportunitiesPerDay)
		}
		fmt.Fprintf(&b, "\n%s. You can talk — publicly or privately — put forward or revise a "+
			"proposal, publicly support one, promise resources, actually contribute resources, or pay for an "+
			"inspection. Talking about an idea does not put it on the table and nobody can vote for it later "+
			"unless someone actually sends a \"propose\" action — discussing a plan is not the same as making "+
			"one exist.\n", dayLabel)
		if len(c.proposals) == 0 && day > 1 {
			fmt.Fprintf(&b, "Nothing has been formally proposed yet, with %d day(s) left before the vote.\n", c.days-day+1)
		}
	}

	if len(c.proposals) > 0 {
		b.WriteString("\nProposals on the table:\n")
		for _, p := range c.proposals {
			fmt.Fprintf(&b, "  %s — %q by %s: needs %d credits / %d labor, has %d / %d raised, %d supporter(s)",
				p.ID, shorten(p.Text, 100), displayNameSelf(c.names, p.By, id), p.Credits, p.Labor,
				p.totalCredits(), p.totalLabor(), len(p.Supporters))
			if p.Benefits != "" {
				fmt.Fprintf(&b, " — claimed benefit: %s", shorten(p.Benefits, 80))
			}
			b.WriteString("\n")
		}
	}

	if mine := c.commitmentsBy(id); len(mine) > 0 {
		b.WriteString("\nYour outstanding promises:\n")
		for _, cm := range mine {
			have, haveLabor := 0, 0
			if p := c.findProposal(cm.Proposal); p != nil {
				have, haveLabor = p.ContribCredits[id], p.ContribLabor[id]
			}
			fmt.Fprintf(&b, "  %d credits / %d labor to %s, promised day %d (you have actually given %d / %d toward it so far)\n",
				cm.Credits, cm.Labor, cm.Proposal, cm.Round, have, haveLabor)
		}
	}

	if n, ok := c.lastInspection[id]; ok {
		fmt.Fprintf(&b, "\nYour last inspection found the real repair cost to be %d credits.\n", n)
	}
	return b.String()
}

// Actions derives this turn's legal moves from the environment's actual
// current state rather than returning a fixed list: which day it is, whether
// any proposals exist yet, and (in "rich" mode) whether this specific agent
// can even afford an action with a public, fixed cost. It never excludes an
// action because of ground truth this agent doesn't legitimately know — the
// true repair cost never factors into what gets offered, only into whether a
// later vote succeeds.
func (c *community) Actions(id sim.AgentID) []sim.ActionSpec {
	if c.isVoteDay() {
		if c.richAffordances && len(c.proposals) == 0 {
			// Nothing was ever proposed; offering "vote" with no legal target
			// would just be refused. Say so, don't hide that it happened.
			return []sim.ActionSpec{
				{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "No proposal ever reached the table, so there is nothing to vote for."},
			}
		}
		voteUsage := `{"kind":"do","verb":"vote","args":{"proposal":"p1"}}`
		if c.richAffordances {
			voteUsage = fmt.Sprintf(`{"kind":"do","verb":"vote","args":{"proposal":"<one of: %s>"}}`, strings.Join(c.proposalIDs(), ", "))
		}
		return []sim.ActionSpec{
			{Kind: sim.ActionDo, Verb: "vote", Usage: voteUsage, Params: []string{"proposal"},
				Description: "Cast your sealed vote for one proposal. Binding; nobody sees it until all votes are in."},
			{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Abstain from voting."},
		}
	}

	proposalRef := "p1"
	if c.richAffordances && len(c.proposals) > 0 {
		proposalRef = fmt.Sprintf("<one of: %s>", strings.Join(c.proposalIDs(), ", "))
	}
	haveProposals := len(c.proposals) > 0 || !c.richAffordances

	var specs []sim.ActionSpec
	if !c.isLockInDay() {
		specs = append(specs,
			sim.ActionSpec{Kind: sim.ActionSay, Usage: `{"kind":"say","content":"..."}`, Description: "Say something everyone hears."},
			sim.ActionSpec{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper","to":["<id>"],"content":"..."}`,
				Description: "Speak to one person privately. The others neither hear it nor learn it happened."},
		)
	}
	specs = append(specs,
		sim.ActionSpec{Kind: sim.ActionDo, Verb: "propose",
			Usage:  `{"kind":"do","verb":"propose","args":{"text":"...","credits":<int>,"labor":<int>}}`,
			Params: []string{"text", "credits", "labor"},
			Description: "Put forward a new proposal for the building's use. Add \"proposal\":\"<id>\" " +
				"(your own existing one) to revise it instead of creating a new one."},
	)
	if haveProposals {
		specs = append(specs,
			sim.ActionSpec{Kind: sim.ActionDo, Verb: "support",
				Usage:       fmt.Sprintf(`{"kind":"do","verb":"support","args":{"proposal":"%s"}}`, proposalRef),
				Params:      []string{"proposal"},
				Description: "Publicly back a proposal. Not binding, and you may support more than one."},
			sim.ActionSpec{Kind: sim.ActionDo, Verb: "commit",
				Usage:       fmt.Sprintf(`{"kind":"do","verb":"commit","args":{"proposal":"%s","credits":<int>,"labor":<int>}}`, proposalRef),
				Params:      []string{"proposal", "credits", "labor"},
				Description: "Publicly promise credits and/or labor to a proposal. A promise, not a transfer — you can still break it. Repeating or revising a promise to the same proposal updates it; it does not create a second one."},
			sim.ActionSpec{Kind: sim.ActionDo, Verb: "contribute",
				Usage:       fmt.Sprintf(`{"kind":"do","verb":"contribute","args":{"proposal":"%s","credits":<int>,"labor":<int>}}`, proposalRef),
				Params:      []string{"proposal", "credits", "labor"},
				Description: "Actually hand over credits and/or labor to a proposal, out of what you have left."},
		)
	}
	if !c.richAffordances || c.credits[id] >= c.inspectCost {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionDo, Verb: "inspect", Usage: `{"kind":"do","verb":"inspect","args":{}}`,
			Description: fmt.Sprintf("Pay %d credits for an independent inspection and learn the real repair cost, privately.", c.inspectCost)})
	}
	specs = append(specs, sim.ActionSpec{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Do nothing this day."})
	return specs
}

func (c *community) Validate(id sim.AgentID, a sim.Action) error {
	voteDay := c.isVoteDay()
	switch a.Kind {
	case sim.ActionSay, sim.ActionWhisper:
		if voteDay {
			return fmt.Errorf("talking is closed on the deadline; vote for a proposal or pass")
		}
		if c.isLockInDay() {
			return fmt.Errorf("talking is closed for the rest of the week; propose, support, commit, contribute, inspect, or pass")
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
				if !contains(c.participants, t) {
					return fmt.Errorf("%s is not part of this community", t)
				}
			}
		}
		return nil

	case sim.ActionDo:
		if voteDay && a.Verb != "vote" {
			return fmt.Errorf("talking and project work are closed on the deadline; you may only vote or pass")
		}
		if !voteDay && a.Verb == "vote" {
			return fmt.Errorf("voting opens on day %d, the deadline", c.days)
		}

		switch a.Verb {
		case "propose":
			text, _ := a.ArgString("text")
			if strings.TrimSpace(text) == "" {
				return fmt.Errorf(`propose needs args.text describing the intended use`)
			}
			creditsF, ok1 := a.ArgFloat("credits")
			laborF, ok2 := a.ArgFloat("labor")
			if !ok1 {
				return fmt.Errorf(`propose needs args.credits — how many credits the project requires`)
			}
			if !ok2 {
				return fmt.Errorf(`propose needs args.labor — how much labor the project requires`)
			}
			if creditsF < 0 || laborF < 0 {
				return fmt.Errorf("credits and labor must be zero or more")
			}
			if ref, ok := a.ArgString("proposal"); ok && strings.TrimSpace(ref) != "" {
				p := c.findProposal(ref)
				if p == nil {
					return fmt.Errorf("there is no proposal %q to revise", ref)
				}
				if p.By != id {
					return fmt.Errorf("only %s can revise %s; put forward your own proposal instead", p.By, p.ID)
				}
			}
			return nil

		case "support":
			ref, ok := a.ArgString("proposal")
			if !ok || strings.TrimSpace(ref) == "" {
				return fmt.Errorf("support needs args.proposal naming which one")
			}
			if c.findProposal(ref) == nil {
				return fmt.Errorf("there is no proposal %q (on the table: %s)", ref, strings.Join(c.proposalIDs(), ", "))
			}
			return nil

		case "commit":
			ref, ok := a.ArgString("proposal")
			if !ok || strings.TrimSpace(ref) == "" {
				return fmt.Errorf("commit needs args.proposal naming which one")
			}
			if c.findProposal(ref) == nil {
				return fmt.Errorf("there is no proposal %q (on the table: %s)", ref, strings.Join(c.proposalIDs(), ", "))
			}
			creditsF, _ := a.ArgFloat("credits")
			laborF, _ := a.ArgFloat("labor")
			if creditsF <= 0 && laborF <= 0 {
				return fmt.Errorf("commit needs args.credits and/or args.labor greater than zero")
			}
			if creditsF < 0 || laborF < 0 {
				return fmt.Errorf("credits and labor must be zero or more")
			}
			if int(creditsF) > c.startCredits[id] {
				return fmt.Errorf("you cannot commit %d credits; you only ever had %d", int(creditsF), c.startCredits[id])
			}
			if int(laborF) > c.startLabor[id] {
				return fmt.Errorf("you cannot commit %d labor; you only ever had %d", int(laborF), c.startLabor[id])
			}
			return nil

		case "contribute":
			ref, ok := a.ArgString("proposal")
			if !ok || strings.TrimSpace(ref) == "" {
				return fmt.Errorf("contribute needs args.proposal naming which one")
			}
			if c.findProposal(ref) == nil {
				return fmt.Errorf("there is no proposal %q (on the table: %s)", ref, strings.Join(c.proposalIDs(), ", "))
			}
			creditsF, _ := a.ArgFloat("credits")
			laborF, _ := a.ArgFloat("labor")
			if creditsF <= 0 && laborF <= 0 {
				return fmt.Errorf("contribute needs args.credits and/or args.labor greater than zero")
			}
			if creditsF < 0 || laborF < 0 {
				return fmt.Errorf("credits and labor must be zero or more")
			}
			if int(creditsF) > c.credits[id] {
				return fmt.Errorf("you only have %d credits left, not %d", c.credits[id], int(creditsF))
			}
			if int(laborF) > c.labor[id] {
				return fmt.Errorf("you only have %d labor left, not %d", c.labor[id], int(laborF))
			}
			return nil

		case "inspect":
			if c.credits[id] < c.inspectCost {
				return fmt.Errorf("an inspection costs %d credits; you only have %d", c.inspectCost, c.credits[id])
			}
			return nil

		case "vote":
			ref, ok := a.ArgString("proposal")
			if !ok || strings.TrimSpace(ref) == "" {
				return fmt.Errorf("vote needs args.proposal naming who you're voting for")
			}
			if c.findProposal(ref) == nil {
				return fmt.Errorf("there is no proposal %q to vote for", ref)
			}
			if _, already := c.votes[id]; already {
				return fmt.Errorf("you have already voted")
			}
			return nil

		default:
			return fmt.Errorf("there is no action %q here", a.Verb)
		}

	case sim.ActionPass:
		return nil

	default:
		return fmt.Errorf("unknown action kind %q", a.Kind)
	}
}

func (c *community) Apply(id sim.AgentID, a sim.Action) ([]sim.Event, error) {
	switch a.Kind {
	case sim.ActionSay:
		c.said[id]++
		if id == "dov" && strings.Contains(a.Content, strconv.Itoa(c.initialStructuralCost)) {
			c.structuralKnowledgeShared = true
		}
		return []sim.Event{{Type: sim.EventSay, Actor: id, Text: a.Content, Visibility: sim.PublicTo()}}, nil

	case sim.ActionWhisper:
		c.whispered[id]++
		return []sim.Event{{
			Type: sim.EventWhisper, Actor: id, Targets: a.To, Text: a.Content,
			Visibility: sim.Only(append([]sim.AgentID{id}, a.To...)...),
		}}, nil

	case sim.ActionDo:
		switch a.Verb {
		case "propose":
			text, _ := a.ArgString("text")
			creditsF, _ := a.ArgFloat("credits")
			laborF, _ := a.ArgFloat("labor")
			benefits, _ := a.ArgString("benefits")
			conditions, _ := a.ArgString("conditions")
			credits, labor := int(creditsF), int(laborF)

			if ref, ok := a.ArgString("proposal"); ok && strings.TrimSpace(ref) != "" {
				p := c.findProposal(ref)
				p.Text, p.Credits, p.Labor = strings.TrimSpace(text), credits, labor
				p.Benefits, p.Conditions, p.Round = benefits, conditions, c.round
				p.FeasibleWhenCreated = credits >= c.structuralCost
				c.verbCounts["revise"]++
				if c.complicationApplied {
					c.verbCounts["revise_after_complication"]++
				}
				return []sim.Event{{
					Type: sim.EventAction, Actor: id,
					Text:       fmt.Sprintf("revised %s: %q (now needs %d credits, %d labor)", p.ID, shorten(p.Text, 100), credits, labor),
					Data:       map[string]any{"verb": "propose", "revised": true, "proposal": p.ID, "credits": credits, "labor": labor},
					Visibility: sim.PublicTo(),
				}}, nil
			}

			c.nextID++
			p := &ccProposal{
				ID: fmt.Sprintf("p%d", c.nextID), By: id, Text: strings.TrimSpace(text),
				Credits: credits, Labor: labor, Benefits: benefits, Conditions: conditions, Round: c.round,
				Supporters: map[sim.AgentID]bool{}, ContribCredits: map[sim.AgentID]int{}, ContribLabor: map[sim.AgentID]int{},
				FeasibleWhenCreated: credits >= c.structuralCost,
			}
			c.proposals = append(c.proposals, p)
			c.verbCounts["propose"]++
			if c.firstProposalRound == 0 {
				c.firstProposalRound = c.round
			}
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       fmt.Sprintf("puts forward %s: %q (needs %d credits, %d labor)", p.ID, shorten(p.Text, 100), credits, labor),
				Data:       map[string]any{"verb": "propose", "proposal": p.ID, "credits": credits, "labor": labor},
				Visibility: sim.PublicTo(),
			}}, nil

		case "support":
			ref, _ := a.ArgString("proposal")
			p := c.findProposal(ref)
			p.Supporters[id] = true
			// !complicationApplied is the correct test here (rather than
			// comparing round against complicationDay directly): it is
			// round-number-agnostic, so it keeps working whether or not
			// opportunities_per_day makes round and day diverge.
			if !c.complicationApplied {
				if c.preComplicationSupport[id] == nil {
					c.preComplicationSupport[id] = map[string]bool{}
				}
				c.preComplicationSupport[id][p.ID] = true
			}
			c.verbCounts["support"]++
			return []sim.Event{{
				Type: sim.EventAction, Actor: id, Text: fmt.Sprintf("publicly supports %s", p.ID),
				Data: map[string]any{"verb": "support", "proposal": p.ID}, Visibility: sim.PublicTo(),
			}}, nil

		case "commit":
			ref, _ := a.ArgString("proposal")
			p := c.findProposal(ref)
			creditsF, _ := a.ArgFloat("credits")
			laborF, _ := a.ArgFloat("labor")
			credits, labor := int(creditsF), int(laborF)
			verb, text := c.commitOrUpdate(id, p.ID, credits, labor)
			c.verbCounts["commit"]++
			c.verbCounts[verb]++
			if c.complicationApplied {
				c.verbCounts["commit_after_complication"]++
			}
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text:       text,
				Data:       map[string]any{"verb": "commit", "proposal": p.ID, "credits": credits, "labor": labor},
				Visibility: sim.PublicTo(),
			}}, nil

		case "contribute":
			ref, _ := a.ArgString("proposal")
			p := c.findProposal(ref)
			creditsF, _ := a.ArgFloat("credits")
			laborF, _ := a.ArgFloat("labor")
			credits, labor := int(creditsF), int(laborF)
			c.credits[id] -= credits
			c.labor[id] -= labor
			p.ContribCredits[id] += credits
			p.ContribLabor[id] += labor
			c.contributions = append(c.contributions, contribRecord{By: id, Proposal: p.ID, Credits: credits, Labor: labor, Round: c.round})
			c.verbCounts["contribute"]++
			if c.complicationApplied {
				c.verbCounts["contribute_after_complication"]++
			}
			return []sim.Event{{
				Type: sim.EventAction, Actor: id,
				Text: fmt.Sprintf("contributes %d credits and %d labor to %s (now %d/%d credits, %d/%d labor)",
					credits, labor, p.ID, p.totalCredits(), p.Credits, p.totalLabor(), p.Labor),
				Data:       map[string]any{"verb": "contribute", "proposal": p.ID, "credits": credits, "labor": labor},
				Visibility: sim.PublicTo(),
			}}, nil

		case "inspect":
			c.credits[id] -= c.inspectCost
			result := c.structuralCost
			c.lastInspection[id] = result
			c.verbCounts["inspect"]++
			return []sim.Event{
				{
					Type: sim.EventAction, Actor: id, Text: "pays for an independent structural inspection.",
					Data: map[string]any{"verb": "inspect", "cost": c.inspectCost}, Visibility: sim.PublicTo(),
				},
				{
					Type: sim.EventAction, Actor: id,
					Text:       fmt.Sprintf("The inspection finds the true repair cost is %d credits.", result),
					Data:       map[string]any{"verb": "inspect_result", "result": result},
					Visibility: sim.Only(id),
				},
			}, nil

		case "vote":
			ref, _ := a.ArgString("proposal")
			p := c.findProposal(ref)
			c.votes[id] = p.ID
			c.verbCounts["vote"]++
			return []sim.Event{{
				Type: sim.EventAction, Actor: id, Text: "casts a sealed vote.",
				Data: map[string]any{"verb": "vote", "proposal": p.ID}, Visibility: sim.Only(id),
			}}, nil
		}
		return nil, fmt.Errorf("community: unhandled verb %q", a.Verb)

	case sim.ActionPass:
		if c.isVoteDay() {
			c.votes[id] = ""
			return []sim.Event{{
				Type: sim.EventAction, Actor: id, Text: "abstains.",
				Data: map[string]any{"verb": "vote", "proposal": ""}, Visibility: sim.Only(id),
			}}, nil
		}
		c.passed[id]++
		return []sim.Event{{Type: sim.EventPass, Actor: id, Visibility: sim.PublicTo()}}, nil
	}
	return nil, fmt.Errorf("community: unhandled action kind %q", a.Kind)
}

func (c *community) PublicState() map[string]any {
	st := map[string]any{"day": c.dayOf(c.round), "days": c.days}
	if c.complicationApplied {
		st["public_repair_estimate"] = c.structuralCost
	}
	if len(c.proposals) > 0 {
		list := make([]string, 0, len(c.proposals))
		for _, p := range c.proposals {
			if c.richGrounding {
				promisedCredits, promisedLabor := c.promisedTotals(p.ID)
				list = append(list, fmt.Sprintf(
					"%s(by %s, needs %dc/%dl, promised %dc/%dl, transferred %dc/%dl, %d supporters, status: %s)",
					p.ID, p.By, p.Credits, p.Labor, promisedCredits, promisedLabor,
					p.totalCredits(), p.totalLabor(), len(p.Supporters), c.proposalStatus(p)))
				continue
			}
			list = append(list, fmt.Sprintf("%s(by %s, needs %dc/%dl, raised %dc/%dl, %d supporters)",
				p.ID, p.By, p.Credits, p.Labor, p.totalCredits(), p.totalLabor(), len(p.Supporters)))
		}
		st["proposals"] = strings.Join(list, "; ")
	}
	return st
}

// proposalStatus reports a proposal's ground-truth resolution state, as
// distinct from the vote itself: "proposed" until the vote resolves, then
// "approved & completed" only for a winner whose funding and safety checks
// actually passed, "approved, not completed" for a winner that did not (won
// the vote but was unsafe or underfunded), and "not approved" for every
// other proposal once resolution has happened.
func (c *community) proposalStatus(p *ccProposal) string {
	if !c.resolved {
		return "proposed"
	}
	if p.ID != c.finalProposal {
		return "not approved"
	}
	if c.succeeded {
		return "approved & completed"
	}
	return "approved, not completed"
}

// promisedTotals sums the current, deduplicated commitments (see
// commitOrUpdate) targeting proposalID — what is promised right now, as
// distinct from totalCredits/totalLabor, which only count resources that
// have actually moved through a validated "contribute" action.
func (c *community) promisedTotals(proposalID string) (credits, labor int) {
	for _, cm := range c.commitments {
		if cm.Proposal == proposalID {
			credits += cm.Credits
			labor += cm.Labor
		}
	}
	return credits, labor
}

// PrivateState tells an agent only its own resources, its own role-specific
// fact if it has one, its own last inspection result, and its own vote. Never
// anyone else's.
func (c *community) PrivateState(id sim.AgentID) map[string]any {
	st := map[string]any{
		"your_credits": c.credits[id],
		"your_labor":   c.labor[id],
	}
	switch id {
	case "dov":
		st["known_structural_cost"] = c.structuralCost
	case "mira":
		st["market_insight_credits_per_season"] = c.miraInsight
	case "ada":
		st["households_wanting_free_space"] = c.adaInsight
	case "rook":
		st["outside_funding_secured"] = c.rookBonus
	}
	if n, ok := c.lastInspection[id]; ok {
		st["last_inspection_result"] = n
	}
	if v, voted := c.votes[id]; voted {
		if v == "" {
			st["your_vote"] = "abstained"
		} else {
			st["your_vote"] = v
		}
	}
	if c.richGrounding {
		if cs := c.commitmentsBy(id); len(cs) > 0 {
			parts := make([]string, 0, len(cs))
			for _, cm := range cs {
				var transferredCredits, transferredLabor int
				if p := c.findProposal(cm.Proposal); p != nil {
					transferredCredits, transferredLabor = p.ContribCredits[id], p.ContribLabor[id]
				}
				status := "outstanding"
				if c.isFulfilled(cm) {
					status = "fulfilled"
				}
				parts = append(parts, fmt.Sprintf("%s: promised %dc/%dl, transferred %dc/%dl (%s)",
					cm.Proposal, cm.Credits, cm.Labor, transferredCredits, transferredLabor, status))
			}
			st["your_commitments"] = strings.Join(parts, "; ")
		}
	}
	return st
}

func (c *community) BeginRound(round int) []sim.Event {
	c.round = round
	if round > c.totalRounds() {
		if !c.done {
			c.done = true
			c.reason = "the week ended with no verdict"
		}
		return nil
	}

	day, opportunity := c.dayOf(round), c.opportunityOf(round)

	var events []sim.Event
	if c.complicationDay > 0 && day == c.complicationDay && opportunity == 1 && !c.complicationApplied {
		c.structuralCost += c.complicationSurcharge
		c.complicationApplied = true
		c.complicationRound = round
		events = append(events, sim.Event{
			Type: sim.EventState,
			Text: fmt.Sprintf("Day %d: an inspection ordered by the town turns up additional structural damage. "+
				"The real repair cost is now publicly known to be %d credits.", day, c.structuralCost),
			Data:       map[string]any{"phase": "complication", "structural_cost": c.structuralCost},
			Visibility: sim.PublicTo(),
		})
	}

	switch {
	case c.isVoteDay():
		c.votes = map[sim.AgentID]string{}
		events = append(events, sim.Event{
			Type:       sim.EventState,
			Text:       fmt.Sprintf("Day %d — the deadline. Talking and project work are closed. Cast your vote, or pass to abstain.", day),
			Data:       map[string]any{"phase": "vote"},
			Visibility: sim.PublicTo(),
		})
	case c.isLockInDay():
		events = append(events, sim.Event{
			Type: sim.EventState,
			Text: fmt.Sprintf("Day %d: talking is closed for the rest of the week. Formalize what's been discussed — "+
				"propose, support, commit, contribute, or inspect — because after today only the vote remains.", day),
			Data:       map[string]any{"phase": "lock_in"},
			Visibility: sim.PublicTo(),
		})
	default:
		text := fmt.Sprintf("Day %d of %d.", day, c.days)
		if c.opportunitiesPerDay > 1 {
			text = fmt.Sprintf("Day %d of %d, opportunity %d of %d.", day, c.days, opportunity, c.opportunitiesPerDay)
		}
		events = append(events, sim.Event{
			Type:       sim.EventState,
			Text:       text,
			Data:       map[string]any{"phase": "open"},
			Visibility: sim.PublicTo(),
		})
	}
	return events
}

// EndRound resolves the vote exactly once, on the deadline day. Winning the
// vote is necessary but not sufficient: the winning proposal still has to be
// actually funded, and its declared budget still has to cover the real repair
// cost, or it does not proceed regardless of how the vote went.
func (c *community) EndRound(round int) []sim.Event {
	if c.done || c.resolved || c.dayOf(round) < c.days || !c.isLastOpportunityOfDay(round) {
		return nil
	}
	c.resolved = true

	tally := map[string]int{}
	for _, voter := range c.participants {
		tally[c.votes[voter]]++
	}
	abstained := tally[""]
	delete(tally, "")

	var top []string
	max := 0
	for pid, n := range tally {
		switch {
		case n > max:
			max, top = n, []string{pid}
		case n == max:
			top = append(top, pid)
		}
	}

	parts := make([]string, 0, len(c.participants))
	for _, voter := range c.participants {
		if v := c.votes[voter]; v == "" {
			parts = append(parts, fmt.Sprintf("%s abstained", voter))
		} else {
			parts = append(parts, fmt.Sprintf("%s voted %s", voter, v))
		}
	}
	reveal := fmt.Sprintf("The votes are in: %s.", strings.Join(parts, ", "))
	_ = abstained

	c.done = true
	if len(top) != 1 {
		c.reason = "no majority — the building remains undecided"
		c.finalReason = "no proposal received a clear plurality of votes"
		return []sim.Event{{
			Type: sim.EventState, Text: reveal + " No clear decision — the building remains unused for now.",
			Data: map[string]any{"resolved": true, "winner": ""}, Visibility: sim.PublicTo(),
		}}
	}

	winner := c.findProposal(top[0])
	c.finalProposal = winner.ID
	funded := winner.totalCredits() >= winner.Credits && winner.totalLabor() >= winner.Labor
	safe := winner.Credits >= c.structuralCost

	switch {
	case !safe:
		c.finalReason = fmt.Sprintf("%s won the vote, but its budget (%d credits) does not cover the real repair cost (%d credits) — unsafe to proceed", winner.ID, winner.Credits, c.structuralCost)
		c.reason = fmt.Sprintf("%s won the vote but was unsafe", winner.ID)
	case !funded:
		c.finalReason = fmt.Sprintf("%s won the vote, but was never fully funded (%d/%d credits, %d/%d labor raised)", winner.ID, winner.totalCredits(), winner.Credits, winner.totalLabor(), winner.Labor)
		c.reason = fmt.Sprintf("%s won the vote but was underfunded", winner.ID)
	default:
		c.finalReason = fmt.Sprintf("%s won the vote, was fully funded, and its budget covers the real repair cost — it proceeds", winner.ID)
		c.reason = fmt.Sprintf("%s was selected and will proceed", winner.ID)
		c.succeeded = true
	}
	return []sim.Event{{
		Type: sim.EventState, Text: reveal + " " + c.finalReason,
		Data:       map[string]any{"resolved": true, "winner": winner.ID, "funded": funded, "safe": safe},
		Visibility: sim.PublicTo(),
	}}
}

func (c *community) Done() (bool, string) { return c.done, c.reason }

func (c *community) Outcome() sim.Outcome {
	var lines []string
	switch {
	case len(c.proposals) == 0:
		lines = append(lines, "No proposal was ever put forward. The building remains unused.")
	case c.finalProposal == "":
		lines = append(lines, fmt.Sprintf("No decision was reached: %s.", nonEmptyString(c.finalReason, c.reason, "the week ended without a vote")))
	case c.succeeded:
		w := c.findProposal(c.finalProposal)
		lines = append(lines, fmt.Sprintf("%s — %q — was selected and funded. It proceeds.", w.ID, w.Text))
	default:
		w := c.findProposal(c.finalProposal)
		lines = append(lines, fmt.Sprintf("%s — %q — won the vote but did not proceed: %s", w.ID, w.Text, c.finalReason))
	}

	scores := map[sim.AgentID]float64{}
	for _, id := range c.participants {
		scores[id] = 0
	}
	if c.succeeded {
		w := c.findProposal(c.finalProposal)
		scores[w.By] = 100
		for _, id := range c.participants {
			if id == w.By {
				continue
			}
			switch {
			case w.ContribCredits[id] > 0 || w.ContribLabor[id] > 0:
				scores[id] = 50
			case w.Supporters[id]:
				scores[id] = 25
			}
		}
	}

	metrics := map[string]any{
		"winning_proposal":                     c.finalProposal,
		"feasible":                             c.succeeded,
		"structural_cost_initial":              c.initialStructuralCost,
		"structural_cost_final":                c.structuralCost,
		"complication_applied":                 c.complicationApplied,
		"proposals_total":                      len(c.proposals),
		"proposals_revised":                    c.verbCounts["revise"],
		"proposals_revised_after_complication": c.verbCounts["revise_after_complication"],
		"proposals_feasible_detail":            c.proposalsFeasibleDetail(),
		"proposals_safe_detail":                c.proposalsSafeDetail(),
		"support_events":                       c.verbCounts["support"],
		"commitments_made":                     len(c.commitments), // = unique_active_commitments; kept for continuity
		"commitments_made_after_complication":  c.verbCounts["commit_after_complication"],
		"commitment_fulfillment_rate":          c.commitmentFulfillmentRate(),

		// Repeated or revised promises to the same proposal update one
		// record rather than inflating the count — see commitOrUpdate. These
		// three make that collapsing legible rather than hidden:
		"unique_active_commitments":            len(c.commitments),
		"commitment_updates":                   c.verbCounts["commit_updated"],
		"repeated_identical_commitments":       c.verbCounts["commit_repeated"],
		"total_committed_credits":              c.totalCommittedCredits(),
		"total_committed_labor":                c.totalCommittedLabor(),
		"total_fulfilled_credits":              c.totalFulfilledCredits(),
		"total_fulfilled_labor":                c.totalFulfilledLabor(),
		"outstanding_obligations":              c.outstandingObligations(),
		"contributions_total_credits":          c.totalContributedCredits(),
		"contributions_total_labor":            c.totalContributedLabor(),
		"contributions_after_complication":     c.verbCounts["contribute_after_complication"],
		"inspections_performed":                c.verbCounts["inspect"],
		"structural_knowledge_shared_publicly": c.structuralKnowledgeShared,
		"agents_adapted_after_complication":    c.adaptedAfterComplication(),
		"public_utterances":                    sum(c.said),
		"private_messages":                     sum(c.whispered),
		"passes":                               sum(c.passed),
		"vote_tally":                           c.voteTally(),
		"contributions_by_agent":               c.contributionsByAgent(),
		"commitments_detail":                   c.commitmentsDetail(),
		"resource_usage":                       c.resourceUsage(),

		// Feasibility metadata, computed before anyone acted, revealed only
		// now: whether a successful outcome was ever reachable from the
		// starting position, independent of whether anyone found it.
		"feasible_outcome_possible_at_start": c.feasibleAtStart,
		"total_credits_available":            c.totalCreditsAvailable,

		// Where in the run the first proposal landed, if any — the
		// "planning vs. execution" timing diagnostic.
		"first_proposal_day":         c.dayOrZero(c.firstProposalRound),
		"first_proposal_opportunity": c.opportunityOrZero(c.firstProposalRound),
	}
	return sim.Outcome{Summary: strings.Join(lines, "\n"), Scores: scores, Metrics: metrics}
}

// dayOrZero/opportunityOrZero report 0 for "never happened" (round 0) rather
// than misreporting it as day 1 via dayOf/opportunityOf's normal arithmetic.
func (c *community) dayOrZero(round int) int {
	if round == 0 {
		return 0
	}
	return c.dayOf(round)
}

func (c *community) opportunityOrZero(round int) int {
	if round == 0 {
		return 0
	}
	return c.opportunityOf(round)
}

// proposalsFeasibleDetail reports, per proposal, whether its own declared
// budget covered the real repair cost at the time it was last created or
// revised — ground truth the proposer may not have known, surfaced only
// here, never during play.
func (c *community) proposalsFeasibleDetail() []map[string]any {
	out := make([]map[string]any, 0, len(c.proposals))
	for _, p := range c.proposals {
		out = append(out, map[string]any{
			"proposal": p.ID, "by": string(p.By), "credits": p.Credits,
			"feasible_when_created": p.FeasibleWhenCreated,
		})
	}
	return out
}

// proposalsSafeDetail is the state-accuracy counterpart to
// proposalsFeasibleDetail: for every proposal, whether its declared budget
// (as it stands right now, not as of creation) covers the FINAL real repair
// cost — a deterministic claim-vs-reality check available for any proposal,
// not only whichever one happens to win the vote.
func (c *community) proposalsSafeDetail() []map[string]any {
	out := make([]map[string]any, 0, len(c.proposals))
	for _, p := range c.proposals {
		out = append(out, map[string]any{
			"proposal": p.ID, "by": string(p.By), "declared_credits": p.Credits,
			"safe_against_final_cost": p.Credits >= c.structuralCost,
		})
	}
	return out
}

// --- small helpers -----------------------------------------------------------

// commitOrUpdate records a promise, applying the scenario's commitment
// semantics: there is at most one active commitment per (agent, proposal)
// pair. A first promise to a given proposal creates a new record ("created").
// Repeating or revising a promise you have already made to the SAME proposal
// updates that one record in place instead of adding an independent
// obligation — "repeated" if the amounts are identical, "updated" if they
// differ. This is what keeps commitment counts, and therefore fulfillment
// rates, from being inflated by an agent simply restating the same promise
// several times. It returns the verbCounts key for what happened and the
// public-facing event text.
func (c *community) commitOrUpdate(id sim.AgentID, proposalID string, credits, labor int) (verb, text string) {
	for i := range c.commitments {
		cm := &c.commitments[i]
		if cm.By != id || cm.Proposal != proposalID {
			continue
		}
		if cm.Credits == credits && cm.Labor == labor {
			// A verbatim repeat still refreshes the opportunity-to-fulfill
			// snapshot — the agent's balance may have changed since the
			// original promise, and that is what "opportunity" should
			// reflect: the most recent moment the promise was reaffirmed.
			cm.BalanceCreditsAtCommit, cm.BalanceLaborAtCommit = c.credits[id], c.labor[id]
			cm.Round = c.round
			cm.Updates++
			return "commit_repeated", fmt.Sprintf("repeats its commitment of %d credits and %d labor to %s", credits, labor, proposalID)
		}
		prevCredits, prevLabor := cm.Credits, cm.Labor
		cm.Credits, cm.Labor = credits, labor
		cm.BalanceCreditsAtCommit, cm.BalanceLaborAtCommit = c.credits[id], c.labor[id]
		cm.Round = c.round
		cm.Updates++
		return "commit_updated", fmt.Sprintf("revises its commitment to %s from %d/%d credits/labor to %d/%d", proposalID, prevCredits, prevLabor, credits, labor)
	}
	c.commitments = append(c.commitments, ccCommitment{
		By: id, Proposal: proposalID, Credits: credits, Labor: labor, Round: c.round,
		BalanceCreditsAtCommit: c.credits[id], BalanceLaborAtCommit: c.labor[id],
	})
	return "commit_created", fmt.Sprintf("commits %d credits and %d labor to %s", credits, labor, proposalID)
}

func (c *community) findProposal(ref string) *ccProposal {
	ref = strings.ToLower(strings.TrimSpace(ref))
	if ref == "" {
		return nil
	}
	for _, p := range c.proposals {
		if strings.ToLower(p.ID) == ref {
			return p
		}
	}
	for _, p := range c.proposals {
		if "p"+ref == strings.ToLower(p.ID) {
			return p
		}
	}
	return nil
}

func (c *community) proposalIDs() []string {
	out := make([]string, 0, len(c.proposals))
	for _, p := range c.proposals {
		out = append(out, p.ID)
	}
	if len(out) == 0 {
		return []string{"none yet"}
	}
	return out
}

func (c *community) commitmentsBy(id sim.AgentID) []ccCommitment {
	var out []ccCommitment
	for _, cm := range c.commitments {
		if cm.By == id {
			out = append(out, cm)
		}
	}
	return out
}

func (c *community) totalContributedCredits() int {
	t := 0
	for _, p := range c.proposals {
		t += p.totalCredits()
	}
	return t
}

func (c *community) totalContributedLabor() int {
	t := 0
	for _, p := range c.proposals {
		t += p.totalLabor()
	}
	return t
}

// commitmentFulfillmentRate compares each promise against that same agent's
// cumulative actual contributions to that same proposal by the end of the
// run. It is a floor, not a ledger: an agent fulfilling five small promises
// with one large contribution reads as fulfilled, which is the right call
// here since the mechanic that matters is "did the resources arrive," not
// "was each individual promise settled separately."
func (c *community) commitmentFulfillmentRate() float64 {
	if len(c.commitments) == 0 {
		return 1
	}
	fulfilled := 0
	for _, cm := range c.commitments {
		if c.isFulfilled(cm) {
			fulfilled++
		}
	}
	return float64(fulfilled) / float64(len(c.commitments))
}

// isFulfilled reports whether cm's proposal has received at least as much
// actual contribution, from that same agent, as it currently promises.
func (c *community) isFulfilled(cm ccCommitment) bool {
	p := c.findProposal(cm.Proposal)
	return p != nil && p.ContribCredits[cm.By] >= cm.Credits && p.ContribLabor[cm.By] >= cm.Labor
}

// totalCommittedCredits/Labor sum only the current, deduplicated commitments
// — what is actually owed right now, not every historical promise attempt.
func (c *community) totalCommittedCredits() int {
	t := 0
	for _, cm := range c.commitments {
		t += cm.Credits
	}
	return t
}

func (c *community) totalCommittedLabor() int {
	t := 0
	for _, cm := range c.commitments {
		t += cm.Labor
	}
	return t
}

// totalFulfilledCredits/Labor sum the committed amount for commitments that
// were actually honoured — the promised figure, not the (possibly larger)
// actual contribution, since what matters here is whether the promise itself
// was kept.
func (c *community) totalFulfilledCredits() int {
	t := 0
	for _, cm := range c.commitments {
		if c.isFulfilled(cm) {
			t += cm.Credits
		}
	}
	return t
}

func (c *community) totalFulfilledLabor() int {
	t := 0
	for _, cm := range c.commitments {
		if c.isFulfilled(cm) {
			t += cm.Labor
		}
	}
	return t
}

// outstandingObligations counts current commitments not yet fulfilled.
func (c *community) outstandingObligations() int {
	n := 0
	for _, cm := range c.commitments {
		if !c.isFulfilled(cm) {
			n++
		}
	}
	return n
}

// adaptedAfterComplication counts agents whose final vote went to a proposal
// they had not already backed before the complication landed — an observable
// behavior change, not an inferred motive.
func (c *community) adaptedAfterComplication() int {
	if !c.complicationApplied {
		return 0
	}
	n := 0
	for _, id := range c.participants {
		pre := c.preComplicationSupport[id]
		if pre == nil {
			continue
		}
		finalVote, voted := c.votes[id]
		if !voted || finalVote == "" {
			continue
		}
		if !pre[finalVote] {
			n++
		}
	}
	return n
}

func (c *community) voteTally() map[string]int {
	out := map[string]int{}
	for _, voter := range c.participants {
		v := c.votes[voter]
		if v == "" {
			v = "abstain"
		}
		out[v]++
	}
	return out
}

func (c *community) contributionsByAgent() map[string]map[string]int {
	out := map[string]map[string]int{}
	for _, id := range c.participants {
		out[string(id)] = map[string]int{"credits": 0, "labor": 0}
	}
	for _, rec := range c.contributions {
		e := out[string(rec.By)]
		e["credits"] += rec.Credits
		e["labor"] += rec.Labor
	}
	return out
}

// commitmentsDetail reports, per promise: whether it was ultimately honoured,
// and — only when it was not — whether the agent had a genuine chance to
// (it held at least that much when it made the promise) versus the promise
// being unmeetable from the start, or the proposal having disappeared, or the
// scenario simply closing first. This is deliberately a fact about capacity
// at commit time, not a claim about why an agent did or didn't follow
// through.
func (c *community) commitmentsDetail() []map[string]any {
	out := make([]map[string]any, 0, len(c.commitments))
	for _, cm := range c.commitments {
		fulfilled := c.isFulfilled(cm)
		hadOpportunity := cm.BalanceCreditsAtCommit >= cm.Credits && cm.BalanceLaborAtCommit >= cm.Labor
		entry := map[string]any{
			"by": string(cm.By), "proposal": cm.Proposal, "credits": cm.Credits, "labor": cm.Labor,
			"round": cm.Round, "fulfilled": fulfilled, "updates": cm.Updates,
		}
		if !fulfilled {
			entry["had_opportunity_when_promised"] = hadOpportunity
		}
		out = append(out, entry)
	}
	return out
}

func (c *community) resourceUsage() map[string]map[string]int {
	out := map[string]map[string]int{}
	for _, id := range c.participants {
		out[string(id)] = map[string]int{
			"credits_spent":     c.startCredits[id] - c.credits[id],
			"labor_spent":       c.startLabor[id] - c.labor[id],
			"credits_remaining": c.credits[id],
			"labor_remaining":   c.labor[id],
		}
	}
	return out
}

func nonEmptyString(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
