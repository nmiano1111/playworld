// Package world is the open-ended, persistent, event-driven simulation mode.
//
// Unlike the round-based scenarios, a world has no shared objective, no
// scripted ending and no turn order. It is a small text town — a square, a
// market, a workshop and one home per resident — with simulated time, a
// deliberately tiny economy (credits and materials), projects anyone can start,
// and a discrete-event scheduler that decides who gets a moment to think, and
// when, based on what actually happened.
//
// The world is persistent: World is saved to disk as plain JSON, including its
// pending event queue, and an execution session is just a bounded slice of
// simulated time over it. Resuming a world continues exactly where the last
// session stopped.
//
// The World type owns all objective state and is the only thing that can change
// it; agents' beliefs about it live in their own memory and can be wrong. The
// Engine (engine.go) drives agents through it and knows nothing about any LLM.
package world

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/nmiano/playworld/internal/sim"
)

// SchemaVersion guards against silently misreading an older world file.
const SchemaVersion = 1

// Goods is the whole economy: two resources. Labor exists only as a project
// requirement, supplied by spending time building.
type Goods struct {
	Credits   int `json:"credits" yaml:"credits"`
	Materials int `json:"materials" yaml:"materials"`
}

func (g Goods) String() string {
	return fmt.Sprintf("%d credits, %d materials", g.Credits, g.Materials)
}

// Covers reports whether g has at least x of everything.
func (g Goods) Covers(x Goods) bool { return g.Credits >= x.Credits && g.Materials >= x.Materials }

func (g Goods) Plus(x Goods) Goods  { return Goods{g.Credits + x.Credits, g.Materials + x.Materials} }
func (g Goods) Minus(x Goods) Goods { return Goods{g.Credits - x.Credits, g.Materials - x.Materials} }
func (g Goods) IsZero() bool        { return g.Credits == 0 && g.Materials == 0 }

// Place kinds. A kind decides what can be done there; everything else about a
// place is description.
const (
	PlaceSquare    = "square"
	PlaceMarket    = "market"
	PlaceWorkshop  = "workshop"
	PlaceResidence = "residence"
)

// PlaceSpec configures one public place.
type PlaceSpec struct {
	ID          string `yaml:"id"`
	Name        string `yaml:"name"`
	Kind        string `yaml:"kind"`
	Description string `yaml:"description"`
}

// Economy holds every number in the economy, so the world can be tuned from
// YAML without touching code. All availability is bounded per day.
type Economy struct {
	ShiftHours            int `yaml:"shift_hours" json:"shift_hours"`
	ShiftCredits          int `yaml:"shift_credits" json:"shift_credits"`
	ShiftsPerDay          int `yaml:"shifts_per_day" json:"shifts_per_day"`
	MaterialPrice         int `yaml:"material_price" json:"material_price"`
	MarketMaterialsPerDay int `yaml:"market_materials_per_day" json:"market_materials_per_day"`
	SalvageHours          int `yaml:"salvage_hours" json:"salvage_hours"`
	SalvageMaterials      int `yaml:"salvage_materials" json:"salvage_materials"`
	ScrapPerDay           int `yaml:"scrap_per_day" json:"scrap_per_day"`
	BuildHours            int `yaml:"build_hours" json:"build_hours"`
	// ProjectYield is credits a completed project pays its owner each dawn —
	// the persistent consequence of having built something.
	ProjectYield int `yaml:"project_yield" json:"project_yield"`
	// Caps on what one project may declare, so costs stay bounded.
	MaxProjectCredits   int `yaml:"max_project_credits" json:"max_project_credits"`
	MaxProjectMaterials int `yaml:"max_project_materials" json:"max_project_materials"`
	MaxProjectLabor     int `yaml:"max_project_labor" json:"max_project_labor"`
	MaxOpenProjects     int `yaml:"max_open_projects" json:"max_open_projects"`
	OfferHours          int `yaml:"offer_hours" json:"offer_hours"`
	MaxNotices          int `yaml:"max_notices" json:"max_notices"`
}

// Scheduling holds every knob of the event-driven scheduler.
type Scheduling struct {
	// StartHour is when a new world's clock starts on Day 1.
	StartHour int `yaml:"start_hour" json:"start_hour"`
	// DayStartHour and NightStartHour bound waking hours. Outside them,
	// ordinary triggers wait until morning; explicitly requested wakeups and
	// appointments are still honoured.
	DayStartHour   int `yaml:"day_start_hour" json:"day_start_hour"`
	NightStartHour int `yaml:"night_start_hour" json:"night_start_hour"`
	TravelMinutes  int `yaml:"travel_minutes" json:"travel_minutes"`
	// IdleMinutes is the default activation policy: an agent that has not
	// been activated for this long gets a quiet opportunity to decide.
	IdleMinutes int `yaml:"idle_minutes" json:"idle_minutes"`
	// MaxIdleMinutes caps the idle interval as it backs off through
	// consecutive idle opportunities the agent declined.
	MaxIdleMinutes int `yaml:"max_idle_minutes" json:"max_idle_minutes"`
	// CooldownMinutes is the minimum simulated time between two activations
	// of the same agent. Triggers inside it coalesce into the next one.
	CooldownMinutes int `yaml:"cooldown_minutes" json:"cooldown_minutes"`
	// ReactionMinutes delays activations triggered by someone else (a
	// message, being spoken to), so a burst of events lands in one
	// activation instead of several.
	ReactionMinutes int `yaml:"reaction_minutes" json:"reaction_minutes"`
	// MaxActivationsPerDay caps one agent's activations per simulated day.
	MaxActivationsPerDay int `yaml:"max_activations_per_day" json:"max_activations_per_day"`
	// WakeOnSpeech: "mention" (default) wakes only listeners named in what
	// was said, or the only listener present; "all" wakes everyone present;
	// "none" wakes nobody — they hear it at their next activation.
	WakeOnSpeech string `yaml:"wake_on_speech" json:"wake_on_speech"`
	// WakeOnArrival wakes the people already at a place when someone arrives.
	WakeOnArrival bool `yaml:"wake_on_arrival" json:"wake_on_arrival"`
	// MaxReplyChain bounds reactive activations: an agent woken by being
	// spoken to (or sent a note, an offer...) that replies can wake the other
	// side again, but only this many reactions deep. Past it, what is said
	// is still delivered and heard at the listener's next activation; it
	// just no longer wakes them. Non-social causes reset the chain.
	MaxReplyChain int `yaml:"max_reply_chain" json:"max_reply_chain"`
	// Bounds on self-scheduled intentions and arranged appointments.
	MinIntentionMinutes        int `yaml:"min_intention_minutes" json:"min_intention_minutes"`
	MaxIntentionHours          int `yaml:"max_intention_hours" json:"max_intention_hours"`
	MaxPendingIntentions       int `yaml:"max_pending_intentions" json:"max_pending_intentions"`
	MaxIntentionsPerActivation int `yaml:"max_intentions_per_activation" json:"max_intentions_per_activation"`
	MaxPendingAppointments     int `yaml:"max_pending_appointments" json:"max_pending_appointments"`
	// MinAppointmentMinutes is how far ahead a meeting must be arranged.
	// A meeting coming due is a fresh stimulus for everyone in it, so
	// allowing them back-to-back would let agents bypass MaxReplyChain.
	MinAppointmentMinutes int `yaml:"min_appointment_minutes" json:"min_appointment_minutes"`
	// MaxRejectRetriggers bounds how many times in a row a refused action
	// earns the agent an immediate follow-up activation.
	MaxRejectRetriggers int `yaml:"max_reject_retriggers" json:"max_reject_retriggers"`
	// MaxConsecutiveInvalid is how many activations in a row may end with a
	// refused action before the agent is cooled off until its idle time.
	MaxConsecutiveInvalid int `yaml:"max_consecutive_invalid" json:"max_consecutive_invalid"`
}

// ResidentSpec gives one resident modest starting differences: what it
// holds, where it starts, and what it already knows. Knowledge is delivered
// once, privately, as the resident's first observations in the world; it is
// meant for true facts and existing familiarity, never for instructions.
type ResidentSpec struct {
	Goods *Goods   `yaml:"goods,omitempty" json:"goods,omitempty"`
	Start string   `yaml:"start,omitempty" json:"start,omitempty"`
	Knows []string `yaml:"knows,omitempty" json:"knows,omitempty"`
}

// Settings is the full configuration of a world. It is copied into the world
// file at creation, so a world keeps its own rules even if the YAML changes.
type Settings struct {
	Places     []PlaceSpec             `yaml:"places" json:"places"`
	Scheduling Scheduling              `yaml:"scheduling" json:"scheduling"`
	Economy    Economy                 `yaml:"economy" json:"economy"`
	Starting   Goods                   `yaml:"starting" json:"starting"`
	PerAgent   map[string]ResidentSpec `yaml:"per_agent,omitempty" json:"per_agent,omitempty"`
	// Agent protocol options, carried with the world so every session of it
	// runs the same way unless a flag overrides them.
	DualAction bool `yaml:"dual_actions" json:"dual_actions"`
	Planning   bool `yaml:"planning" json:"planning"`
	Beliefs    bool `yaml:"beliefs" json:"beliefs"`
	// Cognition selects how residents are grounded: "grounded" gives each
	// activation an ordered, sectioned account of the resident's own state,
	// surroundings, projects and transactions, context-specific constraints,
	// repeat-aware reaction triggers and intention superseding, and turns on
	// the agents' context refinements (see agent.Cognition). "legacy" (or
	// empty, as in worlds created before this setting existed) reproduces
	// the original observation and prompt, which is what makes a
	// before/after comparison possible without editing code.
	Cognition string `yaml:"cognition,omitempty" json:"cognition,omitempty"`
}

// Cognition modes.
const (
	CognitionLegacy   = "legacy"
	CognitionGrounded = "grounded"
)

// Grounded reports whether the world runs the grounded cognition mode.
func (s Settings) Grounded() bool { return s.Cognition == CognitionGrounded }

// DefaultSettings is a small town that works out of the box.
func DefaultSettings() Settings {
	return Settings{
		Places: []PlaceSpec{
			{ID: "square", Name: "the square", Kind: PlaceSquare, Description: "the open heart of town, with benches and a public notice board"},
			{ID: "market", Name: "the market", Kind: PlaceMarket, Description: "a few stalls; there is paid shift work and building materials for sale"},
			{ID: "workshop", Name: "the workshop", Kind: PlaceWorkshop, Description: "a shared workshop with tools and a pile of salvageable scrap"},
		},
		Scheduling: Scheduling{
			StartHour: 7, DayStartHour: 7, NightStartHour: 22, TravelMinutes: 15,
			IdleMinutes: 180, MaxIdleMinutes: 540, CooldownMinutes: 20, ReactionMinutes: 2, MaxActivationsPerDay: 12,
			WakeOnSpeech: "mention", WakeOnArrival: false, MaxReplyChain: 3,
			MinIntentionMinutes: 20, MaxIntentionHours: 72, MaxPendingIntentions: 4,
			MaxIntentionsPerActivation: 2, MaxPendingAppointments: 4, MinAppointmentMinutes: 60, MaxRejectRetriggers: 1,
			MaxConsecutiveInvalid: 4,
		},
		Economy: Economy{
			ShiftHours: 3, ShiftCredits: 6, ShiftsPerDay: 4,
			MaterialPrice: 3, MarketMaterialsPerDay: 8,
			SalvageHours: 2, SalvageMaterials: 2, ScrapPerDay: 6,
			BuildHours: 2, ProjectYield: 2,
			MaxProjectCredits: 60, MaxProjectMaterials: 30, MaxProjectLabor: 24, MaxOpenProjects: 2,
			OfferHours: 24, MaxNotices: 8,
		},
		Starting:   Goods{Credits: 10, Materials: 1},
		DualAction: true,
		Planning:   true,
		Cognition:  CognitionGrounded,
	}
}

// LoadSettings reads a world YAML file over the defaults: anything the file
// leaves out keeps its default value.
func LoadSettings(path string) (Settings, error) {
	s := DefaultSettings()
	raw, err := os.ReadFile(path)
	if err != nil {
		return s, fmt.Errorf("world config: %w", err)
	}
	var file struct {
		Settings `yaml:",inline"`
		// Agents is accepted so one file can name the cast for `world new`.
		Agents []string `yaml:"agents,omitempty"`
		Seed   int64    `yaml:"seed,omitempty"`
	}
	file.Settings = s
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&file); err != nil {
		return s, fmt.Errorf("world config %s: %w", path, err)
	}
	return file.Settings, file.Settings.Validate()
}

// ConfigCast reads just the optional `agents` and `seed` keys of a world
// YAML file.
func ConfigCast(path string) ([]string, int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var file struct {
		Agents []string `yaml:"agents"`
		Seed   int64    `yaml:"seed"`
	}
	if err := yaml.Unmarshal(raw, &file); err != nil {
		return nil, 0, err
	}
	return file.Agents, file.Seed, nil
}

// Validate rejects settings that would make the scheduler misbehave.
func (s Settings) Validate() error {
	sc := s.Scheduling
	switch {
	case sc.DayStartHour < 0 || sc.DayStartHour > 23 || sc.NightStartHour <= sc.DayStartHour || sc.NightStartHour > 24:
		return fmt.Errorf("world config: need 0 <= day_start_hour < night_start_hour <= 24")
	case sc.IdleMinutes < 30:
		return fmt.Errorf("world config: idle_minutes must be at least 30")
	case sc.CooldownMinutes < 1:
		return fmt.Errorf("world config: cooldown_minutes must be at least 1")
	case sc.MinIntentionMinutes < sc.CooldownMinutes:
		return fmt.Errorf("world config: min_intention_minutes must be >= cooldown_minutes")
	case sc.MaxActivationsPerDay < 1:
		return fmt.Errorf("world config: max_activations_per_day must be at least 1")
	case sc.TravelMinutes < 1:
		return fmt.Errorf("world config: travel_minutes must be at least 1")
	case sc.MaxReplyChain < 0:
		return fmt.Errorf("world config: max_reply_chain cannot be negative")
	}
	switch s.Cognition {
	case "", CognitionLegacy, CognitionGrounded:
	default:
		return fmt.Errorf(`world config: cognition must be "grounded" or "legacy"`)
	}
	switch sc.WakeOnSpeech {
	case "mention", "all", "none":
	default:
		return fmt.Errorf(`world config: wake_on_speech must be "mention", "all" or "none"`)
	}
	kinds := map[string]bool{}
	ids := map[string]bool{}
	for _, p := range s.Places {
		if p.ID == "" || ids[p.ID] {
			return fmt.Errorf("world config: place ids must be unique and non-empty")
		}
		ids[p.ID] = true
		switch p.Kind {
		case PlaceSquare, PlaceMarket, PlaceWorkshop:
		default:
			return fmt.Errorf("world config: place %q has unknown kind %q (square, market, workshop)", p.ID, p.Kind)
		}
		kinds[p.Kind] = true
	}
	for _, k := range []string{PlaceSquare, PlaceMarket, PlaceWorkshop} {
		if !kinds[k] {
			return fmt.Errorf("world config: need at least one %s place", k)
		}
	}
	return nil
}

// Location is a place in the world.
type Location struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Kind        string      `json:"kind"`
	Description string      `json:"description"`
	Owner       sim.AgentID `json:"owner,omitempty"`
	// Features are permanent additions: completed projects.
	Features []string `json:"features,omitempty"`
}

// Activity is something an agent is busy with until End. An agent is never
// activated while busy; triggers wait for the activity to finish.
type Activity struct {
	Kind    string    `json:"kind"` // travel, work, salvage, build
	Detail  string    `json:"detail"`
	Start   time.Time `json:"start"`
	End     time.Time `json:"end"`
	To      string    `json:"to,omitempty"`
	Project string    `json:"project,omitempty"`
	ItemID  int64     `json:"item_id"`
}

// Resident is the objective state of one agent in the world.
type Resident struct {
	ID       sim.AgentID `json:"id"`
	Name     string      `json:"name"`
	Home     string      `json:"home"`
	Location string      `json:"location"` // empty while travelling
	Goods    Goods       `json:"goods"`
	Activity *Activity   `json:"activity,omitempty"`

	LastActivation time.Time `json:"last_activation,omitempty"`
	Activations    int       `json:"activations"`
	// ActDay/ActCount enforce MaxActivationsPerDay.
	ActDay   int `json:"act_day,omitempty"`
	ActCount int `json:"act_count,omitempty"`

	// ActivateItem and IdleItem point at this agent's pending activation and
	// idle items, so triggers coalesce into one and idle can be replaced.
	ActivateItem int64 `json:"activate_item,omitempty"`
	IdleItem     int64 `json:"idle_item,omitempty"`
	// RejectStreak counts consecutive activations with a refused action.
	RejectStreak int `json:"reject_streak,omitempty"`
	// IdleStreak counts consecutive idle activations the agent declined;
	// it backs off the idle interval.
	IdleStreak int `json:"idle_streak,omitempty"`
	// Knows is starting knowledge not yet delivered; Briefed marks that it
	// was, so it is never delivered twice across sessions.
	Knows   []string `json:"knows,omitempty"`
	Briefed bool     `json:"briefed,omitempty"`
	// Totals are this resident's cumulative counters (resources earned,
	// given, received, contributed...).
	Totals map[string]int `json:"totals,omitempty"`
	// Said is what this resident recently said or whispered, and to whom.
	Said []Utterance `json:"said,omitempty"`
	// Errand is the resident's latest walk: where from, where to, when, and
	// what it said it was going for — so on arrival it can see why it came.
	Errand *Errand `json:"errand,omitempty"`
	// Setbacks are this resident's recent refused actions, newest last.
	Setbacks []Setback `json:"setbacks,omitempty"`
	// Seen is what the resident last saw of each market and workshop it
	// visited: a remembered glimpse, not a live view.
	Seen map[string]Glimpse `json:"seen,omitempty"`
	// AfterSetback is the kind of move that last failed for this resident,
	// until its next decision (for the strategy-change metric only).
	AfterSetback string `json:"after_setback,omitempty"`
}

// Errand is one walk and its stated purpose.
type Errand struct {
	From    string     `json:"from"`
	To      string     `json:"to"`
	For     string     `json:"for,omitempty"`
	Left    time.Time  `json:"left"`
	Arrived *time.Time `json:"arrived,omitempty"`
}

// Setback is something that did not go the resident's way; for now, a
// refused action.
type Setback struct {
	At   time.Time `json:"at"`
	Kind string    `json:"kind"` // refused
	Code string    `json:"code,omitempty"`
	Text string    `json:"text"`
}

// maxSetbacks bounds how many setbacks a resident keeps.
const maxSetbacks = 10

// Glimpse is what a resident saw of a place's stock at a moment.
type Glimpse struct {
	At        time.Time `json:"at"`
	Materials int       `json:"materials,omitempty"`
	Shifts    int       `json:"shifts,omitempty"`
	Scrap     int       `json:"scrap,omitempty"`
}

// Busy reports whether the resident is mid-activity at t.
func (r *Resident) Busy(t time.Time) bool { return r.Activity != nil && t.Before(r.Activity.End) }

// Project is a self-declared undertaking anyone can contribute to. Its
// requirements are set by its owner and bounded by Economy; it completes
// when every requirement is met and then stays in the world for good.
type Project struct {
	ID           string                `json:"id"`
	Owner        sim.AgentID           `json:"owner"`
	Name         string                `json:"name"`
	Description  string                `json:"description,omitempty"`
	Location     string                `json:"location"`
	Needs        Goods                 `json:"needs"`
	NeedsLabor   int                   `json:"needs_labor"`
	Raised       Goods                 `json:"raised"`
	Labor        int                   `json:"labor"`
	Contributors map[sim.AgentID]Goods `json:"contributors,omitempty"`
	Builders     map[sim.AgentID]int   `json:"builders,omitempty"`
	Created      time.Time             `json:"created"`
	Completed    *time.Time            `json:"completed,omitempty"`
	// Abandoned is when the owner gave the project up unfinished; whatever
	// had been contributed went back to the contributors.
	Abandoned *time.Time `json:"abandoned,omitempty"`
}

// Done reports whether the project was completed.
func (p *Project) Done() bool { return p.Completed != nil }

// Closed reports whether the project can no longer be worked on: completed
// or abandoned.
func (p *Project) Closed() bool { return p.Completed != nil || p.Abandoned != nil }

func (p *Project) satisfied() bool {
	return p.Raised.Covers(p.Needs) && p.Labor >= p.NeedsLabor
}

// remaining describes what an unfinished project still lacks, e.g. "2
// materials and 4 labor hours".
func (p *Project) remaining() string {
	var parts []string
	if c := p.Needs.Credits - p.Raised.Credits; c > 0 {
		parts = append(parts, fmt.Sprintf("%d credits", c))
	}
	if m := p.Needs.Materials - p.Raised.Materials; m > 0 {
		parts = append(parts, fmt.Sprintf("%d materials", m))
	}
	if l := p.NeedsLabor - p.Labor; l > 0 {
		parts = append(parts, fmt.Sprintf("%d labor hours", l))
	}
	if len(parts) == 0 {
		return "nothing"
	}
	return strings.Join(parts, " and ")
}

// Offer is a proposed exchange. Nothing moves until it is accepted, and the
// world checks both sides can actually pay at that moment.
type Offer struct {
	ID      string      `json:"id"`
	From    sim.AgentID `json:"from"`
	To      sim.AgentID `json:"to"`
	Give    Goods       `json:"give"`
	Want    Goods       `json:"want"`
	Note    string      `json:"note,omitempty"`
	Created time.Time   `json:"created"`
	Expires time.Time   `json:"expires"`
	Status  string      `json:"status"` // open, accepted, declined, expired, withdrawn
	// Resolved is when the offer stopped being open (absent in older
	// worlds and while open).
	Resolved *time.Time `json:"resolved,omitempty"`
}

// settle closes an offer with a final status at t.
func (o *Offer) settle(status string, t time.Time) {
	o.Status = status
	o.Resolved = &t
}

// Transfer is one completed movement of goods: a gift, one side of a trade,
// or a contribution to a project. The ledger is the world's authoritative
// record of what actually moved, as opposed to what was offered or said.
type Transfer struct {
	Time  time.Time   `json:"time"`
	From  sim.AgentID `json:"from"`
	To    sim.AgentID `json:"to,omitempty"` // empty for a project contribution
	Goods Goods       `json:"goods"`
	Kind  string      `json:"kind"`          // gift, trade, contribution
	Ref   string      `json:"ref,omitempty"` // the offer or project id
}

// maxLedger bounds the ledger kept in the world file.
const maxLedger = 400

// Utterance is something a resident said or wrote to others, kept briefly so
// a repeat can be told from new information.
type Utterance struct {
	At    time.Time     `json:"at"`
	Text  string        `json:"text"`
	Heard []sim.AgentID `json:"heard"`
}

// maxUtterances bounds how many recent utterances a resident keeps.
const maxUtterances = 8

// Appointment is a meeting one agent arranged with others.
type Appointment struct {
	ID     string        `json:"id"`
	By     sim.AgentID   `json:"by"`
	With   []sim.AgentID `json:"with"`
	Place  string        `json:"place"`
	At     time.Time     `json:"at"`
	What   string        `json:"what"`
	ItemID int64         `json:"item_id"`
	Status string        `json:"status"` // pending, due
}

// Intention is an agent's own request to reconsider something later.
type Intention struct {
	ID      string      `json:"id"`
	Agent   sim.AgentID `json:"agent"`
	At      time.Time   `json:"at"`
	Note    string      `json:"note"`
	Created time.Time   `json:"created"`
	ItemID  int64       `json:"item_id"`
	Status  string      `json:"status"` // pending, due, cancelled, superseded
	// Supersedes names the intention this one replaced, if any.
	Supersedes string `json:"supersedes,omitempty"`
}

// Notice is a post on a square's notice board.
type Notice struct {
	ID     string      `json:"id"`
	By     sim.AgentID `json:"by"`
	Place  string      `json:"place"`
	Text   string      `json:"text"`
	Posted time.Time   `json:"posted"`
}

// HistoryEntry is a completed event, kept (bounded) in the world file so
// `world inspect` can show recent history without a log file.
type HistoryEntry struct {
	Seq        int           `json:"seq"`
	Time       time.Time     `json:"time"`
	Type       sim.EventType `json:"type"`
	Actor      sim.AgentID   `json:"actor,omitempty"`
	Text       string        `json:"text"`
	Visibility string        `json:"visibility"`
}

// maxHistory bounds the history kept in the world file; logs keep it all.
const maxHistory = 400

// Session records one execution session over the world.
type Session struct {
	Number      int            `json:"number"`
	ID          string         `json:"id"`
	Started     time.Time      `json:"started"`
	Ended       time.Time      `json:"ended"`
	SimFrom     time.Time      `json:"sim_from"`
	SimTo       time.Time      `json:"sim_to"`
	StopReason  string         `json:"stop_reason"`
	Activations int            `json:"activations"`
	LLMCalls    int            `json:"llm_calls"`
	LogPath     string         `json:"log_path,omitempty"`
	Metrics     map[string]int `json:"metrics,omitempty"`
}

// Stock is the per-day bounded availability of the economy.
type Stock struct {
	Shifts    map[string]int `json:"shifts"`    // per market place
	Materials map[string]int `json:"materials"` // per market place
	Scrap     map[string]int `json:"scrap"`     // per workshop place
}

// World is the complete persistent state. Every field is exported so the
// whole thing round-trips through JSON; the queue's heap is rebuilt on load.
type World struct {
	Version  int       `json:"version"`
	Name     string    `json:"name"`
	Seed     int64     `json:"seed"`
	Created  time.Time `json:"created"`
	Now      time.Time `json:"now"`
	Settings Settings  `json:"settings"`

	// RosterPath is where the residents' personas were loaded from; the cast
	// itself is Order.
	RosterPath string                    `json:"roster_path"`
	Order      []sim.AgentID             `json:"order"`
	Residents  map[sim.AgentID]*Resident `json:"residents"`
	Locations  []*Location               `json:"locations"`
	Stock      Stock                     `json:"stock"`

	Projects     []*Project     `json:"projects"`
	Offers       []*Offer       `json:"offers"`
	Appointments []*Appointment `json:"appointments"`
	Intentions   []*Intention   `json:"intentions"`
	Notices      []*Notice      `json:"notices"`
	Ledger       []Transfer     `json:"ledger,omitempty"`

	Pending  []*Item        `json:"queue"`
	NextItem int64          `json:"next_item"`
	Counters map[string]int `json:"counters"`
	EventSeq int            `json:"event_seq"`
	History  []HistoryEntry `json:"history"`
	Sessions []Session      `json:"sessions"`
	Totals   map[string]int `json:"totals"`

	queue *Queue
	// chain is the reply-chain depth of the activation currently being
	// applied (0 outside activations). Transient: set by the engine.
	chain int
}

// New creates a fresh world for the given cast. The first activation of each
// resident is staggered a few minutes apart from the start hour, so the town
// wakes up one person at a time rather than all at once.
func New(name string, seed int64, settings Settings, cast []sim.AgentID, names map[sim.AgentID]string) (*World, error) {
	if err := settings.Validate(); err != nil {
		return nil, err
	}
	if len(cast) == 0 {
		return nil, fmt.Errorf("world: no residents")
	}
	w := &World{
		Version: SchemaVersion, Name: name, Seed: seed, Created: time.Now().UTC(),
		Now:       At(1, settings.Scheduling.StartHour, 0),
		Settings:  settings,
		Residents: map[sim.AgentID]*Resident{},
		Stock:     Stock{Shifts: map[string]int{}, Materials: map[string]int{}, Scrap: map[string]int{}},
		Counters:  map[string]int{},
		Totals:    map[string]int{},
	}
	w.queue = newQueue(nil)
	for _, p := range settings.Places {
		w.Locations = append(w.Locations, &Location{ID: p.ID, Name: p.Name, Kind: p.Kind, Description: p.Description})
	}
	for _, id := range cast {
		if _, dup := w.Residents[id]; dup {
			return nil, fmt.Errorf("world: duplicate resident %q", id)
		}
		name := names[id]
		if name == "" {
			name = string(id)
		}
		home := "home_" + string(id)
		w.Locations = append(w.Locations, &Location{
			ID: home, Name: name + "'s home", Kind: PlaceResidence, Owner: id,
			Description: "a small private home",
		})
		w.Residents[id] = &Resident{ID: id, Name: name, Home: home, Location: home, Goods: settings.Starting}
		w.Order = append(w.Order, id)
	}
	for id, spec := range settings.PerAgent {
		r := w.Residents[sim.AgentID(id)]
		if r == nil {
			continue // configured for someone not in this cast
		}
		if spec.Goods != nil {
			r.Goods = *spec.Goods
		}
		if spec.Start != "" {
			l := w.FindPlace(r.ID, spec.Start)
			if l == nil || (l.Kind == PlaceResidence && l.Owner != r.ID) {
				return nil, fmt.Errorf("world config: %s cannot start at %q", id, spec.Start)
			}
			r.Location = l.ID
		}
		r.Knows = append([]string(nil), spec.Knows...)
	}
	w.restock()
	// The first dawn is the next one after the start time.
	dawn := At(Day(w.Now), settings.Scheduling.DayStartHour-1, 0)
	if !dawn.After(w.Now) {
		dawn = dawn.Add(24 * time.Hour)
	}
	w.push(&Item{At: dawn, Kind: KindDawn})
	for i, id := range w.Order {
		r := w.Residents[id]
		at := w.Now.Add(time.Duration(i*7) * time.Minute)
		r.IdleItem = w.push(&Item{At: at, Kind: KindIdle, Agent: id})
	}
	return w, nil
}

// Queue exposes the pending events, for inspection.
func (w *World) Queue() *Queue { return w.queue }

func (w *World) push(it *Item) int64 {
	w.NextItem++
	it.ID = w.NextItem
	w.queue.Push(it)
	return it.ID
}

func (w *World) nextID(prefix string) string {
	w.Counters[prefix]++
	return fmt.Sprintf("%s%d", prefix, w.Counters[prefix])
}

func (w *World) count(key string, n int) {
	if n == 0 {
		return
	}
	if w.Totals == nil {
		w.Totals = map[string]int{}
	}
	w.Totals[key] += n
}

// tally adds to one resident's own counters.
func (w *World) tally(id sim.AgentID, key string, n int) {
	r := w.Residents[id]
	if r == nil || n == 0 {
		return
	}
	if r.Totals == nil {
		r.Totals = map[string]int{}
	}
	r.Totals[key] += n
}

// Location finds a place by id.
func (w *World) Location(id string) *Location {
	for _, l := range w.Locations {
		if l.ID == id {
			return l
		}
	}
	return nil
}

// FindPlace resolves a model's loose reference to a place: an id, a name
// ("the market", "market"), "home" for one's own home, or "<name>'s home".
func (w *World) FindPlace(self sim.AgentID, ref string) *Location {
	ref = strings.ToLower(strings.TrimSpace(ref))
	ref = strings.TrimPrefix(ref, "the ")
	if ref == "" {
		return nil
	}
	if r, ok := w.Residents[self]; ok && (ref == "home" || ref == "my home" || ref == "house" || ref == "my house") {
		return w.Location(r.Home)
	}
	for _, l := range w.Locations {
		if strings.ToLower(l.ID) == ref || strings.TrimPrefix(strings.ToLower(l.Name), "the ") == ref {
			return l
		}
	}
	for _, l := range w.Locations {
		if l.Kind == PlaceResidence {
			owner := strings.ToLower(string(l.Owner))
			name := strings.ToLower(w.Residents[l.Owner].Name)
			for _, form := range []string{owner, name} {
				if ref == form+"'s home" || ref == form+"'s house" || ref == form+"s home" || ref == "home "+form || ref == form+" home" {
					return l
				}
			}
		}
	}
	return nil
}

// Here lists the residents currently at a place (not travelling there).
func (w *World) Here(place string) []sim.AgentID {
	var out []sim.AgentID
	for _, id := range w.Order {
		if w.Residents[id].Location == place {
			out = append(out, id)
		}
	}
	return out
}

func (w *World) name(id sim.AgentID) string {
	if r, ok := w.Residents[id]; ok && r.Name != "" {
		return r.Name
	}
	return string(id)
}

// Names maps resident ids to display names.
func (w *World) Names() map[sim.AgentID]string {
	out := map[sim.AgentID]string{}
	for id, r := range w.Residents {
		out[id] = r.Name
	}
	return out
}

func (w *World) placeName(id string) string {
	if l := w.Location(id); l != nil {
		return l.Name
	}
	return id
}

func (w *World) project(ref string) *Project {
	ref = strings.ToLower(strings.TrimSpace(ref))
	for _, p := range w.Projects {
		if strings.ToLower(p.ID) == ref {
			return p
		}
	}
	for _, p := range w.Projects {
		if strings.ToLower(p.Name) == ref {
			return p
		}
	}
	return nil
}

func (w *World) offer(ref string) *Offer {
	ref = strings.ToLower(strings.TrimSpace(ref))
	for _, o := range w.Offers {
		if strings.ToLower(o.ID) == ref {
			return o
		}
	}
	return nil
}

// restock resets the per-day bounded availability.
func (w *World) restock() {
	e := w.Settings.Economy
	for _, l := range w.Locations {
		switch l.Kind {
		case PlaceMarket:
			w.Stock.Shifts[l.ID] = e.ShiftsPerDay
			w.Stock.Materials[l.ID] = e.MarketMaterialsPerDay
		case PlaceWorkshop:
			w.Stock.Scrap[l.ID] = e.ScrapPerDay
		}
	}
}

func (w *World) record(e sim.Event) {
	w.History = append(w.History, HistoryEntry{
		Seq: e.Seq, Time: e.Time, Type: e.Type, Actor: e.Actor, Text: e.Text, Visibility: e.Visibility.Describe(),
	})
	if len(w.History) > maxHistory {
		w.History = w.History[len(w.History)-maxHistory:]
	}
}

// --- persistence -------------------------------------------------------------

// Store keeps each world in its own directory:
//
//	<root>/<name>/world.json   the world, including its pending queue
//	<root>/<name>/memory/      the residents' agent memory (one JSON each)
//	<root>/<name>/logs/        one JSONL log per execution session
type Store struct{ Root string }

func (s Store) Dir(name string) string       { return filepath.Join(s.Root, safeName(name)) }
func (s Store) path(name string) string      { return filepath.Join(s.Dir(name), "world.json") }
func (s Store) MemoryDir(name string) string { return filepath.Join(s.Dir(name), "memory") }
func (s Store) LogsDir(name string) string   { return filepath.Join(s.Dir(name), "logs") }

// Exists reports whether a world has been created under name.
func (s Store) Exists(name string) bool {
	_, err := os.Stat(s.path(name))
	return err == nil
}

// Save writes the world atomically, so an interrupted session can never
// leave a truncated world file behind.
func (s Store) Save(w *World) error {
	w.Pending = w.queue.Sorted()
	w.Version = SchemaVersion
	raw, err := json.MarshalIndent(w, "", "  ")
	if err != nil {
		return fmt.Errorf("world: encode %s: %w", w.Name, err)
	}
	if err := os.MkdirAll(s.Dir(w.Name), 0o755); err != nil {
		return fmt.Errorf("world: create %s: %w", s.Dir(w.Name), err)
	}
	final := s.path(w.Name)
	tmp := final + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return fmt.Errorf("world: write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, final); err != nil {
		return fmt.Errorf("world: replace %s: %w", final, err)
	}
	return nil
}

// Load reads a saved world and rebuilds its scheduler.
func (s Store) Load(name string) (*World, error) {
	raw, err := os.ReadFile(s.path(name))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("no world named %q in %s (create it with `playworld world new %s`)", name, s.Root, name)
		}
		return nil, fmt.Errorf("world: read %s: %w", s.path(name), err)
	}
	return Decode(raw)
}

// Decode parses a world file's contents.
func Decode(raw []byte) (*World, error) {
	var w World
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("world: parse: %w", err)
	}
	if w.Version > SchemaVersion {
		return nil, fmt.Errorf("world: file was written by a newer version (%d > %d)", w.Version, SchemaVersion)
	}
	if w.Residents == nil {
		w.Residents = map[sim.AgentID]*Resident{}
	}
	if w.Counters == nil {
		w.Counters = map[string]int{}
	}
	if w.Totals == nil {
		w.Totals = map[string]int{}
	}
	for _, m := range []*map[string]int{&w.Stock.Shifts, &w.Stock.Materials, &w.Stock.Scrap} {
		if *m == nil {
			*m = map[string]int{}
		}
	}
	w.queue = newQueue(w.Pending)
	w.Pending = nil
	return &w, nil
}

// List returns the names of saved worlds.
func (s Store) List() ([]string, error) {
	entries, err := os.ReadDir(s.Root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() && s.Exists(e.Name()) {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

// Delete removes a world and everything stored with it.
func (s Store) Delete(name string) error {
	if !s.Exists(name) {
		return fmt.Errorf("no world named %q in %s", name, s.Root)
	}
	return os.RemoveAll(s.Dir(name))
}

// safeName keeps world names from escaping the store directory.
func safeName(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	if b.Len() == 0 {
		return "world"
	}
	return b.String()
}

// ValidName reports whether name is usable as-is for a world directory.
func ValidName(name string) bool { return name != "" && safeName(name) == name }
