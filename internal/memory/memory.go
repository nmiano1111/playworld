// Package memory persists the part of an agent that outlives a single model
// call: its identity, what it episodically observed, what it believes about
// other agents, and what it has committed to.
//
// The MVP deliberately uses one JSON file per agent. That is enough for a
// handful of agents on a laptop, is trivially inspectable with `cat`, and keeps
// the Store interface narrow enough to swap for SQLite or a retrieval index
// later without touching the agent.
package memory

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Kind classifies an episodic entry so that later policies (summarisation,
// relevance ranking) have something to filter on.
type Kind string

const (
	KindObservation Kind = "observation" // something the agent was permitted to see
	KindOwnAction   Kind = "own_action"  // something the agent itself did
	KindNote        Kind = "note"        // a note the agent chose to keep
	KindOutcome     Kind = "outcome"     // a scenario result
)

// Episode is a single remembered event. Importance exists so a future policy
// can prune or rank without a schema change; the MVP sets it to 1.
type Episode struct {
	RunID      string    `json:"run_id"`
	Scenario   string    `json:"scenario"`
	Round      int       `json:"round"`
	Time       time.Time `json:"time"`
	Kind       Kind      `json:"kind"`
	Text       string    `json:"text"`
	Private    bool      `json:"private,omitempty"`
	Importance float64   `json:"importance,omitempty"`
	// Stamp is the environment's own label for when this happened (e.g. a
	// simulated "Day 2 09:30"), kept so a resumed agent can be shown its
	// history in the same terms it originally saw it. Empty for round-based
	// scenarios.
	Stamp string `json:"stamp,omitempty"`
	// Type is the kind of event an observation came from ("say", "action",
	// ...), when known. Older memory files leave it empty.
	Type string `json:"type,omitempty"`
}

// Belief is what an agent thinks about another agent. Only the latest note per
// subject is kept; the update count hints at how much contact they have had.
type Belief struct {
	Subject string    `json:"subject"`
	Note    string    `json:"note"`
	Updated time.Time `json:"updated"`
	Updates int       `json:"updates"`

	// Basis and Confidence are optional provenance: why the agent believes
	// this, and how sure it is. Both are exactly what the agent itself
	// reported — never inferred or corrected by Go code, and never promoted
	// to fact. Empty when the agent didn't say.
	Basis      string `json:"basis,omitempty"`
	Confidence string `json:"confidence,omitempty"`
	// Stamp is when (in the environment's own terms) the note was last
	// revised, so a stale impression reads as stale.
	Stamp string `json:"stamp,omitempty"`
}

// Commitment records a promise or proposal the agent considers itself bound by.
type Commitment struct {
	RunID   string    `json:"run_id"`
	Text    string    `json:"text"`
	Created time.Time `json:"created"`
	Done    bool      `json:"done,omitempty"`
}

// State is the full persisted memory of one agent. Short-term conversational
// context is *not* stored here: it is rebuilt each run from episodes, which
// keeps the on-disk shape stable as context policies change.
type State struct {
	AgentID string `json:"agent_id"`
	Name    string `json:"name"`
	// Persona is a snapshot of the identity the agent last ran with, so a
	// reader of the file can see who this memory belonged to.
	Persona string `json:"persona,omitempty"`

	Episodes    []Episode         `json:"episodes"`
	Beliefs     map[string]Belief `json:"beliefs"`
	Commitments []Commitment      `json:"commitments"`

	// Plan is the agent's own small, self-authored statement of intent —
	// written and revised by the model, never by Go code. Unlike Episodes,
	// Beliefs and Commitments, a plan is scoped to one scenario instance
	// (see ActivePlan): identity and social memory are meant to carry
	// forward between sessions, but an intention formed for one scenario
	// run must not silently resurface as "active" in an unrelated later
	// one just because it shares a memory store.
	Plan *Plan `json:"plan,omitempty"`

	Runs       []string  `json:"runs,omitempty"`
	LastActive time.Time `json:"last_active,omitempty"`

	// Version guards against silently misreading an older on-disk layout.
	Version int `json:"version"`
}

// Plan is an agent's lightweight, persistent statement of intent: a primary
// goal, how it currently means to get there, a few concrete next steps, and
// whatever it currently sees standing in the way. All four fields are
// optional and model-authored; Go code never fills in or corrects any of
// them, and a plan never executes an action by itself — it is carried into
// the prompt as context the model may act on, revise, or ignore.
type Plan struct {
	// RunID scopes the plan to one scenario instance. ActivePlan returns nil
	// for any other run, which is what stops an intention from one scenario
	// quietly becoming "active" in a different one.
	RunID string `json:"run_id"`

	PrimaryGoal     string   `json:"primary_goal,omitempty"`
	CurrentStrategy string   `json:"current_strategy,omitempty"`
	NextSteps       []string `json:"next_steps,omitempty"`
	Obstacles       []string `json:"obstacles,omitempty"`
	// Commitments are promises the agent considers itself bound by, in its
	// own words. Like everything in a plan they are notes, never transfers:
	// only an executed action moves anything.
	Commitments []string `json:"commitments,omitempty"`
	// Done lists steps the agent itself recorded as accomplished.
	Done []string `json:"done,omitempty"`

	Updated time.Time `json:"updated"`
	// At and Stamp are when the plan was written in simulated time (zero
	// and empty for round-based runs), so its age can be shown.
	At    time.Time `json:"at,omitempty"`
	Stamp string    `json:"stamp,omitempty"`
	// Revisions counts replacements after the plan's first creation within
	// this run, so a reader can see how often the agent changed its mind.
	Revisions int `json:"revisions,omitempty"`

	// GoalSince/GoalStamp are when the agent first adopted (roughly) this
	// primary goal, carried across rewrites that keep it; Turns counts the
	// decisions it has taken while holding it. ObstacleSince and
	// ObstacleTurns do the same for each obstacle, index for index. They
	// are bookkeeping about the agent's own notes, never instructions.
	GoalSince     time.Time `json:"goal_since,omitempty"`
	GoalStamp     string    `json:"goal_stamp,omitempty"`
	Turns         int       `json:"turns,omitempty"`
	ObstacleSince []string  `json:"obstacle_since,omitempty"`
	ObstacleTurns []int     `json:"obstacle_turns,omitempty"`
}

// IsEmpty reports whether a plan has no actual content — used to decide
// whether "sending a plan" should count as creating one.
func (p *Plan) IsEmpty() bool {
	return p == nil || (p.PrimaryGoal == "" && p.CurrentStrategy == "" && len(p.NextSteps) == 0 && len(p.Obstacles) == 0 && len(p.Commitments) == 0 && len(p.Done) == 0)
}

// CurrentVersion is the on-disk schema version.
const CurrentVersion = 1

// NewState returns an empty state for an agent.
func NewState(id, name string) *State {
	return &State{
		AgentID: id,
		Name:    name,
		Beliefs: map[string]Belief{},
		Version: CurrentVersion,
	}
}

// AddEpisode appends an episode, capping total retained history.
func (s *State) AddEpisode(e Episode, max int) {
	if e.Importance == 0 {
		e.Importance = 1
	}
	s.Episodes = append(s.Episodes, e)
	if max > 0 && len(s.Episodes) > max {
		s.Episodes = s.Episodes[len(s.Episodes)-max:]
	}
}

// SetBelief records or updates what the agent thinks about a subject.
func (s *State) SetBelief(subject, note string) {
	subject = strings.TrimSpace(subject)
	note = strings.TrimSpace(note)
	if subject == "" || note == "" {
		return
	}
	if s.Beliefs == nil {
		s.Beliefs = map[string]Belief{}
	}
	b := s.Beliefs[subject]
	b.Subject = subject
	b.Note = note
	b.Updated = time.Now().UTC()
	b.Updates++
	s.Beliefs[subject] = b
}

// SetBeliefProvenance records why and how confidently the agent holds a
// belief it already has about subject. It only annotates an existing entry
// (created by SetBelief) — there is nothing to attach provenance to
// otherwise — and never touches Note, Updated or Updates, so recording
// provenance is not itself a belief update.
func (s *State) SetBeliefProvenance(subject, basis, confidence string) {
	subject = strings.TrimSpace(subject)
	if subject == "" || s.Beliefs == nil {
		return
	}
	b, ok := s.Beliefs[subject]
	if !ok {
		return
	}
	if basis = strings.TrimSpace(basis); basis != "" {
		b.Basis = basis
	}
	if confidence = strings.TrimSpace(confidence); confidence != "" {
		b.Confidence = confidence
	}
	s.Beliefs[subject] = b
}

// StampBelief records when a belief was last revised; a no-op for an empty
// stamp or an unknown subject.
func (s *State) StampBelief(subject, stamp string) {
	subject = strings.TrimSpace(subject)
	b, ok := s.Beliefs[subject]
	if !ok || stamp == "" {
		return
	}
	b.Stamp = stamp
	s.Beliefs[subject] = b
}

// ActivePlan returns the agent's plan only if it belongs to runID — a plan
// made during a different scenario instance is not returned, even though it
// is still sitting in the same memory file. This is the one enforcement
// point for "a plan for proposal p1 in one scenario must not silently become
// an active plan in a completely unrelated scenario": every caller that
// wants "my current plan" goes through here rather than reading s.Plan
// directly.
func (s *State) ActivePlan(runID string) *Plan {
	if s.Plan == nil || s.Plan.RunID != runID {
		return nil
	}
	return s.Plan
}

// SetPlan replaces the agent's plan for runID. Replacing an existing plan
// from the SAME run counts as a revision; a plan found to belong to a
// different run is simply overwritten without counting against it, since it
// was never this run's plan to revise in the first place. Passing an empty
// Plan clears it (see ClearPlan for the explicit form).
func (s *State) SetPlan(runID string, p Plan) {
	p.RunID = runID
	p.Updated = time.Now().UTC()
	if s.Plan != nil && s.Plan.RunID == runID {
		p.Revisions = s.Plan.Revisions + 1
	}
	if p.IsEmpty() {
		s.Plan = nil
		return
	}
	s.Plan = &p
}

// ClearPlan abandons the current plan if (and only if) it belongs to runID.
func (s *State) ClearPlan(runID string) {
	if s.Plan != nil && s.Plan.RunID == runID {
		s.Plan = nil
	}
}

// AddCommitment records a commitment, ignoring exact duplicates.
func (s *State) AddCommitment(runID, text string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}
	for _, c := range s.Commitments {
		if c.Text == text {
			return
		}
	}
	s.Commitments = append(s.Commitments, Commitment{RunID: runID, Text: text, Created: time.Now().UTC()})
}

// NoteRun records participation in a run.
func (s *State) NoteRun(runID string) {
	for _, r := range s.Runs {
		if r == runID {
			return
		}
	}
	s.Runs = append(s.Runs, runID)
}

// Recent returns the n most recent episodes, oldest first.
func (s *State) Recent(n int) []Episode {
	if n <= 0 || n >= len(s.Episodes) {
		return append([]Episode(nil), s.Episodes...)
	}
	return append([]Episode(nil), s.Episodes[len(s.Episodes)-n:]...)
}

// OwnNotes returns the most recent n notes the agent wrote to itself during
// runID, oldest first. Long-lived environments surface these alongside the
// transcript, since a note written on Day 1 has long scrolled out of it.
func (s *State) OwnNotes(runID string, n int) []Episode {
	var out []Episode
	for _, e := range s.Episodes {
		if e.RunID == runID && e.Kind == KindNote {
			out = append(out, e)
		}
	}
	if n > 0 && len(out) > n {
		out = out[len(out)-n:]
	}
	return out
}

// Salient returns up to n observations from runID that were marked
// consequential (Importance >= 2) and happened before cutoff — the events a
// long-lived agent should still be able to recall after they have scrolled
// out of its transcript. Exact repeats are kept once (the latest), and the
// result is oldest first.
func (s *State) Salient(runID string, cutoff time.Time, n int) []Episode {
	if n <= 0 {
		return nil
	}
	seen := map[string]bool{}
	var out []Episode
	for i := len(s.Episodes) - 1; i >= 0 && len(out) < n; i-- {
		e := s.Episodes[i]
		if e.RunID != runID || e.Kind != KindObservation || e.Importance < 2 || !e.Time.Before(cutoff) || seen[e.Text] {
			continue
		}
		seen[e.Text] = true
		out = append(out, e)
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// PriorRuns returns episodes from runs other than the given one, oldest first.
// This is the MVP's stand-in for relevance retrieval: "what do I remember from
// before today".
func (s *State) PriorRuns(exclude string, n int) []Episode {
	var out []Episode
	for _, e := range s.Episodes {
		if e.RunID != exclude {
			out = append(out, e)
		}
	}
	if n > 0 && len(out) > n {
		out = out[len(out)-n:]
	}
	return out
}

// SortedBeliefs returns beliefs ordered by subject for stable rendering.
func (s *State) SortedBeliefs() []Belief {
	out := make([]Belief, 0, len(s.Beliefs))
	for _, b := range s.Beliefs {
		out = append(out, b)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Subject < out[j].Subject })
	return out
}

// Store persists agent memory. Keeping this an interface (rather than using
// FileStore directly) is what makes SQLite or an in-memory test double a
// drop-in replacement.
type Store interface {
	Load(agentID string) (*State, error)
	Save(*State) error
	// List returns the agent ids with persisted state.
	List() ([]string, error)
	// Reset deletes persisted state. With no ids, it deletes everything.
	Reset(agentIDs ...string) error
}

// FileStore keeps one JSON file per agent under Dir.
type FileStore struct {
	Dir string
	mu  sync.Mutex
}

// NewFileStore returns a store rooted at dir, creating it if needed.
func NewFileStore(dir string) (*FileStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("memory: create %s: %w", dir, err)
	}
	return &FileStore{Dir: dir}, nil
}

func (f *FileStore) path(agentID string) string {
	return filepath.Join(f.Dir, safeName(agentID)+".json")
}

// Load reads an agent's state, returning a fresh empty state if none exists.
func (f *FileStore) Load(agentID string) (*State, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	raw, err := os.ReadFile(f.path(agentID))
	if err != nil {
		if os.IsNotExist(err) {
			return NewState(agentID, ""), nil
		}
		return nil, fmt.Errorf("memory: read %s: %w", f.path(agentID), err)
	}
	var s State
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("memory: parse %s: %w", f.path(agentID), err)
	}
	if s.Version > CurrentVersion {
		return nil, fmt.Errorf("memory: %s was written by a newer version (%d > %d); run `playworld reset` to clear it", f.path(agentID), s.Version, CurrentVersion)
	}
	if s.Beliefs == nil {
		s.Beliefs = map[string]Belief{}
	}
	s.AgentID = agentID
	return &s, nil
}

// Save writes state atomically so an interrupted run cannot truncate a file.
func (f *FileStore) Save(s *State) error {
	if s == nil || s.AgentID == "" {
		return fmt.Errorf("memory: cannot save state without an agent id")
	}
	f.mu.Lock()
	defer f.mu.Unlock()

	s.Version = CurrentVersion
	s.LastActive = time.Now().UTC()
	raw, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("memory: encode %s: %w", s.AgentID, err)
	}
	final := f.path(s.AgentID)
	tmp := final + ".tmp"
	if err := os.WriteFile(tmp, append(raw, '\n'), 0o644); err != nil {
		return fmt.Errorf("memory: write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, final); err != nil {
		return fmt.Errorf("memory: replace %s: %w", final, err)
	}
	return nil
}

// List returns agent ids that have persisted state.
func (f *FileStore) List() ([]string, error) {
	entries, err := os.ReadDir(f.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("memory: list %s: %w", f.Dir, err)
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		out = append(out, strings.TrimSuffix(e.Name(), ".json"))
	}
	sort.Strings(out)
	return out, nil
}

// Reset removes persisted state for the named agents, or all of them.
func (f *FileStore) Reset(agentIDs ...string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(agentIDs) == 0 {
		entries, err := os.ReadDir(f.Dir)
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return fmt.Errorf("memory: list %s: %w", f.Dir, err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			if err := os.Remove(filepath.Join(f.Dir, e.Name())); err != nil {
				return fmt.Errorf("memory: remove %s: %w", e.Name(), err)
			}
		}
		return nil
	}
	for _, id := range agentIDs {
		if err := os.Remove(f.path(id)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("memory: remove %s: %w", f.path(id), err)
		}
	}
	return nil
}

// MemStore is a non-persistent Store for tests.
type MemStore struct {
	mu     sync.Mutex
	states map[string][]byte
}

// NewMemStore returns an empty in-memory store.
func NewMemStore() *MemStore { return &MemStore{states: map[string][]byte{}} }

func (m *MemStore) Load(agentID string) (*State, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	raw, ok := m.states[agentID]
	if !ok {
		return NewState(agentID, ""), nil
	}
	var s State
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, err
	}
	if s.Beliefs == nil {
		s.Beliefs = map[string]Belief{}
	}
	return &s, nil
}

func (m *MemStore) Save(s *State) error {
	if s == nil || s.AgentID == "" {
		return fmt.Errorf("memory: cannot save state without an agent id")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	raw, err := json.Marshal(s)
	if err != nil {
		return err
	}
	m.states[s.AgentID] = raw
	return nil
}

func (m *MemStore) List() ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, 0, len(m.states))
	for k := range m.states {
		out = append(out, k)
	}
	sort.Strings(out)
	return out, nil
}

func (m *MemStore) Reset(agentIDs ...string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(agentIDs) == 0 {
		m.states = map[string][]byte{}
		return nil
	}
	for _, id := range agentIDs {
		delete(m.states, id)
	}
	return nil
}

// safeName keeps agent ids from escaping the memory directory.
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
		return "agent"
	}
	return b.String()
}
