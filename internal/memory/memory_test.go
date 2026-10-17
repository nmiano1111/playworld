package memory

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFileStoreRoundTrip(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	// A never-seen agent gets a usable empty state rather than an error.
	fresh, err := store.Load("mira")
	if err != nil {
		t.Fatal(err)
	}
	if len(fresh.Episodes) != 0 || fresh.Beliefs == nil {
		t.Fatalf("fresh state should be empty with a usable belief map, got %+v", fresh)
	}

	fresh.Name = "Mira"
	fresh.AddEpisode(Episode{RunID: "run-1", Round: 1, Kind: KindObservation, Text: "Dov asked for 40"}, 0)
	fresh.SetBelief("dov", "says numbers out loud; probably honest")
	fresh.AddCommitment("run-1", "split evenly")
	fresh.NoteRun("run-1")
	if err := store.Save(fresh); err != nil {
		t.Fatal(err)
	}

	// A second store over the same directory sees the persisted state, which
	// is what "state survives between sessions" actually means.
	reopened, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := reopened.Load("mira")
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "Mira" {
		t.Errorf("Name = %q", got.Name)
	}
	if len(got.Episodes) != 1 || got.Episodes[0].Text != "Dov asked for 40" {
		t.Errorf("episodes = %+v", got.Episodes)
	}
	if got.Beliefs["dov"].Note == "" {
		t.Errorf("belief about dov was lost: %+v", got.Beliefs)
	}
	if len(got.Commitments) != 1 {
		t.Errorf("commitments = %+v", got.Commitments)
	}
	if got.Version != CurrentVersion {
		t.Errorf("Version = %d, want %d", got.Version, CurrentVersion)
	}
}

func TestFileStoreIsolatesAgents(t *testing.T) {
	store, err := NewFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"mira", "dov"} {
		s := NewState(id, id)
		s.SetBelief("other", "secret about "+id)
		if err := store.Save(s); err != nil {
			t.Fatal(err)
		}
	}
	mira, _ := store.Load("mira")
	dov, _ := store.Load("dov")
	if mira.Beliefs["other"].Note == dov.Beliefs["other"].Note {
		t.Fatal("agents must not share a memory file")
	}
}

func TestResetClearsSelectivelyAndEntirely(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileStore(dir)
	for _, id := range []string{"a", "b", "c"} {
		if err := store.Save(NewState(id, id)); err != nil {
			t.Fatal(err)
		}
	}

	if err := store.Reset("b"); err != nil {
		t.Fatal(err)
	}
	ids, _ := store.List()
	if len(ids) != 2 || ids[0] != "a" || ids[1] != "c" {
		t.Fatalf("after Reset(b), List = %v", ids)
	}

	if err := store.Reset(); err != nil {
		t.Fatal(err)
	}
	ids, _ = store.List()
	if len(ids) != 0 {
		t.Fatalf("after full Reset, List = %v", ids)
	}
	// Resetting a missing agent is not an error: `reset` must be idempotent.
	if err := store.Reset("never-existed"); err != nil {
		t.Fatalf("Reset of a missing agent returned %v", err)
	}
}

func TestEpisodeCapKeepsNewest(t *testing.T) {
	s := NewState("a", "A")
	for i := 0; i < 10; i++ {
		s.AddEpisode(Episode{Text: string(rune('a' + i))}, 3)
	}
	if len(s.Episodes) != 3 {
		t.Fatalf("len = %d, want 3", len(s.Episodes))
	}
	if s.Episodes[2].Text != "j" {
		t.Errorf("newest episode = %q, want j", s.Episodes[2].Text)
	}
}

func TestPriorRunsExcludesCurrentRun(t *testing.T) {
	s := NewState("a", "A")
	s.AddEpisode(Episode{RunID: "old-1", Text: "last week"}, 0)
	s.AddEpisode(Episode{RunID: "old-2", Text: "yesterday"}, 0)
	s.AddEpisode(Episode{RunID: "now", Text: "this run"}, 0)

	got := s.PriorRuns("now", 0)
	if len(got) != 2 {
		t.Fatalf("PriorRuns returned %d episodes: %+v", len(got), got)
	}
	for _, e := range got {
		if e.RunID == "now" {
			t.Error("PriorRuns leaked an episode from the current run")
		}
	}
	if limited := s.PriorRuns("now", 1); len(limited) != 1 || limited[0].Text != "yesterday" {
		t.Errorf("PriorRuns(n=1) = %+v, want the most recent", limited)
	}
}

func TestSetBeliefOverwritesAndCounts(t *testing.T) {
	s := NewState("a", "A")
	s.SetBelief("rook", "charming")
	s.SetBelief("rook", "charming and lying")
	if got := s.Beliefs["rook"]; got.Note != "charming and lying" || got.Updates != 2 {
		t.Errorf("belief = %+v", got)
	}
	s.SetBelief("", "ignored")
	s.SetBelief("x", "   ")
	if len(s.Beliefs) != 1 {
		t.Errorf("empty subject or note must be ignored, got %+v", s.Beliefs)
	}
}

func TestSetBeliefProvenanceAnnotatesAnExistingBeliefOnly(t *testing.T) {
	s := NewState("a", "A")

	// No belief yet: provenance has nothing to attach to, and must not create one.
	s.SetBeliefProvenance("rook", "they said so", "low")
	if _, ok := s.Beliefs["rook"]; ok {
		t.Fatalf("provenance must not create a belief that doesn't exist, got %+v", s.Beliefs)
	}

	s.SetBelief("rook", "charming")
	s.SetBeliefProvenance("rook", "watched them do it", "high")
	got := s.Beliefs["rook"]
	if got.Basis != "watched them do it" || got.Confidence != "high" {
		t.Errorf("belief = %+v", got)
	}
	if got.Note != "charming" || got.Updates != 1 {
		t.Errorf("provenance must not touch Note or Updates: %+v", got)
	}

	// Blank fields leave the existing values alone rather than clearing them.
	s.SetBeliefProvenance("rook", "", "")
	got = s.Beliefs["rook"]
	if got.Basis != "watched them do it" || got.Confidence != "high" {
		t.Errorf("blank provenance fields should not clear existing ones: %+v", got)
	}
}

func TestAddCommitmentDeduplicates(t *testing.T) {
	s := NewState("a", "A")
	s.AddCommitment("r1", "take the night shift")
	s.AddCommitment("r1", "take the night shift")
	if len(s.Commitments) != 1 {
		t.Errorf("commitments = %+v", s.Commitments)
	}
}

func TestSaveIsAtomicAndRejectsAnonymous(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileStore(dir)
	if err := store.Save(&State{}); err == nil {
		t.Error("saving a state with no agent id must fail")
	}
	s := NewState("mira", "Mira")
	s.LastActive = time.Now()
	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}
	// No temp file should survive a successful save.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".tmp" {
			t.Errorf("left a temp file behind: %s", e.Name())
		}
	}
}

func TestSafeNameCannotEscapeDirectory(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileStore(dir)
	s := NewState("../../escape", "Escapee")
	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "______escape.json")); err != nil {
		t.Errorf("agent id was not sanitised into the memory dir: %v", err)
	}
}

func TestMemStoreMatchesFileStoreSemantics(t *testing.T) {
	var store Store = NewMemStore()
	s, err := store.Load("ghost")
	if err != nil || s.AgentID != "ghost" {
		t.Fatalf("Load of unknown agent = %+v, %v", s, err)
	}
	s.SetBelief("x", "y")
	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}
	back, _ := store.Load("ghost")
	if back.Beliefs["x"].Note != "y" {
		t.Errorf("round trip lost the belief: %+v", back)
	}
}

// --- plans: scenario-instance isolation --------------------------------------

func TestActivePlanIsScopedToItsOwnRun(t *testing.T) {
	s := NewState("mira", "Mira")
	s.SetPlan("run-1", Plan{PrimaryGoal: "get p1 approved"})

	if got := s.ActivePlan("run-1"); got == nil || got.PrimaryGoal != "get p1 approved" {
		t.Fatalf("ActivePlan(run-1) = %+v", got)
	}
	// The same plan must not surface as "active" for a different run, even
	// though it is still sitting in state — this is the whole point: a plan
	// from one scenario instance must not silently become active in another.
	if got := s.ActivePlan("run-2"); got != nil {
		t.Errorf("ActivePlan(run-2) = %+v, want nil (different scenario instance)", got)
	}
}

func TestSetPlanFromADifferentRunOverwritesWithoutCountingAsARevision(t *testing.T) {
	s := NewState("mira", "Mira")
	s.SetPlan("run-1", Plan{PrimaryGoal: "goal for run 1"})
	s.SetPlan("run-2", Plan{PrimaryGoal: "goal for run 2"})

	got := s.ActivePlan("run-2")
	if got == nil || got.PrimaryGoal != "goal for run 2" {
		t.Fatalf("ActivePlan(run-2) = %+v", got)
	}
	if got.Revisions != 0 {
		t.Errorf("Revisions = %d, want 0 — this is a fresh plan for run-2, not a revision of run-1's", got.Revisions)
	}
	if s.ActivePlan("run-1") != nil {
		t.Error("run-1's plan should no longer be active once run-2 replaced it")
	}
}

func TestSetPlanWithinTheSameRunCountsAsARevision(t *testing.T) {
	s := NewState("mira", "Mira")
	s.SetPlan("run-1", Plan{PrimaryGoal: "v1"})
	s.SetPlan("run-1", Plan{PrimaryGoal: "v2"})
	s.SetPlan("run-1", Plan{PrimaryGoal: "v3"})

	got := s.ActivePlan("run-1")
	if got == nil || got.PrimaryGoal != "v3" {
		t.Fatalf("ActivePlan(run-1) = %+v", got)
	}
	if got.Revisions != 2 {
		t.Errorf("Revisions = %d, want 2 (two replacements after the first)", got.Revisions)
	}
}

func TestSetEmptyPlanClearsIt(t *testing.T) {
	s := NewState("mira", "Mira")
	s.SetPlan("run-1", Plan{PrimaryGoal: "something"})
	if s.ActivePlan("run-1") == nil {
		t.Fatal("setup: plan should be active")
	}
	s.SetPlan("run-1", Plan{}) // empty plan abandons it
	if got := s.ActivePlan("run-1"); got != nil {
		t.Errorf("ActivePlan(run-1) = %+v, want nil after clearing", got)
	}
}

func TestClearPlanOnlyAffectsItsOwnRun(t *testing.T) {
	s := NewState("mira", "Mira")
	s.SetPlan("run-1", Plan{PrimaryGoal: "goal"})
	s.ClearPlan("run-2") // a different run clearing has no effect
	if s.ActivePlan("run-1") == nil {
		t.Error("ClearPlan for an unrelated run must not clear this run's plan")
	}
	s.ClearPlan("run-1")
	if s.ActivePlan("run-1") != nil {
		t.Error("ClearPlan for the matching run should abandon the plan")
	}
}

func TestPlanPersistsAcrossLoadAndSave(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	s, _ := store.Load("mira")
	s.SetPlan("run-1", Plan{PrimaryGoal: "fund the market", NextSteps: []string{"talk to dov"}})
	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}

	reloaded, err := store.Load("mira")
	if err != nil {
		t.Fatal(err)
	}
	got := reloaded.ActivePlan("run-1")
	if got == nil || got.PrimaryGoal != "fund the market" || len(got.NextSteps) != 1 {
		t.Fatalf("plan did not survive a save/load round trip: %+v", got)
	}

	// And a *different* run resuming from the same on-disk file does not
	// see it as active, even though it is right there in the file.
	if reloaded.ActivePlan("some-other-run") != nil {
		t.Error("a reloaded plan must still be scoped to its own run")
	}
}
