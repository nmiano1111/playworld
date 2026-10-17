package world

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/nmiano/playworld/internal/sim"
)

// Tests for grounded cognition, most of them reconstructed from failure
// patterns in a four-day qwen3.5:9b run: speaking to an empty room, a gift
// later described as a sale, "finished" projects that still lacked
// materials, new projects past the limit, accepting a closed or one's own
// offer, and conversations that re-triggered themselves with the same lines.

func situation(w *World, id sim.AgentID) string { return w.Situation(id, nil) }

func together(w *World, place string, ids ...sim.AgentID) {
	for _, id := range ids {
		w.Residents[id].Location = place
	}
}

func TestGroundedIsTheDefaultAndLegacyIsPreserved(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	if !w.Settings.Grounded() {
		t.Fatal("new worlds should be grounded")
	}
	turn := w.TurnFor("r", "mira", []Reason{{Kind: "idle", Text: "a quiet moment"}})
	if turn.Situation == "" || turn.PrivateState != nil || turn.Now.IsZero() {
		t.Error("a grounded turn carries a situation, not key/value state")
	}

	// A world file written before cognition existed has no such field: it
	// must load as legacy and produce exactly the original observation.
	w.Settings.Cognition = ""
	raw, err := json.Marshal(w)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(raw, &m)
	settings := m["settings"].(map[string]any)
	delete(settings, "cognition")
	delete(m, "ledger")
	raw, _ = json.Marshal(m)
	old, err := Decode(raw)
	if err != nil {
		t.Fatal(err)
	}
	if old.Settings.Grounded() {
		t.Fatal("an older world must stay legacy unless told otherwise")
	}
	lt := old.TurnFor("r", "mira", nil)
	if lt.Situation != "" || lt.PrivateState["you have"] == nil {
		t.Error("a legacy turn keeps the original key/value state")
	}
	for _, s := range old.Actions("mira") {
		if s.Verb == "abandon" {
			t.Error("legacy worlds do not gain new verbs")
		}
	}
}

func TestAgentSeesItsOwnHoldingsAccurately(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "market", "mira", "rook")
	w.Residents["mira"].Goods = Goods{Credits: 14, Materials: 0}
	w.Residents["rook"].Goods = Goods{Credits: 18, Materials: 0}
	if s := situation(w, "rook"); !strings.Contains(s, "buy more than 6 materials: at 3 credits each, 18 credits buys 6") {
		t.Errorf("what rook can afford should be explicit:\n%s", s)
	}
	mustApply(t, w, "rook", do("buy", map[string]any{"materials": 2}))
	if s := situation(w, "rook"); !strings.Contains(s, "Holding: 12 credits, 2 materials") {
		t.Errorf("rook's holdings after buying 2 at 3 each:\n%s", s)
	}
	mustApply(t, w, "rook", do("give", map[string]any{"to": "mira", "materials": 1}))
	if s := situation(w, "mira"); !strings.Contains(s, "Holding: 14 credits, 1 materials") {
		t.Errorf("mira should see what she received:\n%s", s)
	}
}

func TestOthersPrivateStateStaysHidden(t *testing.T) {
	w := newWorld(t, "mira", "rook", "ada")
	together(w, "square", "mira", "rook")
	w.Residents["mira"].Goods = Goods{Credits: 37, Materials: 5}
	mustApply(t, w, "mira", sim.Action{Kind: sim.ActionWhisper, To: []sim.AgentID{"ada"}, Content: "secret plan"})
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "ada", "give_credits": 3, "want_materials": 1}))
	mustApply(t, w, "mira", sim.Action{Kind: sim.ActionIntend, Args: map[string]any{"when": "in 2h", "note": "sneak to the market"}})

	s := situation(w, "rook")
	for _, leak := range []string{"37 credits", "secret plan", "sneak to the market", "o1"} {
		if strings.Contains(s, leak) {
			t.Errorf("rook can see %q, which is mira's private business:\n%s", leak, s)
		}
	}
	if !strings.Contains(s, "Mira (mira)") {
		t.Error("rook should still see that mira is here")
	}
}

func TestGiftTradeAndProposalAreDistinct(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "square", "mira", "rook")
	w.Residents["mira"].Goods = Goods{Credits: 20, Materials: 4}
	w.Residents["rook"].Goods = Goods{Credits: 20, Materials: 4}

	gift := mustApply(t, w, "rook", do("give", map[string]any{"to": "mira", "credits": 5}))
	if !strings.Contains(gift.Events[0].Text, "as a gift — nothing was asked or given in return") {
		t.Errorf("gift event: %q", gift.Events[0].Text)
	}
	mustApply(t, w, "rook", do("offer", map[string]any{"to": "mira", "give_materials": 2, "want_credits": 6}))

	s := situation(w, "mira")
	if !strings.Contains(s, "Rook gave you 5 credits — a gift") {
		t.Errorf("the gift should be in mira's record of what changed hands:\n%s", s)
	}
	if !strings.Contains(s, "Offers made to you (you can accept or decline; nothing has moved yet)") || !strings.Contains(s, "o1 from Rook: you would get 2 materials and give 6 credits") {
		t.Errorf("the open offer should read as a proposal, from mira's side:\n%s", s)
	}
	if strings.Contains(s, "trade o1") {
		t.Error("nothing has been traded yet")
	}

	mustApply(t, w, "mira", do("accept", map[string]any{"offer": "o1"}))
	s = situation(w, "rook")
	if !strings.Contains(s, "trade o1: you handed Mira 2 materials") || !strings.Contains(s, "trade o1: Mira handed you 6 credits") {
		t.Errorf("both sides of the completed trade should be recorded:\n%s", s)
	}
	if !strings.Contains(s, "you gave Mira 5 credits — a gift") {
		t.Errorf("the earlier gift stays a gift:\n%s", s)
	}
	kinds := map[string]int{}
	for _, l := range w.Ledger {
		kinds[l.Kind]++
	}
	if kinds["gift"] != 1 || kinds["trade"] != 2 {
		t.Errorf("ledger kinds = %v", kinds)
	}
}

func TestProjectWithLaborButNoMaterialsIsStillOpen(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "workshop", "mira", "rook")
	mustApply(t, w, "mira", do("start_project", map[string]any{"name": "Small Repair Kit", "materials": 2, "labor": 2}))
	p := w.Projects[0]
	p.Labor = 2 // the labor is done; the materials are not

	if p.Done() {
		t.Fatal("labor alone must not complete a project")
	}
	mine := situation(w, "mira")
	if !strings.Contains(mine, `p1 "Small Repair Kit" at the workshop — has 0/0 credits, 0/2 materials, 2/2 labor hours; NOT finished: still needs 2 materials`) {
		t.Errorf("owner should see what is still missing:\n%s", mine)
	}
	theirs := situation(w, "rook")
	if !strings.Contains(theirs, "Mira's project, at the workshop") || !strings.Contains(theirs, "NOT finished: still needs 2 materials") {
		t.Errorf("others should see the remaining requirement too:\n%s", theirs)
	}
	for _, a := range w.Actions("rook") {
		if a.Verb == "build" {
			t.Error("a project that needs no more labor should not be offered for building")
		}
	}
	if err := w.Validate("mira", do("build", map[string]any{"project": "p1"})); RefusalCode(err) != "no_labor_needed" {
		t.Errorf("build refusal = %v", err)
	}
}

func TestProjectLimitIsVisibleBeforeTryingAgain(t *testing.T) {
	w := newWorld(t, "mira")
	together(w, "workshop", "mira")
	for _, name := range []string{"Kit", "Shelf"} {
		mustApply(t, w, "mira", do("start_project", map[string]any{"name": name, "materials": 2}))
	}
	s := situation(w, "mira")
	if !strings.Contains(s, "Your unfinished projects (2 of at most 2)") || !strings.Contains(s, "start_project: you already have 2 unfinished projects") {
		t.Errorf("the limit should be stated:\n%s", s)
	}
	for _, a := range w.Actions("mira") {
		if a.Verb == "start_project" {
			t.Error("start_project should not be offered at the limit")
		}
	}
	err := w.Validate("mira", do("start_project", map[string]any{"name": "Post", "materials": 1}))
	if RefusalCode(err) != "project_limit" || !strings.Contains(err.Error(), "abandon") {
		t.Errorf("refusal = %v", err)
	}
}

func TestAbandoningAProjectRefundsContributorsAndFreesTheSlot(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "workshop", "mira", "rook")
	w.Residents["rook"].Goods = Goods{Credits: 5, Materials: 3}
	mustApply(t, w, "mira", do("start_project", map[string]any{"name": "Kit", "materials": 4}))
	mustApply(t, w, "rook", do("contribute", map[string]any{"project": "p1", "materials": 3}))
	if err := w.Validate("rook", do("abandon", map[string]any{"project": "p1"})); RefusalCode(err) != "not_your_project" {
		t.Errorf("only the owner may abandon: %v", err)
	}
	eff := mustApply(t, w, "mira", do("abandon", map[string]any{"project": "p1"}))
	if got := w.Residents["rook"].Goods; got != (Goods{Credits: 5, Materials: 3}) {
		t.Errorf("rook's contribution should come back, has %s", got)
	}
	if len(eff.Triggers) != 1 || eff.Triggers[0].Agent != "rook" {
		t.Errorf("the contributor should be told: %+v", eff.Triggers)
	}
	if err := w.Validate("rook", do("contribute", map[string]any{"project": "p1", "materials": 1})); RefusalCode(err) != "project_abandoned" {
		t.Errorf("an abandoned project takes nothing more: %v", err)
	}
	if w.openProjects("mira") != 0 {
		t.Error("abandoning frees the owner's project slot")
	}
}

func TestSettledOffersAreNotPresentedAsActionable(t *testing.T) {
	w := newWorld(t, "mira", "rook", "ada")
	together(w, "square", "mira", "rook", "ada")
	for _, id := range []sim.AgentID{"mira", "rook", "ada"} {
		w.Residents[id].Goods = Goods{Credits: 30, Materials: 6}
	}
	offer := map[string]any{"to": "ada", "give_credits": 20, "give_materials": 4, "want_materials": 8}
	mustApply(t, w, "rook", do("offer", offer))                                          // o1, declined below
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "ada", "give_credits": 2})) // o2, accepted below
	mustApply(t, w, "rook", do("offer", map[string]any{"to": "ada", "want_credits": 1})) // o3, expires below
	mustApply(t, w, "ada", do("decline", map[string]any{"offer": "o1"}))
	mustApply(t, w, "ada", do("accept", map[string]any{"offer": "o2"}))
	w.Now = w.Now.Add(25 * time.Hour)
	w.dawn()

	s := situation(w, "ada")
	if strings.Contains(s, "Offers made to you") {
		t.Errorf("no offer is open to ada any more:\n%s", s)
	}
	for _, a := range w.Actions("ada") {
		if a.Verb == "accept" || a.Verb == "decline" {
			t.Error("accept/decline should not be offered when nothing is open")
		}
	}
	err := w.Validate("ada", do("accept", map[string]any{"offer": "o1"}))
	if RefusalCode(err) != "offer_closed" || !strings.Contains(err.Error(), "declined") {
		t.Errorf("accepting a declined offer: %v", err)
	}
	if RefusalCode(w.Validate("ada", do("accept", map[string]any{"offer": "o3"}))) != "offer_closed" {
		t.Error("accepting an expired offer should be refused as closed")
	}
	if err := w.Validate("rook", do("accept", map[string]any{"offer": "o1"})); RefusalCode(err) != "own_offer" || !strings.Contains(err.Error(), "your own offer") {
		t.Errorf("accepting one's own offer: %v", err)
	}
}

func TestRepeatedOffersOnDeclinedTermsAreCountedAndShown(t *testing.T) {
	w := newWorld(t, "rook", "ada")
	together(w, "square", "rook", "ada")
	w.Residents["rook"].Goods = Goods{Credits: 30, Materials: 6}
	offer := map[string]any{"to": "ada", "give_credits": 20, "give_materials": 4, "want_materials": 8}
	first := mustApply(t, w, "rook", do("offer", offer))
	if len(first.Triggers) != 1 {
		t.Fatal("a first offer wakes its recipient")
	}
	mustApply(t, w, "ada", do("decline", map[string]any{"offer": "o1"}))
	again := mustApply(t, w, "rook", do("offer", offer))
	if len(again.Triggers) != 0 {
		t.Error("the same terms again are delivered but do not wake ada as if new")
	}
	if !strings.Contains(situation(w, "ada"), "o2 from Rook: you would get 20 credits and 4 materials and give 8 materials — open until") ||
		!strings.Contains(situation(w, "ada"), "(the same terms as o1)") {
		t.Errorf("ada should see the repeat for what it is:\n%s", situation(w, "ada"))
	}
	mustApply(t, w, "ada", do("decline", map[string]any{"offer": "o2"}))
	if s := situation(w, "rook"); !strings.Contains(s, "these same terms have been turned down 2 times") {
		t.Errorf("rook should see how his terms have fared:\n%s", s)
	}
	better := mustApply(t, w, "rook", do("offer", map[string]any{"to": "ada", "give_credits": 24, "give_materials": 4, "want_materials": 8}))
	if len(better.Triggers) != 1 {
		t.Error("new terms are new information and must wake ada")
	}
	if w.Totals["repeated_offers"] != 1 {
		t.Errorf("repeated_offers = %d", w.Totals["repeated_offers"])
	}
}

func TestSayingToAnEmptyRoomIsVisiblyImpossible(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "square", "mira")
	w.Residents["rook"].Location = "market"
	s := situation(w, "mira")
	if !strings.Contains(s, "People: nobody else is here") || !strings.Contains(s, "say: nobody else is here to hear it") {
		t.Errorf("an empty room should be plain:\n%s", s)
	}
	for _, a := range w.Actions("mira") {
		if a.Kind == sim.ActionSay {
			t.Error("say should not be offered with nobody present")
		}
		if a.Verb == "go" && strings.Contains(a.Description, "square,") {
			t.Errorf("go should not list the current place: %s", a.Description)
		}
	}
	if !strings.Contains(s, "go to square: you are already here") {
		t.Error("the current place should be named as no destination")
	}
}

func TestRevisedIntentionsReplaceRatherThanPileUp(t *testing.T) {
	w := newWorld(t, "ada")
	w.Now = At(1, 9, 0)
	intend := func(when, note string, extra ...string) {
		args := map[string]any{"when": when, "note": note}
		if len(extra) > 0 {
			args["replaces"] = extra[0]
		}
		mustApply(t, w, "ada", sim.Action{Kind: sim.ActionIntend, Args: args})
	}
	intend("in 1h", "check whether Dov has finished salvaging the pile")
	intend("in 2h", "check whether Dov finished salvaging the pile yet")
	intend("in 3h", "ask Rook what he needs", "i2")

	pending := 0
	for _, in := range w.Intentions {
		if in.Status == "pending" {
			pending++
		}
	}
	if pending != 1 || w.Intentions[0].Status != "superseded" || w.Intentions[1].Status != "superseded" {
		t.Fatalf("one live intention expected, got %+v", w.Intentions)
	}
	if w.Totals["intentions_superseded"] != 2 {
		t.Errorf("superseded = %d", w.Totals["intentions_superseded"])
	}
	ada := &scripted{id: "ada"}
	engine(t, w, ada).Run(context.Background(), At(1, 14, 0))
	woken := 0
	for _, turn := range ada.turns {
		if strings.Contains(why(turn), "you asked to reconsider") {
			woken++
			if !strings.Contains(why(turn), "ask Rook what he needs") {
				t.Errorf("woken for a superseded intention: %s", why(turn))
			}
		}
	}
	if woken != 1 {
		t.Errorf("woken %d times by intentions, want exactly 1", woken)
	}
	if s := situation(w, "ada"); strings.Contains(s, "Your pending intentions") {
		t.Errorf("a fired intention is no longer pending:\n%s", s)
	}
}

func TestRepeatedRemarksDoNotCascade(t *testing.T) {
	w := newWorld(t, "dov", "ada")
	together(w, "workshop", "dov", "ada")
	// Each answers the other with the same line every time — the "the pile
	// is empty" / "let's check the pile" loop from the four-day run.
	dov := &scripted{id: "dov", decide: func(sim.Turn, int) []sim.Action {
		return []sim.Action{{Kind: sim.ActionSay, Content: "Ada, the scrap pile is empty for today. I will salvage tomorrow."}}
	}}
	ada := &scripted{id: "ada", decide: func(sim.Turn, int) []sim.Action {
		return []sim.Action{{Kind: sim.ActionSay, Content: "Dov, can we all check the pile together to be sure?"}}
	}}
	engine(t, w, dov, ada).Run(context.Background(), At(1, 11, 0))
	reactive := 0
	for _, a := range []*scripted{dov, ada} {
		for _, turn := range a.turns {
			if strings.Contains(why(turn), "spoke to you") {
				reactive++
			}
		}
	}
	// The first exchange is news to each side; after that, the same lines
	// are heard but do not re-trigger.
	if reactive > 2 {
		t.Errorf("%d reactive activations from two repeated lines", reactive)
	}
	if w.Totals["repeat_triggers_suppressed"] == 0 || w.Totals["repeated_speech"] == 0 {
		t.Errorf("repeats should be counted: %v", w.Totals)
	}
	if !ada.saw("Dov (dov) said: Ada, the scrap pile is empty") {
		t.Error("repeated speech must still be heard")
	}
}

func TestRepeatDampingKeepsNewInformationAndNewListeners(t *testing.T) {
	w := newWorld(t, "dov", "ada", "rook")
	together(w, "workshop", "dov", "ada")
	w.Residents["rook"].Location = "market"
	say := func(text string) Effects {
		return mustApply(t, w, "dov", sim.Action{Kind: sim.ActionSay, Content: text})
	}
	if eff := say("Ada, the scrap pile is empty for today."); len(eff.Triggers) != 1 {
		t.Fatal("the first remark wakes ada")
	}
	if eff := say("Ada, the scrap pile is empty for today!"); len(eff.Triggers) != 0 {
		t.Error("the same remark again should not")
	}
	if eff := say("Ada, I just finished salvaging and have 2 materials for you."); len(eff.Triggers) != 1 {
		t.Error("something new must wake ada")
	}
	if eff := say("Ada, the scrap pile now has enough for 4 materials."); len(eff.Triggers) != 1 {
		t.Error("a changed number is new information")
	}
	together(w, "workshop", "rook")
	if eff := say("Rook, the scrap pile is empty for today."); len(eff.Triggers) != 1 || eff.Triggers[0].Agent != "rook" {
		t.Errorf("a listener who never heard it should be woken: %+v", eff.Triggers)
	}
	// Half a day later the same line is no longer a repeat.
	w.Now = w.Now.Add(13 * time.Hour)
	if eff := say("Ada, the scrap pile is empty for today."); len(eff.Triggers) == 0 {
		t.Error("an old remark said again much later should wake again")
	}
	// Whispers follow the same rule.
	n1 := mustApply(t, w, "rook", sim.Action{Kind: sim.ActionWhisper, To: []sim.AgentID{"ada"}, Content: "Dov is lying about the pile."})
	n2 := mustApply(t, w, "rook", sim.Action{Kind: sim.ActionWhisper, To: []sim.AgentID{"ada"}, Content: "Dov is lying about the pile!"})
	if len(n1.Triggers) != 1 || len(n2.Triggers) != 0 {
		t.Errorf("whisper triggers: first %d, repeat %d", len(n1.Triggers), len(n2.Triggers))
	}
}

func TestLegacyModeKeepsOriginalTriggers(t *testing.T) {
	w := newWorld(t, "dov", "ada")
	w.Settings.Cognition = CognitionLegacy
	together(w, "workshop", "dov", "ada")
	for i := 0; i < 2; i++ {
		eff := mustApply(t, w, "dov", sim.Action{Kind: sim.ActionSay, Content: "Ada, the scrap pile is empty for today."})
		if len(eff.Triggers) != 1 {
			t.Fatal("legacy worlds trigger on every remark (only the reply chain damps)")
		}
	}
	if w.Totals["repeated_speech"] != 1 {
		t.Errorf("repeats are still counted in legacy mode, for comparison: %d", w.Totals["repeated_speech"])
	}
}

func TestRefusalsAreCountedByReason(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "square", "mira")
	w.Residents["rook"].Location = "market"
	mira := &scripted{id: "mira", decide: func(_ sim.Turn, n int) []sim.Action {
		if n == 0 {
			return []sim.Action{{Kind: sim.ActionSay, Content: "Hello?"}, do("go", map[string]any{"to": "square"})}
		}
		return nil
	}}
	res := engine(t, w, mira, &scripted{id: "rook"}).Run(context.Background(), At(1, 7, 30))
	if res.Metrics["rejected_nobody_here"] != 1 || res.Metrics["rejected_already_there"] != 1 {
		t.Errorf("refusal codes: %v", res.Metrics)
	}
}

func TestGroundedStatePersistsAcrossSaveAndLoad(t *testing.T) {
	store := Store{Root: t.TempDir()}
	w := newWorld(t, "mira", "rook")
	w.Name = "g"
	together(w, "square", "mira", "rook")
	w.Residents["rook"].Goods = Goods{Credits: 9, Materials: 2}
	mustApply(t, w, "rook", do("give", map[string]any{"to": "mira", "credits": 3}))
	mustApply(t, w, "rook", do("offer", map[string]any{"to": "mira", "want_credits": 1}))
	mustApply(t, w, "mira", do("decline", map[string]any{"offer": "o1"}))
	mustApply(t, w, "mira", sim.Action{Kind: sim.ActionSay, Content: "Rook, thanks for the credits."})
	mustApply(t, w, "mira", do("start_project", map[string]any{"name": "Kit", "materials": 1}))
	mustApply(t, w, "mira", do("abandon", map[string]any{"project": "p1"}))
	before := situation(w, "mira")
	if err := store.Save(w); err != nil {
		t.Fatal(err)
	}
	back, err := store.Load("g")
	if err != nil {
		t.Fatal(err)
	}
	if got := situation(back, "mira"); got != before {
		t.Errorf("the observation should survive a resume unchanged:\n--- before\n%s\n--- after\n%s", before, got)
	}
	if !back.Settings.Grounded() || len(back.Ledger) != 1 || back.Offers[0].Resolved == nil || len(back.Residents["mira"].Said) != 1 || back.Projects[0].Abandoned == nil {
		t.Error("grounded state was not persisted")
	}
	// A repeat right after resuming is still recognised as one.
	eff := mustApply(t, back, "mira", sim.Action{Kind: sim.ActionSay, Content: "Rook, thanks for the credits!"})
	if len(eff.Triggers) != 0 {
		t.Error("repeat recognition should survive a resume")
	}
}

func TestOldRefusalMessagesClassifyLikeLiveCodes(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "square", "mira")
	w.Residents["rook"].Location = "market"
	w.Residents["mira"].Goods = Goods{Credits: 1}
	for _, a := range []sim.Action{
		{Kind: sim.ActionSay, Content: "hello?"},
		do("go", map[string]any{"to": "square"}),
		do("offer", map[string]any{"to": "rook", "give_credits": 9}),
		do("accept", map[string]any{"offer": "o9"}),
		do("salvage", nil),
		do("fly", nil),
	} {
		err := w.Validate("mira", a)
		if err == nil {
			t.Fatalf("%+v should be refused", a)
		}
		if got, want := ClassifyRefusal(err.Error()), RefusalCode(err); got != want {
			t.Errorf("%q classifies as %s, live code %s", err.Error(), got, want)
		}
	}
}
