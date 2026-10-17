package world

import (
	"context"
	"strings"
	"testing"

	"github.com/nmiano/playworld/internal/sim"
)

// Tests for making consequences explicit, reconstructed from a second
// four-day qwen3.5:9b run: offers that put the price into want_credits (so
// credits went both ways), a free offer later called a trap to make someone
// pay, residents planning to "collect dividends" from projects they did not
// own, accepts that could not be paid for, and walks that ended with the
// walker forgetting why it came.

func actionVerbs(w *World, id sim.AgentID) map[string]string {
	out := map[string]string{}
	for _, a := range w.Actions(id) {
		key := a.Verb
		if a.Kind != sim.ActionDo {
			key = string(a.Kind)
		}
		out[key] = a.Description
	}
	return out
}

func TestOfferSaysWhatEachSideGivesAndWhatWouldRemain(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "market", "mira", "rook")
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "rook", "give_credits": 6, "want_materials": 2}))

	mine := situation(w, "mira")
	if !strings.Contains(mine, "o1 to Rook: you would give 6 credits and get 2 materials") ||
		!strings.Contains(mine, "→ if it goes through you will hold 4 credits, 3 materials") {
		t.Errorf("the offerer should see both sides and the result:\n%s", mine)
	}
	theirs := situation(w, "rook") // rook holds 10 credits, 1 material
	if !strings.Contains(theirs, "o1 from Mira: you would get 6 credits and give 2 materials") ||
		!strings.Contains(theirs, "→ you cannot accept it now: it asks you for 2 materials and you hold 10 credits, 1 materials") {
		t.Errorf("the recipient should see it cannot pay its side:\n%s", theirs)
	}
	if !strings.Contains(theirs, "accept o1: it asks you for 2 materials") {
		t.Error("an unaffordable accept should be named as not possible")
	}
	if _, ok := actionVerbs(w, "rook")["accept"]; ok {
		t.Error("accept should not be offered for an offer rook cannot pay for")
	}
	if _, ok := actionVerbs(w, "rook")["decline"]; !ok {
		t.Error("declining stays possible")
	}
	if err := w.Validate("rook", do("accept", map[string]any{"offer": "o1"})); err == nil || RefusalCode(err) != "cannot_afford" {
		t.Errorf("the environment remains the authority: %v", err)
	}

	w.Residents["rook"].Goods.Materials = 3
	theirs = situation(w, "rook")
	if !strings.Contains(theirs, "→ if it goes through you will hold 16 credits, 1 materials") {
		t.Errorf("once affordable, the result should be shown:\n%s", theirs)
	}
	if d := actionVerbs(w, "rook")["accept"]; !strings.Contains(d, "o1 (you give 2 materials, you get 6 credits)") {
		t.Errorf("accept should spell out the exchange: %q", d)
	}
}

func TestUnfavourableTwoSidedOffersAreLegalAndSummed(t *testing.T) {
	w := newWorld(t, "dov", "rook")
	together(w, "workshop", "dov")
	w.Residents["rook"].Location = "market"
	w.Residents["dov"].Goods = Goods{Credits: 6, Materials: 20}
	w.Residents["rook"].Goods = Goods{Credits: 12}
	mustApply(t, w, "dov", do("offer", map[string]any{"to": "rook", "give_credits": 6, "give_materials": 20, "want_credits": 10}))

	s := situation(w, "dov")
	if !strings.Contains(s, "credits go both ways: net, you gain 4") || !strings.Contains(s, "if it goes through you will hold 10 credits, 0 materials") {
		t.Errorf("a two-sided offer should be summed for the offerer:\n%s", s)
	}
	if r := situation(w, "rook"); !strings.Contains(r, "credits go both ways: net, you lose 4") {
		t.Errorf("and for the recipient:\n%s", r)
	}
	// A bad deal is still a deal: nothing stops it, and it moves exactly
	// what it says.
	mustApply(t, w, "rook", do("accept", map[string]any{"offer": "o1"}))
	if d, r := w.Residents["dov"].Goods, w.Residents["rook"].Goods; d != (Goods{Credits: 10}) || r != (Goods{Credits: 8, Materials: 20}) {
		t.Errorf("transfers: dov %v, rook %v", d, r)
	}
}

func TestOneWayOfferIsNotRecordedAsAPaidTrade(t *testing.T) {
	w := newWorld(t, "ada", "rook")
	together(w, "square", "ada", "rook")
	w.Residents["rook"].Goods = Goods{Materials: 6}
	w.Residents["ada"].Goods = Goods{Credits: 16}
	eff := mustApply(t, w, "rook", do("offer", map[string]any{"to": "ada", "give_materials": 6}))
	if !strings.Contains(eff.Events[0].Text, "if Ada accepts, Rook hands over 6 materials and Ada hands over nothing") {
		t.Errorf("the offer event should say who hands over what: %s", eff.Events[0].Text)
	}
	if s := situation(w, "ada"); !strings.Contains(s, "you would give nothing for it") {
		t.Errorf("ada should see she gives nothing:\n%s", s)
	}
	if s := situation(w, "rook"); !strings.Contains(s, "→ if it goes through you will hold 0 credits, 0 materials; you would get nothing back") {
		t.Errorf("rook should see he gets nothing back:\n%s", s)
	}
	eff = mustApply(t, w, "ada", do("accept", map[string]any{"offer": "o1"}))
	text := eff.Events[0].Text
	if strings.Contains(text, "the trade is done") || !strings.Contains(text, "Rook handed over 6 materials and Ada gave nothing back") {
		t.Errorf("a free offer accepted is one-way: %s", text)
	}
	if w.Residents["ada"].Goods != (Goods{Credits: 16, Materials: 6}) || w.Residents["rook"].Goods != (Goods{}) {
		t.Errorf("ada paid nothing: ada %v rook %v", w.Residents["ada"].Goods, w.Residents["rook"].Goods)
	}
	for _, id := range []sim.AgentID{"ada", "rook"} {
		if s := situation(w, id); !strings.Contains(s, "nothing went the other way; the offer asked for nothing in return") {
			t.Errorf("%s's record of what changed hands should say it was one-way:\n%s", id, s)
		}
	}
	// A plain gift is still a gift, never a trade.
	mustApply(t, w, "ada", do("give", map[string]any{"to": "rook", "credits": 2}))
	if s := situation(w, "rook"); !strings.Contains(s, "Ada gave you 2 credits — a gift, nothing was asked of you") {
		t.Errorf("gift record:\n%s", s)
	}
}

func TestOwnOfferThatCanNoLongerBeFundedIsMarked(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "rook", "give_credits": 6, "want_materials": 2}))
	w.Residents["mira"].Goods.Credits = 1 // spent elsewhere since
	if s := situation(w, "mira"); !strings.Contains(s, "→ you no longer hold 6 credits — you hold 1 credits, 1 materials — so it cannot go through as it stands") {
		t.Errorf("a stale offer should not read as executable:\n%s", s)
	}
}

func TestProjectOwnershipAndIncomeAreExplicit(t *testing.T) {
	w := newWorld(t, "ada", "mira")
	together(w, "square", "ada", "mira")
	w.Now = At(1, 9, 0)
	mustApply(t, w, "ada", do("start_project", map[string]any{"name": "Community Garden Bed", "materials": 2, "labor": 2}))
	w.Residents["mira"].Goods.Materials = 2
	mustApply(t, w, "mira", do("contribute", map[string]any{"project": "p1", "materials": 1}))

	m := situation(w, "mira")
	for _, want := range []string{
		`p1 "Community Garden Bed" — Ada's project, at the square`,
		"NOT finished: still needs 1 materials and 2 labor hours",
		"you put in 1 materials",
		"it gives you no share, no say and none of its income",
	} {
		if !strings.Contains(m, want) {
			t.Errorf("contributor view lacks %q:\n%s", want, m)
		}
	}
	if d := actionVerbs(w, "mira")["contribute"]; !strings.Contains(d, "p1, Ada's, needs 1 materials") || !strings.Contains(d, "stays its owner's") {
		t.Errorf("contribute should say whose project it is: %q", d)
	}
	if d := actionVerbs(w, "mira")["build"]; !strings.Contains(d, "p1, Ada's, needs 2 labor hours") {
		t.Errorf("build should say whose project it is: %q", d)
	}

	// Labor done, one material short: still unfinished for everyone.
	w.Projects[0].Labor = 2
	if w.Projects[0].Done() || !strings.Contains(situation(w, "ada"), "NOT finished: still needs 1 materials") {
		t.Error("labor alone does not finish a project")
	}
	mustApply(t, w, "mira", do("contribute", map[string]any{"project": "p1", "materials": 1}))
	if !w.Projects[0].Done() {
		t.Fatal("the project should be complete")
	}
	if _, ok := actionVerbs(w, "mira")["contribute"]; ok {
		t.Error("a completed project takes no more contributions")
	}
	if err := w.Validate("mira", do("contribute", map[string]any{"project": "p1", "materials": 1})); RefusalCode(err) != "project_complete" {
		t.Errorf("contributing to a finished project is refused: %v", err)
	}

	adaBefore, miraBefore := w.Residents["ada"].Goods.Credits, w.Residents["mira"].Goods.Credits
	w.Now = At(2, 6, 0)
	w.dawn()
	if w.Residents["ada"].Goods.Credits-adaBefore != 2 || w.Residents["mira"].Goods.Credits != miraBefore {
		t.Error("income goes to the owner alone")
	}
	w.Now = At(2, 8, 0)
	a := situation(w, "ada")
	if !strings.Contains(a, "Your completed projects (each pays you, its owner, 2 credits every dawn, automatically and wherever you are — there is nothing to collect; next payment Day 3 06:00)") ||
		!strings.Contains(a, `"Community Garden Bed" at the square — completed Day 1 09:00; has paid you 2 credits so far`) {
		t.Errorf("the owner should see its income and when it comes:\n%s", a)
	}
	m = situation(w, "mira")
	if !strings.Contains(m, "Completed (each pays 2 credits every dawn to its owner alone)") ||
		!strings.Contains(m, `"Community Garden Bed" — Ada's, at the square, completed Day 1 09:00 (you put in 2 materials)`) {
		t.Errorf("a contributor should see it is someone else's income:\n%s", m)
	}
	if strings.Contains(m, "Your completed projects") {
		t.Error("a contributor does not own the project")
	}
}

func TestCompletionIsNotedOnARelatedIntention(t *testing.T) {
	w := newWorld(t, "ada", "mira")
	together(w, "square", "ada")
	w.Now = At(1, 9, 0)
	mustApply(t, w, "ada", do("start_project", map[string]any{"name": "Community Garden Bed", "materials": 1}))
	mustApply(t, w, "ada", sim.Action{Kind: sim.ActionIntend, Args: map[string]any{"when": "in 3h", "note": "get the last material for the garden bed"}})
	mustApply(t, w, "ada", sim.Action{Kind: sim.ActionIntend, Args: map[string]any{"when": "in 4h", "note": "see whether Dov wants to talk"}})
	w.Now = At(1, 10, 0)
	mustApply(t, w, "ada", do("contribute", map[string]any{"project": "p1", "materials": 1}))

	s := situation(w, "ada")
	if !strings.Contains(s, `get the last material for the garden bed — since you set it: "Community Garden Bed" (Ada's) was completed Day 1 10:00`) {
		t.Errorf("a pending intention about a finished project should say so:\n%s", s)
	}
	if strings.Contains(s, "see whether Dov wants to talk — since") {
		t.Error("unrelated intentions are left alone")
	}
	// The intention still comes due (the agent decides what it means now),
	// and its reason carries the same fact.
	var in *Intention
	for _, x := range w.Intentions {
		if strings.Contains(x.Note, "garden") {
			in = x
		}
	}
	w.Now = in.At
	eff := w.intentionDue(&Item{Ref: in.ID})
	if len(eff.Triggers) != 1 || !strings.Contains(eff.Triggers[0].Reason.Text, "was completed Day 1 10:00") {
		t.Errorf("the due intention should carry what changed: %+v", eff.Triggers)
	}
	if w.Totals["stale_intentions_due"] != 1 {
		t.Errorf("stale intentions are counted: %v", w.Totals)
	}
}

func TestArrivalRemindsTheWalkerWhyItCame(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "square", "mira")
	w.Now = At(1, 9, 0)
	if d := actionVerbs(w, "mira")["go"]; !strings.Contains(d, "you are at square now: market, workshop") {
		t.Errorf("go should say where you already are: %q", d)
	}
	mustApply(t, w, "mira", do("go", map[string]any{"to": "market", "for": "buy two materials"}))
	act := w.Residents["mira"].Activity
	w.Now = act.End
	eff := w.Process(w.queue.Get(act.ItemID))
	var reason string
	for _, tr := range eff.Triggers {
		if tr.Agent == "mira" {
			reason = tr.Reason.Text
		}
	}
	if reason != "you arrived at the market (you came to: buy two materials)" {
		t.Errorf("arrival reason: %q", reason)
	}
	if s := situation(w, "mira"); !strings.Contains(s, `You walked here from the square, arriving 09:15; you said you were coming to: "buy two materials"`) {
		t.Errorf("the purpose should stay visible after arriving:\n%s", s)
	}
	if err := w.Validate("mira", do("go", map[string]any{"to": "market"})); RefusalCode(err) != "already_there" {
		t.Errorf("going where you are is still refused: %v", err)
	}
	// Leaving again replaces the errand.
	mustApply(t, w, "mira", do("go", map[string]any{"to": "square"}))
	if e := w.Residents["mira"].Errand; e.To != "square" || e.For != "" {
		t.Errorf("errand: %+v", e)
	}
}

func TestSetbacksAndElsewhereAreRememberedNotLive(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "market", "mira")
	w.Residents["rook"].Location = "square"
	w.Now = At(1, 9, 0)
	w.Look("mira")
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "rook", "give_credits": 3, "want_materials": 1}))
	mustApply(t, w, "rook", do("decline", map[string]any{"offer": "o1"}))
	mustApply(t, w, "mira", do("offer", map[string]any{"to": "rook", "give_credits": 3, "want_materials": 1}))
	mustApply(t, w, "rook", do("decline", map[string]any{"offer": "o2"}))
	err := w.Validate("mira", do("buy", map[string]any{"materials": 9}))
	if w.NoteRefusal("mira", "buy", err) {
		t.Error("a first refusal is not a repeat")
	}
	if !w.NoteRefusal("mira", "buy", err) {
		t.Error("the same refusal again is a repeat")
	}
	s := situation(w, "mira")
	for _, want := range []string{
		"Your refused actions in the last day (newest first):",
		"your buy was refused: the market only has 8 materials left today (2 times since Day 1 09:00)",
		"o2 you → Rook (3 credits for 1 materials): declined Day 1 09:00 — these same terms have been turned down 2 times",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("setbacks lack %q:\n%s", want, s)
		}
	}
	if w.Totals["reoffers_after_decline"] != 1 {
		t.Errorf("re-offering declined terms is counted: %v", w.Totals)
	}

	// Away from the market, mira remembers what she saw there — and that it
	// was before the restock.
	w.Residents["mira"].Location = "workshop"
	w.Stock.Materials["market"] = 0
	s = situation(w, "mira")
	if !strings.Contains(s, "the market, as of Day 1 09:00: 8 materials for sale, 4 shifts left") {
		t.Errorf("a remembered glimpse, not the live stock:\n%s", s)
	}
	w.Now = At(2, 10, 0)
	if s = situation(w, "mira"); !strings.Contains(s, "that was before today's restock") || strings.Contains(s, "Your refused actions") {
		t.Errorf("old glimpses and setbacks should age:\n%s", s)
	}
	if strings.Contains(situation(w, "rook"), "the market, as of") {
		t.Error("rook never saw the market")
	}
}

func TestRefusalsTheAffordancesForesawAreTold(t *testing.T) {
	w := newWorld(t, "mira", "rook")
	together(w, "square", "mira")
	w.Residents["rook"].Location = "market"
	mira := &scripted{id: "mira", decide: func(_ sim.Turn, n int) []sim.Action {
		switch n {
		case 0, 1:
			return []sim.Action{{Kind: sim.ActionSay, Content: "Hello?"}}
		case 2:
			return []sim.Action{{Kind: sim.ActionWhisper, To: []sim.AgentID{"rook"}, Content: "Anyone around?"}}
		}
		return nil
	}}
	res := engine(t, w, mira, &scripted{id: "rook"}).Run(context.Background(), At(1, 12, 0))
	m := res.Metrics
	if m["rejected_unoffered"] != 2 || m["rejected_offered"] != 0 {
		t.Errorf("say with nobody here was never offered: %v", m)
	}
	if m["repeat_refusals"] != 1 {
		t.Errorf("the second identical refusal is a repeat: %v", m)
	}
	if m["after_setback_same"] != 1 || m["after_setback_changed"] != 1 {
		t.Errorf("after a refusal: once the same move, then a different one: %v", m)
	}
}
