package world

import (
	"fmt"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/sim"
)

// This file is the grounded observation: what one resident can see at the
// moment it decides, laid out in a fixed order — itself, the place it is in,
// the public record of projects, and what is plainly impossible right now.
// Everything in it is authoritative (the world's own state), and everything
// in it is limited to what that resident can legitimately see: its own
// holdings, offers and transfers, the people and stock in front of it, and
// the public project record. Other residents' holdings, plans, messages and
// memories never appear. What the resident believes about any of it lives in
// its own memory and may differ.

// recentWindow is how far back settled offers and completed transfers stay
// in the observation.
const recentWindow = 24 * time.Hour

// Situation renders id's grounded observation. reasons say why it is
// deciding now.
func (w *World) Situation(id sim.AgentID, reasons []Reason) string {
	r := w.Residents[id]
	if r == nil {
		return ""
	}
	var b strings.Builder
	if why := groupReasons(reasons); why != "" {
		fmt.Fprintf(&b, "Why you are deciding now: %s\n\n", why)
	}

	b.WriteString("YOU (known only to you)\n")
	if loc := w.Location(r.Location); loc != nil {
		fmt.Fprintf(&b, "  Where: %s (%s)\n", loc.Name, loc.ID)
	} else {
		b.WriteString("  Where: on the road\n")
	}
	w.writeErrand(&b, id)
	fmt.Fprintf(&b, "  Holding: %s (yours alone; nobody else's holdings are shown to you)\n", r.Goods)
	w.writeOwnProjects(&b, id)
	w.writeOffers(&b, id)
	w.writeLedger(&b, id)
	w.writeSetbacks(&b, id)
	w.writeMeetings(&b, id)
	w.writeIntentions(&b, id)

	w.writeHere(&b, id)
	w.writeElsewhere(&b, id)
	w.writeProjects(&b, id)

	if limits := w.constraints(id); len(limits) > 0 {
		b.WriteString("\nNOT POSSIBLE RIGHT NOW\n")
		for _, l := range limits {
			fmt.Fprintf(&b, "  - %s\n", l)
		}
	}
	return b.String()
}

// groupReasons joins reason texts, folding repeats ("Mira spoke to you at
// the workshop (3 times)").
func groupReasons(reasons []Reason) string {
	var order []string
	count := map[string]int{}
	for _, rs := range reasons {
		if count[rs.Text] == 0 {
			order = append(order, rs.Text)
		}
		count[rs.Text]++
	}
	for i, t := range order {
		if n := count[t]; n > 1 {
			order[i] = fmt.Sprintf("%s (%d times)", t, n)
		}
	}
	return strings.Join(order, "; ")
}

func (w *World) writeOwnProjects(b *strings.Builder, id sim.AgentID) {
	var open []string
	for _, p := range w.Projects {
		if p.Owner == id && !p.Closed() {
			open = append(open, fmt.Sprintf("%s %q at %s — %s", p.ID, p.Name, w.placeName(p.Location), projectProgress(p)))
		}
	}
	max := w.Settings.Economy.MaxOpenProjects
	switch {
	case len(open) == 0:
		b.WriteString("  Your unfinished projects: none\n")
	default:
		fmt.Fprintf(b, "  Your unfinished projects (%d of at most %d):\n", len(open), max)
		for _, l := range open {
			fmt.Fprintf(b, "    %s\n", l)
		}
	}
	var done []string
	for _, p := range w.Projects {
		if p.Owner == id && p.Done() {
			done = append(done, fmt.Sprintf("%q at %s — completed %s; has paid you %d credits so far",
				p.Name, w.placeName(p.Location), Stamp(*p.Completed), w.payments(*p.Completed)*w.Settings.Economy.ProjectYield))
		}
	}
	if len(done) > 0 {
		if y := w.Settings.Economy.ProjectYield; y > 0 {
			fmt.Fprintf(b, "  Your completed projects (each pays you, its owner, %d credits every dawn, automatically and wherever you are — there is nothing to collect; next payment %s):\n", y, Stamp(w.nextDawn()))
		} else {
			b.WriteString("  Your completed projects:\n")
		}
		for _, l := range done {
			fmt.Fprintf(b, "    %s\n", l)
		}
	}
}

// writeErrand reminds a resident that has just walked somewhere where it
// came from and what it said it came for.
func (w *World) writeErrand(b *strings.Builder, id sim.AgentID) {
	r := w.Residents[id]
	e := r.Errand
	if e == nil || e.Arrived == nil || e.To != r.Location {
		return
	}
	line := fmt.Sprintf("  You walked here from %s, arriving %s", w.placeName(e.From), clockOf(*e.Arrived))
	if Day(*e.Arrived) != Day(w.Now) {
		line = fmt.Sprintf("  You walked here from %s, arriving %s", w.placeName(e.From), Stamp(*e.Arrived))
	}
	if e.For != "" {
		line += fmt.Sprintf("; you said you were coming to: %q", e.For)
	}
	b.WriteString(line + "\n")
}

// projectProgress says where a project stands: what it has, and what it
// still lacks — or that it is complete.
func projectProgress(p *Project) string {
	have := fmt.Sprintf("has %d/%d credits, %d/%d materials, %d/%d labor hours",
		p.Raised.Credits, p.Needs.Credits, p.Raised.Materials, p.Needs.Materials, p.Labor, p.NeedsLabor)
	switch {
	case p.Done():
		return "COMPLETE"
	case p.Abandoned != nil:
		return "abandoned"
	}
	return have + "; NOT finished: still needs " + p.remaining()
}

func (w *World) writeOffers(b *strings.Builder, id sim.AgentID) {
	var mine, theirs, settled []string
	for _, o := range w.Offers {
		live := o.Status == "open" && w.Now.Before(o.Expires)
		switch {
		case live && o.From == id:
			mine = append(mine, fmt.Sprintf("%s to %s: you would give %s and get %s — open until %s%s\n      → %s",
				o.ID, w.name(o.To), goodsPhrase(o.Give), goodsPhrase(o.Want), Stamp(o.Expires), quoteNote(o.Note), w.offerConsequence(o, id)))
		case live && o.To == id:
			line := fmt.Sprintf("%s from %s: you would get %s and give %s — open until %s", o.ID, w.name(o.From), goodsPhrase(o.Give), goodsPhrase(o.Want), Stamp(o.Expires))
			if prev := w.earlierSameTerms(o); prev != "" {
				line += " (the same terms as " + prev + ")"
			} else {
				line += quoteNote(o.Note)
			}
			line += "\n      → " + w.offerConsequence(o, id)
			theirs = append(theirs, line)
		case !live && (o.From == id || o.To == id) && o.Resolved != nil && w.Now.Sub(*o.Resolved) <= recentWindow:
			dir := fmt.Sprintf("%s → you", w.name(o.From))
			if o.From == id {
				dir = fmt.Sprintf("you → %s", w.name(o.To))
			}
			line := fmt.Sprintf("%s %s (%s for %s): %s %s", o.ID, dir, goodsPhrase(o.Give), goodsPhrase(o.Want), o.Status, Stamp(*o.Resolved))
			if n := w.declinedTerms(o); n > 1 {
				line += fmt.Sprintf(" — these same terms have been turned down %d times", n)
			}
			settled = append(settled, line)
		}
	}
	if len(mine) > 0 {
		fmt.Fprintf(b, "  Your open offers (waiting on the other person; nothing has moved; only you can withdraw them — %d of at most %d):\n", len(mine), maxOpenOffers)
		for _, l := range mine {
			fmt.Fprintf(b, "    %s\n", l)
		}
	}
	if len(theirs) > 0 {
		b.WriteString("  Offers made to you (you can accept or decline; nothing has moved yet):\n")
		for _, l := range theirs {
			fmt.Fprintf(b, "    %s\n", l)
		}
	}
	if len(settled) > 0 {
		b.WriteString("  Offers settled in the last day (no longer open):\n")
		for _, l := range settled {
			fmt.Fprintf(b, "    %s\n", l)
		}
	}
}

// earlierSameTerms names an earlier offer on exactly o's terms between the
// same two people, if there is one.
func (w *World) earlierSameTerms(o *Offer) string {
	for _, x := range w.Offers {
		if x == o {
			return ""
		}
		if x.From == o.From && x.To == o.To && x.Give == o.Give && x.Want == o.Want && !x.Created.Before(o.Created.Add(-recentWindow)) {
			return x.ID
		}
	}
	return ""
}

// declinedTerms counts declined offers on exactly o's terms between the same
// two people.
func (w *World) declinedTerms(o *Offer) int {
	n := 0
	for _, x := range w.Offers {
		if x.From == o.From && x.To == o.To && x.Give == o.Give && x.Want == o.Want && x.Status == "declined" {
			n++
		}
	}
	return n
}

// writeLedger lists what actually changed hands involving id recently — the
// authoritative difference between a gift, a trade and a contribution, and
// between something moving and something merely being offered.
func (w *World) writeLedger(b *strings.Builder, id sim.AgentID) {
	var lines []string
	for i := len(w.Ledger) - 1; i >= 0 && len(lines) < 6; i-- {
		t := w.Ledger[i]
		if w.Now.Sub(t.Time) > recentWindow || (t.From != id && t.To != id) {
			continue
		}
		var line string
		switch t.Kind {
		case "gift":
			if t.From == id {
				line = fmt.Sprintf("you gave %s %s — a gift, nothing came back", w.name(t.To), goodsPhrase(t.Goods))
			} else {
				line = fmt.Sprintf("%s gave you %s — a gift, nothing was asked of you", w.name(t.From), goodsPhrase(t.Goods))
			}
		case "trade":
			if t.From == id {
				line = fmt.Sprintf("trade %s: you handed %s %s", t.Ref, w.name(t.To), goodsPhrase(t.Goods))
			} else {
				line = fmt.Sprintf("trade %s: %s handed you %s", t.Ref, w.name(t.From), goodsPhrase(t.Goods))
			}
			if o := w.offer(t.Ref); o != nil && (o.Give.IsZero() || o.Want.IsZero()) {
				line += " — nothing went the other way; the offer asked for nothing in return"
				if o.Give.IsZero() {
					line = strings.TrimSuffix(line, "; the offer asked for nothing in return") + "; the offer gave nothing in return"
				}
			}
		case "contribution":
			line = fmt.Sprintf("you put %s into project %s", goodsPhrase(t.Goods), t.Ref)
		case "refund":
			line = fmt.Sprintf("%s came back to you from abandoned project %s", goodsPhrase(t.Goods), t.Ref)
		default:
			continue
		}
		lines = append(lines, fmt.Sprintf("%s %s", Stamp(t.Time), line))
	}
	if len(lines) == 0 {
		return
	}
	b.WriteString("  What actually changed hands in the last day (newest first):\n")
	for _, l := range lines {
		fmt.Fprintf(b, "    %s\n", l)
	}
}

func (w *World) writeMeetings(b *strings.Builder, id sim.AgentID) {
	var meets []string
	for _, ap := range w.Appointments {
		if ap.Status != "pending" {
			continue
		}
		party := append([]sim.AgentID{ap.By}, ap.With...)
		for _, x := range party {
			if x == id {
				meets = append(meets, fmt.Sprintf("%s %s at %s with %s: %s", ap.ID, Stamp(ap.At), w.placeName(ap.Place), w.nameList(party), ap.What))
				break
			}
		}
	}
	if len(meets) > 0 {
		b.WriteString("  Meetings arranged (not happened yet):\n")
		for _, m := range meets {
			fmt.Fprintf(b, "    %s\n", m)
		}
	}
}

func (w *World) writeIntentions(b *strings.Builder, id sim.AgentID) {
	var mine []string
	for _, in := range w.Intentions {
		if in.Agent == id && in.Status == "pending" {
			line := fmt.Sprintf("%s at %s: %s", in.ID, Stamp(in.At), in.Note)
			if since := w.sinceNoted(in.Note, in.Created); len(since) > 0 {
				line += " — since you set it: " + strings.Join(since, "; ")
			}
			mine = append(mine, line)
		}
	}
	if len(mine) > 0 {
		fmt.Fprintf(b, "  Your pending intentions (%d of at most %d):\n", len(mine), w.Settings.Scheduling.MaxPendingIntentions)
		for _, m := range mine {
			fmt.Fprintf(b, "    %s\n", m)
		}
	}
}

// writeHere describes the place id is in: who is present and what they are
// visibly doing, and what the place itself offers today.
func (w *World) writeHere(b *strings.Builder, id sim.AgentID) {
	r := w.Residents[id]
	loc := w.Location(r.Location)
	if loc == nil {
		return
	}
	e := w.Settings.Economy
	fmt.Fprintf(b, "\nHERE — %s (what you can see)\n", loc.Name)
	var people []string
	for _, x := range w.Here(loc.ID) {
		if x == id {
			continue
		}
		p := fmt.Sprintf("%s (%s)", w.name(x), x)
		if act := w.Residents[x].Activity; act != nil && w.Now.Before(act.End) {
			p += " — " + activityPhrase(w, act) + " until " + clockOf(act.End)
		}
		people = append(people, p)
	}
	if len(people) == 0 {
		b.WriteString("  People: nobody else is here\n")
	} else {
		fmt.Fprintf(b, "  People: %s\n", strings.Join(people, "; "))
	}
	switch loc.Kind {
	case PlaceMarket:
		fmt.Fprintf(b, "  Market today: %d paid shifts left; %d materials for sale at %d credits each (restocks at dawn)\n",
			w.Stock.Shifts[loc.ID], w.Stock.Materials[loc.ID], e.MaterialPrice)
	case PlaceWorkshop:
		fmt.Fprintf(b, "  Scrap pile today: enough left for %d more materials (salvaging takes %dh for %d; refills at dawn)\n",
			w.Stock.Scrap[loc.ID], e.SalvageHours, e.SalvageMaterials)
	case PlaceSquare:
		var notes []string
		for _, n := range w.Notices {
			if n.Place == loc.ID {
				notes = append(notes, fmt.Sprintf("[%s, %s] %s", w.name(n.By), Stamp(n.Posted), n.Text))
			}
		}
		if len(notes) == 0 {
			b.WriteString("  Notice board: empty\n")
		} else {
			b.WriteString("  Notice board (claims by whoever posted them):\n")
			for _, n := range notes {
				fmt.Fprintf(b, "    %s\n", n)
			}
		}
	}
	// Completed projects standing here are listed under PROJECTS.
}

func activityPhrase(w *World, a *Activity) string {
	switch a.Kind {
	case "work":
		return "working a shift"
	case "salvage":
		return "salvaging scrap"
	case "build":
		if p := w.project(a.Project); p != nil {
			return fmt.Sprintf("building %q", p.Name)
		}
		return "building"
	}
	return a.Detail
}

func clockOf(t time.Time) string { return t.Format("15:04") }

// writeProjects is the public project record: every open project with what
// it still lacks, and the finished ones by name.
func (w *World) writeProjects(b *strings.Builder, id sim.AgentID) {
	var open, done []string
	for _, p := range w.Projects {
		switch {
		case p.Abandoned != nil:
		case p.Done() && p.Owner == id:
			// Already shown under YOU.
		case p.Done():
			line := fmt.Sprintf("%q — %s's, at %s, completed %s", p.Name, w.name(p.Owner), w.placeName(p.Location), Stamp(*p.Completed))
			if s := stake(p, id); s != "" {
				line += " (" + s + ")"
			}
			done = append(done, line)
		case p.Owner == id:
			// Already shown under YOU.
		default:
			line := fmt.Sprintf("%s %q — %s's project, at %s: %s", p.ID, p.Name, w.name(p.Owner), w.placeName(p.Location), projectProgress(p))
			if s := stake(p, id); s != "" {
				line += "; " + s
			}
			open = append(open, line)
		}
	}
	if len(open) == 0 && len(done) == 0 {
		return
	}
	b.WriteString("\nOTHER PEOPLE'S PROJECTS (public record; complete only when every requirement is met)\n")
	b.WriteString("  Each belongs to its owner for good. Putting resources or labor into one helps its owner finish it; it gives you no share, no say and none of its income.\n")
	if len(open) == 0 {
		b.WriteString("  none unfinished\n")
	}
	for _, l := range open {
		fmt.Fprintf(b, "  %s\n", l)
	}
	if len(done) > 0 {
		if y := w.Settings.Economy.ProjectYield; y > 0 {
			fmt.Fprintf(b, "  Completed (each pays %d credits every dawn to its owner alone):\n", y)
		} else {
			b.WriteString("  Completed:\n")
		}
		for _, l := range done {
			fmt.Fprintf(b, "    %s\n", l)
		}
	}
}

// maxOpenOffers is how many open offers one resident may have.
const maxOpenOffers = 3

// constraints lists the moves that cannot succeed right now and why, for
// the cases a decision most often trips over. It describes the world; it
// never recommends anything.
func (w *World) constraints(id sim.AgentID) []string {
	r := w.Residents[id]
	loc := w.Location(r.Location)
	if loc == nil {
		return []string{"anything but a private note (whisper): you are on the road"}
	}
	e := w.Settings.Economy
	var out []string
	others := len(w.Here(loc.ID)) - 1
	if others == 0 {
		out = append(out, "say: nobody else is here to hear it (a whisper reaches anyone, wherever they are)")
		out = append(out, "give: there is nobody here to hand anything to")
	}
	out = append(out, fmt.Sprintf("go to %s: you are already here", loc.ID))
	if n := w.openProjects(id); n >= e.MaxOpenProjects {
		out = append(out, fmt.Sprintf("start_project: you already have %d unfinished projects, the most allowed — one has to be completed or abandoned first", n))
	}
	if n := w.openOffers(id); n >= maxOpenOffers {
		out = append(out, fmt.Sprintf("offer: you already have %d open offers, the most allowed — wait for answers or withdraw one", n))
	}
	for _, o := range w.Offers {
		if o.To == id && o.Status == "open" && w.Now.Before(o.Expires) && !r.Goods.Covers(o.Want) {
			out = append(out, fmt.Sprintf("accept %s: it asks you for %s and you hold %s", o.ID, goodsPhrase(o.Want), r.Goods))
		}
	}
	if r.Goods.IsZero() {
		out = append(out, "give or contribute: you hold nothing")
	} else if r.Goods.Materials == 0 {
		out = append(out, "giving, offering or contributing materials: you hold none")
	}
	switch loc.Kind {
	case PlaceMarket:
		if w.Stock.Shifts[loc.ID] == 0 {
			out = append(out, "work: every shift here is taken for today")
		}
		switch {
		case w.Stock.Materials[loc.ID] == 0:
			out = append(out, "buy: the market has no materials left today")
		case r.Goods.Credits < e.MaterialPrice:
			out = append(out, fmt.Sprintf("buy: materials cost %d credits each and you have %d", e.MaterialPrice, r.Goods.Credits))
		case r.Goods.Credits/max(e.MaterialPrice, 1) < w.Stock.Materials[loc.ID]:
			n := r.Goods.Credits / max(e.MaterialPrice, 1)
			out = append(out, fmt.Sprintf("buy more than %d materials: at %d credits each, %d credits buys %d", n, e.MaterialPrice, r.Goods.Credits, n))
		}
	case PlaceWorkshop:
		if w.Stock.Scrap[loc.ID] == 0 {
			out = append(out, "salvage: the scrap pile here is empty until dawn")
		}
	}
	return out
}

func (w *World) openProjects(id sim.AgentID) int {
	n := 0
	for _, p := range w.Projects {
		if p.Owner == id && !p.Closed() {
			n++
		}
	}
	return n
}

func (w *World) openOffers(id sim.AgentID) int {
	n := 0
	for _, o := range w.Offers {
		if o.From == id && o.Status == "open" && w.Now.Before(o.Expires) {
			n++
		}
	}
	return n
}

// groundedActions is Actions for grounded worlds: the same verbs, but each
// described against the current state with the identifiers it can take, and
// moves that cannot succeed left out (they are named under NOT POSSIBLE
// RIGHT NOW instead).
func (w *World) groundedActions(id sim.AgentID) []sim.ActionSpec {
	r := w.Residents[id]
	if r == nil {
		return nil
	}
	e := w.Settings.Economy
	loc := w.Location(r.Location)
	var specs []sim.ActionSpec
	var others []string
	if loc != nil {
		for _, x := range w.Here(loc.ID) {
			if x != id {
				others = append(others, string(x))
			}
		}
	}
	if len(others) > 0 {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionSay, Usage: `{"kind":"say","content":"..."}`,
			Description: "Speak aloud; heard by everyone here (" + strings.Join(others, ", ") + ")."})
	}
	if len(w.Order) > 1 {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper","to":["<id>"],"content":"..."}`,
			Description: "Send a private note; it reaches them wherever they are."})
	}
	do := func(verb, usage, desc string, params ...string) {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionDo, Verb: verb, Usage: usage, Description: desc, Params: params})
	}
	pass := sim.ActionSpec{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Do nothing for now (wait)."}
	if loc == nil {
		return append(specs, pass)
	}

	var dests []string
	for _, l := range w.Locations {
		if l.ID == loc.ID || (l.Kind == PlaceResidence && l.Owner != id) {
			continue
		}
		dests = append(dests, l.ID)
	}
	do("go", `{"kind":"do","verb":"go","args":{"to":"<place id>","for":"..."}}`,
		fmt.Sprintf("Walk to another place (%d min) — you are at %s now: %s. \"for\" (optional, private): what you are going there to do; you are reminded of it when you arrive.", w.Settings.Scheduling.TravelMinutes, loc.ID, strings.Join(dests, ", ")), "to", "for")

	switch loc.Kind {
	case PlaceMarket:
		if w.Stock.Shifts[loc.ID] > 0 {
			do("work", `{"kind":"do","verb":"work"}`, fmt.Sprintf("Work a %d-hour shift for %d credits.", e.ShiftHours, e.ShiftCredits))
		}
		if afford := min(w.Stock.Materials[loc.ID], r.Goods.Credits/max(e.MaterialPrice, 1)); afford > 0 {
			do("buy", `{"kind":"do","verb":"buy","args":{"materials":<n>}}`,
				fmt.Sprintf("Buy materials at %d credits each (you can afford up to %d of the %d for sale).", e.MaterialPrice, afford, w.Stock.Materials[loc.ID]), "materials")
		}
	case PlaceWorkshop:
		if w.Stock.Scrap[loc.ID] > 0 {
			do("salvage", `{"kind":"do","verb":"salvage"}`, fmt.Sprintf("Spend %d hours salvaging; you come away with up to %d materials.", e.SalvageHours, e.SalvageMaterials))
		}
	case PlaceSquare:
		do("post", `{"kind":"do","verb":"post","args":{"text":"..."}}`, "Pin a notice to the board for anyone who comes by.", "text")
	}
	if len(others) > 0 && !r.Goods.IsZero() {
		do("give", `{"kind":"do","verb":"give","args":{"to":"<id>","credits":<n>,"materials":<n>}}`,
			fmt.Sprintf("Hand someone here (%s) part of what you hold, as a gift: it moves at once and nothing comes back. To exchange, make an offer instead.", strings.Join(others, ", ")),
			"to", "credits", "materials")
	}
	if len(w.Order) > 1 && w.openOffers(id) < maxOpenOffers {
		do("offer", `{"kind":"do","verb":"offer","args":{"to":"<id>","give_credits":<n>,"give_materials":<n>,"want_credits":<n>,"want_materials":<n>,"note":"..."}}`,
			fmt.Sprintf("Propose an exchange to anyone. give_* is what would leave YOUR hands; want_* is what you would get from THEM (to buy 2 materials for 6 credits: give_credits 6, want_materials 2). Nothing moves unless they accept, and you must still hold your side then (you hold %s).", r.Goods),
			"to", "give_credits", "give_materials", "want_credits", "want_materials", "note")
	}
	var own, incoming, acceptable []string
	for _, o := range w.Offers {
		if o.Status != "open" || !w.Now.Before(o.Expires) {
			continue
		}
		if o.From == id {
			own = append(own, o.ID)
		}
		if o.To == id {
			incoming = append(incoming, o.ID)
			if r.Goods.Covers(o.Want) {
				acceptable = append(acceptable, fmt.Sprintf("%s (you give %s, you get %s)", o.ID, goodsPhrase(o.Want), goodsPhrase(o.Give)))
			}
		}
	}
	if len(own) > 0 {
		do("withdraw", `{"kind":"do","verb":"withdraw","args":{"offer":"<offer id>"}}`, "Take back one of your own open offers: "+strings.Join(own, ", ")+".", "offer")
	}
	if len(acceptable) > 0 {
		do("accept", `{"kind":"do","verb":"accept","args":{"offer":"<offer id>"}}`, "Accept an offer made to you that you can pay for now: "+strings.Join(acceptable, "; ")+". It goes through only if they still hold their side.", "offer")
	}
	if len(incoming) > 0 {
		do("decline", `{"kind":"do","verb":"decline","args":{"offer":"<offer id>"}}`, "Turn down an offer made to you: "+strings.Join(incoming, ", ")+".", "offer")
	}
	if w.openProjects(id) < e.MaxOpenProjects {
		do("start_project", `{"kind":"do","verb":"start_project","args":{"name":"...","description":"...","place":"<place id>","credits":<n>,"materials":<n>,"labor":<hours>}}`,
			fmt.Sprintf("Start something of your own and say what it needs (at most %d credits, %d materials, %d labor hours).", e.MaxProjectCredits, e.MaxProjectMaterials, e.MaxProjectLabor),
			"name", "description", "place", "credits", "materials", "labor")
	}
	var fundable, buildable, mine []string
	for _, p := range w.Projects {
		if p.Closed() {
			continue
		}
		need := p.Needs.Minus(p.Raised)
		whose := "yours"
		if p.Owner != id {
			whose = w.name(p.Owner) + "'s"
		}
		if (need.Credits > 0 && r.Goods.Credits > 0) || (need.Materials > 0 && r.Goods.Materials > 0) {
			fundable = append(fundable, fmt.Sprintf("%s, %s, needs %s", p.ID, whose, goodsPhrase(Goods{Credits: max(need.Credits, 0), Materials: max(need.Materials, 0)})))
		}
		if p.Location == loc.ID && p.Labor+w.laborUnderway(p.ID) < p.NeedsLabor {
			buildable = append(buildable, fmt.Sprintf("%s, %s, needs %d labor hours", p.ID, whose, p.NeedsLabor-p.Labor))
		}
		if p.Owner == id {
			mine = append(mine, p.ID)
		}
	}
	if len(fundable) > 0 {
		do("contribute", `{"kind":"do","verb":"contribute","args":{"project":"<project id>","credits":<n>,"materials":<n>}}`,
			"Put some of what you hold into a project, from anywhere ("+strings.Join(fundable, "; ")+"). It is gone from you for good, and the project stays its owner's.", "project", "credits", "materials")
	}
	if len(buildable) > 0 {
		do("build", `{"kind":"do","verb":"build","args":{"project":"<project id>"}}`,
			fmt.Sprintf("Put %d hours of labor into a project here (%s); the project stays its owner's.", e.BuildHours, strings.Join(buildable, "; ")), "project")
	}
	if len(mine) > 0 {
		do("abandon", `{"kind":"do","verb":"abandon","args":{"project":"<project id>"}}`,
			"Give up one of your unfinished projects ("+strings.Join(mine, ", ")+"); whatever was contributed goes back to whoever gave it.", "project")
	}
	if len(w.Order) > 1 {
		do("arrange", `{"kind":"do","verb":"arrange","args":{"with":["<id>"],"place":"<place id>","when":"<e.g. tomorrow 18:00>","what":"..."}}`,
			"Arrange a meeting; everyone invited is told, and reminded when it comes.", "with", "place", "when", "what")
	}
	var meetings []string
	for _, ap := range w.Appointments {
		if ap.By == id && ap.Status == "pending" {
			meetings = append(meetings, ap.ID)
		}
	}
	if len(meetings) > 0 {
		do("cancel_meeting", `{"kind":"do","verb":"cancel_meeting","args":{"meeting":"<meeting id>"}}`,
			"Call off a meeting you arranged ("+strings.Join(meetings, ", ")+"); everyone invited is told.", "meeting")
	}
	return append(specs, pass)
}

// groundedBrief is appended to the brief in grounded worlds.
const groundedBrief = `
  - "give" is a gift: it moves at once, one way. An exchange is an offer the
    other person accepts; until then nothing moves. In an offer, give_* is
    what you hand over and want_* is what you get back. A project is complete
    only when its credits, materials and labor are all met. You can abandon
    your own project; whatever was put in goes back to whoever gave it.
  - A project always belongs to the person who started it. Its daily income
    goes to that owner alone, automatically, wherever they are — nobody
    collects it. Putting resources or labor into someone else's project helps
    them finish it; it gives you no ownership, share or income.
  - Your situation (YOU, HERE, PROJECTS) is the world's own record of what
    you can see now. Others' holdings, plans and messages are not in it.`
