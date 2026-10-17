package world

import (
	"fmt"
	"sort"
	"strings"

	"github.com/nmiano/playworld/internal/sim"
)

// Brief is the stable description of how the world works. It deliberately
// prescribes nothing: no goal, no expected activity, no suggestion that any
// particular thing is worth doing. It is identical for every activation so
// the backend can keep the system prompt cached.
func (w *World) Brief(id sim.AgentID) string {
	sc, e := w.Settings.Scheduling, w.Settings.Economy
	var b strings.Builder
	b.WriteString(`You live in a small town with the other people listed below. Nobody gives you
a task and there is no shared goal: what you do, with whom, and when is up to
you. Life goes on whether or not you act. You get a moment to decide when
something concerns you, when something you started finishes, when a time you
asked for arrives, and now and then when things are quiet. Doing nothing is a
perfectly good choice.

Places:
`)
	for _, l := range w.Locations {
		if l.Kind == PlaceResidence {
			continue
		}
		fmt.Fprintf(&b, "  - %s (%s): %s\n", l.ID, l.Name, l.Description)
	}
	fmt.Fprintf(&b, "  - home_<id>: each person's own home. Yours is %s; \"home\" works too.\n", w.Residents[id].Home)
	fmt.Fprintf(&b, `Walking between any two places takes %d minutes.

How things work:
  - "say" is heard only by the people in the same place as you. A "whisper"
    is a private note that reaches the named person wherever they are.
  - There are two resources, credits and materials. You can earn credits with
    a %d-hour shift at the market (+%d credits; a few shifts a day), buy
    materials at the market (%d credits each; limited stock), or salvage them
    at the workshop (%d hours, +%d; limited scrap). Stock resets every dawn.
  - Anyone can start a project anywhere public, or in their own home, saying
    what it needs in credits, materials and hours of labor. Anyone can
    contribute resources to it from anywhere, or build it (%d-hour stints) at
    its place. A completed project stays for good and pays its owner %d
    credits every morning.
  - You can hand things to someone in the same place, make anyone a trade
    offer (they accept or decline; nothing moves until then), pin a notice at
    the square, or arrange a meeting with others at a time and place. You
    can withdraw your own offers and call off meetings you arranged.
  - Working, salvaging, building and walking take time; you cannot do
    anything else until they finish.
  - Only actions change the world. Saying you will do something does not do
    it, a promise is not a payment, and an arranged meeting has not happened
    yet. Lines marked ✔ are things you saw actually happen; anything someone
    said, whispered or posted is only their claim. The situation shown to you
    each time is the truth about what you can see; what you remember or were
    told may be out of date or wrong.
  - Waking hours are %02d:00 to %02d:00.`,
		sc.TravelMinutes, e.ShiftHours, e.ShiftCredits, e.MaterialPrice, e.SalvageHours, e.SalvageMaterials,
		e.BuildHours, e.ProjectYield, sc.DayStartHour, sc.NightStartHour%24)
	if w.Settings.Grounded() {
		b.WriteString(groundedBrief)
	}
	b.WriteString("\n")
	return b.String()
}

// verbs lists the do-verbs currently open to id, for refusal messages.
func (w *World) verbs(id sim.AgentID) []string {
	var out []string
	for _, s := range w.Actions(id) {
		if s.Kind == sim.ActionDo {
			out = append(out, s.Verb)
		}
	}
	return out
}

// Actions lists what id can sensibly do right now. Like the community
// scenario's rich affordances, it leaves out moves whose prerequisites are
// plainly unmet, so a small model is not offered actions that can only fail.
func (w *World) Actions(id sim.AgentID) []sim.ActionSpec {
	if w.Settings.Grounded() {
		return w.groundedActions(id)
	}
	r := w.Residents[id]
	if r == nil {
		return nil
	}
	e := w.Settings.Economy
	loc := w.Location(r.Location)
	var specs []sim.ActionSpec
	others := 0
	for _, x := range w.Here(r.Location) {
		if x != id {
			others++
		}
	}
	if others > 0 {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionSay, Usage: `{"kind":"say","content":"..."}`,
			Description: "Speak aloud to the people here."})
	}
	if len(w.Order) > 1 {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionWhisper, Usage: `{"kind":"whisper","to":["<id>"],"content":"..."}`,
			Description: "Send a private note; it reaches them wherever they are."})
	}
	do := func(verb, usage, desc string, params ...string) {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionDo, Verb: verb, Usage: usage, Description: desc, Params: params})
	}
	if loc == nil {
		specs = append(specs, sim.ActionSpec{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Do nothing for now."})
		return specs
	}
	do("go", `{"kind":"do","verb":"go","args":{"to":"<place id>"}}`, "Walk somewhere else.", "to")
	switch loc.Kind {
	case PlaceMarket:
		if w.Stock.Shifts[loc.ID] > 0 {
			do("work", `{"kind":"do","verb":"work"}`, fmt.Sprintf("Work a %d-hour shift for %d credits.", e.ShiftHours, e.ShiftCredits))
		}
		if w.Stock.Materials[loc.ID] > 0 && r.Goods.Credits >= e.MaterialPrice {
			do("buy", `{"kind":"do","verb":"buy","args":{"materials":<n>}}`, fmt.Sprintf("Buy materials at %d credits each.", e.MaterialPrice), "materials")
		}
	case PlaceWorkshop:
		if w.Stock.Scrap[loc.ID] > 0 {
			do("salvage", `{"kind":"do","verb":"salvage"}`, fmt.Sprintf("Spend %d hours salvaging materials.", e.SalvageHours))
		}
	case PlaceSquare:
		do("post", `{"kind":"do","verb":"post","args":{"text":"..."}}`, "Pin a notice to the board for anyone who comes by.", "text")
	}
	if others > 0 && !r.Goods.IsZero() {
		do("give", `{"kind":"do","verb":"give","args":{"to":"<id>","credits":<n>,"materials":<n>}}`, "Hand someone here some of what you have.", "to", "credits", "materials")
	}
	if len(w.Order) > 1 {
		do("offer", `{"kind":"do","verb":"offer","args":{"to":"<id>","give_credits":<n>,"give_materials":<n>,"want_credits":<n>,"want_materials":<n>,"note":"..."}}`,
			"Propose a trade to anyone; nothing moves unless they accept.", "to", "give_credits", "give_materials", "want_credits", "want_materials", "note")
	}
	hasOffer, ownOffer := false, false
	for _, o := range w.Offers {
		if o.Status == "open" && w.Now.Before(o.Expires) {
			hasOffer = hasOffer || o.To == id
			ownOffer = ownOffer || o.From == id
		}
	}
	if ownOffer {
		do("withdraw", `{"kind":"do","verb":"withdraw","args":{"offer":"<offer id>"}}`, "Take back an open offer you made.", "offer")
	}
	if hasOffer {
		do("accept", `{"kind":"do","verb":"accept","args":{"offer":"<offer id>"}}`, "Accept an offer made to you.", "offer")
		do("decline", `{"kind":"do","verb":"decline","args":{"offer":"<offer id>"}}`, "Turn down an offer made to you.", "offer")
	}
	do("start_project", `{"kind":"do","verb":"start_project","args":{"name":"...","description":"...","place":"<place id>","credits":<n>,"materials":<n>,"labor":<hours>}}`,
		"Start something of your own and say what it needs.", "name", "description", "place", "credits", "materials", "labor")
	openAnywhere, openHere := false, false
	for _, p := range w.Projects {
		if p.Closed() {
			continue
		}
		openAnywhere = true
		if p.Location == loc.ID && p.Labor+w.laborUnderway(p.ID) < p.NeedsLabor {
			openHere = true
		}
	}
	if openAnywhere && !r.Goods.IsZero() {
		do("contribute", `{"kind":"do","verb":"contribute","args":{"project":"<project id>","credits":<n>,"materials":<n>}}`, "Put resources into a project.", "project", "credits", "materials")
	}
	if openHere {
		do("build", `{"kind":"do","verb":"build","args":{"project":"<project id>"}}`, fmt.Sprintf("Put %d hours of labor into a project here.", e.BuildHours), "project")
	}
	if len(w.Order) > 1 {
		do("arrange", `{"kind":"do","verb":"arrange","args":{"with":["<id>"],"place":"<place id>","when":"<e.g. tomorrow 18:00>","what":"..."}}`,
			"Arrange a meeting; everyone invited is told, and reminded when it comes.", "with", "place", "when", "what")
	}
	for _, ap := range w.Appointments {
		if ap.By == id && ap.Status == "pending" {
			do("cancel_meeting", `{"kind":"do","verb":"cancel_meeting","args":{"meeting":"<meeting id>"}}`, "Call off a meeting you arranged; everyone invited is told.", "meeting")
			break
		}
	}
	specs = append(specs, sim.ActionSpec{Kind: sim.ActionPass, Usage: `{"kind":"pass"}`, Description: "Do nothing for now (wait)."})
	return specs
}

// PublicState is what everyone in town knows: the time, and the projects
// underway or finished.
func (w *World) PublicState() map[string]any {
	out := map[string]any{"time": Clock(w.Now)}
	var open, done []string
	for _, p := range w.Projects {
		if p.Abandoned != nil {
			continue
		}
		if p.Done() {
			done = append(done, fmt.Sprintf("%s %q by %s at %s", p.ID, p.Name, w.name(p.Owner), w.placeName(p.Location)))
			continue
		}
		open = append(open, fmt.Sprintf("%s %q by %s at %s — has %d/%d credits, %d/%d materials, %d/%d labor hours",
			p.ID, p.Name, w.name(p.Owner), w.placeName(p.Location),
			p.Raised.Credits, p.Needs.Credits, p.Raised.Materials, p.Needs.Materials, p.Labor, p.NeedsLabor))
	}
	if len(open) > 0 {
		out["projects underway"] = strings.Join(open, "; ")
	}
	if len(done) > 0 {
		out["completed projects"] = strings.Join(done, "; ")
	}
	return out
}

// PrivateState is what only id can see right now: where it is, who is
// there, what it holds, what this place offers, and its own pending offers,
// meetings and intentions. reasons says why it is being asked to decide.
func (w *World) PrivateState(id sim.AgentID, reasons []Reason) map[string]any {
	r := w.Residents[id]
	if r == nil {
		return nil
	}
	e := w.Settings.Economy
	out := map[string]any{}
	loc := w.Location(r.Location)
	if loc == nil {
		out["where you are"] = "on the road"
	} else {
		out["where you are"] = fmt.Sprintf("%s (%s)", loc.Name, loc.ID)
		var others []string
		for _, x := range w.Here(loc.ID) {
			if x != id {
				others = append(others, fmt.Sprintf("%s (%s)", w.name(x), x))
			}
		}
		if len(others) == 0 {
			out["here with you"] = "nobody"
		} else {
			out["here with you"] = strings.Join(others, ", ")
		}
		var place []string
		switch loc.Kind {
		case PlaceMarket:
			place = append(place, fmt.Sprintf("%d shifts left today, %d materials for sale at %d credits each",
				w.Stock.Shifts[loc.ID], w.Stock.Materials[loc.ID], e.MaterialPrice))
		case PlaceWorkshop:
			place = append(place, fmt.Sprintf("enough scrap left today for %d materials", w.Stock.Scrap[loc.ID]))
		case PlaceSquare:
			var notes []string
			for _, n := range w.Notices {
				if n.Place == loc.ID {
					notes = append(notes, fmt.Sprintf("[%s, %s] %s", w.name(n.By), Stamp(n.Posted), n.Text))
				}
			}
			if len(notes) == 0 {
				place = append(place, "the notice board is empty")
			} else {
				place = append(place, "notice board: "+strings.Join(notes, " | "))
			}
		}
		if len(loc.Features) > 0 {
			place = append(place, "here stands: "+strings.Join(loc.Features, "; "))
		}
		if len(place) > 0 {
			out["this place"] = strings.Join(place, "; ")
		}
	}
	out["you have"] = r.Goods.String()

	var toYou, fromYou []string
	for _, o := range w.Offers {
		if o.Status != "open" || !w.Now.Before(o.Expires) {
			continue
		}
		line := fmt.Sprintf("%s: %s%s, until %s", o.ID, offerPhrase(o), quoteNote(o.Note), Stamp(o.Expires))
		if o.To == id {
			toYou = append(toYou, fmt.Sprintf("from %s — %s", w.name(o.From), line))
		}
		if o.From == id {
			fromYou = append(fromYou, fmt.Sprintf("to %s — %s", w.name(o.To), line))
		}
	}
	if len(toYou) > 0 {
		out["offers made to you"] = strings.Join(toYou, "; ")
	}
	if len(fromYou) > 0 {
		out["your open offers"] = strings.Join(fromYou, "; ")
	}

	var meets []string
	for _, ap := range w.Appointments {
		if ap.Status != "pending" {
			continue
		}
		party := append([]sim.AgentID{ap.By}, ap.With...)
		involved := false
		for _, x := range party {
			involved = involved || x == id
		}
		if involved {
			meets = append(meets, fmt.Sprintf("%s %s at %s with %s: %s", ap.ID, Stamp(ap.At), w.placeName(ap.Place), w.nameList(party), ap.What))
		}
	}
	if len(meets) > 0 {
		out["meetings arranged"] = strings.Join(meets, "; ")
	}

	var mine []string
	for _, in := range w.Intentions {
		if in.Agent == id && in.Status == "pending" {
			mine = append(mine, fmt.Sprintf("%s at %s: %s", in.ID, Stamp(in.At), in.Note))
		}
	}
	if len(mine) > 0 {
		out["your intentions"] = strings.Join(mine, "; ")
	}

	if len(reasons) > 0 {
		var why []string
		for _, rs := range reasons {
			why = append(why, rs.Text)
		}
		out["why you are deciding now"] = strings.Join(why, "; ")
	}
	return out
}

// --- inspection --------------------------------------------------------------

// Describe renders the world for `playworld world inspect`.
func (w *World) Describe(events int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "world     %s\n", w.Name)
	fmt.Fprintf(&b, "clock     %s\n", Clock(w.Now))
	fmt.Fprintf(&b, "seed      %d · roster %s\n", w.Seed, w.RosterPath)
	fmt.Fprintf(&b, "sessions  %d\n", len(w.Sessions))
	for _, s := range w.Sessions {
		fmt.Fprintf(&b, "  #%d  %s → %s  %d activations, %d llm calls — %s\n",
			s.Number, Stamp(s.SimFrom), Stamp(s.SimTo), s.Activations, s.LLMCalls, s.StopReason)
	}

	b.WriteString("\nresidents\n")
	for _, id := range w.Order {
		r := w.Residents[id]
		where := w.placeName(r.Location)
		if r.Location == "" {
			where = "on the road"
		}
		fmt.Fprintf(&b, "  %-8s %-10s at %-18s %s", id, r.Name, where, r.Goods)
		if r.Activity != nil {
			fmt.Fprintf(&b, " · %s until %s", r.Activity.Detail, Stamp(r.Activity.End))
		}
		fmt.Fprintf(&b, " · %d activations\n", r.Activations)
	}

	b.WriteString("\nplaces\n")
	for _, l := range w.Locations {
		var here []string
		for _, x := range w.Here(l.ID) {
			here = append(here, string(x))
		}
		fmt.Fprintf(&b, "  %-14s %s", l.ID, strings.Join(here, ", "))
		for _, f := range l.Features {
			fmt.Fprintf(&b, "\n                 + %s", f)
		}
		b.WriteString("\n")
	}

	if len(w.Projects) > 0 {
		b.WriteString("\nprojects\n")
		for _, p := range w.Projects {
			status := "open"
			switch {
			case p.Done():
				status = "completed " + Stamp(*p.Completed)
			case p.Abandoned != nil:
				status = "abandoned " + Stamp(*p.Abandoned)
			}
			fmt.Fprintf(&b, "  %-4s %-24q by %-6s at %-10s credits %d/%d materials %d/%d labor %d/%d  %s\n",
				p.ID, p.Name, p.Owner, p.Location, p.Raised.Credits, p.Needs.Credits, p.Raised.Materials, p.Needs.Materials, p.Labor, p.NeedsLabor, status)
		}
	}
	if len(w.Offers) > 0 {
		b.WriteString("\noffers\n")
		for _, o := range w.Offers {
			fmt.Fprintf(&b, "  %-4s %s → %s: %s [%s]\n", o.ID, o.From, o.To, offerPhrase(o), o.Status)
		}
	}
	if len(w.Appointments) > 0 {
		b.WriteString("\nmeetings\n")
		for _, ap := range w.Appointments {
			fmt.Fprintf(&b, "  %-4s %s at %s, %s with %s: %s [%s]\n", ap.ID, Stamp(ap.At), ap.Place, ap.By, w.nameList(ap.With), ap.What, ap.Status)
		}
	}
	if len(w.Intentions) > 0 {
		b.WriteString("\nintentions\n")
		for _, in := range w.Intentions {
			fmt.Fprintf(&b, "  %-4s %-6s %s: %s [%s]\n", in.ID, in.Agent, Stamp(in.At), in.Note, in.Status)
		}
	}
	if len(w.Notices) > 0 {
		b.WriteString("\nnotice board\n")
		for _, n := range w.Notices {
			fmt.Fprintf(&b, "  %s %s: %s\n", Stamp(n.Posted), n.By, n.Text)
		}
	}

	b.WriteString("\nscheduled\n")
	for i, it := range w.queue.Sorted() {
		if i >= 15 {
			fmt.Fprintf(&b, "  … %d more\n", w.queue.Len()-i)
			break
		}
		fmt.Fprintf(&b, "  %s  %-13s %-6s %s\n", Stamp(it.At), it.Kind, it.Agent, describeItem(it))
	}

	if len(w.Totals) > 0 {
		b.WriteString("\ntotals\n")
		keys := make([]string, 0, len(w.Totals))
		for k := range w.Totals {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Fprintf(&b, "  %-24s %d\n", k, w.Totals[k])
		}
	}

	if events > 0 && len(w.History) > 0 {
		b.WriteString("\nrecent events\n")
		start := max(len(w.History)-events, 0)
		for _, h := range w.History[start:] {
			who := ""
			if h.Actor != "" {
				who = string(h.Actor) + " "
			}
			fmt.Fprintf(&b, "  %s  %-8s %s%s  [%s]\n", Stamp(h.Time), h.Type, who, clip(h.Text, 140), h.Visibility)
		}
	}
	return b.String()
}

func describeItem(it *Item) string {
	switch it.Kind {
	case KindActivate:
		var rs []string
		for _, r := range it.Reasons {
			rs = append(rs, r.Text)
		}
		return clip(strings.Join(rs, "; "), 90)
	case KindIntention, KindAppointment:
		return it.Ref
	}
	return ""
}
