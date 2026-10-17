package world

import (
	"fmt"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/textsim"
)

// This file spells out the practical consequences of the world's moves for
// the one resident who is deciding: what an offer would take from it and
// leave it holding, who a project belongs to and who it pays, what went
// wrong for it lately, and what it last saw elsewhere. Everything here is
// derived from the world's own state and limited to what that resident can
// legitimately know. None of it judges whether a move is wise.

// offerConsequence describes, from id's side, what offer o would do if it
// went through: whether id could do its part right now, and what id would be
// left holding.
func (w *World) offerConsequence(o *Offer, id sim.AgentID) string {
	r := w.Residents[id]
	if r == nil {
		return ""
	}
	var gives, gets Goods
	if o.From == id {
		gives, gets = o.Give, o.Want
	} else {
		gives, gets = o.Want, o.Give
	}
	var parts []string
	if n := netPhrase(gives, gets); n != "" {
		parts = append(parts, n)
	}
	switch {
	case !r.Goods.Covers(gives) && o.From == id:
		parts = append(parts, fmt.Sprintf("you no longer hold %s — you hold %s — so it cannot go through as it stands", goodsPhrase(gives), r.Goods))
	case !r.Goods.Covers(gives):
		parts = append(parts, fmt.Sprintf("you cannot accept it now: it asks you for %s and you hold %s", goodsPhrase(gives), r.Goods))
	default:
		parts = append(parts, fmt.Sprintf("if it goes through you will hold %s", r.Goods.Minus(gives).Plus(gets)))
	}
	switch {
	case gets.IsZero():
		parts = append(parts, "you would get nothing back")
	case gives.IsZero():
		parts = append(parts, "you would give nothing for it")
	}
	return strings.Join(parts, "; ")
}

// netPhrase sums a two-sided exchange that moves the same resource both
// ways ("credits go both ways: net, you lose 2 credits"); empty otherwise.
func netPhrase(gives, gets Goods) string {
	var parts []string
	for _, x := range []struct {
		name       string
		give, gets int
	}{{"credits", gives.Credits, gets.Credits}, {"materials", gives.Materials, gets.Materials}} {
		if x.give == 0 || x.gets == 0 {
			continue
		}
		switch d := x.gets - x.give; {
		case d > 0:
			parts = append(parts, fmt.Sprintf("%s go both ways: net, you gain %d", x.name, d))
		case d < 0:
			parts = append(parts, fmt.Sprintf("%s go both ways: net, you lose %d", x.name, -d))
		default:
			parts = append(parts, fmt.Sprintf("%s go both ways and cancel out", x.name))
		}
	}
	return strings.Join(parts, "; ")
}

// payments counts the dawns at which a project completed at done has paid
// its owner, up to now.
func (w *World) payments(done time.Time) int {
	if w.Settings.Economy.ProjectYield <= 0 {
		return 0
	}
	first := Day(done)
	if !w.dawnOf(first).After(done) {
		first++
	}
	last := Day(w.Now)
	if w.dawnOf(last).After(w.Now) {
		last--
	}
	return max(last-first+1, 0)
}

// dawnOf is when day's restock and project payments happen.
func (w *World) dawnOf(day int) time.Time {
	return At(day, w.Settings.Scheduling.DayStartHour-1, 0)
}

// nextDawn is the next restock and payment time after now.
func (w *World) nextDawn() time.Time {
	d := w.dawnOf(Day(w.Now))
	if !d.After(w.Now) {
		d = w.dawnOf(Day(w.Now) + 1)
	}
	return d
}

// stake describes what id put into project p ("you put in 3 materials and
// 2 labor hours"); empty if nothing.
func stake(p *Project, id sim.AgentID) string {
	var parts []string
	if g, ok := p.Contributors[id]; ok && !g.IsZero() {
		parts = append(parts, goodsPhrase(g))
	}
	if h := p.Builders[id]; h > 0 {
		parts = append(parts, fmt.Sprintf("%d labor hours", h))
	}
	if len(parts) == 0 {
		return ""
	}
	return "you put in " + strings.Join(parts, " and ")
}

// noteSetback records something that went wrong for id.
func (w *World) noteSetback(id sim.AgentID, s Setback) {
	r := w.Residents[id]
	if r == nil {
		return
	}
	s.At = w.Now
	r.Setbacks = append(r.Setbacks, s)
	if len(r.Setbacks) > maxSetbacks {
		r.Setbacks = r.Setbacks[len(r.Setbacks)-maxSetbacks:]
	}
}

// NoteRefusal records a refused action as a setback for id, and reports
// whether it repeats the previous refusal's reason within the last day.
func (w *World) NoteRefusal(id sim.AgentID, what string, err error) (repeat bool) {
	r := w.Residents[id]
	if r == nil {
		return false
	}
	code := RefusalCode(err)
	for i := len(r.Setbacks) - 1; i >= 0; i-- {
		if s := r.Setbacks[i]; s.Kind == "refused" {
			repeat = s.Code == code && w.Now.Sub(s.At) <= recentWindow
			break
		}
	}
	w.noteSetback(id, Setback{Kind: "refused", Code: code, Text: fmt.Sprintf("your %s was refused: %s", what, err.Error())})
	return repeat
}

// writeSetbacks lists id's refused actions in the last day, newest first,
// folding refusals for the same reason. (Declined and expired offers show
// under the settled offers.)
func (w *World) writeSetbacks(b *strings.Builder, id sim.AgentID) {
	r := w.Residents[id]
	type line struct {
		text  string
		first time.Time
		last  time.Time
		n     int
	}
	var lines []*line
	byCode := map[string]*line{}
	for i := len(r.Setbacks) - 1; i >= 0; i-- {
		s := r.Setbacks[i]
		if w.Now.Sub(s.At) > recentWindow {
			break
		}
		if s.Kind == "refused" {
			if l := byCode[s.Code]; l != nil {
				l.n++
				l.first = s.At
				continue
			}
		}
		l := &line{text: s.Text, first: s.At, last: s.At, n: 1}
		lines = append(lines, l)
		if s.Kind == "refused" {
			byCode[s.Code] = l
		}
	}
	if len(lines) == 0 {
		return
	}
	b.WriteString("  Your refused actions in the last day (newest first):\n")
	for _, l := range lines {
		if l.n > 1 {
			fmt.Fprintf(b, "    %s %s (%d times since %s)\n", Stamp(l.last), l.text, l.n, Stamp(l.first))
			continue
		}
		fmt.Fprintf(b, "    %s %s\n", Stamp(l.last), l.text)
	}
}

// Look records what id can see of the stock where it is now.
func (w *World) Look(id sim.AgentID) {
	r := w.Residents[id]
	if r == nil {
		return
	}
	loc := w.Location(r.Location)
	if loc == nil || (loc.Kind != PlaceMarket && loc.Kind != PlaceWorkshop) {
		return
	}
	if r.Seen == nil {
		r.Seen = map[string]Glimpse{}
	}
	r.Seen[loc.ID] = Glimpse{At: w.Now, Materials: w.Stock.Materials[loc.ID], Shifts: w.Stock.Shifts[loc.ID], Scrap: w.Stock.Scrap[loc.ID]}
}

// writeElsewhere lists what id last saw at the market and workshop when it
// is not there now — a memory of the place, which may have changed.
func (w *World) writeElsewhere(b *strings.Builder, id sim.AgentID) {
	r := w.Residents[id]
	var lines []string
	for _, l := range w.Locations {
		g, ok := r.Seen[l.ID]
		if !ok || l.ID == r.Location {
			continue
		}
		var what string
		switch l.Kind {
		case PlaceMarket:
			what = fmt.Sprintf("%d materials for sale, %d shifts left", g.Materials, g.Shifts)
		case PlaceWorkshop:
			what = fmt.Sprintf("scrap enough for %d more materials", g.Scrap)
		default:
			continue
		}
		line := fmt.Sprintf("%s, as of %s: %s", l.Name, Stamp(g.At), what)
		if w.dawnOf(Day(w.Now)).After(g.At) && !w.dawnOf(Day(w.Now)).After(w.Now) {
			line += " — that was before today's restock"
		}
		lines = append(lines, line)
	}
	if len(lines) == 0 {
		return
	}
	b.WriteString("\nELSEWHERE — LAST YOU SAW (others may have bought or salvaged since)\n")
	for _, l := range lines {
		fmt.Fprintf(b, "  %s\n", l)
	}
}

// sinceNoted lists what has happened, since t, to the projects and offers a
// note refers to: by id ("p4", "o21") or, for projects, by name. It is how
// an intention set before a project was finished says so when it comes due.
func (w *World) sinceNoted(note string, t time.Time) []string {
	var out []string
	refs := map[string]bool{}
	for _, r := range textsim.Refs(note) {
		refs[r] = true
	}
	for _, p := range w.Projects {
		if !refs[strings.ToLower(p.ID)] && !textsim.Mentions(note, p.Name) {
			continue
		}
		switch {
		case p.Completed != nil && p.Completed.After(t):
			out = append(out, fmt.Sprintf("%q (%s's) was completed %s", p.Name, w.name(p.Owner), Stamp(*p.Completed)))
		case p.Abandoned != nil && p.Abandoned.After(t):
			out = append(out, fmt.Sprintf("%q (%s's) was abandoned %s", p.Name, w.name(p.Owner), Stamp(*p.Abandoned)))
		}
	}
	for _, o := range w.Offers {
		if refs[strings.ToLower(o.ID)] && o.Resolved != nil && o.Resolved.After(t) {
			out = append(out, fmt.Sprintf("offer %s was %s %s", o.ID, o.Status, Stamp(*o.Resolved)))
		}
	}
	return out
}

// citesClosedProject reports whether text mentions a project that was
// already completed or abandoned (for the metric only).
func (w *World) citesClosedProject(text string) bool {
	for _, p := range w.Projects {
		if p.Closed() && textsim.Mentions(text, p.Name) {
			return true
		}
	}
	return false
}
