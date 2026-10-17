package world

import (
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/nmiano/playworld/internal/sim"
	"github.com/nmiano/playworld/internal/textsim"
)

// Trigger is a request to give an agent a moment to decide. Triggers never
// call a model themselves: they coalesce into at most one pending activation
// per agent, placed no earlier than the agent's cooldown, the end of whatever
// it is busy with, and (unless Explicit) the next waking hour.
type Trigger struct {
	Agent    sim.AgentID
	At       time.Time
	Reason   Reason
	Explicit bool
}

// Effects is what applying an action or processing a scheduled item
// produced: events to deliver, and agents to trigger.
type Effects struct {
	Events   []sim.Event
	Triggers []Trigger
}

func (e *Effects) add(o Effects) {
	e.Events = append(e.Events, o.Events...)
	e.Triggers = append(e.Triggers, o.Triggers...)
}

// TriggerResult says what a trigger did to the schedule.
type TriggerResult struct {
	At        time.Time
	Damped    bool // past the reply-chain limit: woke nobody
	Coalesced bool // joined an activation that was already pending
}

// Trigger schedules (or coalesces into) an activation for t.Agent.
func (w *World) Trigger(t Trigger) TriggerResult {
	r := w.Residents[t.Agent]
	if r == nil {
		return TriggerResult{Damped: true}
	}
	if t.Reason.Depth > w.Settings.Scheduling.MaxReplyChain {
		// Too many reactions deep: the event was delivered, but it does not
		// earn another activation. This is what ends a ping-pong exchange.
		w.count("reactions_damped", 1)
		w.tally(t.Agent, "reactions_damped", 1)
		return TriggerResult{Damped: true}
	}
	at := w.eligible(r, t.At, t.Explicit)
	if it := w.queue.Get(r.ActivateItem); it != nil {
		it.Reasons = appendReason(it.Reasons, t.Reason)
		it.Explicit = it.Explicit || t.Explicit
		if at.Before(it.At) {
			w.queue.Reschedule(it.ID, at)
		}
		w.count("triggers_coalesced", 1)
		return TriggerResult{At: it.At, Coalesced: true}
	}
	r.ActivateItem = w.push(&Item{
		At: at, Kind: KindActivate, Agent: t.Agent, Reasons: []Reason{t.Reason}, Explicit: t.Explicit,
	})
	return TriggerResult{At: at}
}

func appendReason(rs []Reason, r Reason) []Reason {
	for _, x := range rs {
		if x == r {
			return rs
		}
	}
	return append(rs, r)
}

// eligible is the earliest time an activation for r may happen at or after at.
func (w *World) eligible(r *Resident, at time.Time, explicit bool) time.Time {
	sc := w.Settings.Scheduling
	if at.Before(w.Now) {
		at = w.Now
	}
	if !r.LastActivation.IsZero() {
		if c := r.LastActivation.Add(time.Duration(sc.CooldownMinutes) * time.Minute); at.Before(c) {
			at = c
		}
	}
	if r.Activity != nil && at.Before(r.Activity.End) {
		at = r.Activity.End
	}
	if !explicit {
		at = w.awake(r, at)
	}
	if Day(at) == r.ActDay && r.ActCount >= sc.MaxActivationsPerDay {
		at = w.morning(r, Day(at)+1)
	}
	return at
}

// awake moves at into waking hours.
func (w *World) awake(r *Resident, at time.Time) time.Time {
	sc := w.Settings.Scheduling
	h := at.Hour()
	switch {
	case h >= sc.NightStartHour:
		return w.morning(r, Day(at)+1)
	case h < sc.DayStartHour:
		if m := w.morning(r, Day(at)); m.After(at) {
			return m
		}
	}
	return at
}

// morning is r's wake-up time on a day, staggered by roster position so the
// town does not wake in a single burst.
func (w *World) morning(r *Resident, day int) time.Time {
	idx := 0
	for i, id := range w.Order {
		if id == r.ID {
			idx = i
		}
	}
	return At(day, w.Settings.Scheduling.DayStartHour, 0).Add(time.Duration(idx*7) * time.Minute)
}

// BeginActivation is called when an activation item comes due. If the agent
// cannot actually decide right now (it became busy, or has used its daily
// allowance) the item is re-queued at the next eligible time and false is
// returned. Otherwise the activation is booked against the agent's cooldown
// and allowance, its idle opportunity is cancelled, and true is returned.
func (w *World) BeginActivation(it *Item) bool {
	r := w.Residents[it.Agent]
	if r == nil {
		return false
	}
	if r.ActivateItem == it.ID {
		r.ActivateItem = 0
	}
	sc := w.Settings.Scheduling
	if r.Busy(w.Now) || (Day(w.Now) == r.ActDay && r.ActCount >= sc.MaxActivationsPerDay) {
		it.At = w.eligible(r, w.Now, it.Explicit)
		if existing := w.queue.Get(r.ActivateItem); existing != nil {
			for _, rs := range it.Reasons {
				existing.Reasons = appendReason(existing.Reasons, rs)
			}
			return false
		}
		r.ActivateItem = w.push(it)
		w.count("activations_deferred", 1)
		return false
	}
	if Day(w.Now) != r.ActDay {
		r.ActDay, r.ActCount = Day(w.Now), 0
	}
	r.ActCount++
	r.Activations++
	r.LastActivation = w.Now
	w.queue.Remove(r.IdleItem)
	r.IdleItem = 0
	w.count("activations", 1)
	return true
}

// EndActivation schedules the agent's next default, low-frequency
// opportunity, moved into waking hours. quiet reports that this activation
// was only an idle opportunity and the agent did nothing with it; each such
// activation in a row after the first doubles the idle interval (up to MaxIdleMinutes), so a
// world where nothing is happening does not spend calls on agents
// repeatedly declining to act. Anything else resets the interval.
func (w *World) EndActivation(id sim.AgentID, quiet bool) {
	r := w.Residents[id]
	if r == nil {
		return
	}
	sc := w.Settings.Scheduling
	if quiet {
		r.IdleStreak++
	} else {
		r.IdleStreak = 0
	}
	mins := sc.IdleMinutes
	// The first declined opportunity keeps the base interval; each further
	// one in a row doubles it.
	for i := 1; i < r.IdleStreak && mins < sc.MaxIdleMinutes; i++ {
		mins *= 2
	}
	if sc.MaxIdleMinutes > 0 {
		mins = min(mins, max(sc.MaxIdleMinutes, sc.IdleMinutes))
	}
	w.queue.Remove(r.IdleItem)
	at := w.awake(r, w.Now.Add(time.Duration(mins)*time.Minute))
	r.IdleItem = w.push(&Item{At: at, Kind: KindIdle, Agent: id})
}

// CoolOff defers an agent whose actions keep being refused: its pending
// activation (if any) is dropped and it next gets a chance at its idle time.
// This is an operational safeguard against a confused model spending the
// session's budget on invalid moves, not a judgement of its behaviour.
func (w *World) CoolOff(id sim.AgentID) time.Time {
	r := w.Residents[id]
	w.queue.Remove(r.ActivateItem)
	r.ActivateItem = 0
	w.count("invalid_cooloffs", 1)
	w.tally(id, "invalid_cooloffs", 1)
	if it := w.queue.Get(r.IdleItem); it != nil {
		return it.At
	}
	return time.Time{}
}

// Process resolves a scheduled item that is not an activation. None of
// these call a model: they change the world and produce triggers.
func (w *World) Process(it *Item) Effects {
	switch it.Kind {
	case KindDawn:
		return w.dawn()
	case KindActivityDone:
		return w.finishActivity(it)
	case KindAppointment:
		return w.appointmentDue(it)
	case KindIntention:
		return w.intentionDue(it)
	case KindIdle:
		r := w.Residents[it.Agent]
		if r == nil {
			return Effects{}
		}
		if r.IdleItem == it.ID {
			r.IdleItem = 0
		}
		return Effects{Triggers: []Trigger{{Agent: it.Agent, At: w.Now, Reason: Reason{Kind: "idle", Text: "a quiet moment; nothing in particular prompted it"}}}}
	}
	return Effects{}
}

func (w *World) dawn() Effects {
	var eff Effects
	day := Day(w.Now)
	w.restock()
	text := fmt.Sprintf("Day %d begins. The market has restocked and the workshop has fresh scrap.", day)
	eff.Events = append(eff.Events, sim.Event{Type: sim.EventSystem, Text: text, Visibility: sim.PublicTo()})

	// Completed projects pay their owners: the persistent consequence of
	// having built something.
	if y := w.Settings.Economy.ProjectYield; y > 0 {
		for _, p := range w.Projects {
			if !p.Done() {
				continue
			}
			if r := w.Residents[p.Owner]; r != nil {
				r.Goods.Credits += y
				w.count("yield_credits", y)
				eff.Events = append(eff.Events, sim.Event{
					Type: sim.EventState, Text: fmt.Sprintf("Your completed project %q earned you %d credits overnight.", p.Name, y),
					Visibility: sim.Only(p.Owner),
				})
			}
		}
	}
	for _, o := range w.Offers {
		if o.Status == "open" && !w.Now.Before(o.Expires) {
			o.settle("expired", w.Now)
			eff.Events = append(eff.Events, sim.Event{
				Type: sim.EventState, Text: fmt.Sprintf("Offer %s from %s to %s expired unanswered.", o.ID, w.name(o.From), w.name(o.To)),
				Visibility: sim.Only(o.From, o.To), Importance: 2,
			})
		}
	}
	w.push(&Item{At: w.Now.Add(24 * time.Hour), Kind: KindDawn})
	return eff
}

func (w *World) finishActivity(it *Item) Effects {
	var eff Effects
	r := w.Residents[it.Agent]
	if r == nil || r.Activity == nil || r.Activity.ItemID != it.ID {
		return eff
	}
	act := r.Activity
	r.Activity = nil
	e := w.Settings.Economy
	reason := Reason{Kind: "activity"}
	switch act.Kind {
	case "travel":
		r.Location = act.To
		here := w.Here(act.To)
		eff.Events = append(eff.Events, sim.Event{
			Type: sim.EventAction, Actor: r.ID, Text: "arrived at " + w.placeName(act.To) + ".",
			Visibility: sim.Only(here...),
		})
		reason.Text = "you arrived at " + w.placeName(act.To)
		if er := r.Errand; er != nil && er.To == act.To {
			now := w.Now
			er.Arrived = &now
			if er.For != "" && w.Settings.Grounded() {
				reason.Text += fmt.Sprintf(" (you came to: %s)", er.For)
			}
		}
		w.Look(r.ID)
		if w.Settings.Scheduling.WakeOnArrival {
			for _, other := range here {
				if other != r.ID {
					eff.Triggers = append(eff.Triggers, w.reactTrigger(other, Reason{Kind: "arrival", Text: r.Name + " arrived where you are"}))
				}
			}
		}
	case "work":
		r.Goods.Credits += e.ShiftCredits
		w.count("credits_earned", e.ShiftCredits)
		w.tally(r.ID, "credits_earned", e.ShiftCredits)
		eff.Events = append(eff.Events, sim.Event{
			Type: sim.EventAction, Actor: r.ID, Text: fmt.Sprintf("finished a shift and earned %d credits.", e.ShiftCredits),
			Visibility: sim.Only(r.ID),
		})
		reason.Text = "you finished your shift"
	case "salvage":
		var n int
		fmt.Sscanf(act.Detail, "%d", &n)
		r.Goods.Materials += n
		w.count("materials_salvaged", n)
		w.tally(r.ID, "materials_salvaged", n)
		eff.Events = append(eff.Events, sim.Event{
			Type: sim.EventAction, Actor: r.ID, Text: fmt.Sprintf("finished salvaging and came away with %d materials.", n),
			Visibility: sim.Only(r.ID),
		})
		reason.Text = "you finished salvaging"
	case "build":
		p := w.project(act.Project)
		reason.Text = "you finished a stint of building"
		if p == nil || p.Closed() {
			eff.Events = append(eff.Events, sim.Event{
				Type: sim.EventAction, Actor: r.ID, Text: "stopped building; the project no longer needed the work.",
				Visibility: sim.Only(r.ID),
			})
			break
		}
		hours := e.BuildHours
		if rem := p.NeedsLabor - p.Labor; hours > rem {
			hours = max(rem, 0)
		}
		p.Labor += hours
		if p.Builders == nil {
			p.Builders = map[sim.AgentID]int{}
		}
		p.Builders[r.ID] += hours
		w.count("labor_hours", hours)
		w.tally(r.ID, "labor_hours", hours)
		eff.Events = append(eff.Events, sim.Event{
			Type: sim.EventAction, Actor: r.ID,
			Text:       fmt.Sprintf("put %d hours of work into %q (%s): labor %d/%d%s.", hours, p.Name, p.ID, p.Labor, p.NeedsLabor, w.stillNeeds(p)),
			Visibility: sim.Only(appendUnique(w.Here(r.Location), p.Owner)...), Importance: 2,
		})
		if p.Owner != r.ID {
			eff.Triggers = append(eff.Triggers, w.reactTrigger(p.Owner, Reason{Kind: "project", Text: r.Name + " worked on your project " + p.ID}))
		}
		eff.add(w.maybeComplete(p))
	}
	eff.Triggers = append(eff.Triggers, Trigger{Agent: r.ID, At: w.Now, Reason: reason})
	return eff
}

func (w *World) appointmentDue(it *Item) Effects {
	var eff Effects
	var ap *Appointment
	for _, a := range w.Appointments {
		if a.ID == it.Ref {
			ap = a
		}
	}
	if ap == nil || ap.Status != "pending" {
		return eff
	}
	ap.Status = "due"
	w.count("appointments_due", 1)
	party := append([]sim.AgentID{ap.By}, ap.With...)
	eff.Events = append(eff.Events, sim.Event{
		Type: sim.EventSystem,
		Text: fmt.Sprintf("It is time for the meeting %s arranged at %s with %s: %q (%s).",
			w.name(ap.By), w.placeName(ap.Place), w.nameList(party), ap.What, ap.ID),
		Visibility: sim.Only(party...),
	})
	for _, id := range party {
		eff.Triggers = append(eff.Triggers, Trigger{
			Agent: id, At: w.Now, Explicit: true,
			Reason: Reason{Kind: "appointment", Text: fmt.Sprintf("the meeting at %s (%s) is now: %s", w.placeName(ap.Place), ap.ID, ap.What)},
		})
	}
	return eff
}

func (w *World) intentionDue(it *Item) Effects {
	for _, in := range w.Intentions {
		if in.ID != it.Ref || in.Status != "pending" {
			continue
		}
		in.Status = "due"
		w.count("intentions_due", 1)
		text := fmt.Sprintf("you asked to reconsider now (%s): %s", in.ID, in.Note)
		if w.Settings.Grounded() {
			if since := w.sinceNoted(in.Note, in.Created); len(since) > 0 {
				text += " — since you set it: " + strings.Join(since, "; ")
				w.count("stale_intentions_due", 1)
				w.tally(in.Agent, "stale_intentions_due", 1)
			}
		}
		return Effects{Triggers: []Trigger{{
			Agent: in.Agent, At: w.Now, Explicit: true,
			Reason: Reason{Kind: "intention", Text: text},
		}}}
	}
	return Effects{}
}

// reactTrigger is a trigger caused by someone else, delayed by the reaction
// window so that a burst of events coalesces into one activation, and one
// step deeper in the reply chain than whatever caused it.
func (w *World) reactTrigger(id sim.AgentID, r Reason) Trigger {
	r.Depth = w.chain + 1
	return Trigger{Agent: id, At: w.Now.Add(time.Duration(w.Settings.Scheduling.ReactionMinutes) * time.Minute), Reason: r}
}

// SetChain tells the world the reply-chain depth of the activation whose
// actions are about to be applied.
func (w *World) SetChain(depth int) { w.chain = depth }

func (w *World) maybeComplete(p *Project) Effects {
	var eff Effects
	if p.Closed() || !p.satisfied() {
		return eff
	}
	now := w.Now
	p.Completed = &now
	w.count("projects_completed", 1)
	if l := w.Location(p.Location); l != nil {
		l.Features = append(l.Features, fmt.Sprintf("%s — %s's project, completed %s", p.Name, w.name(p.Owner), Stamp(now)))
	}
	eff.Events = append(eff.Events, sim.Event{
		Type: sim.EventState,
		Text: fmt.Sprintf("News: %s's project %q at %s is complete. It now stands there for good.",
			w.name(p.Owner), p.Name, w.placeName(p.Location)),
		Visibility: sim.PublicTo(), Importance: 2,
	})
	involved := []sim.AgentID{p.Owner}
	for id := range p.Contributors {
		involved = appendUnique(involved, id)
	}
	for id := range p.Builders {
		involved = appendUnique(involved, id)
	}
	for _, id := range w.Order { // deterministic order
		for _, x := range involved {
			if x == id {
				eff.Triggers = append(eff.Triggers, w.reactTrigger(id, Reason{Kind: "project", Text: fmt.Sprintf("project %q (%s) was completed", p.Name, p.ID)}))
			}
		}
	}
	return eff
}

// --- validation and application ----------------------------------------------

// Validate checks a proposed action against the objective state of the world.
// It never changes anything; a refusal explains itself so the agent can
// correct course.
func (w *World) Validate(id sim.AgentID, a sim.Action) error {
	_, err := w.plan(id, a)
	return err
}

// Apply executes a validated action.
func (w *World) Apply(id sim.AgentID, a sim.Action) (Effects, error) {
	run, err := w.plan(id, a)
	if err != nil {
		return Effects{}, err
	}
	return run(), nil
}

// plan validates an action and, if it is legal, returns the closure that
// performs it. Keeping validation and execution in one place means the two
// can never disagree about what an argument meant.
func (w *World) plan(id sim.AgentID, a sim.Action) (func() Effects, error) {
	r := w.Residents[id]
	if r == nil {
		return nil, refuse("bad_args", "you are not a resident of this world")
	}
	if r.Location == "" && a.Kind != sim.ActionWhisper && a.Kind != sim.ActionPass && a.Kind != sim.ActionIntend {
		return nil, refuse("travelling", "you are on your way somewhere and can only send a private note right now")
	}
	switch a.Kind {
	case sim.ActionPass:
		return func() Effects {
			return Effects{Events: []sim.Event{{Type: sim.EventPass, Actor: id, Text: "chose to do nothing for now", Visibility: sim.Only()}}}
		}, nil
	case sim.ActionSay:
		return w.planSay(r, a)
	case sim.ActionWhisper:
		return w.planWhisper(r, a)
	case sim.ActionIntend:
		return w.planIntend(r, a)
	case sim.ActionDo:
		return w.planDo(r, a)
	}
	return nil, refuse("bad_args", "unknown action kind %q", a.Kind)
}

func (w *World) planSay(r *Resident, a sim.Action) (func() Effects, error) {
	text := strings.TrimSpace(a.Content)
	if text == "" {
		return nil, refuse("bad_args", "say needs something to say")
	}
	here := w.Here(r.Location)
	if len(here) < 2 {
		return nil, refuse("nobody_here", "nobody else is at %s to hear you; go where people are, or send a private note (whisper)", w.placeName(r.Location))
	}
	return func() Effects {
		eff := Effects{Events: []sim.Event{{Type: sim.EventSay, Actor: r.ID, Text: text, Visibility: sim.Only(here...)}}}
		var listeners []sim.AgentID
		for _, x := range here {
			if x != r.ID {
				listeners = append(listeners, x)
			}
		}
		w.count("speech", 1)
		heard := w.remember(r, text, listeners)
		for _, l := range listeners {
			if heard[l] && w.Settings.Grounded() {
				// They already heard the same thing from r: it is delivered,
				// but it is not news, so it does not earn a fresh reaction.
				w.count("repeat_triggers_suppressed", 1)
				w.tally(l, "repeat_triggers_suppressed", 1)
				continue
			}
			wake := false
			switch w.Settings.Scheduling.WakeOnSpeech {
			case "all":
				wake = true
			case "mention":
				wake = len(listeners) == 1 || mentions(text, l, w.name(l))
			}
			if wake {
				eff.Triggers = append(eff.Triggers, w.reactTrigger(l, Reason{Kind: "speech", Text: fmt.Sprintf("%s spoke to you at %s", r.Name, w.placeName(r.Location))}))
			}
		}
		return eff
	}, nil
}

func (w *World) planWhisper(r *Resident, a sim.Action) (func() Effects, error) {
	text := strings.TrimSpace(a.Content)
	if text == "" {
		return nil, refuse("bad_args", "a private note needs a message")
	}
	var to []sim.AgentID
	for _, t := range a.To {
		target := w.resident(string(t))
		if target == nil {
			return nil, refuse("no_such_person", "nobody called %q lives here", t)
		}
		if target.ID != r.ID {
			to = appendUnique(to, target.ID)
		}
	}
	if len(to) == 0 {
		return nil, refuse("bad_args", "a private note needs a recipient other than yourself")
	}
	return func() Effects {
		eff := Effects{Events: []sim.Event{{
			Type: sim.EventWhisper, Actor: r.ID, Targets: to, Text: text,
			Visibility: sim.Only(append([]sim.AgentID{r.ID}, to...)...),
		}}}
		w.count("messages", 1)
		heard := w.remember(r, text, to)
		for _, t := range to {
			if heard[t] && w.Settings.Grounded() {
				w.count("repeat_triggers_suppressed", 1)
				w.tally(t, "repeat_triggers_suppressed", 1)
				continue
			}
			eff.Triggers = append(eff.Triggers, w.reactTrigger(t, Reason{Kind: "message", Text: "a private note from " + r.Name}))
		}
		return eff
	}, nil
}

func (w *World) planIntend(r *Resident, a sim.Action) (func() Effects, error) {
	if ref, ok := a.ArgString("cancel"); ok && strings.TrimSpace(ref) != "" {
		ref = strings.TrimSpace(ref)
		var in *Intention
		for _, x := range w.Intentions {
			if strings.EqualFold(x.ID, ref) && x.Agent == r.ID && x.Status == "pending" {
				in = x
			}
		}
		if in == nil {
			return nil, refuse("intention_not_found", "you have no pending intention %q to cancel", ref)
		}
		return func() Effects {
			in.Status = "cancelled"
			w.queue.Remove(in.ItemID)
			w.count("intentions_cancelled", 1)
			return Effects{Events: []sim.Event{{Type: sim.EventState, Actor: r.ID,
				Text: fmt.Sprintf("You dropped your intention %s (%s).", in.ID, in.Note), Visibility: sim.Only(r.ID)}}}
		}, nil
	}
	when, _ := a.ArgString("when")
	note, _ := a.ArgString("note")
	note = clip(strings.TrimSpace(note), 200)
	if note == "" {
		return nil, refuse("bad_args", `an intention needs a "note" saying what you mean to do or reconsider`)
	}
	at, err := w.within(when, w.Settings.Scheduling.MinIntentionMinutes)
	if err != nil {
		return nil, err
	}
	// In grounded mode a new intention replaces the pending one it
	// revises — named explicitly with "replaces", or recognisably the same
	// note — instead of piling up a second wakeup for the same thing.
	var old *Intention
	if w.Settings.Grounded() {
		ref, _ := a.ArgString("replaces")
		print := textsim.Of(note)
		for _, x := range w.Intentions {
			if x.Agent != r.ID || x.Status != "pending" {
				continue
			}
			if strings.EqualFold(x.ID, strings.TrimSpace(ref)) || (old == nil && textsim.Same(textsim.Of(x.Note), print, intentionOverlap)) {
				old = x
			}
		}
	}
	pending := 0
	for _, x := range w.Intentions {
		if x.Agent == r.ID && x.Status == "pending" && x != old {
			pending++
		}
	}
	if max := w.Settings.Scheduling.MaxPendingIntentions; pending >= max {
		return nil, refuse("intention_limit", "you already have %d pending intentions (the most you can keep); cancel one first", pending)
	}
	return func() Effects {
		in := &Intention{ID: w.nextID("i"), Agent: r.ID, At: at, Note: note, Created: w.Now, Status: "pending"}
		in.ItemID = w.push(&Item{At: at, Kind: KindIntention, Agent: r.ID, Ref: in.ID})
		w.Intentions = append(w.Intentions, in)
		w.count("intentions_set", 1)
		text := fmt.Sprintf("You mean to come back to this at %s: %s (%s).", Stamp(at), note, in.ID)
		if old != nil {
			old.Status = "superseded"
			w.queue.Remove(old.ItemID)
			in.Supersedes = old.ID
			w.count("intentions_superseded", 1)
			w.tally(r.ID, "intentions_superseded", 1)
			text = fmt.Sprintf("You mean to come back to this at %s: %s (%s, replacing %s, which was set for %s).", Stamp(at), note, in.ID, old.ID, Stamp(old.At))
		}
		return Effects{Events: []sim.Event{{Type: sim.EventState, Actor: r.ID, Text: text, Visibility: sim.Only(r.ID)}}}
	}, nil
}

// within parses a "when" and enforces the scheduling horizon: too-soon
// requests are moved out to minMinutes (so an agent cannot schedule a stream
// of immediate wakeups), too-distant ones are refused.
func (w *World) within(when string, minMinutes int) (time.Time, error) {
	sc := w.Settings.Scheduling
	at, err := ParseWhen(when, w.Now)
	if err != nil {
		return time.Time{}, err
	}
	if min := w.Now.Add(time.Duration(minMinutes) * time.Minute); at.Before(min) {
		at = min
	}
	if max := w.Now.Add(time.Duration(sc.MaxIntentionHours) * time.Hour); at.After(max) {
		return time.Time{}, refuse("too_far_ahead", "%s is too far ahead; you can plan at most %d hours out", Stamp(at), sc.MaxIntentionHours)
	}
	return at, nil
}

func (w *World) planDo(r *Resident, a sim.Action) (func() Effects, error) {
	loc := w.Location(r.Location)
	e := w.Settings.Economy
	switch a.Verb {
	case "wait", "rest", "stay", "idle", "relax":
		return w.plan(r.ID, sim.Action{Kind: sim.ActionPass})

	case "cancel_meeting", "cancel_appointment":
		ref := firstArg(a, "meeting", "appointment", "id")
		var ap *Appointment
		for _, x := range w.Appointments {
			if strings.EqualFold(x.ID, ref) && x.Status == "pending" {
				ap = x
			}
		}
		if ap == nil {
			return nil, refuse("meeting_not_found", "there is no pending meeting %q", ref)
		}
		if ap.By != r.ID {
			return nil, refuse("not_your_meeting", "only %s, who arranged %s, can cancel it (you can simply not go)", w.name(ap.By), ap.ID)
		}
		return func() Effects {
			ap.Status = "cancelled"
			w.queue.Remove(ap.ItemID)
			w.count("appointments_cancelled", 1)
			w.tally(r.ID, "appointments_cancelled", 1)
			party := append([]sim.AgentID{r.ID}, ap.With...)
			eff := Effects{Events: []sim.Event{{Type: sim.EventAction, Actor: r.ID,
				Text: fmt.Sprintf("called off the meeting %s (%s at %s).", ap.ID, Stamp(ap.At), w.placeName(ap.Place)), Visibility: sim.Only(party...)}}}
			for _, x := range ap.With {
				eff.Triggers = append(eff.Triggers, w.reactTrigger(x, Reason{Kind: "invite", Text: fmt.Sprintf("%s called off the meeting %s", r.Name, ap.ID)}))
			}
			return eff
		}, nil

	case "withdraw", "withdraw_offer", "retract":
		ref := firstArg(a, "offer", "id")
		o := w.offer(ref)
		if o == nil || o.From != r.ID || o.Status != "open" {
			return nil, refuse("offer_not_found", "you have no open offer %q", ref)
		}
		return func() Effects {
			o.settle("withdrawn", w.Now)
			w.count("offers_withdrawn", 1)
			return Effects{Events: []sim.Event{{Type: sim.EventAction, Actor: r.ID,
				Text: fmt.Sprintf("withdrew offer %s.", o.ID), Visibility: sim.Only(r.ID, o.To), Importance: 2}}}
		}, nil

	case "go", "move", "walk", "travel", "visit":
		ref := firstArg(a, "to", "place", "destination", "location", "where")
		dest := w.FindPlace(r.ID, ref)
		if dest == nil {
			return nil, refuse("no_such_place", "there is no place called %q (places: %s)", ref, w.placeIDs())
		}
		if dest.ID == r.Location {
			return nil, refuse("already_there", "you are already at %s", dest.Name)
		}
		purpose := clip(firstArg(a, "for", "purpose", "why", "to_do"), 160)
		return func() Effects {
			here := w.Here(r.Location)
			mins := time.Duration(w.Settings.Scheduling.TravelMinutes) * time.Minute
			ev := sim.Event{Type: sim.EventAction, Actor: r.ID, Text: "set off for " + dest.Name + ".", Visibility: sim.Only(here...)}
			r.Errand = &Errand{From: r.Location, To: dest.ID, For: purpose, Left: w.Now}
			if purpose != "" {
				w.count("trips_with_purpose", 1)
			}
			r.Location = ""
			w.startActivity(r, &Activity{Kind: "travel", Detail: "walking to " + dest.Name, End: w.Now.Add(mins), To: dest.ID})
			w.count("trips", 1)
			return Effects{Events: []sim.Event{ev}}
		}, nil

	case "work":
		if loc.Kind != PlaceMarket {
			return nil, refuse("wrong_place", "paid shifts are only available at the market")
		}
		if w.Stock.Shifts[loc.ID] <= 0 {
			return nil, refuse("sold_out", "every shift at %s is taken for today; more open tomorrow", loc.Name)
		}
		return func() Effects {
			w.Stock.Shifts[loc.ID]--
			w.startActivity(r, &Activity{Kind: "work", Detail: "working a shift at " + loc.Name, End: w.Now.Add(time.Duration(e.ShiftHours) * time.Hour)})
			w.count("shifts", 1)
			return Effects{Events: []sim.Event{w.local(r, fmt.Sprintf("started a %d-hour shift at the stalls.", e.ShiftHours))}}
		}, nil

	case "buy":
		if loc.Kind != PlaceMarket {
			return nil, refuse("wrong_place", "materials are sold at the market")
		}
		n := argInt(a, 1, "materials", "amount", "quantity", "count", "n")
		if n < 1 {
			return nil, refuse("bad_args", "buy at least 1 material")
		}
		if stock := w.Stock.Materials[loc.ID]; n > stock {
			return nil, refuse("sold_out", "the market only has %d materials left today", stock)
		}
		cost := n * e.MaterialPrice
		if r.Goods.Credits < cost {
			return nil, refuse("cannot_afford", "%d materials cost %d credits and you have %d", n, cost, r.Goods.Credits)
		}
		return func() Effects {
			w.Stock.Materials[loc.ID] -= n
			r.Goods.Credits -= cost
			r.Goods.Materials += n
			w.count("materials_bought", n)
			w.tally(r.ID, "materials_bought", n)
			return Effects{Events: []sim.Event{w.local(r, fmt.Sprintf("bought %d materials for %d credits.", n, cost))}}
		}, nil

	case "salvage", "gather", "scavenge":
		if loc.Kind != PlaceWorkshop {
			return nil, refuse("wrong_place", "there is only scrap to salvage at the workshop")
		}
		scrap := w.Stock.Scrap[loc.ID]
		if scrap <= 0 {
			return nil, refuse("sold_out", "the scrap pile at %s is picked clean for today", loc.Name)
		}
		return func() Effects {
			n := min(e.SalvageMaterials, scrap)
			w.Stock.Scrap[loc.ID] -= n
			w.startActivity(r, &Activity{Kind: "salvage", Detail: fmt.Sprintf("%d", n), End: w.Now.Add(time.Duration(e.SalvageHours) * time.Hour)})
			return Effects{Events: []sim.Event{w.local(r, "started sorting through the scrap pile.")}}
		}, nil

	case "give":
		target := w.resident(firstArg(a, "to", "recipient", "target", "who"))
		if target == nil || target.ID == r.ID {
			return nil, refuse("bad_args", `give needs "to" naming someone else`)
		}
		if target.Location != r.Location {
			return nil, refuse("not_here", "%s is not here; you can only hand things to someone in the same place", target.Name)
		}
		g := Goods{Credits: argInt(a, 0, "credits"), Materials: argInt(a, 0, "materials")}
		if g.Credits < 0 || g.Materials < 0 || g.IsZero() {
			return nil, refuse("bad_args", `give needs a positive "credits" and/or "materials"`)
		}
		if !r.Goods.Covers(g) {
			return nil, refuse("cannot_afford", "you only have %s", r.Goods)
		}
		return func() Effects {
			r.Goods = r.Goods.Minus(g)
			target.Goods = target.Goods.Plus(g)
			w.count("gifts", 1)
			w.transfer(r.ID, target.ID, g, "given")
			w.book(Transfer{From: r.ID, To: target.ID, Goods: g, Kind: "gift"})
			text := fmt.Sprintf("gave %s %s.", target.Name, goodsPhrase(g))
			if w.Settings.Grounded() {
				text = fmt.Sprintf("gave %s %s as a gift — nothing was asked or given in return.", target.Name, goodsPhrase(g))
			}
			ev := w.local(r, text)
			ev.Importance = 2
			return Effects{
				Events:   []sim.Event{ev},
				Triggers: []Trigger{w.reactTrigger(target.ID, Reason{Kind: "gift", Text: r.Name + " gave you " + goodsPhrase(g)})},
			}
		}, nil

	case "offer", "propose_trade", "trade":
		target := w.resident(firstArg(a, "to", "recipient", "target", "who"))
		if target == nil || target.ID == r.ID {
			return nil, refuse("bad_args", `offer needs "to" naming someone else`)
		}
		give := Goods{Credits: argInt(a, 0, "give_credits"), Materials: argInt(a, 0, "give_materials")}
		want := Goods{Credits: argInt(a, 0, "want_credits"), Materials: argInt(a, 0, "want_materials")}
		if give.Credits < 0 || give.Materials < 0 || want.Credits < 0 || want.Materials < 0 || (give.IsZero() && want.IsZero()) {
			return nil, refuse("bad_args", "an offer needs non-negative give_credits/give_materials and want_credits/want_materials, not all zero")
		}
		if !r.Goods.Covers(give) {
			return nil, refuse("cannot_afford", "you cannot offer %s; you only have %s", goodsPhrase(give), r.Goods)
		}
		open := 0
		for _, o := range w.Offers {
			if o.From == r.ID && o.Status == "open" {
				open++
			}
		}
		if open >= 3 {
			return nil, refuse("offer_limit", "you already have 3 open offers; wait for answers first")
		}
		note := clip(firstArg(a, "note", "message", "why"), 160)
		return func() Effects {
			repeats := w.sameTerms(r.ID, target.ID, give, want)
			for _, x := range w.Offers {
				if x.From == r.ID && x.To == target.ID && x.Give == give && x.Want == want && x.Status == "declined" {
					w.count("reoffers_after_decline", 1)
					w.tally(r.ID, "reoffers_after_decline", 1)
					break
				}
			}
			if w.citesClosedProject(note) {
				w.count("offers_citing_closed_project", 1)
				w.tally(r.ID, "offers_citing_closed_project", 1)
			}
			o := &Offer{ID: w.nextID("o"), From: r.ID, To: target.ID, Give: give, Want: want, Note: note,
				Created: w.Now, Expires: w.Now.Add(time.Duration(e.OfferHours) * time.Hour), Status: "open"}
			w.Offers = append(w.Offers, o)
			w.count("offers", 1)
			text := fmt.Sprintf("offered %s %s (offer %s)%s.", target.Name, offerPhrase(o), o.ID, quoteNote(note))
			if w.Settings.Grounded() {
				text = fmt.Sprintf("offered %s %s (offer %s): if %s accepts, %s hands over %s and %s hands over %s%s.",
					target.Name, offerPhrase(o), o.ID, target.Name, r.Name, goodsPhrase(give), target.Name, goodsPhrase(want), quoteNote(note))
			}
			eff := Effects{Events: []sim.Event{{Type: sim.EventAction, Actor: r.ID, Text: text, Visibility: sim.Only(r.ID, target.ID), Importance: 2}}}
			if repeats > 0 {
				w.count("repeated_offers", 1)
				w.tally(r.ID, "repeated_offers", 1)
			}
			if repeats > 0 && w.Settings.Grounded() {
				// The same terms were already put to them and answered (or
				// are still waiting). The new offer stands and they will see
				// it, but it does not wake them as if it were news.
				w.count("repeat_triggers_suppressed", 1)
				w.tally(target.ID, "repeat_triggers_suppressed", 1)
				return eff
			}
			eff.Triggers = []Trigger{w.reactTrigger(target.ID, Reason{Kind: "offer", Text: fmt.Sprintf("%s made you an offer (%s)", r.Name, o.ID)})}
			return eff
		}, nil

	case "accept", "decline", "reject":
		ref := firstArg(a, "offer", "id", "offer_id")
		o := w.offer(ref)
		if o != nil && o.From == r.ID {
			return nil, refuse("own_offer", "%s is your own offer to %s; only %s can accept or decline it (you can withdraw it)", o.ID, w.name(o.To), w.name(o.To))
		}
		if o == nil || o.To != r.ID {
			return nil, refuse("offer_not_found", "there is no offer %q made to you", ref)
		}
		if o.Status != "open" || !w.Now.Before(o.Expires) {
			status := o.Status
			if status == "open" {
				status = "expired"
			}
			when := ""
			if o.Resolved != nil {
				when = " at " + Stamp(*o.Resolved)
			}
			return nil, refuse("offer_closed", "offer %s is no longer open (it was %s%s)", o.ID, status, when)
		}
		from := w.Residents[o.From]
		if a.Verb != "accept" {
			return func() Effects {
				o.settle("declined", w.Now)
				w.count("offers_declined", 1)
				// Declines already show under the offerer's settled offers;
				// this only marks the next decision for the strategy metric.
				from.AfterSetback = "offer"
				return Effects{
					Events:   []sim.Event{{Type: sim.EventAction, Actor: r.ID, Text: fmt.Sprintf("declined %s's offer %s.", from.Name, o.ID), Visibility: sim.Only(r.ID, from.ID), Importance: 2}},
					Triggers: []Trigger{w.reactTrigger(from.ID, Reason{Kind: "offer", Text: fmt.Sprintf("%s declined your offer %s", r.Name, o.ID)})},
				}
			}, nil
		}
		if !from.Goods.Covers(o.Give) {
			return nil, refuse("offer_unfunded", "%s no longer has %s, so the offer cannot go through", from.Name, goodsPhrase(o.Give))
		}
		if !r.Goods.Covers(o.Want) {
			return nil, refuse("cannot_afford", "you do not have %s to complete the trade (you have %s)", goodsPhrase(o.Want), r.Goods)
		}
		return func() Effects {
			from.Goods = from.Goods.Minus(o.Give).Plus(o.Want)
			r.Goods = r.Goods.Minus(o.Want).Plus(o.Give)
			o.settle("accepted", w.Now)
			w.count("trades", 1)
			w.transfer(from.ID, r.ID, o.Give, "traded")
			w.transfer(r.ID, from.ID, o.Want, "traded")
			if !o.Give.IsZero() {
				w.book(Transfer{From: from.ID, To: r.ID, Goods: o.Give, Kind: "trade", Ref: o.ID})
			}
			if !o.Want.IsZero() {
				w.book(Transfer{From: r.ID, To: from.ID, Goods: o.Want, Kind: "trade", Ref: o.ID})
			}
			text := fmt.Sprintf("accepted %s's offer %s: %s.", from.Name, o.ID, offerPhrase(o))
			if w.Settings.Grounded() {
				text = fmt.Sprintf("accepted %s's offer %s — the trade is done: %s.", from.Name, o.ID, tradePhrase(o, from.Name, r.Name))
				switch {
				case o.Want.IsZero():
					text = fmt.Sprintf("accepted %s's offer %s — it is done: %s handed over %s and %s gave nothing back (the offer asked for nothing in return).", from.Name, o.ID, from.Name, goodsPhrase(o.Give), r.Name)
				case o.Give.IsZero():
					text = fmt.Sprintf("accepted %s's offer %s — it is done: %s handed over %s and got nothing back (the offer gave nothing in return).", from.Name, o.ID, r.Name, goodsPhrase(o.Want))
				}
			}
			return Effects{
				Events:   []sim.Event{{Type: sim.EventAction, Actor: r.ID, Text: text, Visibility: sim.Only(r.ID, from.ID), Importance: 2}},
				Triggers: []Trigger{w.reactTrigger(from.ID, Reason{Kind: "offer", Text: fmt.Sprintf("%s accepted your offer %s", r.Name, o.ID)})},
			}
		}, nil

	case "start_project", "propose", "project":
		name := clip(firstArg(a, "name", "title", "project"), 60)
		if name == "" {
			return nil, refuse("bad_args", `start_project needs a "name"`)
		}
		if w.project(name) != nil {
			return nil, refuse("duplicate_project", "there is already a project called %q", name)
		}
		place := loc
		if ref := firstArg(a, "place", "location", "where", "at"); ref != "" {
			if place = w.FindPlace(r.ID, ref); place == nil {
				return nil, refuse("no_such_place", "there is no place called %q (places: %s)", ref, w.placeIDs())
			}
		}
		if place.Kind == PlaceResidence && place.Owner != r.ID {
			return nil, refuse("not_your_home", "you cannot build in someone else's home")
		}
		needs := Goods{Credits: argInt(a, 0, "credits"), Materials: argInt(a, 0, "materials")}
		labor := argInt(a, 0, "labor", "hours")
		switch {
		case needs.Credits < 0 || needs.Materials < 0 || labor < 0 || (needs.IsZero() && labor == 0):
			return nil, refuse("bad_args", "a project must need some credits, materials and/or labor hours")
		case needs.Credits > e.MaxProjectCredits || needs.Materials > e.MaxProjectMaterials || labor > e.MaxProjectLabor:
			return nil, refuse("bad_args", "a project can need at most %d credits, %d materials and %d labor hours", e.MaxProjectCredits, e.MaxProjectMaterials, e.MaxProjectLabor)
		}
		open := 0
		for _, p := range w.Projects {
			if p.Owner == r.ID && !p.Closed() {
				open++
			}
		}
		if open >= e.MaxOpenProjects {
			if w.Settings.Grounded() {
				return nil, refuse("project_limit", "you already have %d unfinished projects (the most you can have); finish or abandon one first", open)
			}
			return nil, refuse("project_limit", "you already have %d unfinished projects", open)
		}
		desc := clip(firstArg(a, "description", "purpose", "text"), 200)
		return func() Effects {
			p := &Project{ID: w.nextID("p"), Owner: r.ID, Name: name, Description: desc, Location: place.ID,
				Needs: needs, NeedsLabor: labor, Created: w.Now}
			w.Projects = append(w.Projects, p)
			w.count("projects_started", 1)
			w.tally(r.ID, "projects_started", 1)
			ev := w.local(r, fmt.Sprintf("started a project, %q (%s), at %s; it needs %s and %d labor hours.",
				p.Name, p.ID, place.Name, goodsPhrase(needs), labor))
			ev.Importance = 2
			return Effects{Events: []sim.Event{ev}}
		}, nil

	case "contribute", "fund", "donate":
		ref := firstArg(a, "project", "id", "to")
		p := w.project(ref)
		if p == nil {
			return nil, refuse("project_not_found", "there is no project %q", ref)
		}
		if err := closedRefusal(p); err != nil {
			return nil, err
		}
		g := Goods{Credits: argInt(a, 0, "credits"), Materials: argInt(a, 0, "materials")}
		if g.Credits < 0 || g.Materials < 0 || g.IsZero() {
			return nil, refuse("bad_args", `contribute needs a positive "credits" and/or "materials"`)
		}
		// Cap at what the project still needs before checking affordability,
		// so offering "50" toward an 8-credit shortfall moves 8.
		g.Credits = min(g.Credits, max(p.Needs.Credits-p.Raised.Credits, 0))
		g.Materials = min(g.Materials, max(p.Needs.Materials-p.Raised.Materials, 0))
		if g.IsZero() {
			return nil, refuse("nothing_needed", "%q needs no more of that (it still needs %d labor hours: build it)", p.Name, max(p.NeedsLabor-p.Labor, 0))
		}
		if !r.Goods.Covers(g) {
			return nil, refuse("cannot_afford", "that would take %s and you only have %s", goodsPhrase(g), r.Goods)
		}
		return func() Effects {
			r.Goods = r.Goods.Minus(g)
			p.Raised = p.Raised.Plus(g)
			if p.Contributors == nil {
				p.Contributors = map[sim.AgentID]Goods{}
			}
			p.Contributors[r.ID] = p.Contributors[r.ID].Plus(g)
			w.count("contributions", 1)
			w.count("credits_contributed", g.Credits)
			w.count("materials_contributed", g.Materials)
			w.tally(r.ID, "credits_contributed", g.Credits)
			w.tally(r.ID, "materials_contributed", g.Materials)
			w.book(Transfer{From: r.ID, Goods: g, Kind: "contribution", Ref: p.ID})
			eff := Effects{Events: []sim.Event{{Type: sim.EventAction, Actor: r.ID,
				Text:       fmt.Sprintf("contributed %s to %q (%s)%s.", goodsPhrase(g), p.Name, p.ID, w.stillNeeds(p)),
				Visibility: sim.Only(appendUnique(w.Here(r.Location), p.Owner)...), Importance: 2}}}
			if p.Owner != r.ID {
				eff.Triggers = append(eff.Triggers, w.reactTrigger(p.Owner, Reason{Kind: "project", Text: fmt.Sprintf("%s contributed to your project %s", r.Name, p.ID)}))
			}
			eff.add(w.maybeComplete(p))
			return eff
		}, nil

	case "build", "work_on":
		ref := firstArg(a, "project", "id")
		p := w.project(ref)
		if p == nil {
			return nil, refuse("project_not_found", "there is no project %q", ref)
		}
		if err := closedRefusal(p); err != nil {
			return nil, err
		}
		if p.Location != r.Location {
			return nil, refuse("project_elsewhere", "%q is at %s; go there to work on it", p.Name, w.placeName(p.Location))
		}
		if p.Labor+w.laborUnderway(p.ID) >= p.NeedsLabor {
			if p.Labor >= p.NeedsLabor {
				return nil, refuse("no_labor_needed", "%q needs no more labor, only %s", p.Name, goodsPhrase(p.Needs.Minus(p.Raised)))
			}
			return nil, refuse("no_labor_needed", "the labor %q still needs is already being done by someone else", p.Name)
		}
		return func() Effects {
			w.startActivity(r, &Activity{Kind: "build", Detail: "building " + p.Name, Project: p.ID, End: w.Now.Add(time.Duration(e.BuildHours) * time.Hour)})
			return Effects{Events: []sim.Event{w.local(r, fmt.Sprintf("started working on %q (%s).", p.Name, p.ID))}}
		}, nil

	case "abandon", "abandon_project", "drop_project":
		if !w.Settings.Grounded() {
			break
		}
		ref := firstArg(a, "project", "id", "name")
		p := w.project(ref)
		if p == nil {
			return nil, refuse("project_not_found", "there is no project %q", ref)
		}
		if p.Owner != r.ID {
			return nil, refuse("not_your_project", "%q is %s's project; only its owner can abandon it", p.Name, w.name(p.Owner))
		}
		if err := closedRefusal(p); err != nil {
			return nil, err
		}
		return func() Effects {
			now := w.Now
			p.Abandoned = &now
			w.count("projects_abandoned", 1)
			w.tally(r.ID, "projects_abandoned", 1)
			eff := Effects{}
			for _, id := range w.Order { // deterministic order
				g, ok := p.Contributors[id]
				if !ok || g.IsZero() {
					continue
				}
				w.Residents[id].Goods = w.Residents[id].Goods.Plus(g)
				w.book(Transfer{From: "", To: id, Goods: g, Kind: "refund", Ref: p.ID})
				if id != r.ID {
					eff.Events = append(eff.Events, sim.Event{Type: sim.EventState, Importance: 2,
						Text:       fmt.Sprintf("%s abandoned the project %q (%s); the %s you contributed came back to you.", r.Name, p.Name, p.ID, goodsPhrase(g)),
						Visibility: sim.Only(id)})
					eff.Triggers = append(eff.Triggers, w.reactTrigger(id, Reason{Kind: "project", Text: fmt.Sprintf("%s abandoned project %s", r.Name, p.ID)}))
				}
			}
			refund := ""
			if g, ok := p.Contributors[r.ID]; ok && !g.IsZero() {
				refund = fmt.Sprintf("; your own %s came back to you", goodsPhrase(g))
			}
			eff.Events = append([]sim.Event{{Type: sim.EventAction, Actor: r.ID, Importance: 2,
				Text:       fmt.Sprintf("abandoned the project %q (%s) unfinished%s.", p.Name, p.ID, refund),
				Visibility: sim.Only(appendUnique(w.Here(r.Location), r.ID)...)}}, eff.Events...)
			return eff
		}, nil

	case "post", "notice":
		if loc.Kind != PlaceSquare {
			return nil, refuse("wrong_place", "the notice board is at the square")
		}
		text := clip(firstArg(a, "text", "message", "content", "notice"), 280)
		if text == "" {
			return nil, refuse("bad_args", `post needs "text"`)
		}
		return func() Effects {
			n := &Notice{ID: w.nextID("n"), By: r.ID, Place: loc.ID, Text: text, Posted: w.Now}
			w.Notices = append(w.Notices, n)
			w.trimNotices(loc.ID)
			w.count("notices", 1)
			return Effects{Events: []sim.Event{w.local(r, fmt.Sprintf("pinned a notice to the board: %q", text))}}
		}, nil

	case "arrange", "meet", "invite":
		var with []sim.AgentID
		for _, ref := range listArg(a, "with", "who", "to", "invite") {
			t := w.resident(ref)
			if t == nil {
				return nil, refuse("no_such_person", "nobody called %q lives here", ref)
			}
			if t.ID != r.ID {
				with = appendUnique(with, t.ID)
			}
		}
		if len(with) == 0 {
			return nil, refuse("bad_args", `arrange needs "with" naming at least one other person`)
		}
		ref := firstArg(a, "place", "where", "location", "at_place")
		place := loc
		if ref != "" {
			if place = w.FindPlace(r.ID, ref); place == nil {
				return nil, refuse("no_such_place", "there is no place called %q (places: %s)", ref, w.placeIDs())
			}
		}
		at, err := w.within(firstArg(a, "when", "time", "at"), w.Settings.Scheduling.MinAppointmentMinutes)
		if err != nil {
			return nil, err
		}
		pending := 0
		for _, ap := range w.Appointments {
			if ap.Status != "pending" {
				continue
			}
			if ap.By == r.ID {
				pending++
			}
			party := append([]sim.AgentID{ap.By}, ap.With...)
			if slices.Contains(party, r.ID) {
				for _, x := range with {
					if slices.Contains(party, x) {
						return nil, refuse("meeting_exists", "you already have a meeting with %s (%s, %s at %s)", w.name(x), ap.ID, Stamp(ap.At), w.placeName(ap.Place))
					}
				}
			}
		}
		if pending >= w.Settings.Scheduling.MaxPendingAppointments {
			return nil, refuse("meeting_limit", "you already have %d meetings arranged", pending)
		}
		what := clip(firstArg(a, "what", "about", "purpose", "note"), 160)
		return func() Effects {
			ap := &Appointment{ID: w.nextID("a"), By: r.ID, With: with, Place: place.ID, At: at, What: what, Status: "pending"}
			ap.ItemID = w.push(&Item{At: at, Kind: KindAppointment, Origin: r.ID, Audience: append([]sim.AgentID{r.ID}, with...), Ref: ap.ID})
			w.Appointments = append(w.Appointments, ap)
			w.count("appointments", 1)
			eff := Effects{Events: []sim.Event{{Type: sim.EventAction, Actor: r.ID,
				Text: fmt.Sprintf("arranged a meeting with %s at %s on %s: %q (%s).",
					w.nameList(with), place.Name, Stamp(at), what, ap.ID),
				Visibility: sim.Only(append([]sim.AgentID{r.ID}, with...)...), Importance: 2}}}
			for _, t := range with {
				eff.Triggers = append(eff.Triggers, w.reactTrigger(t, Reason{Kind: "invite", Text: fmt.Sprintf("%s arranged a meeting with you (%s)", r.Name, ap.ID)}))
			}
			return eff
		}, nil
	}
	return nil, refuse("unknown_verb", "%q is not something you can do here (available: %s)", a.Verb, strings.Join(w.verbs(r.ID), ", "))
}

// closedRefusal explains why a completed or abandoned project cannot be
// worked on or funded; nil while it is open.
func closedRefusal(p *Project) error {
	switch {
	case p.Done():
		return refuse("project_complete", "%q is already complete", p.Name)
	case p.Abandoned != nil:
		return refuse("project_abandoned", "%q was abandoned on %s", p.Name, Stamp(*p.Abandoned))
	}
	return nil
}

// intentionOverlap is how alike two intention notes must be for the newer
// to count as a revision of the older.
const intentionOverlap = 0.6

// book appends a completed transfer to the ledger.
func (w *World) book(t Transfer) {
	t.Time = w.Now
	w.Ledger = append(w.Ledger, t)
	if len(w.Ledger) > maxLedger {
		w.Ledger = w.Ledger[len(w.Ledger)-maxLedger:]
	}
}

// remember records what r just said to listeners and reports which of them
// had already heard r say the same thing within the last half day. It is
// the only memory the world keeps of conversation, and it exists purely so
// a repeated remark can be counted and, in grounded mode, not treated as a
// fresh stimulus.
func (w *World) remember(r *Resident, text string, listeners []sim.AgentID) map[sim.AgentID]bool {
	heard := map[sim.AgentID]bool{}
	print := textsim.Of(text)
	since := w.Now.Add(-12 * time.Hour)
	for _, u := range r.Said {
		if u.At.Before(since) || !textsim.Same(textsim.Of(u.Text), print, textsim.Speech) {
			continue
		}
		for _, x := range u.Heard {
			heard[x] = true
		}
	}
	repeat := false
	for _, l := range listeners {
		repeat = repeat || heard[l]
	}
	if repeat {
		w.count("repeated_speech", 1)
		w.tally(r.ID, "repeated_speech", 1)
	}
	r.Said = append(r.Said, Utterance{At: w.Now, Text: clip(text, 280), Heard: append([]sim.AgentID(nil), listeners...)})
	if len(r.Said) > maxUtterances {
		r.Said = r.Said[len(r.Said)-maxUtterances:]
	}
	return heard
}

// sameTerms counts earlier offers from one resident to another on exactly
// these terms within the last day.
func (w *World) sameTerms(from, to sim.AgentID, give, want Goods) int {
	n := 0
	since := w.Now.Add(-24 * time.Hour)
	for _, o := range w.Offers {
		if o.From == from && o.To == to && o.Give == give && o.Want == want && !o.Created.Before(since) {
			n++
		}
	}
	return n
}

// stillNeeds describes what an unfinished project lacks, e.g. "; still
// needs 2 materials" — empty once it is complete.
func (w *World) stillNeeds(p *Project) string {
	if p.Done() || p.satisfied() {
		return ""
	}
	return "; still needs " + p.remaining()
}

// laborUnderway is the labor that stints already in progress will add.
func (w *World) laborUnderway(project string) int {
	n := 0
	for _, id := range w.Order {
		if a := w.Residents[id].Activity; a != nil && a.Kind == "build" && a.Project == project {
			n += w.Settings.Economy.BuildHours
		}
	}
	return n
}

// transfer records resources moving between residents, per kind of
// transfer ("given", "traded"), globally and for both parties.
func (w *World) transfer(from, to sim.AgentID, g Goods, how string) {
	w.count("credits_"+how, g.Credits)
	w.count("materials_"+how, g.Materials)
	w.tally(from, "credits_"+how+"_out", g.Credits)
	w.tally(from, "materials_"+how+"_out", g.Materials)
	w.tally(to, "credits_"+how+"_in", g.Credits)
	w.tally(to, "materials_"+how+"_in", g.Materials)
}

func (w *World) startActivity(r *Resident, act *Activity) {
	act.Start = w.Now
	act.ItemID = w.push(&Item{At: act.End, Kind: KindActivityDone, Agent: r.ID})
	r.Activity = act
}

// local is an action event observed by everyone at the actor's location.
func (w *World) local(r *Resident, text string) sim.Event {
	return sim.Event{Type: sim.EventAction, Actor: r.ID, Text: text, Visibility: sim.Only(w.Here(r.Location)...)}
}

func (w *World) trimNotices(place string) {
	max := w.Settings.Economy.MaxNotices
	var at []int
	for i, n := range w.Notices {
		if n.Place == place {
			at = append(at, i)
		}
	}
	if max <= 0 || len(at) <= max {
		return
	}
	drop := map[int]bool{}
	for _, i := range at[:len(at)-max] {
		drop[i] = true
	}
	kept := w.Notices[:0]
	for i, n := range w.Notices {
		if !drop[i] {
			kept = append(kept, n)
		}
	}
	w.Notices = kept
}

// resident resolves an id or display name, case-insensitively.
func (w *World) resident(ref string) *Resident {
	ref = strings.ToLower(strings.TrimSpace(ref))
	if ref == "" {
		return nil
	}
	for _, id := range w.Order {
		r := w.Residents[id]
		if strings.ToLower(string(id)) == ref || strings.ToLower(r.Name) == ref {
			return r
		}
	}
	return nil
}

func (w *World) placeIDs() string {
	ids := make([]string, 0, len(w.Locations))
	for _, l := range w.Locations {
		ids = append(ids, l.ID)
	}
	return strings.Join(ids, ", ")
}

func (w *World) nameList(ids []sim.AgentID) string {
	out := make([]string, 0, len(ids))
	for _, id := range ids {
		out = append(out, w.name(id))
	}
	return strings.Join(out, ", ")
}

// mentions reports whether text addresses someone by id or name, as a word.
func mentions(text string, id sim.AgentID, name string) bool {
	words := strings.FieldsFunc(strings.ToLower(text), func(c rune) bool {
		return !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-')
	})
	for _, w := range words {
		if w == strings.ToLower(string(id)) || w == strings.ToLower(name) {
			return true
		}
	}
	return false
}

func firstArg(a sim.Action, keys ...string) string {
	for _, k := range keys {
		v, ok := a.Args[k]
		if !ok || v == nil {
			continue
		}
		switch s := v.(type) {
		case string:
			if strings.TrimSpace(s) != "" {
				return strings.TrimSpace(s)
			}
		case float64, int:
			return fmt.Sprintf("%v", s)
		}
	}
	return ""
}

// listArg reads a list of names given as a JSON list or a comma-separated
// string.
func listArg(a sim.Action, keys ...string) []string {
	for _, k := range keys {
		switch v := a.Args[k].(type) {
		case string:
			var out []string
			for _, p := range strings.Split(v, ",") {
				if p = strings.TrimSpace(p); p != "" {
					out = append(out, p)
				}
			}
			if len(out) > 0 {
				return out
			}
		case []any:
			var out []string
			for _, x := range v {
				if s, ok := x.(string); ok && strings.TrimSpace(s) != "" {
					out = append(out, strings.TrimSpace(s))
				}
			}
			if len(out) > 0 {
				return out
			}
		}
	}
	return nil
}

func argInt(a sim.Action, def int, keys ...string) int {
	for _, k := range keys {
		if f, ok := a.ArgFloat(k); ok {
			return int(f)
		}
	}
	return def
}

func appendUnique(ids []sim.AgentID, more ...sim.AgentID) []sim.AgentID {
	out := append([]sim.AgentID(nil), ids...)
	for _, m := range more {
		dup := false
		for _, x := range out {
			if x == m {
				dup = true
			}
		}
		if !dup {
			out = append(out, m)
		}
	}
	return out
}

func goodsPhrase(g Goods) string {
	var parts []string
	if g.Credits != 0 {
		parts = append(parts, fmt.Sprintf("%d credits", g.Credits))
	}
	if g.Materials != 0 {
		parts = append(parts, fmt.Sprintf("%d materials", g.Materials))
	}
	if len(parts) == 0 {
		return "nothing"
	}
	return strings.Join(parts, " and ")
}

// tradePhrase says plainly who handed over what in a completed trade.
func tradePhrase(o *Offer, from, to string) string {
	var parts []string
	if !o.Give.IsZero() {
		parts = append(parts, fmt.Sprintf("%s handed over %s", from, goodsPhrase(o.Give)))
	}
	if !o.Want.IsZero() {
		parts = append(parts, fmt.Sprintf("%s handed over %s", to, goodsPhrase(o.Want)))
	}
	return strings.Join(parts, " and ")
}

func offerPhrase(o *Offer) string {
	switch {
	case o.Want.IsZero():
		return goodsPhrase(o.Give) + " for nothing in return"
	case o.Give.IsZero():
		return "nothing, asking for " + goodsPhrase(o.Want)
	}
	return goodsPhrase(o.Give) + " in exchange for " + goodsPhrase(o.Want)
}

func quoteNote(n string) string {
	if n == "" {
		return ""
	}
	return fmt.Sprintf(" — %q", n)
}

func clip(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}
