package world

import (
	"container/heap"
	"sort"
	"time"

	"github.com/nmiano/playworld/internal/sim"
)

// ItemKind classifies a scheduled item. Its order in kindRank is the
// tie-break for items due at the same instant: the world settles itself
// (dawn, finished activities) before anyone is asked to decide, explicit
// appointments come before self-scheduled intentions, and the low-frequency
// idle opportunity comes last.
type ItemKind string

const (
	// KindDawn is the daily world update: restock, yields, offer expiry.
	KindDawn ItemKind = "dawn"
	// KindActivityDone resolves something an agent started earlier
	// (travel, a work shift, building).
	KindActivityDone ItemKind = "activity_done"
	// KindAppointment is a meeting arranged between agents coming due.
	KindAppointment ItemKind = "appointment"
	// KindIntention is an agent's own requested wakeup coming due.
	KindIntention ItemKind = "intention"
	// KindIdle is the default, low-frequency opportunity to decide when
	// nothing else has happened for a while.
	KindIdle ItemKind = "idle"
	// KindActivate is the one item that costs inference: an agent being
	// given a moment to decide. Every trigger above funnels into at most one
	// pending KindActivate per agent, which is where coalescing happens.
	KindActivate ItemKind = "activate"
)

var kindRank = map[ItemKind]int{
	KindDawn: 0, KindActivityDone: 1, KindAppointment: 2, KindIntention: 3, KindIdle: 4, KindActivate: 5,
}

// Reason is one cause of an activation. Several are coalesced into a single
// activation when they arrive while the agent is busy, cooling down or asleep.
type Reason struct {
	// Kind is a short category for metrics: message, speech, activity,
	// intention, appointment, idle, offer, gift, project, refused, invite.
	Kind string `json:"kind"`
	Text string `json:"text"`
	// Depth is how many agent-to-agent reactions deep this trigger is: 0
	// for anything the world or the agent itself caused (idle, a finished
	// activity, an intention, an appointment), and one more than the
	// activation that caused it for a reaction to another agent (being
	// spoken to, a note, an offer). It is what damps conversational
	// feedback loops — see Scheduling.MaxReplyChain.
	Depth int `json:"depth,omitempty"`
}

// chainDepth is the depth of an activation: the shallowest of its reasons,
// so anything non-reactive among them resets the chain.
func chainDepth(rs []Reason) int {
	if len(rs) == 0 {
		return 0
	}
	d := rs[0].Depth
	for _, r := range rs[1:] {
		d = min(d, r.Depth)
	}
	return d
}

// Item is one scheduled future event. It carries enough to be processed in
// a later process after a save and resume: when, what kind, who caused it,
// who it concerns, and kind-specific metadata.
type Item struct {
	// ID is unique within a world and is the final, stable tie-break.
	ID   int64     `json:"id"`
	At   time.Time `json:"at"`
	Kind ItemKind  `json:"kind"`
	// Agent is who the item concerns (the one to resolve for or activate).
	Agent sim.AgentID `json:"agent,omitempty"`
	// Origin is who caused it, when that is someone else.
	Origin sim.AgentID `json:"origin,omitempty"`
	// Audience lists who may observe the item's consequences when that is
	// not just Agent (e.g. every party to an appointment).
	Audience []sim.AgentID `json:"audience,omitempty"`
	// Ref names the intention, appointment or project the item belongs to.
	Ref string `json:"ref,omitempty"`
	// Reasons accumulate on a KindActivate item as triggers coalesce.
	Reasons []Reason `json:"reasons,omitempty"`
	// Explicit marks an activation the agent (or an appointment) asked for
	// at a specific time, which is honoured even during the night.
	Explicit bool `json:"explicit,omitempty"`

	index int // position in the heap; maintained by Queue
}

// Queue is a min-heap of items ordered by (At, kind rank, ID). The order is
// total, so processing is deterministic regardless of insertion history, and
// it survives a save/load round trip unchanged.
type Queue struct {
	items []*Item
	byID  map[int64]*Item
}

func newQueue(items []*Item) *Queue {
	q := &Queue{items: items, byID: map[int64]*Item{}}
	for i, it := range q.items {
		it.index = i
		q.byID[it.ID] = it
	}
	heap.Init((*queueHeap)(q))
	return q
}

// Len reports how many items are pending.
func (q *Queue) Len() int { return len(q.items) }

// Push schedules an item.
func (q *Queue) Push(it *Item) {
	heap.Push((*queueHeap)(q), it)
	q.byID[it.ID] = it
}

// Peek returns the next item without removing it, or nil.
func (q *Queue) Peek() *Item {
	if len(q.items) == 0 {
		return nil
	}
	return q.items[0]
}

// Pop removes and returns the next item, or nil.
func (q *Queue) Pop() *Item {
	if len(q.items) == 0 {
		return nil
	}
	it := heap.Pop((*queueHeap)(q)).(*Item)
	delete(q.byID, it.ID)
	return it
}

// Get finds a pending item by id.
func (q *Queue) Get(id int64) *Item { return q.byID[id] }

// Remove cancels a pending item. Removing an id that is not pending is a
// no-op, so callers need not track whether it already fired.
func (q *Queue) Remove(id int64) {
	it, ok := q.byID[id]
	if !ok {
		return
	}
	heap.Remove((*queueHeap)(q), it.index)
	delete(q.byID, id)
}

// Reschedule moves a pending item to a new time.
func (q *Queue) Reschedule(id int64, at time.Time) {
	if it, ok := q.byID[id]; ok {
		it.At = at
		heap.Fix((*queueHeap)(q), it.index)
	}
}

// Sorted returns the pending items in processing order, for persistence and
// inspection. It does not disturb the heap.
func (q *Queue) Sorted() []*Item {
	out := append([]*Item(nil), q.items...)
	sortItems(out)
	return out
}

func less(a, b *Item) bool {
	if !a.At.Equal(b.At) {
		return a.At.Before(b.At)
	}
	if ra, rb := kindRank[a.Kind], kindRank[b.Kind]; ra != rb {
		return ra < rb
	}
	return a.ID < b.ID
}

func sortItems(items []*Item) {
	sort.Slice(items, func(i, j int) bool { return less(items[i], items[j]) })
}

// queueHeap adapts Queue to container/heap.
type queueHeap Queue

func (h *queueHeap) Len() int           { return len(h.items) }
func (h *queueHeap) Less(i, j int) bool { return less(h.items[i], h.items[j]) }
func (h *queueHeap) Swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i
	h.items[j].index = j
}
func (h *queueHeap) Push(x any) {
	it := x.(*Item)
	it.index = len(h.items)
	h.items = append(h.items, it)
}
func (h *queueHeap) Pop() any {
	old := h.items
	n := len(old)
	it := old[n-1]
	old[n-1] = nil
	it.index = -1
	h.items = old[:n-1]
	return it
}
