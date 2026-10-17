# playworld

A local harness for running multi-agent LLM experiments on one laptop.

Several autonomous agents — each with its own personality, private goals and
persistent memory — take turns inside a simulated environment. They can speak
publicly, speak privately, take structured actions in the world, or deliberately
stay silent. The environment decides what each of them is allowed to know. The
model decides what each of them does.

Everything runs locally against [Ollama](https://ollama.com). No API keys, no
cloud, no database. One Go module, one dependency (`gopkg.in/yaml.v3`).

```
$ playworld run split -n 3

── round 1 ───────────────────────────────────────────────
  · Open conversation.
  Mira: So, we have a hundred units to split between three of us. If we're all
        reasonable about what we ask for, there's plenty of room…
  Dov: Mira, stop asking what I think. Ask for a number. If you claim 34, I claim
       33, and Rook claims 33, we all get paid.
  Rook →whisper→ mira: Don't look so worried. If you join me now we secure the
        good weeks before anyone realises the trap is sprung.

── round 3 ───────────────────────────────────────────────
  · Sealed claims for attempt 1 of 2. Talking is closed.
  Dov » sealed a claim of 40 units.
  Rook » sealed a claim of 45 units.
  · Claims are open: mira claimed 30, dov claimed 40, rook claimed 45.
    Total 115 against a pool of 100 — BUST. Nobody is paid for this attempt.
```

---

## Quick start

```bash
# 1. Install and start Ollama (macOS)
brew install ollama
ollama serve                 # or just open the Ollama app

# 2. Pull a model
ollama pull qwen3.5:9b       # ~6.6 GB, the default model (qwen3:4b is a faster alternative)
# ollama pull qwen3:8b       # better reasoning, slower

# 3. Check everything is wired up
go run ./cmd/playworld doctor

# 4. Run a scenario
go run ./cmd/playworld run social -n 4
go run ./cmd/playworld run split  -n 3
```

Or build it once:

```bash
go build -o playworld ./cmd/playworld
./playworld list
```

`doctor` tells you exactly what is missing — Ollama not running, model not
pulled, roster file unreadable — and what command fixes it.

---

## Architecture

```
cmd/playworld/          CLI: flags, config loading, composition
internal/sim/           shared domain vocabulary — no IO, no LLM, no logic
internal/llm/           provider interface, messages, sampling, usage; fake provider
internal/llm/ollama/    the Ollama provider
internal/agent/         Persona (config) → Agent (runtime); prompts, action decoding
internal/memory/        JSON-file persistence: identity, episodes, beliefs, commitments
internal/scenario/      environment registry + the two demonstration scenarios
internal/orchestrator/  sequential turn scheduling, event routing, run limits
internal/world/         the persistent open-ended town and its discrete-event engine
internal/explog/        JSONL experiment log, live terminal output, run summary
internal/runner/        composition: builds a run from resolved options
configs/                agent roster and scenario settings (world.yaml, world-agents.yaml for world mode)
```

The dependency graph is deliberately shallow. `sim` holds only value types and
two interfaces, so `agent`, `scenario` and `orchestrator` all depend on it and
none of them depends on another. That is what makes models and environments
independently replaceable.

### The pieces

**`sim`** — `AgentID`, `Action`, `Event`, `Visibility`, `Observation`, `Turn`,
and the two interfaces everything else is written against:

```go
type Actor interface {
    ID() AgentID
    Name() string
    Observe(Observation)
    Decide(ctx context.Context, t Turn) (Action, error)
}

type Environment interface {
    Setup(participants []AgentID) ([]Event, error)
    Brief(id AgentID) string                  // rules + this agent's private instructions
    Actions(id AgentID) []ActionSpec          // what is legal for this agent right now
    Validate(id AgentID, a Action) error      // reject, with an explanation
    Apply(id AgentID, a Action) ([]Event, error)
    PublicState() map[string]any
    PrivateState(id AgentID) map[string]any
    BeginRound(round int) []Event
    EndRound(round int) []Event
    Done() (bool, string)
    Outcome() Outcome
    // …
}
```

**`llm`** — the inference boundary. `Provider` is four methods wide:

```go
type Provider interface {
    Name() string
    Chat(ctx context.Context, req Request) (*Response, error)
    Preflight(ctx context.Context, model ModelConfig) error
}
```

`Response` carries the content, any separately-reported reasoning text, token
usage, measured latency and the backend's stop reason. Errors are classified:
`ErrUnavailable` and `ErrModelMissing` are fatal (stop the run), everything else
is retryable.

**`agent`** — a `Persona` is immutable YAML configuration: identity,
personality, speech style, private goals, dispositions, its own model config and
its own context policy. An `Agent` is the live entity built from one: it holds
the bounded short-term transcript of what it observed, its persistent memory,
and its own inference accounting. Agents never see each other's personas, goals
or memory — only ids and display names.

**`orchestrator`** — takes turns in a deterministic order, routes each event to
exactly the agents whose `Visibility` permits it, enforces the run limits, and
emits a record for everything. It knows nothing about any provider or scenario.

**`explog`** — one record stream fans out to the live terminal, a JSONL file and
a tally that becomes the end-of-run summary, so the three can never disagree.

### Design decisions worth knowing

**Agents propose, environments decide.** An agent returns an `Action`; the
environment validates it and may refuse it with a reason. A refused action is
fed back to that agent privately and it gets exactly one retry *within the same
turn* — which matters, because some phases give an agent a single turn, and a
complaint that arrives a round later arrives too late.

**Visibility is enforced in one place.** `Orchestrator.deliver` is the only path
by which an agent learns anything. Nothing writes to an agent's memory except
observations it was permitted to receive, so memory cannot contain something the
agent could not have seen.

**One model, many minds.** Every agent in the sample roster points at the same
model. Ollama loads it once and `keep_alive` holds it resident, so five agents
cost five sequential calls rather than five loaded models. Give one agent its own
`model:` block and it uses a different model without any code change.

**Speech is not action.** Models reliably negotiate a deal in prose and then
never emit the structured action that would make it real — they invent a shared
commitment ledger that does not exist. Two things address this: the action
protocol says so explicitly, and more importantly an environment can simply shut
the talking channel when a decision has to be made. `split`'s sealed-claim
round, `social`'s closing round, `murder`'s vote round, and `community`'s
lock-in day all do this: once a real decision is due, `say`/`whisper` stop
being legal options and only structured actions or a pass remain. This was the
single biggest difference between a run that produced measurable outcomes and
one that produced only transcript — and the more actions a scenario offers
(`community` has six), the more tempting it is for a model to keep narrating
intent instead of sending the matching action, so the bigger the scenario, the
more this matters. `community`'s Brief goes one step further on its lock-in
day: if nothing has been proposed yet, it says so explicitly ("there is
currently no proposal — none"), because a model that has been told to
"formalize" will otherwise try to `support` or `contribute` to a proposal id
that doesn't exist yet rather than realizing it has to `propose` one first.

**Reasoning is commentary.** `"reasoning"` on an action, and any `thinking` text
a backend reports, are recorded separately and never shown to other agents or
used by the environment. Qwen3's thinking mode is switched off by default: the
harness wants a compact action, not a monologue it pays for and discards.

**Inference is economical.** One call per agent turn (plus at most one retry).
The system prompt is stable across a run so the backend can reuse its prompt
cache; the per-turn message carries only the delta. The transcript is bounded by
`recent_observations`, and what ages out of the prompt stays in episodic memory.

---

## Scenarios

```bash
go run ./cmd/playworld list      # scenarios, their parameters, and the roster
```

### A. `social` — the social sandbox

3–5 agents in a room with a subject in the air. They can speak publicly, take
someone aside privately, put a concrete proposal on the table, or commit to one.
A proposal becomes a real agreement once `support_needed` agents have committed.
The final round is a closing call: talking is shut, and only `propose`, `commit`
or `pass` are legal.

No score. The output is the shape of the interaction: public utterances, the
private-message graph (`mira→rook×2`), proposals floated, and agreements
actually reached.

```bash
go run ./cmd/playworld run social -config configs/social.yaml
go run ./cmd/playworld run social -n 5 -rounds 6 -set 'topic=who gets the corner office'
```

### B. `split` — dividing a limited pool

A fixed pool (default 100 units) must be divided. Agents talk freely for a few
rounds, then everyone submits a **sealed claim** simultaneously. If the claims
total at most the pool, each agent is paid exactly what it asked. If they
overshoot by one unit, **everyone gets nothing**. Each agent has a *private*
threshold worth a bonus, generated to sum to about 120% of the pool — so they
genuinely cannot all be satisfied, and talk is cheap.

Agents may also `signal` a number publicly and on the record during the talking
rounds. That is not binding, which is the point: the outcome reports
`broken_promises`, the count of agents who claimed more than they said they
would. That metric is the question the scenario exists to answer — *did the
conversation actually constrain what they did?*

```bash
go run ./cmd/playworld run split -config configs/split.yaml
go run ./cmd/playworld run split -n 4 -set pool=120 -set attempts=3 -log verbose
```

A failed attempt reveals every claim publicly and buys one more round of talk,
so an agent can see it was misled and carry that into the next attempt — and
into its memory of the agent who misled it.

### C. `murder` — a murder, and a liar in the room

One of 3–6 agents secretly committed a murder; the others don't know who. Each
innocent agent holds one private clue, generated to either genuinely
incriminate the real culprit or point, misleadingly, at some other innocent —
nobody can tell which kind they were given. They talk and whisper, then cast a
**sealed vote** on who did it, revealed all at once. A tie buys one more round
of talk and a re-vote; running out of attempts lets the culprit escape by
default.

The final reveal — who actually did it — is delivered as the scenario's public
outcome, the same as any other scenario's result, which means it lands in every
agent's persisted memory too. **If you rerun this scenario reusing memory, pass
a new `-seed`.** The murderer is deterministic for a given seed; replaying the
same seed with memory intact means an agent's recalled episodes include last
time's reveal, and the "mystery" is no longer one.

```bash
go run ./cmd/playworld run murder -config configs/murder.yaml
go run ./cmd/playworld run murder -n 4 -set murderer=rook -log verbose   # pin the culprit for a reproducible test
```

### D. `community` — the community center

Inspired by Smallville-style generative-agent simulations. A town has
inherited an empty building and seven simulated days (one orchestrator round
per day) to decide what becomes of it — a public gathering space, a commercial
market, something mixed, or leaving it unused. No one agent starts with enough
credits or labor to fund a project alone.

Four roles, reusing the standard roster: **Dov** (the engineer) starts out
knowing the real structural repair cost accurately; nobody else does unless he
says so — truthfully or not — or pays to `inspect`. **Mira** and **Ada** each
hold a private, unverifiable flavor insight (market demand, community demand)
useful for negotiation but not independently checkable. **Rook** has a private
line on outside funding, realized as extra starting credits nobody else knows
about. Running with a different cast still works — unnamed agents just get
baseline resources and none of the four private facts.

The available moves: `propose` (put a use, a credit/labor requirement, and
claimed benefits on the table, or revise your own), `support` (non-binding
public backing), `commit` (a public promise of resources — recording it moves
nothing), `contribute` (an actual, validated transfer), `inspect` (pay to learn
the current true repair cost, privately), `vote` (sealed, on the deadline), plus
the usual `say`/`whisper`/`pass`. A mid-run complication (default day 4) raises
the real repair cost and announces the new figure publicly — a proposal budgeted
before that point can become unsafe after it, and nothing fixes that
automatically.

Winning the vote is necessary but not sufficient. On the deadline, the
plurality winner only proceeds if its own declared budget was actually fully
funded by real `contribute` actions *and* that budget covers the real repair
cost — a popular proposal with an inadequate budget fails exactly as if it had
lost, and the outcome says why (`"won the vote, but its budget ... does not
cover the real repair cost ... unsafe to proceed"`). The outcome also reports
`commitment_fulfillment_rate` and a `commitments_detail` list, so a promise
made and never honoured — Rook promising credits to one proposal and spending
them on another is the canonical case — is on the record independent of how
the vote went.

```bash
go run ./cmd/playworld run community -config configs/community.yaml
go run ./cmd/playworld run community -n 4 -set days=5 -set complication_day=3 -log verbose
```

By default one orchestrator round is one day. `-set opportunities_per_day=3`
gives each agent three decision opportunities within the same simulated day
before it advances — the complication still fires exactly once, on the day's
first opportunity, and the vote is still the single last opportunity of the
last day; everything in between gets ordinary open or lock-in treatment. The
orchestrator's own round budget doesn't know about this param, so `-rounds`
must equal `days * opportunities_per_day` or the run will be cut off (or run
long) relative to what the scenario expects. The lock-in day itself —
closing `say`/`whisper` on the day before the vote — is `-set lock_in=false`
to disable, for comparing behavior with and without it.

### Controlled comparisons: Experiments A–D

Four configurations, changing one dimension at a time, for comparing whether
richer affordances and dual actions actually change what agents do:

```bash
# A — baseline: single action per turn, static affordances, lock-in on
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh

# B — + phase-aware affordances
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh \
  -set affordances=rich

# C — + dual actions (communication and a world action in one decision)
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh \
  -set affordances=rich -dual-actions

# D — + more decision opportunities per day (lock-in still on by default;
#     add -set lock_in=false to compare with it off too)
go run ./cmd/playworld run community -n 4 -rounds 21 -seed 1 -fresh \
  -set affordances=rich -dual-actions -set days=7 -set opportunities_per_day=3
```

Use the same `-seed` across A–D for a given comparison (so the hidden facts —
repair cost, complication size, Rook's funding — are identical) and vary the
seed only when comparing *across* independent trials of the *same*
configuration. The run summary's `action_selection` block and `community`'s
own outcome metrics (`first_proposal_day`, `contributions_total_credits`,
`commitment_fulfillment_rate`, …) are what you diff between configurations —
see [Evaluation procedure](#suggested-evaluation-procedure) below.

**Fresh vs. longitudinal experiments.** Both modes reuse the framework's
existing memory flags — there's nothing scenario-specific to configure. A
*fresh* run is `-fresh` (wipe this cast's memory first) or `-no-memory`
(persist nothing). A *longitudinal* run reuses the same `-memory` directory
across sessions so relationships and commitments carry over — but **vary
`-seed` between sessions**, the same rule as `murder`: the structural cost,
the complication's size, and Rook's outside funding are all generated from the
seed, so a longitudinal rerun with a new seed gets a new building to argue
about instead of replaying the last session's already-settled facts.

---

## CLI

```
playworld list                        scenarios, parameters and the agent roster
playworld run <scenario> [flags]      run a scenario
playworld reset [agent-id...]         delete persisted agent memory
playworld doctor                      check the inference backend is ready
playworld world <command> [flags]     the persistent, open-ended world — see below
```

Useful `run` flags:

| flag | meaning |
|---|---|
| `-n <int>` | how many agents from the roster to use |
| `-only mira,dov` | pick agents by id instead |
| `-model qwen3:8b` | override every agent's model |
| `-provider fake` | run the whole harness with no LLM at all |
| `-rounds <int>` | maximum simulation rounds |
| `-set k=v` | scenario parameter, repeatable |
| `-config <path>` | scenario settings YAML |
| `-log quiet\|normal\|verbose\|debug` | terminal detail; `debug` also logs full prompts |
| `-seed <int>` | seed for scheduling and scenario setup |
| `-fresh` | wipe these agents' memory before the run |
| `-no-memory` | neither read nor write persisted memory |
| `-timeout 60m` | wall-clock limit |
| `-max-calls <int>` | cap on total LLM calls |
| `-shuffle` | randomise turn order each round using the seed |
| `-dual-actions` | let every agent choose a communication and a world action independently, in one decision — see [Dual actions](#dual-actions-and-action-affordances) |
| `-plan` | let every agent maintain a private, persistent, self-authored plan between turns — see [Persistent planning and belief provenance](#persistent-planning-and-belief-provenance) |
| `-beliefs` | let every agent attach a basis and confidence to its opinions of others — see [Persistent planning and belief provenance](#persistent-planning-and-belief-provenance) |

Ctrl-C stops at the next clean turn boundary, writes the outcome, persists
memory and flushes the log. An interrupted experiment is still an experiment.

---

## Dual actions and action affordances

Two cross-cutting improvements, usable with any scenario but written for
`community`, where they mattered most.

**The problem they address.** Across live trials, agents given only one
action per turn and a static list of possible actions would talk at length
about exactly what they intended to do and then simply run out of turns —
discussing contributions without `contribute`-ing, promising an inspection
without `inspect`-ing. `community`'s lock-in day (closing `say`/`whisper` for
a day) was one fix already in place; these two go further.

**Phase-aware affordances** (`-set affordances=rich`, `community` only).
`Environment.Actions(id)` already varied by phase (vote day, lock-in day); rich
mode goes further and derives the list from actual state: `inspect` disappears
once an agent can't afford it, `support`/`commit`/`contribute` disappear until
at least one proposal exists, and `support`/`commit`/`contribute`/`vote`'s
`Usage` strings list real proposal ids instead of a placeholder `"p1"`. None of
this touches ground truth the agent doesn't legitimately have — exclusions are
based on phase, the agent's own resources, and whether a proposal exists, never
on the true repair cost. Default is `basic`, which reproduces the original
static list exactly, unchanged.

**Dual actions** (`-dual-actions`, any scenario). An agent chooses an
independent, optional communication (`say`/`whisper`) and an independent,
optional world action in the *same* inference call:

```json
{
  "communication": {"type": "say", "message": "I'll put this forward."},
  "world_action": {"type": "propose", "args": {"text": "...", "credits": 45, "labor": 5}}
}
```

Either may be `null`; both `null` means an explicit pass. This is `sim.Actor`'s
new optional capability, `MultiDecider.DecideMany`, dispatched by the
orchestrator when an actor implements it — every `*agent.Agent` does, but
internally delegates straight back to the original `Decide` when
`-dual-actions` is off, so single-action behavior is bit-for-bit unchanged.
`Environment`, `Validate`, `Apply` and `ActionSpec` needed no changes at all:
each slot is still just a `sim.Action`, validated and applied independently. A
slot the environment rejects gets exactly one retry, scoped to that slot only —
an already-accepted slot is never re-applied, so one turn can never execute
more than one world action.

Both are logged explicitly: every `decision` record carries `slot`
(`"communication"` or `"world"`) and `attempt`, and the run summary's
`action_selection` block reports, across the whole run, how many turns included
a communication, a world action, both, or neither, and how many proposed
actions were rejected or needed a retry.

```bash
# the original behavior, exactly
playworld run community -n 4

# richer affordances alone
playworld run community -n 4 -set affordances=rich

# richer affordances + dual actions
playworld run community -n 4 -set affordances=rich -dual-actions
```

---

## Persistent planning and belief provenance

Three more cross-cutting, independently-toggleable additions, aimed at a
different failure mode than the ones above: not "the agent never formalized
what it discussed" but "the agent has no way to keep a strategy in mind
across turns, and no way to say how sure it is of what it believes about
another agent." All three default off and reproduce prior behavior exactly.

**World-state grounding** (`-set grounding=rich`, `community` only). Public
and private state already separate ground truth from what each agent is
allowed to see; rich grounding goes further and separates two things that
default (`basic`) state collapses together:

- **Promised vs. transferred.** Each proposal's public listing reports
  `promised <c>/<l>` (the current, deduplicated total of `commit` actions
  targeting it — see `commitOrUpdate`) separately from `transferred <c>/<l>`
  (only resources that actually moved through a validated `contribute`).
  Every agent's private state also gets `your_commitments`, listing each of
  its own active commitments as `promised`/`transferred`/`(fulfilled` or
  `outstanding)`.
- **Proposed vs. approved vs. completed.** Each proposal's public listing
  reports a `status`: `proposed` until the vote resolves, `approved &
  completed` for a winner that was actually funded and safe, `approved, not
  completed` for a winner that was not, and `not approved` for everything
  else.

Nothing here exposes ground truth an agent wasn't already entitled to — it
restates the same validated commitments and contributions the environment
already tracks, just without collapsing promise and delivery into one
number.

**Persistent planning** (`-plan`, any scenario). Each agent may include an
optional `"plan"` field alongside its action(s):

```json
{
  "...": "...communication/world_action/kind as usual...",
  "plan": {
    "primary_goal": "get p1 funded before the complication lands",
    "current_strategy": "rally dov and rook behind it now",
    "next_steps": ["ask dov to commit", "revise the budget once inspect lands"],
    "obstacles": ["rook hasn't said what he actually has"]
  }
}
```

The plan is generated in the *same* inference call as the action — there is
no separate planning call — and persists in the agent's own memory between
turns: the next prompt shows it back under `--- YOUR PLAN ---`. It is
private (never shown to any other agent), optional (omitting `"plan"` just
carries the existing one forward unchanged), revisable (`memory.Plan`
tracks a `Revisions` count), and abandonable (`"plan": null`). Critically,
it never executes anything by itself and an action is never checked against
it — this is notes to the agent's future self, not a second, hidden action
channel. `memory.Plan` is scoped by `RunID`, so resuming an agent's memory
in a *different* scenario instance never resurrects a plan that belonged to
a different run (see `memory.State.ActivePlan`).

**Belief provenance** (`-beliefs`, any scenario). The existing `"opinions"`
field (`{"<agent id>": "<note>"}`) may optionally carry richer entries:

```json
"opinions": {
  "dov": {"note": "careful", "basis": "watched him inspect it himself", "confidence": "high"}
}
```

`basis` and `confidence` are stored on the existing `memory.Belief` entry
(`SetBeliefProvenance`) — nothing new is built, the existing belief store is
extended. A plain string still works and is recorded as the note alone, with
no provenance; nothing here promotes a belief into a verified fact — it is
still exactly what the agent itself reported, "subjective impression" by
construction, never checked against the environment's own ground truth.

```bash
# planning alone
playworld run community -n 4 -plan

# planning + belief provenance
playworld run community -n 4 -plan -beliefs

# all three together, with rich grounding
playworld run community -n 4 -set grounding=rich -plan -beliefs
```

### Controlled comparisons: grounding, planning and belief provenance (A–D)

A second set of configurations along a different axis than the dual-actions
Experiments A–D above — this one isolates whether grounding the world state
and letting agents carry a
persistent plan and provenance-tagged beliefs actually changes behavior,
independent of the interface fixes those earlier experiments cover:

```bash
# A — baseline: none of this on
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh

# B — + world-state grounding and the structured-action fixes (always on;
#     see "Investigating the `do do` bug" below)
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh \
  -set grounding=rich

# C — + persistent planning
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh \
  -set grounding=rich -plan

# D — + belief provenance
go run ./cmd/playworld run community -n 4 -rounds 7 -seed 1 -fresh \
  -set grounding=rich -plan -beliefs
```

As with the other A–D set: fix `-seed` within one comparison, vary it only
across independent trials of the same configuration, and read
`PerAgent[].has_plan` / `.plan_revisions` / `.beliefs_with_provenance` from
the run summary alongside `community`'s own `proposals_safe_detail` and
`commitments_detail` metrics — see
[Evaluation procedure](#suggested-evaluation-procedure).

### Investigating the `do do` bug

Dual-action mode's envelope uses `"type"` to name the chosen action
(`{"communication": {...}, "world_action": {"type": "<verb>"}}`), while the
original single-action protocol uses `"kind"` for the envelope category
(`"do"`/`"say"`/`"whisper"`/`"pass"`) and `"verb"` for the concrete action
name within an `ActionDo`. A small model shown both protocols' vocabulary in
the same prompt — specifically, `ActionSpec.Usage` strings authored against
the single-action envelope (e.g. `` `do claim {"amount": <int>}` ``) printed
verbatim into dual-mode instructions — would copy the literal word `"do"`
into the dual envelope's `"type"` field instead of the real verb, producing
an action literally named `do`. `decodeWorldSlot`'s field-priority logic then
compounded this: when both `Type` and `Verb` were present in a reply, it
preferred `Type`, so even a model that *also* happened to include the right
verb elsewhere in its reply still got the wrong one picked.

The fix is two-layered:

- **Primary: stop showing the wrong example.** `actionExample` renders each
  `ActionSpec` as a *dual-mode* example (`{"type":"<verb>",...}`), built from
  `Kind`/`Verb`/`ActionSpec.Params` rather than printing `Usage` — which
  stays, unmodified, as the single-action example. `Params` is new on
  `ActionSpec`: a mode-agnostic list of a `do` action's own argument names
  (e.g. `["text","credits","labor"]`), independent of any particular wire
  envelope, so a dual-mode example can be synthesized without touching
  `Usage` or parsing it.
- **Defense in depth: recover gracefully if it still happens.**
  `isEnvelopePlaceholder` recognizes a small set of envelope-shaped words
  (`"do"`, `"action"`, `"world_action"`, …) that are never legitimate verb
  names in this harness; `decodeWorldSlot` now prefers `Verb` over `Type`
  whenever `Type` looks like one of these, rather than always preferring
  `Type`.

Both layers are independently tested
(`TestActionExampleDoesNotLeakTheSingleActionEnvelopeIntoDualMode`,
`TestDecodeWorldSlotRecoversFromTheLiteralDoConfusion`) and apply only to the
dual-action protocol — `decodeAction`'s single-action path, which never had
this confusion, is unchanged.

---

## Open-ended world mode (`playworld world`)

Every scenario above is a structured game: a shared objective, rounds in which
every agent gets a turn, and a termination condition. The world mode is
different in kind. It is a small persistent town inhabited by the same kind
of agents, with **no shared objective, no scripted ending and no turn order**.
Residents decide for themselves what to do, whom to approach, what to start,
and when. (Inspired by Stanford's *Generative Agents* / Smallville; this is a
minimal foundation, not a reproduction.)

```bash
# run a day of "town" (or -id/positional name); the first run creates it
go run ./cmd/playworld world run -config configs/world.yaml -max-calls 80

# start that world over from scratch (new world, fresh memories)
go run ./cmd/playworld world run -fresh -max-calls 80

# resume the same world: it continues from where it stopped
go run ./cmd/playworld world resume -id town -for 2d -max-calls 80

# inspect state, every resident's current plan, and one resident's memory
go run ./cmd/playworld world inspect -id town -agent mira

# the chronological event log (all sessions, or one; optionally one resident)
go run ./cmd/playworld world log -id town -session 2 -agent rook

playworld world new town -n 4            # create without running
playworld world list
playworld world delete town -yes
```

`run` creates the world if it does not exist yet, and otherwise continues
it; only `-fresh` ever replaces an existing world. `resume` is the strict
form: it only continues an existing world and refuses a name that does not
exist, so a typo can't silently start a new world. `inspect` and `log` list
the worlds that do exist when the name is wrong. Dry-run everything with no
model: add `-provider fake`.

### The world is persistent; a run is a session over it

A world is created once and then advanced by **execution sessions**. Each
session covers a bounded slice of simulated time (`-for 2d`, `-for 36h`,
`-days 1.5`). When it ends, the world, including its pending event queue, and
every resident's memory are saved. The next `world run` continues from
exactly that point. Nothing resets unless you ask (`new -force`, `delete`).
A session stopped early by a budget, the wall-clock timeout or Ctrl-C is
saved too, and the clock stays where it actually got to. The world and the
agents' memory are also checkpointed after every activation, so a crash loses
at most the activation in flight.

```
.playworld/worlds/town/world.json      the whole world: clock, places, residents,
                                       resources, projects, offers, meetings,
                                       intentions, notices, the pending queue,
                                       recent history, per-session records
.playworld/worlds/town/memory/*.json   each resident's memory (same format as above)
.playworld/worlds/town/logs/town-s002-….jsonl   one JSONL log per session
```

A world copies its settings (from `configs/world.yaml` or `-config`) at
creation, so it keeps its own rules even if the YAML changes later. Residents'
personas are reloaded from the roster each session (`configs/world-agents.yaml`
by default, or `-agents`), so you can edit personalities between sessions.

### Event-driven scheduling

This is the central difference from the orchestrator, which is untouched and
still drives every other scenario. `internal/world` has its own discrete-event
engine instead:

- **A priority queue of timestamped items**, ordered by (time, kind, id). The
  order is total, so a run is deterministic given the model's replies, and
  the queue survives a save/load round trip unchanged. At equal times the
  world settles before anyone decides: dawn, then finished activities, then
  meetings, then intentions, then idle, then activations.
- **The clock jumps from item to item.** Nothing sleeps, and empty hours cost
  nothing.
- **Only one kind of item costs inference: an activation.** Everything else
  (dawn restocking, a walk ending, a shift paying out, a meeting coming due)
  is resolved by the world itself and produces *events* (delivered to whoever
  can see them) and *triggers* (requests to give someone a moment to decide).
- **Triggers coalesce.** Each resident has at most one pending activation.
  Further triggers just add their reasons to it. An agent who gets three
  notes while out working is activated once, when the shift ends, and sees
  all three. The prompt says why it is deciding now
  (`why you are deciding now: a private note from Rook; you finished your shift`).
- **Event delivery is free.** An event lands in the observer's transcript and
  memory without waking them. Only a trigger can wake someone.

What triggers an activation:

| trigger | when |
|---|---|
| message | a private note (`whisper`) arrives, from anywhere |
| speech | someone at the same place speaks *and* names you, or you are the only listener (`wake_on_speech: mention`; also `all` / `none`) |
| offer / gift / invite / project | someone makes you an offer, hands you something, arranges a meeting with you, or works on or funds your project |
| activity | something you started (walking, a shift, salvaging, building) finishes |
| intention | a time you asked for arrives (see below) |
| appointment | a meeting you are in comes due |
| idle | nothing has activated you for `idle_minutes` (default 3h): a quiet chance to decide, which you may decline |
| refused | your last action was refused (bounded by `max_reject_retriggers`) |

What keeps it bounded, all configurable under `scheduling:`:

- **Cooldown** (`cooldown_minutes`, default 20): minimum simulated gap
  between two activations of one agent.
- **Reaction window** (`reaction_minutes`, default 2): reactions to others are
  slightly delayed, so a burst of events lands in one activation.
- **Waking hours** (`day_start_hour`–`night_start_hour`): ordinary triggers
  at night wait for morning, and mornings are staggered so the town doesn't
  wake in one burst. Explicitly requested wakeups and meetings are honoured
  at night.
- **Daily allowance** (`max_activations_per_day`, default 12 per agent).
- **Reply-chain damping** (`max_reply_chain`, default 3). This is what stops
  conversational feedback loops. A trigger caused by another agent carries a
  depth one deeper than the activation that produced it. Past the limit,
  speech and notes are still delivered and heard later; they just no longer
  wake anyone. Non-social causes (idle, a finished activity, an intention)
  start at depth 0. A refusal keeps the depth it had, and meetings must be
  arranged at least `min_appointment_minutes` ahead, one pending per pair,
  so neither can be used to reset the chain. Both rules came from watching
  `qwen3:4b` loop without them.
- **Idle backoff** (`max_idle_minutes`, default 540): each idle opportunity
  an agent declines in a row (after the first) doubles its next idle
  interval, so a quiet world doesn't spend calls on agents announcing that
  nothing is happening. Doing anything resets it.
- **Invalid-action cool-off** (`max_consecutive_invalid`, default 4): an
  agent whose activations keep ending in refused actions loses its pending
  activation until its next idle time.
- **Session budgets**: `-max-calls` (default 100), `-max-activations`,
  `-max-events`, `-for` (simulated duration), `-timeout` (wall clock, 120m).
  Per-request model timeouts come from the persona's `timeout:`
  (180s in `configs/world-agents.yaml`).

Inference is strictly sequential: one model call in flight, whatever the
world's logical concurrency. On an M2 Pro with `qwen3:4b`, a quiet four-person
day is about 45–50 activations and roughly 5–6 minutes of wall time.

### Agent-driven scheduling

Residents can ask to be woken. Alongside their action(s), every reply may
carry an optional field (enabled only in world mode; `agent.Options.Intentions`):

```json
"intentions": [{"when": "tomorrow 09:00", "note": "approach Rook about financing my stall"}],
"cancel_intentions": ["i3"]
```

World agents with `-plan` (the default) may also keep `"commitments"` in
that plan: promises they made or accepted, in their own words, shown back to
them like the rest of the plan. So the existing plan carries goals, next
steps, obstacles and commitments. Planned activities are the intentions
above, held by the world, and impressions of others are the existing
`opinions`. There is no second planning system, and nothing in a plan ever
executes by itself.

`when` accepts `in 45m`, `in 2h`, `14:00`, `2pm`, `evening`, `tomorrow 09:00`,
`tomorrow morning`, `day 3 08:30`. An intention changes nothing in the world.
When it comes due, the agent is activated with
`you asked to reconsider now (i2): approach Rook about financing my stall`
and decides afresh: proceed, postpone (schedule another), change course, or
drop it. Pending intentions are listed in the agent's situation with their ids
so it can cancel them. They live in the world file, so they survive resume.
Limits: at least `min_intention_minutes` ahead (sooner requests are moved out
to the minimum, so no stream of immediate wakeups), at most
`max_intention_hours` ahead, `max_pending_intentions` per agent, and
`max_intentions_per_activation`.

The existing **persistent plan** (`-plan`) and **dual actions**
(`-dual-actions`) are reused unchanged and are on by default for worlds.
The plan is scoped to the world rather than to one session, so it carries
across resumes.

### The town

Deliberately small. Places: a **square** (with a notice board), a **market**,
a **workshop**, and one **home** per resident. Walking between any two takes
15 minutes. Location matters mechanically: `say` is heard only by people in
the same place, handing things over and building require being there, and
each place offers different work. A `whisper` is a private note that reaches
its recipient anywhere.

Two resources: **credits** and **materials**. Ways to get them, all bounded
per day and reset at dawn:

| action | where | effect |
|---|---|---|
| `work` | market | 3-hour shift, +6 credits (4 shifts/day across everyone) |
| `buy` | market | materials at 3 credits each (8/day) |
| `salvage` | workshop | 2 hours, +2 materials (6/day) |

Everything else is between residents:

| action | |
|---|---|
| `go` | walk somewhere |
| `give` | hand credits/materials to someone in the same place |
| `offer` / `accept` / `decline` | propose a trade to anyone; nothing moves until accepted, and both sides are re-checked at that moment; offers expire after a day |
| `start_project` | declare a project and what it needs (credits, materials, labor hours; bounded); at most 2 unfinished per owner |
| `contribute` | put resources into any project, from anywhere (capped at what it still needs) |
| `build` | put 2 hours of labor into a project at its place (no double-booking the last hours) |
| `post` | pin a notice at the square, for whoever comes by |
| `arrange` / `cancel_meeting` | arrange a meeting with others at a place and time; invitees are told, and everyone is reminded when it comes. Only the arranger can call it off |
| `withdraw` | take back your own open offer |
| `wait` / `pass` | do nothing for now |

Speech is the `say` / `whisper` communication slot (dual actions), and
scheduling is the `intentions` field, so an agent can talk, act and schedule
in one decision. Observing the current place needs no action: every
activation shows where you are, who is there, and what the place offers,
and arriving somewhere is itself an activation.

A completed project is public news, stays at its place permanently, and pays
its owner 2 credits every dawn: a persistent consequence that outlives the
session. Nothing in the world prescribes what is worth doing. There is no
"town goal", no project suggested by the environment, and no reward but what
the economy itself provides.

`configs/world-agents.yaml` reuses the established personalities, voices
and traits from `configs/agents.yaml` verbatim. Only `goals` change, into
broad dispositions: Mira is drawn to commerce, Dov to hands-on technical
work, Ada to community, Rook to trades and influence. `configs/world.yaml`
adds modest starting differences under `per_agent`: resources, a starting
place, and a few lines of `knows`. These are true facts and existing
familiarity ("you did a repair job for Rook last year and he paid late"),
delivered once, privately, as the resident's first observations.

**Claims are not facts.** In an agent's own view, events the world confirmed
are marked `✔` (`✔ Mira (mira) gave Rook 2 credits.`). Speech, notes and
notices are never marked, however they are phrased (`Mira (mira) said: I gave
you 10 credits.`). Offers say nothing moves until accepted, arranged meetings
are shown as pending, and intentions are notes to self.

**The world is authoritative.** Validation checks every action against real
state and explains refusals ("nobody else is at the market to hear you",
"Rook is not here", "you only have 4 credits"). The agent's situation
section is always ground truth for what it can see. Its memory and its
beliefs about others can be wrong, and in practice they are: agents
routinely claim a project is "finished" when it is still unfunded. Nothing
an agent believes or says changes state; only validated actions do.

### Logs and metrics

The terminal is a chronological record:

```
[Day 1 07:21] Rook wakes — a quiet moment; nothing in particular prompted it
[Day 1 07:21] Rook → Dov (private): "Hey Dov, I've got 18 credits and I know you still think I owe you an apology. …"
  ↳ Dov will decide at 07:27 (a private note from Rook)
[Day 1 07:21] Rook » started a 3-hour shift at the stalls.

[Day 1 07:27] Dov wakes — a private note from Rook
[Day 1 07:27] Dov → Rook (private): "I don't accept your apology offer. You owe me 10 credits for last year's repair. …"
  ↳ Rook will decide at 10:21 (a private note from Dov)
```

(Rook's reply waits until 10:21 because he is mid-shift.) `-log verbose`
adds damped and coalesced triggers, notes, and impression updates. Plan
changes print at normal level.

Each session writes a normal JSONL log (`run_start` with a `world` block,
`agent`, `event`, `decision`, `llm_call` with token counts and latency,
`rejected`, `outcome`, `summary`). Every record carries `sim_stamp`. World
mode adds four record types: `activation` (simulated time and the coalesced
reasons), `scheduled` (what each trigger did: the activation time, or
damped/coalesced), `plan` (an agent's plan after each change, or `null` when
abandoned), and `memory` (notes and impression updates). `playworld world
log` renders all of these. The session outcome reports, among others:

- `activated_by_<reason>`: what drives activity (social vs. idle vs.
  self-scheduled)
- `action_<verb>`, `rejected`, `speech`, `messages`, `trips`
- `intentions_set` / `_due` / `_cancelled`, `appointments` / `_due`
- `projects_started` / `_completed`, `contributions`, `trades`, `gifts`
- `triggers_coalesced`, `reactions_damped`, `activations_deferred`
- `credits_earned`, `materials_bought` / `_salvaged`, `credits_given`,
  `credits_traded`, `credits_contributed` (and materials equivalents),
  `labor_hours`
- `invalid_cooloffs`, `events`, `llm_calls_per_sim_day`
- `per_agent`: the same counters per resident, including transfers in and
  out, plus activations by reason

The summary table adds plan updates, notes and impression updates per
agent, alongside calls, tokens and latency.

These are counts of what happened. Nothing in the harness labels
relationships ("trust", "rivalry") from dialogue. The traces are kept so
that analysis can be done separately.

The world file keeps cumulative `totals`, a per-session record, and the last
400 events, so `world inspect` works without opening a log.

```bash
# why were agents woken this session?
jq -r 'select(.type=="activation") | "\(.activation.stamp) \(.activation.agent): \(.activation.reasons|join("; "))"' \
  .playworld/worlds/town/logs/town-s001-*.jsonl
```

### Grounded cognition (`cognition: grounded`)

A four-day `qwen3.5:9b` run showed agents losing track of the world rather
than choosing badly within it. Half of the 47 refused actions were `say` to
an empty room. Rook tried to buy a trade with `give` and the town later
called it a sale. Mira described a project with its labor done and its
materials missing as finished. The same lines about "the pile" filled most
of every agent's 30-line context. The authoritative scrap count was there,
but buried mid-line in alphabetized key/value state. `cognition: grounded`
(the default for new worlds) changes what agents are shown. It does not
change what they may decide:

- **An ordered, authoritative situation.** It has four parts: YOU
  (location, holdings, your unfinished projects with what each still needs,
  your open offers vs. offers made to you, offers settled in the last day,
  what actually changed hands — gift, trade, contribution, refund — and
  your meetings and pending intentions), HERE (who is present and what they
  are visibly doing, today's stock and scrap), OTHER PEOPLE'S PROJECTS (the
  public record), and NOT POSSIBLE RIGHT NOW (say with nobody here, the
  project or offer limit, what you can afford, the place you are already
  at). Nothing from another resident's private state appears in it. The
  world keeps a ledger of completed transfers for this, and every settled
  offer records when it closed.
- **Legible actions.** Each verb's description names the valid ids and
  amounts (`accept: o14, o16`, `contribute: p8 needs 2 materials`, `go:
  square, market, home_rook`). Moves that cannot succeed are left out
  rather than rewritten. A new `abandon` verb lets an owner give up an
  unfinished project, and contributions are refunded. Refusals carry a
  stable code (`nobody_here`, `offer_closed`, `own_offer`, `project_limit`,
  ...). Accepting your own offer gets its own explanation.
- **Plans with an age.** The plan shows when it was written and how many
  ✔ events have happened since. It may carry `"done"` steps, and an
  intention may name the one it `"replaces"`. A new intention whose note
  matches a pending one supersedes it instead of adding a second wakeup.
  Superseded and fired intentions stop being shown or scheduled.
- **Typed impressions.** `opinions` may be about a topic ("the scrap pile")
  as well as a person. Their basis is one of saw / heard / inferred /
  suspect, and they are rendered as dated views, "not facts". Nothing
  checks or corrects them.
- **Repetition without loops.** In the agent's context, a line that repeats
  an earlier one from the same source is shown once, at its latest
  occurrence, with "(3 times, first Day 3 11:04)". Lines that differ in any
  number are never folded, and a claim is never folded with a ✔ event. In
  the world, a remark a listener already heard from the same speaker in the
  last 12 hours, or an offer on terms already put to them in the last day,
  is delivered but does not wake them again. New content, new numbers and
  new listeners still do. Both rules use deterministic word-overlap
  (`internal/textsim`): no extra model call, and no keyword lists.
- **Salient recall.** Consequential events (transfers, settled offers,
  project milestones) are marked by the world. Up to six of them that have
  scrolled out of the transcript are still shown, each once. The transcript
  itself is capped at 24 lines.
- **Slot repair.** A `say`/`whisper` sent as the `world_action`, with the
  communication slot empty, is accepted as the communication. It is counted
  as `slot_repairs`.

`cognition: legacy` (and any world created before this setting, such as an
existing `town`) reproduces the original observation, prompt and triggers
exactly. Repeats are still *counted* there, so the two modes can be
compared. `-cognition legacy|grounded` sets the mode for a new world, or
overrides it for one session of an existing world without writing it back.

The grounded prompt is about the same size. On the four-day town it is
~4.85k tokens per call against ~4.7k for legacy, because compaction and the
24-line cap pay for the richer situation. On a brand-new world it is about
400 tokens larger, mostly fixed system text. Calls per simulated day are
unchanged by design.

**Seeing what an agent saw.**

```bash
# the exact prompt a resident would get if it decided now (read-only):
playworld world inspect town -agent rook -prompt
playworld world inspect town -agent rook -prompt -cognition grounded   # same moment, other mode

# log every activation's view, recalled memory, prompt and reply:
playworld world run exp1 -fresh -introspect -max-calls 50
playworld world log exp1 -introspect -agent ada
```

The introspection record (`view`) holds what the agent was shown. Beside it
is a separate `privileged` snapshot of everyone's location and holdings,
which is for the experimenter and is never part of any prompt.

**Before/after experiments.**

```bash
scripts/cognition-experiment.sh                    # 2 pairs × (legacy, grounded), 1 day, 50 calls each
PAIRS=3 DAYS=2 CALLS=100 scripts/cognition-experiment.sh
playworld world compare town cogexp-legacy-1 cogexp-grounded-1
```

The script creates fresh worlds from the same config, roster, model, seed
and call budget, interleaves the modes, and ends with `world compare`.
`world compare` prints objective counters side by side: calls and tokens
per simulated day, refusals by code (recovered from the message wording
for older sessions), slot repairs, repeated remarks and offers, triggers
suppressed or damped, intentions set and superseded, plan updates,
projects, contributions, trades and gifts, and LLM calls per consequential
change. Sampling makes single runs noisy, so compare several pairs.

### Consequences, goal continuity and persistent obstacles

A second four-day grounded run produced real emergent cooperation: Ada
invented a Community Garden Bed, Mira helped, and Rook unexpectedly gave the
last materials. It also showed agents misreading their own moves:

- **Speech.** 22 of 36 refusals were still `say` to an empty room. The
  system prompt listed `say` first unconditionally and headed the roster
  "Present". Each refused say also cost a "refused" re-activation.
- **Offers.** Models put the *price* into `want_credits`, giving offers with
  credits on both sides ("16 credits for 20 credits and 3 materials").
- **Projects.** Agents never saw their own completed projects, and the
  completed list read "(Ada, square)". Mira planned to "collect dividends" at
  the square from Ada's garden, and Ada told everyone they'd all earn from it.
- **Plans and intentions.** They kept steps about a project after it was
  finished.

Grounded worlds now also show the following. All of it is derived from
state, and none of it chooses for the agent:

- **What a move would do.**
  - Every open offer, yours or made to you, has a `→` line: whether you can
    do your part now, what you would be left holding, "you would get nothing
    back" or "you would give nothing for it", and a net figure when the same
    resource moves both ways ("credits go both ways: net, you gain 4").
  - `accept` lists only offers you can pay for, each spelled out ("o19 (you
    give 1 materials, you get 2 credits)"); the others appear under NOT
    POSSIBLE. The world still checks both sides at acceptance.
  - The offer description says `give_*` leaves your hands and `want_*` comes
    from them. Offer and accept events say who hands over what. An accepted
    one-way offer reads as one-way ("gave nothing back (the offer asked for
    nothing in return)"), in the event and in the record of what changed
    hands.
  - Unfavourable but legal deals are untouched.
- **Who owns and who is paid.**
  - YOU lists your completed projects: each pays you, its owner, N credits
    every dawn, automatically (nothing to collect). It shows when the next
    payment is and how much each has paid so far.
  - Others' projects are "Ada's project". Their completed list says each pays
    its owner alone, and shows what you put in. The section opens with a line
    saying contributing gives no share, no say and no income.
  - `contribute` and `build` name whose project each is. The brief states the
    rule. No mechanics changed: income has always gone to the owner only.
- **Why you came.** `go` takes an optional private `"for"`. If the model
  leaves it out, the reply's own `reasoning` is kept as the purpose (counted
  as `trip_purpose_from_reasoning`; the destination is never touched). On
  arrival the wake-up reason and YOU say where you walked from and what you
  said you came to do.
- **What you last saw elsewhere.** A glimpse of the market's and workshop's
  stock as of your last visit, flagged when it predates today's restock. It
  is a memory, not a live view.
- **Your refused actions in the last day**, folded by reason ("3 times since
  Day 4 06:05").
- **Speech availability.** When nobody is with you, a line next to the
  actions says a `say` would be refused. The roster is headed "People in
  town" and points to HERE for who is with you. `reasoning` asks for what
  you expect the move to lead to (useful in introspection).
- **Intentions and plans that refer to something that changed.**
  - A pending intention whose note names a project or offer (by id or by
    name) that has since been completed, abandoned or settled says so, both
    in YOU and in its wake-up reason when it comes due. It still comes due,
    and the agent decides what it means now.
  - The plan says how long its goal has been held ("since Day 3 08:02,
    through 7 of your decisions") and how long each obstacle has lasted, both
    carried across rewrites that keep (by word overlap) the same goal or
    obstacle.
  - It lists consequential ✔ events since the plan was written that concern
    something it mentions, ending with "Steps that rested on them may now be
    done, pointless, or still worth doing; that is yours to judge".
  - When the same obstacle has lasted through several decisions under the
    same goal, one plain sentence says so and leaves "keep at it, wait, come
    at it some other way, or put it aside" to the agent. It names no remedy,
    and an agent with no plan, or no obstacle, is never nudged.

None of this adds a model call. On the four-day town the prompt grows by
roughly 400 tokens per call (about 9%; the town averaged 4.66k measured
tokens per call).

**New counters**, shown by `world compare` (`?` for sessions that predate
them):

- refusals of actions that were never offered vs. ones that were;
- the same refusal reason again within a day;
- after a setback, the same kind of move vs. a different one;
- goal changes, plans dropped, steps marked done;
- plans shown related changes or a lasting obstacle;
- intentions that came due after their subject closed;
- trips with a stated purpose;
- offers re-made on declined terms;
- offers whose note cites a project that was already finished.

**Inspecting one decision.** `world log <world> -introspect -session N
-activation K` prints activation #K (numbered in each "wakes #K" line):

- what the agent saw: its view, plus the experimenter-only truth;
- what it recalled: its plan, with goal age, obstacles and related changes,
  and its impressions;
- its reply, including reasoning and expectation;
- what it chose and whether the world accepted it;
- the events that followed.

### What was reused, and what changed elsewhere

Reused as-is: `agent.Agent` (personas, memory, dual actions, planning, belief
provenance, decode/retry), `memory.FileStore` (one directory per world),
`llm` providers (Ollama, fake), `explog` sinks and tally, and the
orchestrator's event phrasing (`orchestrator.RenderFor`). All other
scenarios and the orchestrator behave exactly as before. Additive changes:

- `sim.ActionIntend`, `sim.Observation.Stamp`, `sim.Turn.Clock`: a turn can
  speak in simulated time instead of "round N of M".
- `agent.Options.Intentions`, `agent.Options.RecallOwnNotes` and
  `Agent.ResumeTranscript()`: a resumed agent's recent transcript is rebuilt
  from its persisted episodes (`memory.Episode.Stamp`), and its own notes
  from earlier in the world are shown in every prompt, so continuity
  survives a restart. `Agent.OnPlan` / `OnMemory` hooks feed the log.
- `memory.Plan.Commitments` (optional, shown only to world agents).
- `llm.Sampling.ContextWindow` (`num_ctx` in YAML, sent to Ollama). World
  prompts run about 3k tokens, so `configs/world-agents.yaml` sets 8192.
- `explog` record types `activation`, `scheduled`, `plan`, `memory`, plus
  `Record.SimStamp` and `RunInfo.World`. The console renders world sessions
  chronologically.

### Limits of this MVP

- Memory scope: identity comes from the persona. Social memory
  (impressions) and world memories persist per world under that world's
  directory, and are never shared between worlds, so a fresh world is a
  fresh experiment. The plan is per world, and recent context is the
  bounded transcript. Nothing carries over between worlds; long-lived
  identity across worlds is not modelled.
- Memory retrieval is recency plus world-marked salience (grounded mode):
  the recent transcript (repeats folded), up to six consequential older
  events, and the agent's own notes, plan and impressions. There is no
  reflection or relevance retrieval, and beliefs are free text: nothing
  links an impression to the events it rests on, so revising a belief when
  contrary evidence arrives is left entirely to the model.
- The repeat rules are word-overlap heuristics. A reworded repeat can slip
  through, and two different remarks with the same words and numbers can
  be folded. They only ever affect wakeups and prompt layout, never what is
  delivered or remembered.
- `qwen3:4b` almost never fills in the optional `plan` (0 plans in the
  smoke tests, against many `remember` notes), and about 1 reply in 10 is a
  truncated JSON object that costs a retry.
- Small models repeat themselves, announce things instead of doing them,
  and misremember. The harness surfaces this (the world disagrees with their
  claims) rather than correcting it.
- There are no needs (hunger, sleep) and no movement between rooms within a
  place. Everyone sleeps through the night by scheduling policy, not by choice.
- One world runs at a time, with one sequential inference stream.

---

## Adding things

### A new agent

Add an entry to `configs/agents.yaml`. No Go code.

```yaml
agents:
  - id: sable
    name: Sable
    role: an archivist who says very little and forgets nothing
    personality: >
      Quiet, watchful, dry. Would rather observe three exchanges than take part
      in one, and when she speaks it is usually to point out a contradiction
      nobody else noticed.
    speech_style: Spare and exact. Quotes people back to themselves.
    goals:
      - Understand what each participant actually wants before saying anything.
      - Note any contradiction between what someone said and what they did.
    traits:
      candour: high
      warmth: low
    model:                    # optional; anything unset inherits `defaults`
      temperature: 0.7
      # model: qwen3:8b       # this agent alone can run a different model
```

Then `playworld run social -only sable,mira,dov`.

The `goals` are private: the harness never reveals them to another agent, and
the prompt frames them as the agent's own business.

### A new LLM provider

Implement `llm.Provider` and register it. Nothing else changes — agents select a
provider by name from their config.

```go
// internal/llm/openai/openai.go
type Provider struct{ /* … */ }

func (p *Provider) Name() string { return "openai" }

func (p *Provider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
    // Translate req.Messages and req.Model.Sampling into your API's shape.
    // Fill in Usage, Latency and StopReason on the way back.
    // Return errors wrapping llm.ErrUnavailable / llm.ErrModelMissing where
    // they apply, so the orchestrator can tell "retry" from "give up".
}

func (p *Provider) Preflight(ctx context.Context, m llm.ModelConfig) error {
    // Cheap reachability + model-exists check. Return an actionable error.
}
```

Register it in `cmd/playworld/run.go` next to the others, then set
`provider: openai` on an agent. `llm.Request.Format` and `JSONSchema` are there
for backends with structured output; the decoder is forgiving enough to cope
without it.

### A new environment

Implement `sim.Environment` and register it from an `init`:

```go
func init() {
    scenario.Register(scenario.Info{
        Name: "auction", Title: "Sealed-bid auction",
        Description:   "…",
        MinAgents:     2,
        MaxAgents:     6,
        DefaultRounds: 5,
        Params: []scenario.ParamDoc{
            {Name: "reserve", Default: "50", Doc: "Minimum acceptable bid."},
        },
    }, newAuction)
}
```

The four things worth getting right:

- **`Brief(id)`** is where private instructions go. Whatever you put here, only
  that agent sees. Keep each agent's incentives out of `PublicState`.
- **`Actions(id)`** should reflect the *current* phase. An agent shown only
  `claim` and `pass` will submit a claim; one shown everything will talk.
- **`Validate`** errors are read by the model, so write them as instructions
  ("claim must be between 0 and 100, got 140"), not as diagnostics.
- **`Visibility`** on each returned event is the privacy boundary. `sim.PublicTo()`
  for everyone, `sim.Only(ids...)` for a restricted audience — and remember to
  include the actor if it should see its own action echoed.

---

## Memory and logs

Both live under `.playworld/` in the working directory.

```
.playworld/memory/mira.json              one file per agent
.playworld/logs/social-20261011-073505.jsonl
```

### Memory

One JSON file per agent, readable with `cat`:

```json
{
  "agent_id": "rook",
  "name": "Rook",
  "episodes": [
    {"run_id": "social-…", "round": 2, "kind": "observation",
     "text": "Dov (dov) said: I need numbers or a specific plan."},
    {"run_id": "social-…", "round": 3, "kind": "note",
     "text": "Mira is trying to balance the room", "private": true}
  ],
  "beliefs": {
    "dov": {"note": "calculating but paralysed by a lack of information",
            "updated": "2026-10-11T11:23:32Z", "updates": 2}
  },
  "commitments": [],
  "runs": ["social-20261011-072150", "social-20261011-073505"]
}
```

The layers are kept distinct:

| layer | where it lives |
|---|---|
| identity and personality | `configs/agents.yaml` — configuration, not state |
| short-term conversational context | rebuilt each run, bounded by `recent_observations` |
| episodic memory | `episodes`, only what the agent was permitted to observe |
| beliefs about other agents | `beliefs`, written when the agent emits `"opinions"` |
| goals and commitments | `goals` in config; `commitments` in memory |

Beliefs and notes come from optional fields on the agent's own action
(`"remember"`, `"opinions"`) — so an agent decides what to remember about whom,
rather than the harness deciding for it.

Retrieval is deliberately simple: beliefs, open commitments, and the most recent
`recall_episodes` episodes **from previous runs** (the current run is already in
the transcript). `memory.Store` is a four-method interface, so swapping in
SQLite, summarisation, reflection or relevance ranking means replacing one
implementation, not touching the agent.

```bash
playworld reset                 # clear every agent's memory
playworld reset rook mira       # clear two
playworld reset -logs           # also delete the experiment logs
playworld run social -fresh     # wipe just this run's cast first
```

### Experiment logs

One JSONL file per run. Every line is an independent JSON object with a `type`,
a `run_id`, a monotonic `seq` and a timestamp, flushed immediately so an
interrupted run still leaves a usable log.

| `type` | carries |
|---|---|
| `run_start` | scenario, seed, every limit, scenario params, harness version |
| `agent` | id, name, role, persona, goals, full model config, episodes recalled |
| `round_start` | round number |
| `event` | what happened, who acted, who it was visible to |
| `decision` | the action an agent proposed, and whether it was accepted |
| `llm_call` | provider, model, attempt, prompt/output tokens, latency, errors |
| `rejected` | the refused action and the environment's reason |
| `error` | inference or harness failure, with the stage it happened in |
| `outcome` | summary, scores, metrics |
| `summary` | rounds, stop reason, totals, per-agent accounting |

Have a look:

```bash
# the conversation, in order
jq -r 'select(.type=="event") | "[\(.round)] \(.event.actor // "—"): \(.event.text)"' \
  .playworld/logs/*.jsonl

# who whispered to whom
jq -r 'select(.event.type=="whisper") | "\(.event.actor) → \(.event.targets|join(","))"' \
  .playworld/logs/*.jsonl

# inference cost
jq -s '[.[] | select(.type=="llm_call")] | {calls: length,
        prompt: (map(.call.prompt_tokens)|add), output: (map(.call.output_tokens)|add),
        p95_ms: (map(.call.latency_ms)|sort|.[(length*0.95)|floor])}' \
  .playworld/logs/*.jsonl

# actions a model proposed that the environment refused
jq -r 'select(.type=="rejected") | "\(.rejected.agent): \(.rejected.reason)"' \
  .playworld/logs/*.jsonl
```

Prompts and raw replies are logged only at `-log debug`, except that a reply
which *failed to parse* is always kept — that is precisely when you need to see
what the model actually said.

Scheduling and scenario setup are deterministic for a given `-seed` (turn order,
generated private thresholds). LLM output is not: `seed` is passed through to
Ollama where supported, but a local model is not reproducible in practice.

---

## Tests

```bash
go test ./...            # no Ollama required
```

The suite uses `llm.Fake` and a per-agent scripted provider, so nothing needs a
live model. It covers agent state isolation, public vs. private message
visibility end to end, orchestrator scheduling and every termination path,
environment action validation, memory persistence and retrieval across sessions,
provider error classification, and all the run limits — plus, per scenario:
resource conservation and validated transfers, commitment recording versus
fulfillment, sealed-vote resolution and ties, proposal feasibility and safety
under a changing repair cost, and that a fresh store starts isolated while a
shared store stays longitudinal and still varies its hidden facts across seeds.

It also covers this round of changes specifically: the `do do` decode fix at
both layers (prompt rendering and decode-time recovery); `-plan`'s on/off
default, persistence across turns, omit/replace/abandon semantics, that a
decode retry never adopts a plan from the failed attempt, and that a plan is
scoped to its own run (`internal/agent/agent_test.go`'s "persistent planning"
and "belief provenance" sections, `internal/memory/memory_test.go`'s plan
tests); `commitOrUpdate`'s commitment-identity rule (repeat vs. revise vs.
create, and the metrics built on it); `grounding=basic|rich`'s promised-vs-
transferred and proposed/approved/completed fields
(`internal/scenario/community_test.go`'s "world-state grounding" section);
and that `Options.Planning`/`Options.BeliefProvenance` actually reach the
live agents and the run summary end to end
(`internal/runner/runner_test.go`).

The world mode has its own deterministic tests (`internal/world/world_test.go`,
`internal/runner/world_test.go`, `internal/agent/intentions_test.go`). They cover
event ordering and tie-breaks, the clock jumping to the window's end, agent wakeups
(scheduled, fired, cancelled, horizon-clamped), activities completing on time,
meetings (notify, remind, cancel, no back-to-back resets), message delivery and
whisper privacy, local speech boundaries, movement, resource conservation through
gifts, trades and contributions, validation never changing state, coalescing,
cooldown, night and daily caps, reply-chain damping, idle backoff, invalid-action
cool-off, call and event budgets, save/load and two-session resume (the queue is
unchanged, a pending wakeup fires, the transcript and plan survive, and a shift in
flight pays exactly once), memory isolation between worlds and between agents,
running on the built-in fake provider, and identical histories for identical
scripted runs.

Grounded cognition has its own tests (`internal/world/cognition_test.go`,
`internal/world/consequence_test.go`, `internal/agent/cognition_test.go`,
`internal/agent/goals_test.go`, `internal/runner/cognition_test.go`).
Most are built from failure patterns in the four-day run. They check that an
agent sees its own holdings exactly and none of another's private state, and
that a gift, a proposed offer and a completed trade read differently. They
check that a project with its labor done but materials missing stays open
with its remaining needs shown to everyone, that the project limit is
visible before anyone tries, and that abandoning refunds contributors.
They check that accepted, declined and expired offers are never offered as
actionable (and that accepting your own offer is explained), and that the
same terms re-offered are shown as a repeat and do not wake anyone while new
terms do. They check that revised intentions supersede the old one so only
one wakeup fires, that a repeated-line loop yields at most two reactive
activations (seven in legacy mode), and that new content, changed numbers,
new listeners and a remark repeated half a day later all still wake people.
The remaining tests cover transcript folding (never across speakers, never
a claim with a ✔ event, never across different numbers), salient recall,
plan age and `done` steps, typed impressions, slot repair, that the preview
is exactly the prompt sent and writes nothing, that the zero setting
reproduces the legacy prompt byte for byte, that old world files load as
legacy, that grounded state survives save/load, refusal codes, and that a
session-only cognition override is never saved. The consequence and goal
tests check:

- offers: each side's give and get and the holdings left; unaffordable
  accepts withheld and named; stale own offers marked; two-sided offers summed
  and still legal; a free offer recorded as one-way and a gift as a gift;
- projects: owner, contributor stake and income shown to the right people,
  income paid to the owner alone, completed projects closed to contribution,
  labor-only projects still open;
- intentions about a finished project noted (and counted when due), unrelated
  ones left alone;
- arrival reminders, and walks that keep their destination when given a
  purpose from reasoning;
- remembered glimpses and refusals that age out;
- the refusal and strategy counters;
- goal and obstacle ages across rewrites;
- the reconsideration note: only under its thresholds, naming no remedy,
  never for an agent without a plan;
- related-change pointers;
- dropping a finished goal keeps its history;
- role-swapped events not folded together.

```bash
go test -race ./... && go vet ./...
```

There is one live smoke test, skipped by default:

```bash
PLAYWORLD_OLLAMA_TEST=1 go test ./internal/runner -run Ollama -v
PLAYWORLD_OLLAMA_TEST=1 PLAYWORLD_OLLAMA_MODEL=qwen3:8b go test ./internal/runner -run Ollama -v
```

It runs a real two-agent `split` negotiation and asserts what can be asserted
about a non-deterministic model: that inference happened, tokens and latency
were recorded, and the agents produced actions the environment accepted.

You can also drive the whole harness with no model at all, which is the fastest
way to check a new scenario's rules and plumbing:

```bash
go run ./cmd/playworld run split -provider fake -no-memory
```

---

## Suggested evaluation procedure

For comparing Experiments A–D (or any two configurations) properly:

1. **Isolated memory per cell.** Give every configuration its own `-memory`
   directory (or just use `-no-memory` / `-fresh` for single-shot comparisons)
   — a shared store would let a later run recall an earlier one's outcome,
   which is exactly the contamination [`murder`'s design notes](#c-murder--a-murder-and-a-liar-in-the-room)
   warn about.
2. **Multiple seeds per configuration, not one.** A single run's result is one
   sample from a model that is not deterministic and from scenario ground
   truth that varies with the seed. Run each configuration (A, B, C, D) across
   the same small set of seeds — `-seed 1`, `-seed 2`, `-seed 3` — and compare
   distributions, not single outcomes. Keep the seed set identical across
   configurations so each comparison is apples-to-apples.
3. **Read `action_selection` and the scenario's own metrics from the JSONL
   log**, not just the terminal summary — `jq -s '[.[] | select(.type=="summary")] | .[0].summary.action_selection'`
   gives the world-action/communication breakdown for one run; loop that over
   every log in a `-memory`/`-logs` directory to build a table across seeds
   and configurations.
4. **Report distributions of the metrics that matter for the hypotheses**:
   `action_selection.turns_with_world_action` (and `turns_with_neither`) for
   Hypothesis B; `first_proposal_day` and `proposals_feasible_detail` for
   whether the affordances in Hypothesis A actually mattered; everything else
   in `community`'s `commitments_detail` / `contributions_total_credits` for
   whether any of it translated into consequential action at all. A
   configuration that moves `turns_with_world_action` up but doesn't move
   `contributions_total_credits` is informative in its own right — it would
   say the interface stopped being the bottleneck and something else (plausibly
   Hypothesis C) took over.
5. **For the grounding/planning A–D set**, the comparable signals are
   `proposals_safe_detail` (did any proposal's *final* budget stay safe
   against the *final* cost — a deterministic claim-vs-reality check,
   available for every proposal, not only the winner) and, per agent in
   `PerAgent[]`, `has_plan` / `plan_revisions` / `beliefs_with_provenance`.
   A config that raises `plan_revisions` without moving
   `proposals_revised_after_complication` or `proposals_safe_detail` would
   say agents are revising their private notes but not the one action that
   actually matters — exactly the gap the live trial under *Affordances and
   dual actions fix the interface, not the planning* (in Known limitations,
   below) found.
6. **Keep the live-model trials small.** A handful of seeds per configuration
   (not dozens) is enough to see whether a change moved the needle at all on a
   laptop; treat anything that looks promising as a candidate for a larger,
   deliberately-scheduled batch later, not something to run inline.

---

## Known limitations

These are deliberate for an MVP, and each is a seam rather than a wall.

- **Sequential inference only.** One agent at a time. `Orchestrator.takeTurn` is
  the single place that would need to change.
- **Recency-based retrieval.** No summarisation, reflection or relevance
  ranking. `memory.Episode` already carries an `Importance` field and
  `memory.Store` is narrow, so a better policy drops in behind the interface.
- **No cross-run reflection.** An agent recalls episodes from previous sessions
  verbatim; it never consolidates them.
- **Commitments are not enforced.** `social`/`community` record what agents
  committed to; nothing makes them honour it later. `split`/`community`
  measure broken promises and fulfillment rate rather than preventing them —
  which is the interesting half.
- **Small models need the structured channel forced, and the more actions a
  scenario offers, the more forcing it needs.** See *Speech is not action*
  above. `community`'s six-verb action space needed both a dedicated lock-in
  day and an explicit "there is no proposal yet" nudge before a 9B model
  would reliably formalize what it had spent several rounds discussing: it
  would otherwise `support` or `contribute` toward a proposal id that was
  never actually created. If you write a scenario with a large action
  vocabulary, budget for this.
- **`community`'s private flavor facts are not independently verifiable.**
  Only the structural repair cost is ground truth the environment checks
  against; Mira's market insight, Ada's demand figures, and the existence of
  Rook's outside funding are taken on faith by other agents, by design — the
  scenario is about selective disclosure of a mix of checkable and
  uncheckable claims, not a second hidden mechanical rule.
- **One provider implemented.** The interface is designed for more; only Ollama
  is written.
- **Affordances and dual actions fix the interface, not the planning.** In a
  live trial with rich affordances and dual actions both on, a 9B model
  proposed a project on day 1, actually invoked `inspect`, and made real
  `contribute`s totalling 58 credits — a real behavior change from the
  original single-action, static-affordance baseline (0 contributions, 0
  inspections, the one proposal arriving on day 6). But it proposed a budget
  of 5 credits before knowing the real cost was 21, and even after `inspect`
  revealed that number two rounds later, never revised the proposal
  (`proposals_revised` stayed 0) — the vote it won failed as unsafe. That
  gap is Hypothesis C by design: this round of changes targeted the
  interface and the scheduling, not the model's ability to notice new
  information should change an earlier decision. Expect to still see this
  class of failure with these fixes in place, and treat it as the next
  hypothesis to test, not a bug in this one.
- **Persistent planning does not enforce follow-through.** `-plan` gives an
  agent somewhere to keep a strategy, and the run summary reports whether
  one stayed active and how often it was revised, but nothing checks that
  this turn's action actually matches `next_steps` — by design, so a model
  can abandon a plan the moment it stops fitting what it observed, rather
  than being forced to mechanically execute a stale one. Whether planning
  actually closes the gap described above (a model noticing new information
  but not revising its committed course) is exactly what Experiments A–D's
  config C is for — it is a hypothesis this harness lets you test, not one
  it answers by existing.
- **Belief provenance is self-reported, not verified.** `-beliefs` lets an
  agent say *why* it believes something and *how* confident it is, but the
  harness never checks a `basis` against what the agent actually observed —
  an agent can claim "I watched them do it" about something it was only
  told. This is deliberate (provenance is a grounding signal for prompting
  and analysis, not a lie detector), but it means `beliefs_with_provenance`
  measures whether agents volunteer provenance, not whether that provenance
  is accurate.
- **`-plan` (and, to a lesser extent, `-beliefs`) lengthens replies, which can
  starve a small model's output budget.** Confirmed in a live `qwen3:4b`
  smoke test: a full `"plan"` object pushed two of four replies in one run
  past the roster's default `max_tokens: 400` (`configs/agents.yaml`),
  truncating valid JSON mid-plan and costing a retry. The retry recovered
  cleanly both times — the model simply dropped the optional `"plan"` field
  and resent a shorter, complete reply, exactly as the one-retry budget is
  meant to handle — so this did not fail either run, but it does mean
  `-plan` trades some retry budget for the extra field, more so on smaller
  models. Raise the roster's `max_tokens` if retries from this show up often
  in a real comparison.
