package agent

import (
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/nmiano/playworld/internal/memory"
	"github.com/nmiano/playworld/internal/sim"
)

// decodeAction parses a model reply into a structured action.
//
// Small local models are not reliably well-behaved, so this is forgiving about
// packaging (code fences, leading prose, trailing text) and about synonyms for
// the action kinds — but it never invents an action: if nothing parses, the
// caller gets an error and feeds it back to the model.
func decodeAction(raw string, participants map[string]sim.AgentID, self sim.AgentID) (sim.Action, error) {
	obj, err := extractJSONObject(raw)
	if err != nil {
		return sim.Action{}, err
	}

	var wire struct {
		Kind      string                     `json:"kind"`
		Action    string                     `json:"action"`
		Content   string                     `json:"content"`
		Message   string                     `json:"message"`
		Text      string                     `json:"text"`
		To        json.RawMessage            `json:"to"`
		Target    json.RawMessage            `json:"target"`
		Verb      string                     `json:"verb"`
		Args      map[string]any             `json:"args"`
		Reasoning string                     `json:"reasoning"`
		Remember  string                     `json:"remember"`
		Opinions  map[string]json.RawMessage `json:"opinions"`
	}
	if err := json.Unmarshal([]byte(obj), &wire); err != nil {
		return sim.Action{}, fmt.Errorf("reply was not a JSON object matching the protocol: %w", err)
	}

	kindStr := strings.ToLower(strings.TrimSpace(wire.Kind))
	if kindStr == "" {
		kindStr = strings.ToLower(strings.TrimSpace(wire.Action))
	}

	content := firstNonEmpty(wire.Content, wire.Message, wire.Text)
	to := append(parseIDs(wire.To), parseIDs(wire.Target)...)
	notes, basis, confidence := parseOpinions(wire.Opinions)

	a := sim.Action{
		Content:           strings.TrimSpace(content),
		Verb:              strings.ToLower(strings.TrimSpace(wire.Verb)),
		Args:              wire.Args,
		Reasoning:         strings.TrimSpace(wire.Reasoning),
		Remember:          strings.TrimSpace(wire.Remember),
		Opinions:          notes,
		OpinionBasis:      basis,
		OpinionConfidence: confidence,
	}

	switch kindStr {
	case "say", "speak", "public", "talk", "broadcast":
		a.Kind = sim.ActionSay
	case "whisper", "private", "dm", "private_message", "privately":
		a.Kind = sim.ActionWhisper
	case "do", "act", "action", "env", "move":
		a.Kind = sim.ActionDo
	case "pass", "idle", "none", "nothing", "wait", "silence", "listen", "skip":
		a.Kind = sim.ActionPass
	case "":
		// Infer from shape rather than failing outright: a reply with a verb is
		// an action, one with recipients is a whisper, one with text is speech.
		switch {
		case a.Verb != "":
			a.Kind = sim.ActionDo
		case len(to) > 0:
			a.Kind = sim.ActionWhisper
		case a.Content != "":
			a.Kind = sim.ActionSay
		default:
			return sim.Action{}, fmt.Errorf(`missing "kind": expected one of say, whisper, do, pass`)
		}
	default:
		// A model sometimes puts the environment verb in "kind".
		if _, ok := participants[kindStr]; !ok && a.Verb == "" {
			a.Kind = sim.ActionDo
			a.Verb = kindStr
			break
		}
		return sim.Action{}, fmt.Errorf("unknown kind %q: expected one of say, whisper, do, pass", kindStr)
	}

	if a.Kind == sim.ActionWhisper {
		resolved := make([]sim.AgentID, 0, len(to))
		seen := map[sim.AgentID]bool{}
		for _, name := range to {
			id, ok := participants[strings.ToLower(strings.TrimSpace(string(name)))]
			if !ok {
				return sim.Action{}, fmt.Errorf("no participant named %q to whisper to", name)
			}
			if id == self || seen[id] {
				continue
			}
			seen[id] = true
			resolved = append(resolved, id)
		}
		if len(resolved) == 0 {
			return sim.Action{}, fmt.Errorf(`whisper needs a "to" field naming another participant`)
		}
		a.To = resolved
	}

	return a, nil
}

// dualDecoded is the result of parsing a dual-mode reply: an optional
// communication action and an optional world action, plus the same
// self-reported memory fields the single-action protocol supports.
type dualDecoded struct {
	Communication     *sim.Action
	World             *sim.Action
	Reasoning         string
	Remember          string
	Opinions          map[string]string
	OpinionBasis      map[string]string
	OpinionConfidence map[string]string
}

// decodeDualAction parses a reply in the two-slot protocol: an optional
// "communication" (say/whisper) and an optional "world_action" (a scenario
// verb, or "pass"). Either or both may be null or absent, meaning the agent
// chose not to use that slot. A slot present but malformed is a decode error
// for the whole reply — the retry loop re-asks for the complete object again,
// the same way the single-action protocol does, rather than trying to salvage
// half a reply.
func decodeDualAction(raw string, participants map[string]sim.AgentID, self sim.AgentID) (*dualDecoded, error) {
	obj, err := extractJSONObject(raw)
	if err != nil {
		return nil, err
	}

	var wire struct {
		Communication json.RawMessage            `json:"communication"`
		WorldAction   json.RawMessage            `json:"world_action"`
		Reasoning     string                     `json:"reasoning"`
		Remember      string                     `json:"remember"`
		Opinions      map[string]json.RawMessage `json:"opinions"`
	}
	if err := json.Unmarshal([]byte(obj), &wire); err != nil {
		return nil, fmt.Errorf("reply was not a JSON object matching the protocol: %w", err)
	}
	notes, basis, confidence := parseOpinions(wire.Opinions)

	out := &dualDecoded{
		Reasoning:         strings.TrimSpace(wire.Reasoning),
		Remember:          strings.TrimSpace(wire.Remember),
		Opinions:          notes,
		OpinionBasis:      basis,
		OpinionConfidence: confidence,
	}

	comm, err := decodeCommSlot(wire.Communication, participants, self)
	if err != nil {
		return nil, fmt.Errorf("communication: %w", err)
	}
	out.Communication = comm

	world, err := decodeWorldSlot(wire.WorldAction)
	if err != nil {
		return nil, fmt.Errorf("world_action: %w", err)
	}
	out.World = world

	return out, nil
}

// decodeCommSlot parses the "communication" slot: null/absent/empty means no
// communication this turn.
func decodeCommSlot(raw json.RawMessage, participants map[string]sim.AgentID, self sim.AgentID) (*sim.Action, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var wire struct {
		Type      string          `json:"type"`
		Kind      string          `json:"kind"`
		Message   string          `json:"message"`
		Content   string          `json:"content"`
		Text      string          `json:"text"`
		Recipient string          `json:"recipient"`
		To        json.RawMessage `json:"to"`
		Target    json.RawMessage `json:"target"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("not a JSON object: %w", err)
	}

	kindStr := strings.ToLower(strings.TrimSpace(firstNonEmpty(wire.Type, wire.Kind)))
	content := strings.TrimSpace(firstNonEmpty(wire.Message, wire.Content, wire.Text))

	var to []sim.AgentID
	if wire.Recipient != "" {
		to = append(to, sim.AgentID(strings.TrimSpace(wire.Recipient)))
	}
	to = append(to, parseIDs(wire.To)...)
	to = append(to, parseIDs(wire.Target)...)

	switch kindStr {
	case "", "none", "pass", "null":
		return nil, nil
	case "say", "speak", "public", "talk", "broadcast":
		if content == "" {
			return nil, fmt.Errorf(`"say" needs a non-empty "message"`)
		}
		return &sim.Action{Kind: sim.ActionSay, Content: content}, nil
	case "whisper", "private", "dm", "private_message", "privately":
		if content == "" {
			return nil, fmt.Errorf(`"whisper" needs a non-empty "message"`)
		}
		resolved, err := resolveWhisperTargets(to, participants, self)
		if err != nil {
			return nil, err
		}
		return &sim.Action{Kind: sim.ActionWhisper, Content: content, To: resolved}, nil
	default:
		return nil, fmt.Errorf(`unknown communication type %q: expected "say", "whisper", or null`, kindStr)
	}
}

// decodeWorldSlot parses the "world_action" slot: null/absent/empty or an
// explicit "pass" means no world action this turn. Parameters may arrive
// nested under "args" (matching the single-action protocol) or as flat
// sibling fields (matching the illustrative dual-action examples) — both are
// accepted and merged, so a scenario's existing ActionSpec.Usage strings stay
// correct regardless of which shape a model reaches for.
func decodeWorldSlot(raw json.RawMessage) (*sim.Action, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var typed struct {
		Type string         `json:"type"`
		Verb string         `json:"verb"`
		Args map[string]any `json:"args"`
	}
	if err := json.Unmarshal(raw, &typed); err != nil {
		return nil, fmt.Errorf("not a JSON object: %w", err)
	}
	var flat map[string]any
	_ = json.Unmarshal(raw, &flat) // best-effort; typed unmarshal above already validated the shape

	// A model primed on the single-action protocol's "kind":"do" sometimes
	// copies that literal word into "type" while still (correctly) putting
	// the real verb in "verb" — e.g. {"type":"do","verb":"propose",...}.
	// "do"/"action"/"act" are never themselves legal verbs, so when Type is
	// one of these placeholder words and Verb holds something else, Verb is
	// almost certainly what was actually meant.
	typ := strings.ToLower(strings.TrimSpace(typed.Type))
	vb := strings.ToLower(strings.TrimSpace(typed.Verb))
	if isEnvelopePlaceholder(typ) && vb != "" {
		typ = ""
	}
	verb := strings.ToLower(strings.TrimSpace(firstNonEmpty(typ, vb)))
	switch verb {
	case "", "none", "null":
		return nil, nil
	case "pass":
		return &sim.Action{Kind: sim.ActionPass}, nil
	}

	args := map[string]any{}
	maps.Copy(args, typed.Args)
	for k, v := range flat {
		switch k {
		case "type", "verb", "args", "kind":
			continue
		}
		if _, already := args[k]; !already {
			args[k] = v
		}
	}
	return &sim.Action{Kind: sim.ActionDo, Verb: verb, Args: args}, nil
}

// isEnvelopePlaceholder reports whether s is one of the generic words that
// name the *concept* of a world action rather than any real verb — exactly
// what a model echoes when it confuses the dual protocol's "type" field with
// the single-action protocol's "kind":"do".
func isEnvelopePlaceholder(s string) bool {
	switch s {
	case "do", "action", "act", "world_action", "world", "move", "env":
		return true
	}
	return false
}

// resolveWhisperTargets mirrors decodeAction's whisper-recipient handling:
// drop self-addresses, de-duplicate, and require at least one real recipient.
func resolveWhisperTargets(to []sim.AgentID, participants map[string]sim.AgentID, self sim.AgentID) ([]sim.AgentID, error) {
	resolved := make([]sim.AgentID, 0, len(to))
	seen := map[sim.AgentID]bool{}
	for _, name := range to {
		id, ok := participants[strings.ToLower(strings.TrimSpace(string(name)))]
		if !ok {
			return nil, fmt.Errorf("no participant named %q to whisper to", name)
		}
		if id == self || seen[id] {
			continue
		}
		seen[id] = true
		resolved = append(resolved, id)
	}
	if len(resolved) == 0 {
		return nil, fmt.Errorf(`whisper needs a "recipient" or "to" naming another participant`)
	}
	return resolved, nil
}

// extractPlan parses the optional "plan" field from a model's raw reply,
// independent of decodeAction/decodeDualAction: "plan" is a sibling field on
// the same top-level JSON object in both protocols, so this just re-reads
// that one field rather than changing either decode function's signature or
// return shape. Called only after the reply's action(s) have already decoded
// successfully — a plan is never adopted from a reply the agent is about to
// be asked to redo.
//
// present distinguishes three cases: the field was omitted (present=false,
// existing plan carries forward untouched), it was explicitly null
// (present=true, plan=nil — abandon it), or it was an object (present=true,
// plan populated — replace it outright).
func extractPlan(raw string) (plan *memory.Plan, present bool, err error) {
	obj, err := extractJSONObject(raw)
	if err != nil {
		return nil, false, err
	}
	var wire struct {
		Plan json.RawMessage `json:"plan"`
	}
	if err := json.Unmarshal([]byte(obj), &wire); err != nil {
		return nil, false, err
	}
	if len(wire.Plan) == 0 {
		return nil, false, nil
	}
	if string(wire.Plan) == "null" {
		return nil, true, nil
	}
	var fields struct {
		PrimaryGoal     string   `json:"primary_goal"`
		CurrentStrategy string   `json:"current_strategy"`
		NextSteps       []string `json:"next_steps"`
		Obstacles       []string `json:"obstacles"`
		Commitments     []string `json:"commitments"`
		Done            []string `json:"done"`
	}
	if err := json.Unmarshal(wire.Plan, &fields); err != nil {
		return nil, true, fmt.Errorf(`"plan" was not a JSON object: %w`, err)
	}
	return &memory.Plan{
		PrimaryGoal:     strings.TrimSpace(fields.PrimaryGoal),
		CurrentStrategy: strings.TrimSpace(fields.CurrentStrategy),
		NextSteps:       trimmedNonEmpty(fields.NextSteps),
		Obstacles:       trimmedNonEmpty(fields.Obstacles),
		Commitments:     trimmedNonEmpty(fields.Commitments),
		Done:            trimmedNonEmpty(fields.Done),
	}, true, nil
}

// extractIntentions parses the optional "intentions" and
// "cancel_intentions" fields into sim.ActionIntend actions. Each intention
// becomes {"when","note"} args and each cancellation {"cancel": id}; the
// agent package does not interpret "when" at all — the environment owns time
// and is the one that validates and schedules it. Entries that are not
// usable objects or strings are skipped. A single object (rather than a list)
// is tolerated, since small models often send one.
func extractIntentions(raw string) []sim.Action {
	obj, err := extractJSONObject(raw)
	if err != nil {
		return nil
	}
	var wire struct {
		Intentions json.RawMessage `json:"intentions"`
		Cancel     json.RawMessage `json:"cancel_intentions"`
	}
	if err := json.Unmarshal([]byte(obj), &wire); err != nil {
		return nil
	}
	type entry struct {
		When     string `json:"when"`
		At       string `json:"at"`
		Note     string `json:"note"`
		What     string `json:"what"`
		Replaces string `json:"replaces"`
	}
	var entries []entry
	if len(wire.Intentions) > 0 && string(wire.Intentions) != "null" {
		if err := json.Unmarshal(wire.Intentions, &entries); err != nil {
			var one entry
			if json.Unmarshal(wire.Intentions, &one) == nil {
				entries = []entry{one}
			}
		}
	}
	var out []sim.Action
	for _, e := range entries {
		when := strings.TrimSpace(firstNonEmpty(e.When, e.At))
		if when == "" {
			continue
		}
		args := map[string]any{"when": when, "note": strings.TrimSpace(firstNonEmpty(e.Note, e.What))}
		if r := strings.TrimSpace(e.Replaces); r != "" {
			args["replaces"] = r
		}
		out = append(out, sim.Action{Kind: sim.ActionIntend, Args: args})
	}
	for _, id := range parseIDs(wire.Cancel) {
		out = append(out, sim.Action{Kind: sim.ActionIntend, Args: map[string]any{"cancel": string(id)}})
	}
	return out
}

func trimmedNonEmpty(ss []string) []string {
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		if t := strings.TrimSpace(s); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// parseOpinions accepts each subject's value in either shape — a plain
// string (the existing, unchanged form: {"dov":"blunt"}) or a richer object
// carrying provenance ({"dov":{"note":"blunt","basis":"...","confidence":"..."}}).
// Returning three parallel maps rather than a richer sim.Action.Opinions
// value keeps the existing field's shape — and every existing caller of it —
// completely unchanged; the two provenance maps are empty unless an agent
// actually provided that detail.
func parseOpinions(raw map[string]json.RawMessage) (notes, basis, confidence map[string]string) {
	notes = map[string]string{}
	basis = map[string]string{}
	confidence = map[string]string{}
	for subject, v := range raw {
		var plain string
		if err := json.Unmarshal(v, &plain); err == nil {
			if n := strings.TrimSpace(plain); n != "" {
				notes[subject] = n
			}
			continue
		}
		var rich struct {
			Note       string `json:"note"`
			Basis      string `json:"basis"`
			Confidence string `json:"confidence"`
		}
		if err := json.Unmarshal(v, &rich); err != nil {
			continue // neither shape parsed; skip rather than fail the whole reply
		}
		if n := strings.TrimSpace(rich.Note); n != "" {
			notes[subject] = n
		}
		if b := strings.TrimSpace(rich.Basis); b != "" {
			basis[subject] = b
		}
		if c := strings.TrimSpace(rich.Confidence); c != "" {
			confidence[subject] = c
		}
	}
	if len(notes) == 0 {
		notes = nil
	}
	if len(basis) == 0 {
		basis = nil
	}
	if len(confidence) == 0 {
		confidence = nil
	}
	return notes, basis, confidence
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// parseIDs accepts a string, a list of strings, or null.
func parseIDs(raw json.RawMessage) []sim.AgentID {
	if len(raw) == 0 {
		return nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		if strings.TrimSpace(one) == "" {
			return nil
		}
		// Tolerate "ava, mira".
		var out []sim.AgentID
		for _, part := range strings.Split(one, ",") {
			if p := strings.TrimSpace(part); p != "" {
				out = append(out, sim.AgentID(p))
			}
		}
		return out
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		var out []sim.AgentID
		for _, m := range many {
			if p := strings.TrimSpace(m); p != "" {
				out = append(out, sim.AgentID(p))
			}
		}
		return out
	}
	return nil
}

// extractJSONObject pulls the first balanced top-level JSON object out of a
// reply, ignoring code fences and surrounding commentary.
func extractJSONObject(raw string) (string, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", fmt.Errorf("empty reply")
	}
	// Strip ```json fences.
	if i := strings.Index(s, "```"); i >= 0 {
		rest := s[i+3:]
		if j := strings.Index(rest, "\n"); j >= 0 {
			if head := strings.TrimSpace(rest[:j]); head == "" || !strings.ContainsAny(head, "{}") {
				rest = rest[j+1:]
			}
		}
		if k := strings.Index(rest, "```"); k >= 0 {
			rest = rest[:k]
		}
		s = strings.TrimSpace(rest)
	}

	start := strings.IndexByte(s, '{')
	if start < 0 {
		return "", fmt.Errorf("reply contained no JSON object: %.160s", s)
	}
	depth, inStr, esc := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1], nil
			}
		}
	}
	return "", fmt.Errorf("reply contained an unterminated JSON object: %.160s", s)
}
