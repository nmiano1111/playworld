// Package textsim is a tiny, deterministic notion of "says the same thing":
// word-set overlap plus identical numbers. It is shared by the agent (to fold
// repeated lines in a prompt) and the world (to tell a repeated remark from
// new information) so both sides agree on what a repeat is. It is
// deliberately simple — no stemming, no embeddings — and errs towards
// treating lines as different.
package textsim

import (
	"regexp"
	"strings"
)

var (
	wordRE = regexp.MustCompile(`[a-z]+`)
	numRE  = regexp.MustCompile(`\d+`)
	// refRE matches object ids such as o12, p3 or i4, which differ between
	// otherwise identical offers or notices.
	refRE = regexp.MustCompile(`\b[a-z]\d+\b`)
)

// Print is a line reduced to what matters for comparison.
type Print struct {
	Words   map[string]bool
	Numbers string
}

// Of fingerprints s: its distinct words (two letters or more, ids removed)
// and its numbers in order.
func Of(s string) Print {
	low := refRE.ReplaceAllString(strings.ToLower(s), " ")
	words := map[string]bool{}
	for _, w := range wordRE.FindAllString(low, -1) {
		if len(w) > 1 {
			words[w] = true
		}
	}
	return Print{Words: words, Numbers: strings.Join(numRE.FindAllString(low, -1), ",")}
}

// Same reports whether a and b say the same thing at the given overlap
// threshold (0..1). Any difference in numbers makes them different, and
// very short lines must match exactly.
func Same(a, b Print, threshold float64) bool {
	if a.Numbers != b.Numbers {
		return false
	}
	if len(a.Words) == 0 || len(b.Words) == 0 {
		return len(a.Words) == len(b.Words)
	}
	if min(len(a.Words), len(b.Words)) < 4 {
		threshold = 1
	}
	return Overlap(a, b) >= threshold
}

// Overlap is the Jaccard index of the two word sets.
func Overlap(a, b Print) float64 {
	inter := 0
	for w := range a.Words {
		if b.Words[w] {
			inter++
		}
	}
	union := len(a.Words) + len(b.Words) - inter
	if union == 0 {
		return 1
	}
	return float64(inter) / float64(union)
}

// Thresholds used across the harness: speech tolerates rewording, anything
// else must match almost exactly.
const (
	Speech = 0.7
	Strict = 0.9
)

// minor are words too common to identify anything by themselves.
var minor = map[string]bool{"the": true, "and": true, "for": true, "of": true, "a": true, "an": true, "my": true, "our": true, "new": true}

// Mentions reports whether text refers to the thing called name: either
// name appears in it verbatim (case-insensitive), or most of name's
// distinctive words do ("the garden bed" mentions "Community Garden Bed").
// A one-word name has to appear as that word. It is a heuristic for linking
// a note to a named thing, and errs towards not linking.
func Mentions(text, name string) bool {
	lt, ln := strings.ToLower(text), strings.ToLower(strings.TrimSpace(name))
	if ln == "" {
		return false
	}
	if strings.Contains(lt, ln) {
		return true
	}
	have := map[string]bool{}
	for _, w := range wordRE.FindAllString(lt, -1) {
		have[w] = true
	}
	var key []string
	for _, w := range wordRE.FindAllString(ln, -1) {
		if len(w) > 2 && !minor[w] {
			key = append(key, w)
		}
	}
	if len(key) < 2 {
		return false // a one-word name only counts verbatim (above)
	}
	hit := 0
	for _, w := range key {
		if have[w] {
			hit++
		}
	}
	return hit >= 2 && float64(hit)/float64(len(key)) >= 2.0/3
}

// Refs lists the object ids (o12, p3, ...) that appear in text, in order.
func Refs(text string) []string {
	return refRE.FindAllString(strings.ToLower(text), -1)
}
