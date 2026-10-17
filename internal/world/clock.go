package world

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Epoch is Day 1, 00:00 of every world. Simulated time is an ordinary
// time.Time measured from here, so durations, ordering and JSON encoding come
// for free; only rendering ("Day 3 09:30") knows that the calendar date is
// meaningless.
var Epoch = time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)

// Day returns the 1-based simulated day containing t.
func Day(t time.Time) int {
	return int(t.Sub(Epoch)/(24*time.Hour)) + 1
}

// At returns the instant of hour:minute on the given simulated day.
func At(day, hour, minute int) time.Time {
	return Epoch.Add(time.Duration(day-1)*24*time.Hour + time.Duration(hour)*time.Hour + time.Duration(minute)*time.Minute)
}

// Stamp renders t compactly: "Day 2 09:30".
func Stamp(t time.Time) string {
	return fmt.Sprintf("Day %d %02d:%02d", Day(t), t.Hour(), t.Minute())
}

// Clock renders t for a prompt: "Day 2, 09:30 (morning)".
func Clock(t time.Time) string {
	return fmt.Sprintf("Day %d, %02d:%02d (%s)", Day(t), t.Hour(), t.Minute(), partOfDay(t))
}

func partOfDay(t time.Time) string {
	switch h := t.Hour(); {
	case h >= 5 && h < 12:
		return "morning"
	case h >= 12 && h < 17:
		return "afternoon"
	case h >= 17 && h < 22:
		return "evening"
	default:
		return "night"
	}
}

// ParseSpan parses a session length such as "2d", "36h", "1d12h" or "90m".
func ParseSpan(s string) (time.Duration, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return 0, fmt.Errorf("empty duration")
	}
	if d, err := parseCompound(s); err == nil && d > 0 {
		return d, nil
	}
	return 0, fmt.Errorf("cannot read %q as a duration (try 2d, 36h, 1d12h or 90m)", s)
}

var spanPart = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*(days?|d|hours?|hrs?|h|minutes?|mins?|m)`)

// parseCompound reads "1d12h", "2 hours", "1h30m", "45 minutes".
func parseCompound(s string) (time.Duration, error) {
	matches := spanPart.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return 0, fmt.Errorf("no duration in %q", s)
	}
	// Everything outside the matches must be whitespace or "and", so that
	// "2h of work" is not silently read as 2h.
	rest := spanPart.ReplaceAllString(s, " ")
	rest = strings.ReplaceAll(rest, "and", " ")
	if strings.TrimSpace(rest) != "" {
		return 0, fmt.Errorf("unexpected text in %q", s)
	}
	var total time.Duration
	for _, m := range matches {
		n, _ := strconv.ParseFloat(s[m[2]:m[3]], 64)
		unit := s[m[4]:m[5]]
		switch unit[0] {
		case 'd':
			total += time.Duration(n * float64(24*time.Hour))
		case 'h':
			total += time.Duration(n * float64(time.Hour))
		case 'm':
			total += time.Duration(n * float64(time.Minute))
		}
	}
	return total, nil
}

var (
	clockRe = regexp.MustCompile(`^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$`)
	dayRe   = regexp.MustCompile(`^day\s*(\d+)(?:\s*(?:,|at)?\s*(.+))?$`)
)

// namedTimes are the loose time-of-day words a model reaches for.
var namedTimes = map[string][2]int{
	"morning": {8, 0}, "noon": {12, 0}, "midday": {12, 0}, "afternoon": {14, 0},
	"evening": {18, 0}, "tonight": {19, 0}, "night": {21, 0},
}

// ParseWhen reads an agent's "when" relative to now. Accepted forms:
//
//	"in 45m", "in 2 hours", "1h30m"     a delay from now
//	"14:00", "2pm", "9:30am", "evening" the next time the clock reads that
//	"tomorrow 09:00", "tomorrow morning"
//	"day 3 08:30", "day 3"              an absolute simulated day (08:00 default)
//
// It never returns a time at or before now: a bare clock time that has
// already passed today means tomorrow.
func ParseWhen(s string, now time.Time) (time.Time, error) {
	raw := s
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimPrefix(s, "at ")
	if s == "" {
		return time.Time{}, fmt.Errorf(`"when" is empty`)
	}

	if strings.HasPrefix(s, "in ") || spanPart.MatchString(s) && !strings.Contains(s, ":") && !strings.HasPrefix(s, "day") {
		d, err := parseCompound(strings.TrimPrefix(s, "in "))
		if err == nil && d > 0 {
			return now.Add(d), nil
		}
	}

	if m := dayRe.FindStringSubmatch(s); m != nil {
		day, _ := strconv.Atoi(m[1])
		h, min := 8, 0
		if m[2] != "" {
			var err error
			if h, min, err = parseTimeOfDay(m[2]); err != nil {
				return time.Time{}, fmt.Errorf("cannot read %q: %v", raw, err)
			}
		}
		t := At(day, h, min)
		if !t.After(now) {
			return time.Time{}, fmt.Errorf("%s is already in the past (it is now %s)", Stamp(t), Stamp(now))
		}
		return t, nil
	}

	offset := 0
	switch {
	case strings.HasPrefix(s, "tomorrow"):
		offset = 1
		s = strings.TrimSpace(strings.TrimPrefix(s, "tomorrow"))
		s = strings.TrimSpace(strings.TrimPrefix(s, "at"))
		if s == "" {
			s = "morning"
		}
	case strings.HasPrefix(s, "today"):
		s = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(s, "today"), " at"))
	case strings.HasPrefix(s, "this "):
		s = strings.TrimPrefix(s, "this ")
	}
	h, min, err := parseTimeOfDay(s)
	if err != nil {
		return time.Time{}, fmt.Errorf(`cannot read "when" %q: use "in 2h", "14:00", "tomorrow 09:00" or "day 3 08:30"`, raw)
	}
	t := At(Day(now)+offset, h, min)
	if offset == 0 && !t.After(now) {
		t = t.Add(24 * time.Hour)
	}
	return t, nil
}

func parseTimeOfDay(s string) (hour, minute int, err error) {
	s = strings.TrimSpace(s)
	if hm, ok := namedTimes[s]; ok {
		return hm[0], hm[1], nil
	}
	m := clockRe.FindStringSubmatch(s)
	if m == nil {
		return 0, 0, fmt.Errorf("not a time of day: %q", s)
	}
	hour, _ = strconv.Atoi(m[1])
	if m[2] != "" {
		minute, _ = strconv.Atoi(m[2])
	}
	switch m[3] {
	case "pm":
		if hour < 12 {
			hour += 12
		}
	case "am":
		if hour == 12 {
			hour = 0
		}
	}
	if hour > 23 || minute > 59 {
		return 0, 0, fmt.Errorf("not a time of day: %q", s)
	}
	return hour, minute, nil
}
