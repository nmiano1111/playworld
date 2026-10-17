package world

import (
	"errors"
	"fmt"
	"strings"
)

// Refusal is why the world would not carry out an action: a stable code for
// metrics ("cannot_afford", "offer_closed", ...) and the explanation the
// agent is shown.
type Refusal struct {
	Code string
	Msg  string
}

func (r *Refusal) Error() string { return r.Msg }

func refuse(code, format string, args ...any) error {
	return &Refusal{Code: code, Msg: fmt.Sprintf(format, args...)}
}

// RefusalCode classifies a validation error; errors that are not refusals
// are "other".
func RefusalCode(err error) string {
	var r *Refusal
	if errors.As(err, &r) {
		return r.Code
	}
	return "other"
}

// legacyRefusals maps the fixed wording of refusal messages to their codes,
// for logs written before refusals carried one. Only the harness's own
// messages are matched — never anything an agent wrote.
var legacyRefusals = []struct{ contains, code string }{
	{"nobody else is at", "nobody_here"},
	{"you are already at", "already_there"},
	{"is not something you can do here", "unknown_verb"},
	{"open offers; wait for answers", "offer_limit"},
	{"unfinished projects", "project_limit"},
	{"is already complete", "project_complete"},
	{"no longer open", "offer_closed"},
	{"so the offer cannot go through", "offer_unfunded"},
	{"made to you", "offer_not_found"},
	{"you cannot offer", "cannot_afford"},
	{"and you only have", "cannot_afford"},
	{"you only have", "cannot_afford"},
	{"credits and you have", "cannot_afford"},
	{"to complete the trade", "cannot_afford"},
	{"picked clean", "sold_out"},
	{"is taken for today", "sold_out"},
	{"only has", "sold_out"},
	{"needs no more", "no_labor_needed"},
	{"already being done", "no_labor_needed"},
	{"go there to work on it", "project_elsewhere"},
	{"there is no place called", "no_such_place"},
	{"there is no project", "project_not_found"},
	{"is not here; you can only hand", "not_here"},
	{"you are on your way", "travelling"},
	{"only available at the market", "wrong_place"},
	{"sold at the market", "wrong_place"},
	{"only scrap to salvage at the workshop", "wrong_place"},
	{"notice board is at the square", "wrong_place"},
	{"pending intentions", "intention_limit"},
}

// ClassifyRefusal recovers the code of a refusal message from a log written
// before codes were recorded; "other" if the wording is not recognised.
func ClassifyRefusal(reason string) string {
	for _, r := range legacyRefusals {
		if strings.Contains(reason, r.contains) {
			return r.code
		}
	}
	return "other"
}
