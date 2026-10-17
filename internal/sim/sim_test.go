package sim

import "testing"

func TestVisibilityCanObserve(t *testing.T) {
	cases := []struct {
		name string
		vis  Visibility
		id   AgentID
		want bool
	}{
		{"public reaches anyone", PublicTo(), "stranger", true},
		{"audience member sees it", Only("a", "b"), "b", true},
		{"outsider does not", Only("a", "b"), "c", false},
		{"empty audience reaches nobody", Visibility{}, "a", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.vis.CanObserve(c.id); got != c.want {
				t.Errorf("CanObserve(%q) = %v, want %v", c.id, got, c.want)
			}
		})
	}
}

func TestOnlyCopiesAudience(t *testing.T) {
	ids := []AgentID{"a", "b"}
	v := Only(ids...)
	ids[0] = "mutated"
	if !v.CanObserve("a") {
		t.Fatal("Only must copy its audience; mutating the caller's slice changed the visibility")
	}
}

func TestVisibilityDescribe(t *testing.T) {
	if got := PublicTo().Describe(); got != "public" {
		t.Errorf("public describe = %q", got)
	}
	// Sorted, so log output is stable regardless of argument order.
	if got := Only("b", "a").Describe(); got != "private:a,b" {
		t.Errorf("private describe = %q, want private:a,b", got)
	}
	if got := (Visibility{}).Describe(); got != "hidden" {
		t.Errorf("empty describe = %q", got)
	}
}

func TestActionArgCoercion(t *testing.T) {
	a := Action{Args: map[string]any{
		"float": 42.0, "int": 7, "string": " 13 ", "words": "lots", "name": "mira",
	}}
	for _, c := range []struct {
		key  string
		want float64
		ok   bool
	}{
		{"float", 42, true},
		{"int", 7, true},
		{"string", 13, true},
		{"words", 0, false},
		{"missing", 0, false},
	} {
		got, ok := a.ArgFloat(c.key)
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("ArgFloat(%q) = %v, %v; want %v, %v", c.key, got, ok, c.want, c.ok)
		}
	}
	if s, ok := a.ArgString("name"); !ok || s != "mira" {
		t.Errorf(`ArgString("name") = %q, %v`, s, ok)
	}
	if _, ok := a.ArgString("float"); ok {
		t.Error("ArgString must not coerce a number")
	}
}

func TestFatalWrapping(t *testing.T) {
	if Fatal(nil) != nil {
		t.Error("Fatal(nil) must be nil")
	}
	var u *Unrecoverable
	err := Fatal(errAny{})
	if !asUnrecoverable(err, &u) {
		t.Fatal("Fatal must produce an *Unrecoverable")
	}
	if u.Error() != "boom" {
		t.Errorf("Error() = %q", u.Error())
	}
}

type errAny struct{}

func (errAny) Error() string { return "boom" }

func asUnrecoverable(err error, target **Unrecoverable) bool {
	u, ok := err.(*Unrecoverable)
	if ok {
		*target = u
	}
	return ok
}
