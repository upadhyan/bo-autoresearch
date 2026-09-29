"""Subagent records: one schema per kind, and the quote check against verdict records.

Pure functions; the CLI turns their ValueError into a refusal.
"""
import re
from decimal import Decimal

from . import directives, hypotheses

# a decimal in prose; a leading '-' is its sign only after a non-word character ("0.1-0.5" is a range)
# ponytail: version-like text ("Python 3.11") reads as a decimal too; exempt it if analysts trip on it
DECIMAL = re.compile(r"(?<![\w.])[-−]?\d*\.\d+(?:[eE][+-]?\d+)?")
REMOVED, UNTESTED = ("rejected", "inconclusive", "parked"), ("proposed", "registered")


def _hypothesis(d: dict, key: str, hyps: dict, where: str = "", statuses: tuple = ()) -> str:
    """An existing hypothesis id (in one of `statuses`, if given)."""
    h = d.get(key)
    if not isinstance(h, str) or h not in hyps:
        raise ValueError(f"{where}{key}: no hypothesis {h!r}")
    if statuses and hyps[h]["status"] not in statuses:
        raise ValueError(f"{where}{key}: {h} is {hyps[h]['status']}, not {' or '.join(statuses)}")
    return h


def _text(d: dict, key: str, where: str, one_line: bool = False) -> None:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{where}{key} is required and must be non-empty text")
    if one_line and "\n" in v.strip():
        raise ValueError(f"{where}{key} must be one line")


def _choice(d: dict, key: str, options: tuple, where: str = "") -> None:
    if d.get(key) not in options:
        raise ValueError(f"{where}{key} must be {', '.join(options[:-1])} or {options[-1]}")


def _ids(d: dict, key: str, known, where: str = "", what: str = "", nonempty: bool = False) -> None:
    v = d.get(key)
    if not isinstance(v, list) or not all(isinstance(i, str) for i in v) or (nonempty and not v):
        raise ValueError(f"{where}{key} must list {what} ids" + (" (at least one)" if nonempty else ""))
    if unknown := [i for i in v if i not in known]:
        raise ValueError(f"{where}{key}: no {what} {', '.join(unknown)}")


def _texts(d: dict, key: str) -> None:
    v = d.get(key)
    if not isinstance(v, list) or not all(isinstance(s, str) and s.strip() for s in v):
        raise ValueError(f"{key} must list non-empty text items ([] for none)")


def _proposal(r: dict, st: dict) -> list[str]:
    hs = r.get("hypotheses")
    if not isinstance(hs, list):
        raise ValueError("hypotheses must list the proposed hypothesis specs ([] when your lane holds nothing new)")
    if not hs:
        return []  # nothing new in this lane: how a final pass shows exhaustion
    for i, s in enumerate(hs):
        try:
            hypotheses.validate(s)
        except ValueError as e:
            raise ValueError(f"hypotheses[{i}]: {e}")
    if len({s["mechanism"].strip() for s in hs}) < 2:
        raise ValueError("hypotheses must span at least 2 mechanisms")
    return []


def _review(r: dict, st: dict) -> list[str]:
    hyps = st["hypotheses"]
    h = _hypothesis(r, "hypothesis", hyps)
    _choice(r, "directive_verdict", ("allow", "prune", "deprioritize"))
    if r["directive_verdict"] != "allow" or r.get("directive") is not None:
        _text(r, "directive", "")  # the directive id behind a prune or deprioritize
    _choice(r, "intent", ("fits", "stretch", "off-intent"))
    _choice(r, "conflict", ("shared-lever", "exclusive", "rival", "none"))
    _ids(r, "conflict_with", hyps, what="hypothesis", nonempty=r["conflict"] != "none")
    if r["conflict"] == "none" and r["conflict_with"]:
        raise ValueError("conflict_with must be [] when conflict is none")
    if h in r["conflict_with"]:
        raise ValueError(f"conflict_with: {h} can't conflict with itself")
    _text(r, "rationale", "")
    if not isinstance(r.get("strict"), bool):
        raise ValueError("strict must be true or false: whether this is a strict re-review")
    registry = {d["id"]: d for d in st["registry"]["directives"]}
    if r.get("directive") is not None:
        d = registry.get(r["directive"])
        if d is None:
            raise ValueError(f"directive: no directive {r['directive']!r} in the registry")
        need = {"prune": "prohibited", "deprioritize": "discouraged"}.get(r["directive_verdict"])
        if need and d["severity"] != need:
            raise ValueError(f"directive: a {r['directive_verdict']} verdict names a {need} directive, "
                             f"and {d['id']} is {d['severity']}")
    if (hits := similar_to(hyps[h], st)) and not r["strict"]:
        raise ValueError(f"strict: {h}'s mechanism is similar to {', '.join(hits)}: laundering is "
                         "caught by mechanism, so this needs a strict re-review (strict: true)")
    return []


def similar_to(h: dict, st: dict) -> list[str]:
    """Prohibited directives and pruned hypotheses (other numbers) whose statement or mechanism is
    close to H's mechanism."""
    mech = h["spec"]["mechanism"]
    return ([f"prohibited directive {d['id']}" for d in st["registry"]["directives"]
             if d["severity"] == "prohibited" and directives.similar(mech, d["statement"])]
            + [f"pruned {x['id']}" for x in st["hypotheses"].values() if x["status"] == "pruned"
               and x["number"] != h["number"] and directives.similar(mech, x["spec"]["mechanism"])])


def _interplay(r: dict, st: dict) -> list[str]:
    hyps, verdicts = st["hypotheses"], _verdicts(st)
    if [r.get("removed"), r.get("newcomer")].count(None) != 1:
        raise ValueError("exactly one of removed (a removal's review) or newcomer (at registration) "
                         "must name a hypothesis")
    # a removal is checked against the untested list; a newcomer against past removals
    key, (own, partners) = (("removed", (REMOVED, UNTESTED)) if r.get("removed") is not None
                            else ("newcomer", (UNTESTED, REMOVED)))
    h = _hypothesis(r, key, hyps, statuses=own)
    flags = r.get("flags")
    if not isinstance(flags, list):
        raise ValueError("flags must list the flagged pairs ([] for none)")
    for i, f in enumerate(flags):
        where = f"flags[{i}]."
        if not isinstance(f, dict) or set(f) - {"partner", "reason", "cites"}:
            raise ValueError(f"flags[{i}] holds partner, reason and cites only")
        _hypothesis(f, "partner", {k: v for k, v in hyps.items() if k != h}, where, partners)
        _text(f, "reason", where)
        _ids(f, "cites", verdicts, where, "verdict record", nonempty=True)
    return [f["reason"] for f in flags]


def _narrative(r: dict, st: dict) -> list[str]:
    if not isinstance(r.get("round"), int) or isinstance(r["round"], bool) \
            or r["round"] not in st["rounds"]:
        raise ValueError(f"round: no round {r.get('round')!r}")
    _text(r, "text", "")
    # the analyst's narrative must cite the round's verdict records, when it has any
    ran = any(v["round"] == r["round"] for v in _verdicts(st).values())
    _ids(r, "cites", _verdicts(st), what="verdict record", nonempty=ran)
    _texts(r, "diagnostics")
    _texts(r, "suggestions")
    if not isinstance(r.get("generation"), bool):
        raise ValueError("generation must be true or false: whether to request a generation pass")
    return [r["text"], *r["diagnostics"], *r["suggestions"]]


def _expected(r: dict, st: dict) -> list[str]:
    _hypothesis(r, "hypothesis", st["hypotheses"])
    _choice(r, "verdict", ("retain", "reject", "undecided"))
    _text(r, "reason", "", one_line=True)
    return [r["reason"]]


def _distill_spec(r: dict, st: dict) -> list[str]:
    _text(r, "content", "")
    _ids(r, "cites", _verdicts(st), what="verdict record")
    # every retained mechanism rests on its latest verdict record
    last = [h["verdicts"][-1]["id"] for h in st["hypotheses"].values() if h["status"] == "retained"]
    if missing := [v for v in last if v not in r["cites"]]:
        raise ValueError(f"cites must include each retained hypothesis's latest verdict record: {', '.join(missing)}")
    # a keep-or-drop decision, with a reason, for each non-lever change (commit-change, add-dependency)
    changes, owed = r.get("changes"), [c["commit"] for c in st["commits"] if "id" not in c]
    if not isinstance(changes, list) or not all(isinstance(c, dict) and set(c) == {"commit", "decision", "reason"}
                                                for c in changes):
        raise ValueError("changes must list {commit, decision, reason} items, one per non-lever change ([] for none)")
    for i, c in enumerate(changes):
        _choice(c, "decision", ("keep", "drop"), f"changes[{i}].")
        _text(c, "reason", f"changes[{i}].")
    if sorted(c["commit"] for c in changes) != sorted(owed):
        raise ValueError(f"changes must decide each non-lever change exactly once: commits {owed or 'none'}")
    return [r["content"], *(c["reason"] for c in changes)]


TAKEAWAYS = 5  # bullets at most in REPORT.md's Takeaways
VERDICT_ID = re.compile(r"V-R\d+-H\d+\.v\d+-\d+")


def _takeaways(r: dict, st: dict) -> list[str]:
    verdicts = _verdicts(st)
    _ids(r, "cites", verdicts, what="verdict record")
    items = r.get("takeaways")
    if not isinstance(items, list) or not 1 <= len(items) <= TAKEAWAYS:
        raise ValueError(f"takeaways must list 1 to {TAKEAWAYS} bullets")
    for i, t in enumerate(items):
        if not isinstance(t, str) or not t.strip() or "\n" in t.strip():
            raise ValueError(f"takeaways[{i}] must be one line of text")
        if verdicts and not set(VERDICT_ID.findall(t)) & set(r["cites"]):
            raise ValueError(f"takeaways[{i}] must name the verdict record it rests on (one of cites)")
    return items


def _distill_review(r: dict, st: dict) -> list[str]:
    w = st.get("wrapup")
    if not w or not any(x["kind"] == "distill_spec" and x["seq"] > w["seq"] for x in st["records"]):
        raise ValueError("kind: there is no distillation spec to review yet (record distill_spec comes first)")
    _choice(r, "verdict", ("approve", "revise"))
    _text(r, "rationale", "")
    return []


# kind -> (fields, validator returning the text fields whose numbers must be quoted)
KINDS = {
    "proposal": ({"hypotheses"}, _proposal),
    "review": ({"hypothesis", "directive_verdict", "directive", "intent", "conflict", "conflict_with",
                "rationale", "strict"}, _review),
    "interplay": ({"removed", "newcomer", "flags", "quotes"}, _interplay),
    "narrative": ({"round", "text", "cites", "quotes", "diagnostics", "suggestions", "generation"},
                  _narrative),
    "expected": ({"hypothesis", "verdict", "reason", "quotes"}, _expected),
    "distill_spec": ({"content", "cites", "changes", "quotes"}, _distill_spec),
    "takeaways": ({"takeaways", "cites", "quotes"}, _takeaways),
    "distill_review": ({"verdict", "rationale"}, _distill_review),
}
WRAPUP = "wrapup"  # the wrap-up's start (the research incumbent and its config): quotable in wrap-up records


def _verdicts(st: dict) -> dict:
    return {v["id"]: v for h in st["hypotheses"].values() for v in h["verdicts"]}


def _field(record: dict, path: str):
    """A dotted path into a verdict record; lever names hold dots, so the longest key wins."""
    cur, parts = record, path.split(".")
    while parts:
        for j in range(len(parts), 0, -1):
            if isinstance(cur, dict) and (k := ".".join(parts[:j])) in cur:
                cur, parts = cur[k], parts[j:]
                break
        else:
            raise KeyError(path)
    return cur


def _quotes(r: dict, st: dict, kind: str) -> list:
    """Check each structured quote against its verdict record; return the quoted values.

    A quote comes from a record the output cites (an expected verdict cites none: any record). A
    wrap-up record may also quote record "wrapup": the tuned values (`config.<lever>`) and the research
    incumbent (`incumbent.mean`) as the wrap-up's start logged them.
    """
    quotes, verdicts = r.get("quotes", []), _verdicts(st)
    cited = (list(r["cites"]) if "cites" in r else [c for f in r["flags"] for c in f["cites"]]
             if "flags" in r else list(verdicts))
    if kind in ("distill_spec", "takeaways") and st.get("wrapup"):
        verdicts, cited = {**verdicts, WRAPUP: st["wrapup"]}, cited + [WRAPUP]
    if not isinstance(quotes, list):
        raise ValueError("quotes must list {record, field, value} items")
    for i, q in enumerate(quotes):
        where = f"quotes[{i}]"
        if not isinstance(q, dict) or set(q) != {"record", "field", "value"}:
            raise ValueError(f"{where} must hold exactly record, field and value")
        if not isinstance(q["record"], str) or q["record"] not in verdicts:
            raise ValueError(f"{where}.record: no verdict record {q['record']!r}")
        if q["record"] not in cited:
            raise ValueError(f"{where}.record: {q['record']} is quoted but not cited")
        try:
            actual = _field(verdicts[q["record"]], q["field"])
        except (KeyError, TypeError, AttributeError):  # AttributeError: a field that isn't text
            raise ValueError(f"{where}.field: {q['record']} has no field {q['field']!r}")
        if actual != q["value"] or isinstance(actual, bool) != isinstance(q["value"], bool):
            raise ValueError(f"{where}: {q['record']} {q['field']} is {actual!r}, not {q['value']!r}")
    return [q["value"] for q in quotes]


def _matches(text: str, value) -> bool:
    """A decimal in prose quotes a value when it is that value rounded to the digits shown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    t = Decimal(text)
    return abs(Decimal(repr(value)) - t) <= Decimal(5).scaleb(int(t.as_tuple().exponent) - 1)


def batch(kind: str, r) -> bool:
    """A batch of expected verdicts, `{"expected": [item, ...]}`: each item is a record of its own."""
    return kind == "expected" and isinstance(r, dict) and "expected" in r


def items(kind: str, r) -> list:
    """The records a validated body logs: a batch's items, else the body."""
    return r["expected"] if batch(kind, r) else [r]


def validate(kind: str, r, st: dict) -> None:
    """Raise ValueError naming the first failing field of a `kind` record (a batch's item by index:
    one bad item refuses the whole batch)."""
    if not batch(kind, r):
        return _validate(kind, r, st)
    xs, seen = r["expected"], dict[str, int]()
    if set(r) != {"expected"} or not isinstance(xs, list) or not xs:
        raise ValueError("expected must list one or more {hypothesis, verdict, reason} items, alone in the body")
    for i, x in enumerate(xs):
        if batch(kind, x):
            raise ValueError(f"expected[{i}]: a batch item is one expected verdict, not a batch")
        try:
            _validate(kind, x, st)
        except ValueError as e:
            raise ValueError(f"expected[{i}]: {e}")
        if (j := seen.setdefault(x["hypothesis"], i)) != i:
            raise ValueError(f"expected[{i}].hypothesis: {x['hypothesis']} is already at expected[{j}]")


def _validate(kind: str, r, st: dict) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not isinstance(r, dict):
        raise ValueError("the record must be a JSON object")
    fields, check = KINDS[kind]
    if unknown := set(r) - fields:
        raise ValueError(f"unknown {kind} fields: {sorted(unknown)}")
    texts = check(r, st)
    values = _quotes(r, st, kind) if "quotes" in fields else []
    for text in texts:
        for m in DECIMAL.finditer(text):
            num = m.group().replace("−", "-")
            if not any(_matches(num, v) for v in values):
                raise ValueError(f"{m.group()} is not quoted: every decimal must match a quotes value "
                                 "taken from a cited verdict record")
