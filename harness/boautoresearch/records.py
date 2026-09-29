"""Subagent records: one schema per kind, and the quote check against verdict records.

Pure functions; the CLI turns their ValueError into a refusal.
"""
import re
from decimal import Decimal

from . import hypotheses

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
    if not isinstance(hs, list) or not hs:
        raise ValueError("hypotheses must list the proposed hypothesis specs")
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
    return []


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
    return [r["content"]]


# kind -> (fields, validator returning the text fields whose numbers must be quoted)
KINDS = {
    "proposal": ({"hypotheses"}, _proposal),
    "review": ({"hypothesis", "directive_verdict", "directive", "intent", "conflict", "conflict_with",
                "rationale", "strict"}, _review),
    "interplay": ({"removed", "newcomer", "flags", "quotes"}, _interplay),
    "narrative": ({"round", "text", "cites", "quotes", "diagnostics", "suggestions", "generation"},
                  _narrative),
    "expected": ({"hypothesis", "verdict", "reason", "quotes"}, _expected),
    "distill_spec": ({"content", "cites", "quotes"}, _distill_spec),
}


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


def _quotes(r: dict, st: dict) -> list:
    """Check each structured quote against its verdict record; return the quoted values.

    A quote comes from a record the output cites (an expected verdict cites none: any record).
    """
    quotes, verdicts = r.get("quotes", []), _verdicts(st)
    cited = (r["cites"] if "cites" in r else [c for f in r["flags"] for c in f["cites"]]
             if "flags" in r else list(verdicts))
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


def validate(kind: str, r, st: dict) -> None:
    """Raise ValueError naming the first failing field of a `kind` record."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not isinstance(r, dict):
        raise ValueError("the record must be a JSON object")
    fields, check = KINDS[kind]
    if unknown := set(r) - fields:
        raise ValueError(f"unknown {kind} fields: {sorted(unknown)}")
    texts = check(r, st)
    values = _quotes(r, st) if "quotes" in fields else []
    for text in texts:
        for m in DECIMAL.finditer(text):
            num = m.group().replace("−", "-")
            if not any(_matches(num, v) for v in values):
                raise ValueError(f"{m.group()} is not quoted: every decimal must match a quotes value "
                                 "taken from a cited verdict record")
