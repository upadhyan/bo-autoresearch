"""Which earlier trials carry into a round's study (`optimizer` copies them in).

A trial joins round r's warm-start set only if its measurement still means the same thing in
round r's search space; the first rule it fails is the one it is counted under.
"""

from __future__ import annotations

import math
from typing import Any

from boar import stats, store
from boar.optimizer import build_frozen_trials, distributions, quiet_optuna  # noqa: F401 - moved there, still importable here

RULE1, RULE2, RULE3 = "rule1_state", "rule2_outside_space", "rule3_fixed"


def allowed(lever: dict, value: Any) -> bool:
    """Whether `value` is a point of the lever's distribution."""
    kind = lever["type"]
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "categorical":
        return any(type(value) is type(c) and value == c for c in lever["choices"])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if kind == "int" and not float(value).is_integer():
        return False
    return lever["low"] <= value <= lever["high"]


def _fixed_after(hyps: list[dict], trial_round: int) -> set[str]:
    """Levers of hypotheses marked `fix` in round `trial_round` or later, i.e. after the trial ran."""
    out: set[str] = set()
    for h in hyps:
        if any(d.get("decision") == "fix" and int(s) >= trial_round for s, d in (h.get("decisions") or {}).items()):
            out.update(lever["name"] for lever in h["levers"])
    return out


def _has_metric(t: dict) -> bool:
    m = t.get("metric")
    return isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m)


def _rule(t: dict, hyps: list[dict], space: dict[str, dict], defaults: dict[str, Any]) -> str | None:
    """The first rule `t` fails, or None if it joins the warm-start set."""
    if t["state"] not in ("complete", "infeasible") or not _has_metric(t):
        return RULE1
    return _space_rule(t, hyps, space, defaults)


def _space_rule(t: dict, hyps: list[dict], space: dict[str, dict], defaults: dict[str, Any]) -> str | None:
    """Rule 2 or 3 if `t`'s config no longer means the same thing in `space`, else None."""
    moved = store.non_default(t["config"], defaults)
    outside = any(name not in space for name in moved)
    if outside or not all(allowed(lever, t["config"].get(n, defaults[n])) for n, lever in space.items()):
        return RULE2
    if set(moved) & _fixed_after(hyps, t["round"]):
        return RULE3
    return None


def select(trials: list[dict], hyps: list[dict], round_r: int) -> tuple[list[dict], dict[str, int]]:
    """The warm-start set for round `round_r` over the current search space, and the exclusion counts."""
    space = store.search_space(hyps)
    defaults = store.lever_defaults(hyps)
    counts = {RULE1: 0, RULE2: 0, RULE3: 0}
    valid: list[dict] = []
    for t in trials:
        if t["round"] >= round_r:
            continue
        rule = _rule(t, hyps, space, defaults)
        if rule:
            counts[rule] += 1
        else:
            valid.append(t)
    return valid, counts


def incumbent(trials: list[dict], hyps: list[dict], round_r: int, direction: str) -> dict | None:
    """`stats.incumbent` over round `round_r`'s warm-start set.

    Eligibility is judged on every trial that still means the same thing in the space, failed ones too:
    a config that crashed when measured again can't win on its earlier repeats.
    """
    space, defaults = store.search_space(hyps), store.lever_defaults(hyps)
    valid, _ = select(trials, hyps, round_r)
    history = [t for t in trials if t["round"] < round_r and not _space_rule(t, hyps, space, defaults)]
    return stats.incumbent(valid, defaults, direction, history)
