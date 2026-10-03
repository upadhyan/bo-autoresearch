"""Which earlier trials carry into a round's study, and how they enter it.

A trial joins round r's warm-start set only if its measurement still means the same thing in
round r's search space; the first rule it fails is the one it is counted under.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import FrozenTrial, TrialState

from boar import stats, store

RULE1, RULE2, RULE3 = "rule1_state", "rule2_outside_space", "rule3_fixed"

# The single constraint every trial carries: 0 feasible, 1 a guard failed. TPE reads it from
# system_attrs ("constraints:guards"), which is where create_trial and Trial.set_constraint put it.
GUARDS = "guards"


def quiet_optuna() -> None:
    """Optuna's per-trial INFO lines and API-churn warnings would drown the worker log."""
    optuna.logging.set_verbosity(logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"optuna(\.|$)")


def distributions(space: dict[str, dict]) -> dict[str, BaseDistribution]:
    out: dict[str, BaseDistribution] = {}
    for name, lever in space.items():
        kind = lever["type"]
        if kind == "bool":
            out[name] = CategoricalDistribution([False, True])
        elif kind == "int":
            out[name] = IntDistribution(int(lever["low"]), int(lever["high"]), log=bool(lever.get("log", False)))
        elif kind == "float":
            out[name] = FloatDistribution(float(lever["low"]), float(lever["high"]), log=bool(lever.get("log", False)))
        elif kind == "categorical":
            out[name] = CategoricalDistribution(list(lever["choices"]))
        else:
            raise ValueError(f"lever {name!r} has unknown type {kind!r}")
    return out


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


def build_frozen_trials(valid: list[dict], space: dict[str, dict], defaults: dict[str, Any]) -> list[FrozenTrial]:
    """Warm trials for `study.add_trials`. Levers new to the space are filled in at their default,
    which is exact: their code didn't exist when the trial ran."""
    dists = distributions(space)
    return [
        optuna.trial.create_trial(
            state=TrialState.COMPLETE,
            value=float(t["metric"]),
            params={name: t["config"].get(name, defaults[name]) for name in space},
            distributions=dists,
            constraints={GUARDS: 0.0 if t["state"] == "complete" else 1.0},
            user_attrs={"boar_trial": t["trial"], "warm": True},
        )
        for t in valid
    ]
