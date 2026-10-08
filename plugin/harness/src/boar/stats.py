"""Statistics over trial records: medians, spreads, rank correlation, pooling and the incumbent.

`trials` are trials.jsonl records; `defaults` is `store.lever_defaults(hyps)`. Trials that share
a config (the same non-default levers) are pooled before ranking, so one lucky run can't crown
a config that other runs of it contradict.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from boar import store

BASELINE_KEY = store.config_key({}, {})


def median(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    return statistics.median(vals) if vals else None


def spread(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    return max(vals) - min(vals) if vals else None


def _ranks(xs: list[float]) -> list[float]:
    """1-based ranks; tied values share the average of the ranks they span."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    """Spearman's rank correlation; None when it says nothing (< 3 points or a constant side)."""
    xs, ys = [float(x) for x in xs], [float(y) for y in ys]
    if len(xs) != len(ys):
        raise ValueError("spearman needs paired samples")
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def pooled(trials: Iterable[dict], defaults: dict[str, Any]) -> dict[str, dict]:
    """Complete (feasible) trials grouped by config key, with their repeats pooled.

    Each group: {"key", "config" (non-default levers), "repeats", "trials" (ids), "metric" (median)}.
    """
    groups: dict[str, dict] = {}
    for t in trials:
        if t.get("state") != "complete" or not t.get("repeats"):
            continue
        key = store.config_key(t["config"], defaults)
        g = groups.setdefault(
            key, {"key": key, "config": store.non_default(t["config"], defaults), "repeats": [], "trials": []}
        )
        g["repeats"].extend(t["repeats"])
        g["trials"].append(t["trial"])
    for g in groups.values():
        g["metric"] = median(g["repeats"])
    return groups


def better(direction: str) -> int:
    """Sign that turns a metric into a lower-is-better score."""
    return 1 if direction == "min" else -1


def incumbent(
    trials: Iterable[dict], defaults: dict[str, Any], direction: str, history: Iterable[dict] | None = None,
) -> dict | None:
    """The best pooled group of an eligible config. Ties go to fewer non-default levers, then to the earliest trial.

    A config is eligible when its latest trial among `history` (default: `trials`) is complete: one that
    broke a guard or failed when measured again can't win on its earlier passing repeats, since a guard
    is a correctness condition. The baseline is always eligible: default levers can't break a guard, so
    its guard failure is eval noise. `warmstart.incumbent` passes the failed trials as `history`.
    """
    trials = list(trials)
    latest: dict[str, dict] = {}
    for t in trials if history is None else history:
        key = store.config_key(t["config"], defaults)
        if key not in latest or t["trial"] > latest[key]["trial"]:
            latest[key] = t
    groups = [
        g for g in pooled(trials, defaults).values()
        if g["key"] == BASELINE_KEY or latest.get(g["key"], {}).get("state") == "complete"
    ]
    if not groups:
        return None
    sign = better(direction)
    return min(groups, key=lambda g: (sign * g["metric"], len(g["config"]), min(g["trials"])))


def noise_trials(trials: Iterable[dict], defaults: dict[str, Any]) -> list[dict]:
    """Complete baseline-config trials with at least two repeats: the ones the noise floor is measured on."""
    return [
        t for t in trials
        if t.get("state") == "complete" and len(t.get("repeats") or []) >= 2
        and store.config_key(t["config"], defaults) == BASELINE_KEY
    ]


def noise_floor(trials: Iterable[dict], defaults: dict[str, Any]) -> float | None:
    """Median over baseline trials of each one's own repeat spread; None until one trial has two repeats.

    Within a trial, not across trials: pooling repeats from different rounds would fold machine drift
    into the floor, and the drift check could then never fire. Also None while the median spread is 0: at
    least half those trials repeated bit for bit, and a floor of 0 would count every difference as an effect.
    """
    return median(spread(t["repeats"]) for t in noise_trials(trials, defaults)) or None


def noise_base(trials: Iterable[dict], defaults: dict[str, Any]) -> float | None:
    """The baseline's metric where the noise floor was measured: the median of those trials' medians."""
    return median(median(t["repeats"]) for t in noise_trials(trials, defaults))


def floor_at(floor: float | None, base: float | None, at: float | None) -> float | None:
    """The noise floor scaled from the baseline's metric `base` to `at` (repeat spread grows with the metric).

    A speed task's incumbent can run 20x faster than the baseline with a far smaller spread, so the
    baseline's absolute floor would hide real effects among optimized configs. Scaled only when `base`
    and `at` are nonzero with the same sign (a metric that can cross zero has no scale); else `floor`.
    """
    if floor is None or base is None or at is None or base * at <= 0:
        return floor
    return floor * abs(at) / abs(base)
