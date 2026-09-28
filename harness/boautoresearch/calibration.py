"""Calibration-round statistics: σ from baseline replicates, and rung validation and choice."""
import math
import statistics

RHO_MIN, SPREAD_SIGMAS, CONFIGS = 0.8, 3, 5
N_FACTOR = 2 * (1.96 + 0.84) ** 2  # two-sample n per arm at α=0.05, power 0.8: N·σ²/effect²


def sigma(values: list[float]) -> float:
    """Sample sd of the replicates; 0 when they all agree within float tolerance."""
    if all(math.isclose(v, values[0], rel_tol=1e-9, abs_tol=1e-12) for v in values):
        return 0.0
    return statistics.stdev(values)


def configs(baseline: dict, rng) -> list[dict]:
    """CONFIGS diverse configs: a Latin hypercube over a box around each lever's baseline."""
    # ponytail: box = baseline ± max(|b|, 1)/2 because levers.json carries no ranges yet;
    # use the registered hypotheses' lever boxes once they exist.
    cols = {}
    for name, b in baseline.items():
        u = [(i + rng.random()) / CONFIGS for i in range(CONFIGS)]
        rng.shuffle(u)
        if isinstance(b, bool):
            cols[name] = [x >= 0.5 for x in u]
        elif isinstance(b, (int, float)):
            w = max(abs(b), 1) / 2
            cols[name] = [(round if isinstance(b, int) else float)(b - w + 2 * w * x) for x in u]
        else:
            cols[name] = [b] * CONFIGS
    return [{n: cols[n][i] for n in baseline} for i in range(CONFIGS)]


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(xs: list, ys: list) -> float | None:
    """Spearman ρ over the pairs where both trials finished; None when undefined."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    try:
        return statistics.correlation(rx, ry)
    except statistics.StatisticsError:  # a constant side
        return None


def _spread(objectives: list) -> float:
    done = [o for o in objectives if o is not None]
    return max(done) - min(done) if done else 0.0


def _trials_needed(sd: float, effect: float) -> float:
    if sd == 0:
        return 1.0
    return max(1.0, N_FACTOR * sd**2 / effect**2) if effect > 0 else math.inf


def choose(reference: dict, rungs: list[dict]) -> tuple[dict, bool]:
    """Validate each rung against the reference and pick the cheapest to a verdict.

    Rows carry fidelity, objectives (per config, None if failed), sigma and cost_s; this adds
    spread, rho, passed and expected_cost_s. δ is not set during R0, so the effect is the
    suggested δ = 2σ at the reference, scaled to a rung by its spread relative to the reference's.
    Returns (chosen fidelity, fallback), fallback meaning a ladder was given and no rung passed.
    """
    delta = 2 * reference["sigma"]
    reference["spread"] = _spread(reference["objectives"])
    candidates = [(reference["cost_s"] * _trials_needed(reference["sigma"], delta), 0, reference)]
    for i, r in enumerate(rungs, 1):
        r["spread"] = _spread(r["objectives"])
        r["rho"] = spearman(r["objectives"], reference["objectives"])
        r["passed"] = (r["rho"] is not None and r["rho"] >= RHO_MIN
                       and r["spread"] >= SPREAD_SIGMAS * r["sigma"])
        if r["passed"]:
            effect = delta * r["spread"] / reference["spread"]
            candidates.append((r["cost_s"] * _trials_needed(r["sigma"], effect), i, r))
    for cost, _, row in candidates:
        row["expected_cost_s"] = cost if math.isfinite(cost) else None
    chosen = min(candidates, key=lambda c: c[:2])[2]
    return chosen["fidelity"], bool(rungs) and len(candidates) == 1
