"""Before/after Monte Carlo for the harness's three noise checks (cli._equivalence_check,
cli._advance_wrapup's verification, cli._drift_check), seeds 0..19.

True σ = 1 (shifts in σ units); δ = 2σ, the harness's suggested δ. "before" is the rule up to
v1-followups@67af636: fixed 2-sd thresholds on the calibration round's σ̂ (k = 5 replicates, df = 4).
"after" is the rule this change ships: a t quantile at α = 5% on σ̂ pooled within configs over the
epoch, shown at df = 4 (right after the calibration round) and df = 20 (after a round or two).
Each cell: mean over seeds 0..19 of 20 000 draws each. Run: python3 docs/research/check_rules_mc.py
"""
import numpy as np
from scipy import stats

N, SEEDS, ALPHA = 20_000, range(20), 0.05


def sd_hat(rng, df):
    return np.sqrt(rng.chisquare(df, N) / df)


def equivalence(rng, rule, df, shift):
    """Baseline (m = 5 logged: R0's replicates) and incumbent. before: m = 3 (its trial + 2
    confirmations), n = 2, fail if |Δ| > 2σ̂ at either. after: m = 2 (its replicates only), stage 1 as
    before, a failing config gets 4 more and fails if |Δ₆| > t(1 − α/4, df)·σ̂·√(1/6 + 1/m)."""
    if rule == "before":  # σ̂ from the same 5 baseline replicates as the baseline's logged mean
        base = rng.standard_normal((N, 5))
        logged, s, ms = [base.mean(1), rng.standard_normal(N) / np.sqrt(3)], base.std(1, ddof=1), (5, 3)
    else:
        logged, s, ms = [rng.standard_normal(N) / np.sqrt(5), rng.standard_normal(N) / np.sqrt(2)], sd_hat(rng, df), (5, 2)
    q = stats.t.ppf(1 - ALPHA / 4, df)
    fail, cost = np.zeros(N, bool), np.full(N, 4.0)
    for c in (0, 1):
        new = shift + rng.standard_normal((N, 6))
        first = np.abs(new[:, :2].mean(1) - logged[c]) > 2 * s
        if rule == "before":
            fail |= first
            continue
        cost += 4 * first
        fail |= first & (np.abs(new.mean(1) - logged[c]) > q * s * np.sqrt(1 / 6 + 1 / ms[c]))
    return fail, cost


def verification(rng, rule, df, shift):
    """r = 3 replicates per branch, fail if |gap| > 2σ̂·√(2/3). after: a failing gap gets 3 more
    distilled replicates and fails if |gap₆| > t(1 − α/2, df)·σ̂·√(1/6 + 1/3)."""
    s = sd_hat(rng, df)
    research, distilled = rng.standard_normal((N, 3)).mean(1), shift + rng.standard_normal((N, 6))
    first = np.abs(distilled[:, :3].mean(1) - research) > 2 * s * np.sqrt(2 / 3)
    if rule == "before":
        return first, np.full(N, 6.0)
    q = stats.t.ppf(1 - ALPHA / 2, df)
    return first & (np.abs(distilled.mean(1) - research) > q * s * np.sqrt(1 / 6 + 1 / 3)), 6 + 3.0 * first


def drift(rng, rule, df, shift):
    """One reference trial per config; the proxy says the incumbent is better. Broken iff the
    reference reverses that by more than 2√2·σ̂ (before) or t(1 − α, df)·√2·σ̂ (after). `shift` is the
    true reversal at the reference (0: the configs tie there, the least favourable faithful case)."""
    s = sd_hat(rng, df)
    d_ref = -shift + np.sqrt(2) * rng.standard_normal(N)  # > 0 agrees with the proxy
    k = 2 if rule == "before" else stats.t.ppf(1 - ALPHA, df)
    return (d_ref < 0) & (np.abs(d_ref) > k * np.sqrt(2) * s), np.full(N, 2.0)


if __name__ == "__main__":
    print("check | rule | σ̂ df | false alarm | power δ/2 | power δ | trials/check (no shift)")
    for name, check in (("equivalence", equivalence), ("verification", verification), ("drift", drift)):
        for rule, df in (("before", 4), ("after", 4), ("after", 20)):
            cells, cost = [], []
            for shift in (0.0, 1.0, 2.0):
                rates = []
                for seed in SEEDS:
                    fail, c = check(np.random.default_rng(seed), rule, df, shift)
                    rates.append(fail.mean())
                    if shift == 0.0:
                        cost.append(c.mean())
                cells.append(np.mean(rates))
            print(f"{name} | {rule} | {df} | {cells[0]:.3f} | {cells[1]:.2f} | {cells[2]:.2f} | {np.mean(cost):.2f}")
