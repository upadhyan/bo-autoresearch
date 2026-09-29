"""Monte Carlo for docs/research/equivalence-check-threshold.md.

Models harness/boautoresearch/cli.py::_equivalence_check: two configs (baseline, incumbent); each gets
n new replicates on the new commit, and the new mean is compared with the logged mean of m trials
at that config (m_b = 5: the calibration round's k = 5 baseline replicates; m_i = 3: the incumbent's
trial plus CONFIRMATIONS = 2). σ̂ is the sample sd of the k = 5 baseline replicates (df = 4), the same
samples as the baseline's logged mean (independent of it under normality). The check fails if either
config fails. True σ = 1, so shifts are in σ units; δ = 2σ is the harness's suggested δ.

Run: python3 docs/research/equivalence_check_mc.py   (numpy + scipy; ~1 min)
"""
import numpy as np
from scipy import stats

N = 400_000
M_B, M_I = 5, 3
RNG = np.random.default_rng(0)


def draw(df, shift=0.0, bias=0.0, n_max=8):
    """Everything one check might need: logged means, σ̂, and n_max new replicates per config."""
    if df == 4:  # σ̂ from the same 5 baseline replicates that make the logged baseline mean
        base = RNG.standard_normal((N, M_B))
        logged_b, s = base.mean(1), base.std(1, ddof=1)
    else:  # σ̂ pooled over the epoch's replicated configs: df independent of the logged means
        logged_b = RNG.standard_normal(N) / np.sqrt(M_B)
        s = np.sqrt(RNG.chisquare(df, N) / df)
    logged_i = RNG.standard_normal(N) / np.sqrt(M_I) - bias  # winner's curse: logged mean looks better
    new = shift + RNG.standard_normal((2, N, n_max))
    return (logged_b, logged_i), s, new


def diff(logged, new, c, n):
    return new[c, :, :n].mean(1) - logged[c]


def se(n, m):
    return np.sqrt(1 / n + 1 / m)


MS = (M_B, M_I)


def rule_fixed(logged, s, new, n=2, k=2.0, df=None):
    """Current rule (n=2, k=2): fail if |Δ| > k·σ̂ at either config."""
    fail = np.zeros(N, bool)
    for c in (0, 1):
        fail |= np.abs(diff(logged, new, c, n)) > k * s
    return fail, np.full(N, 2 * n)


def rule_alpha(logged, s, new, n=2, alpha=0.01, df=4):
    """Two-sided t test per config at α/2 (Bonferroni over the 2 configs)."""
    q = stats.t.ppf(1 - alpha / 4, df)
    fail = np.zeros(N, bool)
    for c in (0, 1):
        fail |= np.abs(diff(logged, new, c, n)) > q * s * se(n, MS[c])
    return fail, np.full(N, 2 * n)


def rule_tost(logged, s, new, n=2, eps=2.0, alpha=0.05, df=4):
    """TOST (Schuirmann 1987): equivalent iff the 1-2α CI of Δ lies inside ±ε, at both configs."""
    q = stats.t.ppf(1 - alpha, df)
    fail = np.zeros(N, bool)
    for c in (0, 1):
        fail |= np.abs(diff(logged, new, c, n)) + q * s * se(n, MS[c]) >= eps
    return fail, np.full(N, 2 * n)


def rule_two_stage(logged, s, new, n1=2, n2=4, alpha=0.01, df=4):
    """Stage 1 = the current rule. A config that fails it gets n2 more replicates and fails the check
    only if the pooled n1+n2 mean still fails a two-sided t test at α/2."""
    q = stats.t.ppf(1 - alpha / 4, df)
    fail, cost = np.zeros(N, bool), np.full(N, 2 * n1)
    for c in (0, 1):
        again = np.abs(diff(logged, new, c, n1)) > 2 * s
        cost = cost + n2 * again
        fail |= again & (np.abs(diff(logged, new, c, n1 + n2)) > q * s * se(n1 + n2, MS[c]))
    return fail, cost


def run(rule, df=4, bias=0.0, **kw):
    """(false-alarm rate, power at shift 1σ, 2σ, 4σ, mean trials per check at no shift)."""
    out = []
    for shift in (0.0, 1.0, 2.0, 4.0):
        logged, s, new = draw(df, shift, bias)
        fail, cost = rule(logged, s, new, df=df, **kw)
        out.append(fail.mean())
        if shift == 0.0:
            trials = cost.mean()
    return out, trials


def analytic_current(m, df):
    """P(|Δ| > 2σ̂) at one config: Δ/(σ̂·se) ~ t_df, so P(|t_df| > 2/se(2, m))."""
    return 2 * stats.t.sf(2 / se(2, m), df) if df else 2 * stats.norm.sf(2 / se(2, m))


if __name__ == "__main__":
    print("Analytic, current rule, one config:")
    for df in (None, 2, 4, 20):
        pb, pi = analytic_current(M_B, df), analytic_current(M_I, df)
        print(f"  σ {'known' if df is None else f'df={df}'}: baseline {pb:.4f}  incumbent {pi:.4f}  "
              f"either (indep. approx) {1 - (1 - pb) * (1 - pi):.4f}")
    rows = [
        ("current: n=2, 2σ̂", rule_fixed, {}),
        ("(a) n=3, 2σ̂", rule_fixed, {"n": 3}),
        ("(a) n=5, 2σ̂", rule_fixed, {"n": 5}),
        ("(b) n=2, t-test α=1%", rule_alpha, {}),
        ("(b) n=4, t-test α=1%", rule_alpha, {"n": 4}),
        ("(b) n=2, t-test α=5%", rule_alpha, {"alpha": 0.05}),
        ("(c) TOST ε=δ=2σ, n=2", rule_tost, {"eps": 2.0}),
        ("(c) TOST ε=δ=2σ, n=6", rule_tost, {"eps": 2.0, "n": 6}),
        ("(c) TOST ε=δ=4σ, n=2", rule_tost, {"eps": 4.0}),
        ("(d) 2 + 4 if stage 1 fails, α=1%", rule_two_stage, {}),
        ("(d) 2 + 4 if stage 1 fails, α=5%", rule_two_stage, {"alpha": 0.05}),
        ("(d) 2 + 8 if stage 1 fails, α=5%", rule_two_stage, {"n2": 8, "alpha": 0.05}),
    ]
    print("\nrule | σ̂ df | FA | power 1σ | 2σ | 4σ | trials/check")
    for df in (4, 20):
        for name, rule, kw in rows:
            (fa, p1, p2, p4), trials = run(rule, df, **kw)
            print(f"{name} | {df} | {fa:.4f} | {p1:.3f} | {p2:.3f} | {p4:.3f} | {trials:.2f}")
    print("\nWinner's curse: incumbent's logged mean optimistic by 0.5σ (false-alarm rate only)")
    for df in (4, 20):
        for name, rule, kw in (rows[0], rows[3], rows[9], rows[10]):
            (fa, *_), _ = run(rule, df, bias=0.5, **kw)
            print(f"  {name} | df {df} | FA {fa:.4f}")
    print("\nP(≥1 false epoch in 10 no-op checks), σ̂ fixed for the epoch:")
    for df in (4, 20):
        _, s0, _ = draw(df)
        for name, rule, kw in (rows[0], rows[3], rows[9], rows[10]):
            any_fail = np.zeros(N, bool)
            for _ in range(10):  # fresh logged means/new replicates, the same σ̂
                logged, _, new = draw(df)
                any_fail |= rule(logged, s0, new, df=df, **kw)[0]
            print(f"  {name} | df {df}: {any_fail.mean():.3f}")
