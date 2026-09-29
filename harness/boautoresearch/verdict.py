"""The fixed reject form's statistics: a noise-aware verdict GP (numpy/scipy, outside Optuna), and
from its joint latent posterior sample paths Δ (best with the group free minus best with it at
baseline, the other levers optimised in both), M_u (the total-effect range: the largest change in
the objective as the group moves, over settings of the other levers) and, as telemetry, √V_T
(unnormalised group total-order Sobol variance, Jansen).

Inputs are scaled to [0, 1] (log levers in log space, categoricals by option index); the output is
oriented so larger is better and standardised. Everything is seeded by the caller.
"""
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize
from scipy.stats import qmc

from .hypotheses import options

N_SOBOL, N_CANDIDATES, N_SAMPLES, RESTARTS = 128, 128, 256, 3
# M_u's grid: N_GRID + 2 settings of the other levers × N_GRID + 1 of the group
# ponytail: a fixed 16 × 16 grid (plus each candidate against its baseline slice) under-reads a range
# confined to a corner; refine locally from the top rows if `irrelevant` labels prove too generous
N_GRID = 16
N_GRID_POINTS = (N_GRID + 2) * (N_GRID + 1)
FIT_MAX_Z2, HOMOGENEITY_MAX, HOMOGENEITY_MIN_TRIALS = 2.0, 3.0, 8


def encode(lv: dict, value) -> float:
    if lv["kind"] in ("float", "int"):
        lo, hi, v = lv["low"], lv["high"], value
        if lv.get("log"):
            lo, hi, v = math.log(lo), math.log(hi), math.log(v)
        return (v - lo) / (hi - lo)
    opts = options(lv)
    return next(i for i, o in enumerate(opts) if type(o) is type(value) and o == value) / max(len(opts) - 1, 1)


def decode(lv: dict, u: float):
    if lv["kind"] in ("float", "int"):
        lo, hi = lv["low"], lv["high"]
        if lv.get("log"):
            x = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            x = lo + u * (hi - lo)
        return min(max(round(x), lo), hi) if lv["kind"] == "int" else x
    opts = options(lv)
    return opts[min(int(u * len(opts)), len(opts) - 1)]


def _snap(space: dict, U: np.ndarray) -> np.ndarray:
    """Quasi-random unit points onto the levers' own grids (ints and options), re-encoded."""
    out = U.copy()
    for j, lv in enumerate(space.values()):
        if lv["kind"] != "float":
            out[:, j] = [encode(lv, decode(lv, u)) for u in U[:, j]]
    return out


def _kernel(A, B, ls, var):
    d2 = np.zeros((len(A), len(B)))
    for j in range(A.shape[1]):  # one lever at a time: no (len A, len B, D) array
        d2 += ((A[:, None, j] - B[None, :, j]) / ls[j]) ** 2
    s = math.sqrt(5) * np.sqrt(d2)
    return var * (1 + s + s * s / 3) * np.exp(-s)


class GP:
    """Matern-5/2 ARD GP, hyperparameters by maximising the log marginal likelihood (L-BFGS-B, a few
    seeded restarts), the noise variance floored at `noise_floor` (standardised units)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, noise_floor: float, rng: np.random.Generator):
        self.X, self.y = X, y
        n, D = X.shape
        floor = max(noise_floor, 1e-6)
        lo = np.r_[np.full(D, math.log(1e-2)), math.log(1e-3), math.log(floor)]
        hi = np.r_[np.full(D, math.log(1e2)), math.log(1e2), math.log(max(10.0, 2 * floor))]

        def nll(theta):
            try:
                L = self._chol(theta)
            except np.linalg.LinAlgError:
                return 1e10
            a = cho_solve(L, y)
            return 0.5 * y @ a + np.log(np.diag(L[0])).sum() + 0.5 * n * math.log(2 * math.pi)

        starts = [np.r_[np.zeros(D), 0.0, max(math.log(0.1), lo[-1])]]
        starts += [rng.uniform(lo, hi) for _ in range(RESTARTS - 1)]
        best = min((minimize(nll, s, method="L-BFGS-B", bounds=list(zip(lo, hi))) for s in starts),
                   key=lambda r: r.fun)
        self.theta = best.x
        self.L = self._chol(self.theta)
        self.alpha = cho_solve(self.L, y)

    def _parts(self, theta):
        D = self.X.shape[1]
        return np.exp(theta[:D]), math.exp(theta[D]), math.exp(theta[D + 1])

    def _chol(self, theta):
        ls, var, noise = self._parts(theta)
        K = _kernel(self.X, self.X, ls, var) + (noise + 1e-9) * np.eye(len(self.X))
        return cho_factor(K, lower=True)

    def loo(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out residuals and predictive variances, closed form."""
        Kinv = cho_solve(self.L, np.eye(len(self.X)))
        d = np.diag(Kinv)
        return self.alpha / d, 1 / d

    def samples(self, P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Joint posterior samples of the latent function at P: (n, len(P))."""
        ls, var, _ = self._parts(self.theta)
        Ks = _kernel(P, self.X, ls, var)
        mean = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        cov = _kernel(P, P, ls, var) - Ks @ v
        jitter = 1e-8 * var
        while True:
            try:
                C = cholesky(cov + jitter * np.eye(len(P)), lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        return mean + rng.standard_normal((n, len(P))) @ C.T


def _bounds(xs: np.ndarray) -> dict:
    """Point estimate (median) and 95% one-sided bounds."""
    return {"estimate": float(np.median(xs)), "lower": float(np.quantile(xs, 0.05)),
            "upper": float(np.quantile(xs, 0.95))}


def judge(space: dict, baseline: dict, group: list[str], trials: list[dict], sign: int,
          sigma: float, fresh_sampler: list[int], seed: int) -> dict:
    """The statistics and gates of one verdict check. `trials` are the eligible trials. Δ, M_u and
    √V_T (telemetry), for the group and (d ≥ 2) each lever, come back in objective units, larger =
    better, all from the same latent posterior sample paths."""
    rng = np.random.default_rng(seed)
    names = list(space)
    gi = [names.index(g) for g in group]
    X = np.array([[encode(space[n], t["levers"][n]) for n in names] for t in trials])
    obj = -sign * np.array([t["objective"] for t in trials])
    mu, sd = obj.mean(), obj.std()
    sd = sd if sd > 0 else 1.0
    gp = GP(X, (obj - mu) / sd, sigma**2 / sd**2, rng)
    base = np.array([encode(space[n], baseline[n]) for n in names])

    ls, var, _ = gp._parts(gp.theta)
    # the candidates: quasi-random points, the trials and the baseline (Δ and M_u share them)
    C = np.vstack([_snap(space, qmc.Sobol(len(names), seed=rng).random(N_CANDIDATES)), X, base])
    best = C[int(np.argmax(_kernel(C, X, ls, var) @ gp.alpha))]  # the best posterior-mean point

    # Sobol (Jansen), telemetry: V_T(G) = E[(f(A) - f(A_B^G))²] / 2 over the in-search box
    sob = qmc.Sobol(2 * len(names), seed=rng)
    AB = _snap({**space, **{f"{n}#": space[n] for n in names}}, sob.random(N_SOBOL))
    A, B = AB[:, :len(names)], AB[:, len(names):]
    sets = [gi] + ([[j] for j in gi] if len(gi) > 1 else [])
    blocks = [A]
    for s in sets:
        M = A.copy()
        M[:, s] = B[:, s]
        # Δ: the same candidates with s at baseline (the other levers stay free, so the max
        # optimises them in both terms)
        Cb = C.copy()
        Cb[:, s] = base[s]
        blocks += [M, Cb, _grid(C, base, best, s)]
    F = gp.samples(np.vstack([*blocks, C]), N_SAMPLES, rng) * sd  # latent paths, objective units
    nA, nC = len(A), len(C)
    fA, fC = F[:, :nA], F[:, -nC:]
    stats, vts, at = [], [], nA
    for s in sets:
        fM, fCb = F[:, at:at + nA], F[:, at + nA:at + nA + nC]
        fG = F[:, at + nA + nC:at + nA + nC + N_GRID_POINTS].reshape(len(F), -1, N_GRID + 1)
        at += nA + nC + N_GRID_POINTS
        vt = np.sqrt(0.5 * ((fA - fM) ** 2).mean(1))
        vts.append(vt)
        free = np.maximum(fC.max(1), fCb.max(1))  # the free max covers the baseline set
        # M_s: the largest range over s at any setting of the others: the grid's rows, and each
        # candidate against itself with s at baseline (Δ's maximisers, so Δ ≤ M_s on every path)
        m = np.maximum((fG.max(2) - fG.min(2)).max(1), np.abs(fC - fCb).max(1))
        stats.append({"sqrt_vt": _bounds(vt), "delta_stat": _bounds(free - fCb.max(1)), "m_u": _bounds(m)})
    total = fA.var(1)
    best_point = {n: decode(space[n], float(best[j])) for j, n in enumerate(names) if n in group}

    resid, loo_var = gp.loo()
    z2 = resid**2 / loo_var
    gates = {
        "coverage": _coverage(space, group, [x for x, t in zip(X, trials)
                                             if t["trial"] in set(fresh_sampler)], names),
        "fit": {"passed": bool(z2.mean() <= FIT_MAX_Z2), "loo_mean_z2": float(z2.mean())},
        "homogeneity": _homogeneity(X, resid / np.sqrt(loo_var), names),
    }
    return {
        **stats[0], "levers": {group[i]: x for i, x in enumerate(stats[1:])},
        "sobol_index": float(np.median(vts[0] ** 2 / np.maximum(total, 1e-300))),
        "best_point": best_point, "gates": gates,
        "gp": {"lengthscales": [float(x) for x in np.exp(gp.theta[:len(names)])],
               "signal_var": float(var), "noise_var": float(math.exp(gp.theta[-1])),
               "noise_floor": float(sigma**2 / sd**2)},
    }


def _grid(C: np.ndarray, base: np.ndarray, best: np.ndarray, s: list[int]) -> np.ndarray:
    """Rows = settings of the other levers (N_GRID candidates, the baseline, the best point), each
    with s moved over N_GRID other candidates' values and its baseline: the ICE curves M_s reads."""
    rows = np.vstack([C[:N_GRID], base, best])
    cols = np.vstack([C[N_GRID:2 * N_GRID], base])
    G = np.repeat(rows, len(cols), axis=0)
    G[:, s] = np.tile(cols[:, s], (len(rows), 1))
    return G


def _coverage(space: dict, group: list[str], xs: list, names: list[str]) -> dict:
    """The sampler trials span each graded lever's range (some in its bottom and its top third) and
    try every option of a categorical."""
    # ponytail: the span, not every third: BO exploits, so a lever whose best is at an end of its
    # range would never fill the middle third and could never be retained
    per = {}
    for n in group:
        j, lv = names.index(n), space[n]
        if lv["kind"] in ("float", "int"):
            need = {"bottom third": any(x[j] <= 1 / 3 for x in xs),
                    "top third": any(x[j] >= 2 / 3 for x in xs)}
        else:
            opts = options(lv)
            need = {repr(o): any(abs(x[j] - encode(lv, o)) < 1e-9 for x in xs) for o in opts}
        per[n] = {"missing": [k for k, hit in need.items() if not hit]}
    return {"passed": not any(p["missing"] for p in per.values()), "levers": per}


def _homogeneity(X: np.ndarray, z: np.ndarray, names: list[str]) -> dict:
    """The spread of the standardised LOO residuals in the two halves of each lever's range; a ratio
    ≥ 3 fails the gate. Standardised, a sparse or steep region's misfit is not read as noise."""
    # ponytail: regions are lever half-ranges and spread is the GP's LOO residuals, not replicate
    # spread (replicates cluster near the incumbent, so few regions have any); bin by posterior
    # region if this misfires
    ratio, worst = 1.0, None
    for j, n in enumerate(names):
        lo, hi = z[X[:, j] < 0.5], z[X[:, j] >= 0.5]
        if min(len(lo), len(hi)) < HOMOGENEITY_MIN_TRIALS:
            continue
        a, b = math.sqrt((lo**2).mean()), math.sqrt((hi**2).mean())
        r = max(a, b) / max(min(a, b), 1e-12)
        if r > ratio:
            ratio, worst = r, n
    return {"passed": ratio < HOMOGENEITY_MAX, "ratio": ratio, "lever": worst}
