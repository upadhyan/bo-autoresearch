"""The round's search: eligibility, the disposable Optuna study rebuilt from the log, the incumbent.

Trials here are the state fold's trial dicts. Optuna (and torch, behind GPSampler) is imported
only by the rounds that search, so every other command stays fast.
"""
import warnings
from pathlib import Path

from .hypotheses import GRADED, in_range, options


def eligible(t: dict, space: dict, baseline: dict, fidelity: dict, epoch: int = 0,
             dropped: frozenset = frozenset(), merges: dict | None = None) -> dict | None:
    """THE eligibility rule, for the sampler and the verdict GP: same epoch, same fidelity, in range
    after the warm-start mapping. Returns the trial rewritten into the round's levers, or None.

    Mapping: a trial that ran a merged-away lever (not searched now) but not the lever `merges` maps
    it to takes the mapped value (`values` pairs old -> new; none = the same value), and a value the
    mapping can't express makes it ineligible. A `dropped` lever (credibly irrelevant: rejected `irrelevant`, or frozen) out of the
    search loses its key, whatever its value; a lever the trial predates is backfilled at its
    baseline. Then the searched levers must
    lie inside their boxes (never clipped), every other lever at its baseline, and no lever the round
    doesn't know.
    """
    if t["status"] != "finished" or t["fidelity"] != fidelity or t.get("epoch", 0) != epoch:
        return None
    gone = set(dropped) - set(space)
    levers = dict(t["levers"])
    for new, rule in (merges or {}).items():
        old = rule["lever"]
        if new in levers or old not in levers or old in space:
            continue  # the trial ran the new lever, or predates the old one, or it is still searched
        v = levers.pop(old)
        gone.add(old)  # expressed by the new lever from here on
        if "values" in rule:
            hits = [b for a, b in rule["values"] if type(a) is type(v) and a == v]
            if not hits:
                return None
            v = hits[0]
        levers[new] = v
    levers = {n: v for n, v in levers.items() if n not in gone}
    known = (set(baseline) | set(space)) - gone
    for n in known:
        if n not in levers:  # an added lever: its baseline reproduces the code the trial ran
            levers[n] = baseline[n] if n in baseline else space[n]["baseline"]
    if set(levers) != known:
        return None
    ok = all(in_range(levers[n], space[n]) if n in space else levers[n] == baseline[n] for n in levers)
    return {**t, "levers": levers} if ok else None


def _distributions(space: dict) -> dict:
    import optuna.distributions as d
    out: dict = {}
    for name, lv in space.items():
        if lv["kind"] in GRADED:
            cls = d.FloatDistribution if lv["kind"] == "float" else d.IntDistribution
            out[name] = cls(lv["low"], lv["high"], log=bool(lv.get("log")))
        else:
            out[name] = d.CategoricalDistribution(options(lv))
    return out


class Study:
    """The round's Optuna study in a disposable SQLite file, seeded from the log's eligible trials."""

    def __init__(self, path: Path, direction: str, space: dict, trials: list[dict], seed: int):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        self._optuna, self.space, self.dists = optuna, space, _distributions(space)
        path.unlink(missing_ok=True)  # the log is the source of truth: always rebuilt
        self.study = optuna.create_study(
            storage=f"sqlite:///{path}", direction=direction,  # type: ignore[arg-type]
            sampler=optuna.samplers.GPSampler(seed=seed, deterministic_objective=False))
        self.study.add_trials([self._frozen(t) for t in trials])

    def _frozen(self, t: dict):
        return self._optuna.trial.create_trial(
            params={n: t["levers"][n] for n in self.space}, distributions=self.dists,
            value=t["objective"], user_attrs={"trial": t["trial"]})

    def ask(self):
        return self.study.ask(self.dists)

    def tell(self, asked, t: dict) -> None:
        """Called after the trial is logged: the log first, then Optuna."""
        if t["status"] == "finished":
            asked.set_user_attr("trial", t["trial"])
            self.study.tell(asked, t["objective"])
        else:
            self.study.tell(asked, state=self._optuna.trial.TrialState.FAIL)

    def add(self, t: dict) -> None:
        self.study.add_trial(self._frozen(t))


def groups(trials: list[dict]) -> dict[int, list[dict]]:
    """Each config's trials: the original and its replicates (never averaged into the original)."""
    out: dict[int, list[dict]] = {}
    for t in trials:
        out.setdefault(t.get("replicate_of", t["trial"]), []).append(t)
    return out


def ranked(trials: list[dict], sign: int) -> list[dict]:
    """Configs best first by mean objective (sign 1 minimises, -1 maximises)."""
    rows = []
    for root, ts in groups(trials).items():
        mean = sum(t["objective"] for t in ts) / len(ts)
        rows.append({"trial": root, "levers": ts[0]["levers"], "mean": mean,
                     "replicates": len(ts) - 1})
    return sorted(rows, key=lambda r: (sign * r["mean"], r["trial"]))


def incumbent(trials: list[dict], sign: int, confirmations: int) -> dict | None:
    """The best config with at least `confirmations` replicates, else the best one unconfirmed."""
    rows = ranked(trials, sign)
    for r in rows:
        if r["replicates"] >= confirmations:
            return {**r, "confirmed": True}
    return {**rows[0], "confirmed": False} if rows else None
