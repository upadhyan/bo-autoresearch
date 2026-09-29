"""The round's search: eligibility, the disposable Optuna study rebuilt from the log, the incumbent.

Trials here are the state fold's trial dicts. Optuna (and torch, behind GPSampler) is imported
only by the rounds that search, so every other command stays fast.
"""
import warnings
from pathlib import Path

from .hypotheses import GRADED, options


def in_range(value, lv: dict) -> bool:
    """The harness's own bounds check of a lever value against its declaration."""
    if lv["kind"] == "bool":
        return isinstance(value, bool)
    if lv["kind"] == "categorical":
        return any(type(o) is type(value) and o == value for o in lv["options"])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return (lv["kind"] == "float" or isinstance(value, int)) and lv["low"] <= value <= lv["high"]


def eligible(t: dict, space: dict, baseline: dict, fidelity: dict, epoch: int = 0) -> bool:
    """THE eligibility rule, for the sampler and the verdict GP: same epoch, same fidelity, in range.

    In range: the searched levers inside their boxes, every other lever at its baseline, and no
    lever the round doesn't know. #28 extends this with the warm-start mapping rules (backfill,
    dropped and filtered levers, narrowing, merges).
    """
    # ponytail: trials carry no epoch until #28 starts epochs; a missing one is epoch 0
    if t["status"] != "finished" or t["fidelity"] != fidelity or t.get("epoch", 0) != epoch:
        return False
    levers = t["levers"]
    if set(levers) != set(baseline) | set(space):
        return False
    return all(in_range(levers[n], space[n]) if n in space else levers[n] == baseline[n]
               for n in levers)


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
