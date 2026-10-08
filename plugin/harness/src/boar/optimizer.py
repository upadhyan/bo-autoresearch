"""The optimizer backend: a round's study, and the only module that imports Optuna."""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import Any

import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import FrozenTrial, TrialState

_DIRECTION = {"min": "minimize", "max": "maximize"}

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


def _sampler(n_startup: int, seed: int) -> optuna.samplers.TPESampler:
    # numpy takes seeds in 0 .. 2**32 - 1 only; `seed` may be any integer.
    return optuna.samplers.TPESampler(multivariate=True, n_startup_trials=n_startup, seed=seed % 2**32)


class OptunaBackend:
    """A round's study over `space`, opened with `create` or `resume`, asked and told one trial at a time."""

    def __init__(self, study: optuna.Study, space: dict[str, dict]) -> None:
        self._study = study
        self._space = space
        self._dists = distributions(space)
        self._trial: optuna.Trial | None = None

    @classmethod
    def create(
        cls, storage: str, name: str, direction: str, space: dict[str, dict], seed: int, n_startup: int
    ) -> OptunaBackend:
        """A new study, replacing any study of that name; `direction` is min or max."""
        quiet_optuna()
        with contextlib.suppress(KeyError):
            optuna.delete_study(study_name=name, storage=storage)
        study = optuna.create_study(
            study_name=name, storage=storage, sampler=_sampler(n_startup, seed), direction=_DIRECTION[direction]
        )
        return cls(study, space)

    @classmethod
    def resume(cls, storage: str, name: str, space: dict[str, dict], seed: int, n_startup: int) -> OptunaBackend:
        """The existing study (KeyError if there is none), with the trial an interrupted run left in flight failed."""
        quiet_optuna()
        study = optuna.load_study(study_name=name, storage=storage, sampler=_sampler(n_startup, seed))
        # Re-seeded from progress: the same seed would replay the random startup draws the round already ran.
        # Every attempt adds trials (an interrupted one stays as FAIL), so each resume gets a new seed.
        study.sampler = _sampler(n_startup, seed + 1_000_003 * len(study.trials))
        for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study.tell(t.number, state=TrialState.FAIL)
        return cls(study, space)

    def add_warm(self, valid: list[dict], defaults: dict[str, Any]) -> None:
        """Copy in earlier trial records (warmstart.select's set) as finished trials."""
        self._study.add_trials(build_frozen_trials(valid, self._space, defaults))

    def enqueue(self, label: str, params: dict[str, Any]) -> None:
        """Queue a full config of the space; queued configs are asked for first, in order."""
        self._study.enqueue_trial(params, user_attrs={"queued": label})

    def waiting(self) -> set[str]:
        """The labels of queued configs not asked for yet."""
        return {t.user_attrs.get("queued") for t in self._study.get_trials(deepcopy=False, states=(TrialState.WAITING,))}

    def ask(self, boar_trial: int) -> tuple[dict[str, Any], str | None]:
        """The next config to measure as trial `boar_trial`, and its queue label (None if sampled)."""
        self._trial = self._study.ask(self._dists)
        self._trial.set_user_attr("boar_trial", boar_trial)
        return self._trial.params, self._trial.user_attrs.get("queued")

    def tell(self, state: str, metric: float | None) -> None:
        """The asked trial's result: failed, or its metric and whether the guards held (complete, infeasible)."""
        trial, self._trial = self._trial, None
        if state == "failed":
            self._study.tell(trial, state=TrialState.FAIL)
        else:
            trial.set_constraint(GUARDS, 0.0 if state == "complete" else 1.0)
            self._study.tell(trial, metric)
