"""#47: a GP through Ax's Client (BoTorch underneath) in place of TPE, compared with the harness on the same seeds.

Needs Ax, so it runs in benchmark/'s own environment (docs/experiments/47-ax-gp.md):

    uv run --frozen --project benchmark python benchmark/ax_gp.py [--seeds 30] [--trials 12] [--repeats 2] \
        [--variant harness --variant ax …]
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

import bench
from boar import rounds

# Ax logs each trial at INFO and BoTorch a failed GP fit attempt at DEBUG: `_Log` reads both, the console neither.
for _name in ("ax", "botorch"):
    for _handler in logging.getLogger(_name).handlers:
        _handler.setLevel(logging.WARNING)
logging.getLogger("botorch").setLevel(logging.DEBUG)

# How a GP suggestion fell back: BoTorch refit from resampled hyperparameters, or Ax drew it from Sobol instead.
FALLBACK = re.compile(r"Fit attempt #\d+ of \d+ (triggered retry|failed)|switching to fallback model")


def _parameter(name: str, lever: dict) -> RangeParameterConfig | ChoiceParameterConfig:
    if lever["type"] == "bool":
        return ChoiceParameterConfig(name=name, values=[False, True], parameter_type="bool")
    if lever["type"] == "categorical":
        choices = list(lever["choices"])
        return ChoiceParameterConfig(name=name, values=choices, parameter_type=type(choices[0]).__name__,
                                     is_ordered=False)
    return RangeParameterConfig(name=name, bounds=(lever["low"], lever["high"]), parameter_type=lever["type"],
                                scaling="log" if lever["log"] else "linear")


class _Log(logging.Handler):
    """What Ax and BoTorch log while Ax makes one suggestion."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.text = ""

    def emit(self, record: logging.LogRecord) -> None:
        self.text += record.getMessage() + "\n"

    def __enter__(self) -> _Log:
        for name in ("ax", "botorch"):
            logging.getLogger(name).addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        for name in ("ax", "botorch"):
            logging.getLogger(name).removeHandler(self)


class AxBackend:
    """A round's optimizer with OptunaBackend's methods, on an Ax Client with its default GP.

    `fit` says how the last suggestion went: None if Ax didn't fit a GP for it (its startup draws), "fallback" if
    BoTorch or Ax had to fall back (`FALLBACK`), else "ok". A fit that raises stops the run.
    """

    def __init__(self, space: dict, seed: int, n_startup: int) -> None:
        self._client = Client(random_seed=seed)
        self._client.configure_experiment(parameters=[_parameter(name, lever) for name, lever in space.items()])
        # TPE gets a guard as a constraint; Ax gets it as a second metric that must stay below 0.5.
        self._client.configure_optimization(objective="-metric", outcome_constraints=["guards <= 0.5"])
        # As TPE's n_startup_trials: warm and queued trials count towards it. Ax's centre point is skipped once a
        # trial exists, but still costs a Sobol draw on every new Client, so it is off.
        self._client.configure_generation_strategy(initialization_budget=n_startup, initialization_random_seed=seed,
                                                   initialize_with_center=False)
        self._space = space
        self._queue: list[tuple[str, dict]] = []
        self._trial: int | None = None
        self.fit: str | None = None

    def add_warm(self, valid: list[dict], defaults: dict) -> None:
        """Attach earlier trial records as finished trials; levers new to the space are at default."""
        for t in valid:
            params = {name: t["config"].get(name, defaults[name]) for name in self._space}
            self._trial = self._client.attach_trial(params)
            self.tell(t["state"], t["metric"])

    def enqueue(self, label: str, params: dict) -> None:
        self._queue.append((label, params))

    def ask(self, boar_trial: int) -> tuple[dict, str | None]:
        if self._queue:
            label, params = self._queue.pop(0)
            self._trial = self._client.attach_trial(params)
            return params, label
        with _Log() as log:
            ((self._trial, params),) = self._client.get_next_trials(max_trials=1).items()
        if not re.search(r"using GenerationNode MBM\.", log.text):
            self.fit = None
        else:
            self.fit = "fallback" if FALLBACK.search(log.text) else "ok"
        return params, None

    def tell(self, state: str, metric: float | None) -> None:
        if state == "failed":
            self._client.mark_trial_failed(self._trial)
        else:
            self._client.complete_trial(self._trial, {"metric": metric, "guards": float(state == "infeasible")})


class CrashAsGuard(AxBackend):
    """Post hoc: a crash reaches Ax as a broken guard with no metric. Ax drops a failed trial's data and may suggest
    its config again, and with the GP unchanged it does, crash after crash."""

    def tell(self, state: str, metric: float | None) -> None:
        if state == "failed":
            self._client.complete_trial(self._trial, {"guards": 1.0})
        else:
            super().tell(state, metric)


class _Recorded:
    """`backend`'s methods, recording each ask the queue didn't answer: its seconds, and `backend.fit` if any."""

    def __init__(self, backend, records: list[dict]) -> None:
        self._backend, self._records = backend, records

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def ask(self, boar_trial: int) -> tuple[dict, str | None]:
        start = time.perf_counter()
        params, label = self._backend.ask(boar_trial)
        if label is None:
            self._records.append({"seconds": time.perf_counter() - start, "fit": getattr(self._backend, "fit", None)})
        return params, label


class Recorded(bench.Harness):
    """The harness on `backend` (TPE if None), recording each suggestion in `records`."""

    def __init__(self, backend=None) -> None:
        self._backend = backend
        self.records: list[dict] = []

    def backend(self, space: dict, seed: int, n_startup: int):
        make = self._backend or super().backend
        return _Recorded(make(space, seed, n_startup), self.records)


def suggestions(records: dict[str, list[dict]]) -> str:
    """A table of each variant's suggestions: how many, their seconds (median [IQR]), and its GP fits and fallbacks."""
    rows = []
    for name, recs in records.items():
        fits = [r["fit"] for r in recs if r["fit"]]
        fell_back = fits.count("fallback")
        rows.append([name, len(recs), bench._spread([r["seconds"] for r in recs]), len(fits),
                     f"{fell_back} ({fell_back / len(fits):.0%})" if fits else "0"])
    return rounds.md_table(["variant", "suggestions", "seconds each", "GP fits", "fell back"], rows)


VARIANTS = {"harness": None, "ax": AxBackend, "ax-crash": CrashAsGuard}  # the harness on TPE or on Ax


def _run(name: str, seed: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    """One seed's run through a variant, and the suggestions it made."""
    warnings.simplefilter("ignore")  # Ax warns about its own deprecations and every constant guard column
    variant = Recorded(VARIANTS[name])
    return bench.simulate(variant, seed, cfg), variant.records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="#47: the harness on Ax's GP against TPE, on the same seeds.")
    ap.add_argument("--seeds", type=int, default=30, help="runs per variant, seeded 0, 1, … (default 30)")
    ap.add_argument("--rounds", type=int, default=len(bench.SCENARIO), help=f"default {len(bench.SCENARIO)}")
    ap.add_argument("--trials", type=int, default=12, help="trials per round (default 12)")
    ap.add_argument("--repeats", type=int, default=2, help="eval runs per trial (default 2)")
    ap.add_argument("--jobs", type=int, default=8, help="processes, one thread each (default 8)")
    ap.add_argument("--variant", action="append", choices=list(VARIANTS),
                    help="repeat to compare with the first (default harness, ax)")
    args = ap.parse_args(argv)
    names = args.variant or ["harness", "ax"]
    cfg = {"rounds": args.rounds, "trials_per_round": args.trials, "repeats": args.repeats}
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")  # read by torch and numpy in each process
    runs, records = {}, {}
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name in names:
            start = time.monotonic()
            out = list(pool.map(_run, [name] * args.seeds, range(args.seeds), [cfg] * args.seeds))
            runs[name], records[name] = [run for run, _ in out], [rec for _, recs in out for rec in recs]
            print(f"{name}: {args.seeds} seeds in {time.monotonic() - start:.0f} s on {args.jobs} processes; "
                  "median [IQR] over seeds\n")
            print(bench.report(runs[name]) + "\n")
    for name in names[1:]:
        print(f"{name} − {names[0]}, per seed:\n\n{bench.compare(runs[names[0]], runs[name])}\n")
    print(f"Suggestions (asks the queue didn't answer), over all seeds:\n\n{suggestions(records)}")


if __name__ == "__main__":
    main()
