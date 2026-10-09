"""#47: a GP through Ax's Client (BoTorch underneath) in place of TPE, compared with the harness on the same seeds.

Needs Ax, so it runs in benchmark/'s own environment (docs/experiments/47-ax-gp.md):

    uv run --frozen --project benchmark python benchmark/ax_gp.py [--target training] [--seeds 30] [--trials 12] \
        [--repeats 2] [--variant harness --variant ax …]
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import multiprocessing
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

import bench
import training
from boar import rounds

# Ax logs each trial at INFO and BoTorch a failed GP fit attempt at DEBUG: `_LOG` keeps both, the console neither.
_LOG = logging.handlers.BufferingHandler(10**9)
for _name in ("ax", "botorch"):
    for _handler in logging.getLogger(_name).handlers:
        _handler.setLevel(logging.WARNING)
    logging.getLogger(_name).addHandler(_LOG)
logging.getLogger("botorch").setLevel(logging.DEBUG)

FIT_FAILED = re.compile(r"Fit attempt #\d+ of \d+ (triggered retry|failed)")
REPEATED = re.compile(r"MAX_GEN_ATTEMPTS.*switching to fallback model")


def _parameter(name: str, lever: dict) -> RangeParameterConfig | ChoiceParameterConfig:
    if lever["type"] == "bool":
        return ChoiceParameterConfig(name=name, values=[False, True], parameter_type="bool")
    if lever["type"] == "categorical":
        choices = list(lever["choices"])
        return ChoiceParameterConfig(name=name, values=choices, parameter_type=type(choices[0]).__name__,
                                     is_ordered=False)
    return RangeParameterConfig(name=name, bounds=(lever["low"], lever["high"]), parameter_type=lever["type"],
                                scaling="log" if lever["log"] else "linear")


class AxBackend:
    """A round's optimizer with OptunaBackend's methods, on an Ax Client with its default GP.

    `fit` says how the last suggestion went: None if Ax didn't fit a GP for it (its startup draws), "failed" if
    BoTorch logged a failed fit attempt and refit (`FIT_FAILED`; a fit that raises stops the run), "repeated" if Ax
    drew it from Sobol because the GP's candidates repeated trials it has (`REPEATED`), else "ok".
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
        _LOG.buffer.clear()
        ((self._trial, params),) = self._client.get_next_trials(max_trials=1).items()
        log = "\n".join(r.getMessage() for r in _LOG.buffer)
        if not re.search(r"using GenerationNode MBM\.", log):
            self.fit = None
        elif FIT_FAILED.search(log):
            self.fit = "failed"
        else:
            self.fit = "repeated" if REPEATED.search(log) else "ok"
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


def suggestions(runs: dict[str, list[list[dict]]], records: dict[str, list[dict]]) -> str:
    """A table of each variant's suggestions: how many, their seconds (median [IQR]), how many of their trials broke
    the guard or crashed, and how many Ax made with a GP, a failed fit or a Sobol draw (`fit`)."""
    rows = []
    for name, recs in records.items():
        states = [t["state"] for run in runs[name] for rec in run for t in rec["trials"] if t["queued"] is None]
        fits = [r["fit"] for r in recs]
        rows.append([name, len(recs), bench._spread([r["seconds"] for r in recs]), states.count("infeasible"),
                     states.count("failed"), len(fits) - fits.count(None), fits.count("failed"),
                     fits.count("repeated")])
    return rounds.md_table(["variant", "suggestions", "seconds each", "broke guard", "crashed", "GP fits", "fit failed",
                            "Sobol instead"], rows)


VARIANTS = {"harness": None, "ax": AxBackend, "ax-crash": CrashAsGuard}  # the harness on TPE or on Ax
TARGETS = {"default": bench, **training.TARGETS}  # bench's own scenario, or the shared hard one


def _run(name: str, target: str, seed: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    """One seed's run through a variant on a target, and the suggestions it made."""
    warnings.simplefilter("ignore")  # Ax warns about its own deprecations and every constant guard column
    variant = Recorded(VARIANTS[name])
    return bench.simulate(variant, seed, cfg, TARGETS[target]), variant.records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="#47: the harness on Ax's GP against TPE, on the same seeds.")
    ap.add_argument("--target", choices=list(TARGETS), default="default", help="what is tuned (default bench's)")
    ap.add_argument("--seeds", type=int, default=30, help="runs per variant (default 30)")
    ap.add_argument("--first-seed", type=int, default=0, help="seed of the first run, then 1 more each (default 0)")
    ap.add_argument("--rounds", type=int, help="default the target's")
    ap.add_argument("--trials", type=int, default=12, help="trials per round (default 12)")
    ap.add_argument("--repeats", type=int, default=2, help="eval runs per trial (default 2)")
    ap.add_argument("--jobs", type=int, default=8, help="processes, one thread each (default 8)")
    ap.add_argument("--variant", action="append", choices=list(VARIANTS),
                    help="repeat to compare with the first (default harness, ax)")
    ap.add_argument("--window", help="compare over rounds A-B only (default every round)")
    ap.add_argument("--level", type=float, default=0.95, help="the comparison's CI level (default 0.95)")
    ap.add_argument("--out", help="also save every run and suggestion to this JSON file")
    args = ap.parse_args(argv)
    names = args.variant or ["harness", "ax"]
    cfg = {"rounds": args.rounds or len(TARGETS[args.target].SCENARIO), "trials_per_round": args.trials,
           "repeats": args.repeats}
    window = None  # checked before the runs, which can take hours
    if args.window:
        first, _, last = args.window.partition("-")
        if not (first.isdigit() and last.isdigit() and 1 <= int(first) <= int(last) <= cfg["rounds"]):
            ap.error(f"--window {args.window}: need A-B with 1 <= A <= B <= {cfg['rounds']}, the rounds")
        window = set(range(int(first), int(last) + 1))
    if not 0 < args.level < 1:
        ap.error(f"--level {args.level}: a fraction, e.g. 0.95")
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")  # read by torch and numpy in each process
    runs, records = {}, {}
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name in names:
            start = time.monotonic()
            seeds = range(args.first_seed, args.first_seed + args.seeds)
            out = list(pool.map(_run, [name] * args.seeds, [args.target] * args.seeds, seeds, [cfg] * args.seeds))
            runs[name], records[name] = [run for run, _ in out], [rec for _, recs in out for rec in recs]
            print(f"{name}: {args.seeds} seeds in {time.monotonic() - start:.0f} s on {args.jobs} processes; "
                  "median [IQR] over seeds\n")
            print(bench.report(runs[name]) + "\n")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "runs": runs, "suggestions": records}, f)
    for name in names[1:]:
        print(f"{name} − {names[0]}, per seed{f', rounds {args.window}' if window else ''}:\n\n"
              f"{bench.compare(runs[names[0]], runs[name], window, args.level)}\n")
    print(f"Suggestions (asks the queue didn't answer), over all seeds:\n\n{suggestions(runs, records)}")


if __name__ == "__main__":
    main()
