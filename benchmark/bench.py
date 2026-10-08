"""Offline BO benchmark: simulated multi-round BOAR runs on a synthetic objective, scored by regret.

Run from the repo root (README.md says how to add a variant):

    uv run --frozen --project plugin/harness python benchmark/bench.py [--seeds 100] [--variant harness …]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np

from boar import control, evalrun, optimizer, rounds, schema, stats, store, warmstart

DIRECTION = "min"

# --- the target: examples/toy's log summariser, in seconds -----------------------------------------------------


def _proposal(statement: str, *levers: dict) -> dict:
    return {"statement": statement, "mechanism": statement, "source": "user", "levers": list(levers)}


PROPOSALS = {
    "H1": _proposal("A set makes the duplicate check O(1)", {"name": "dedup_set", "type": "bool", "default": False}),
    "H2": _proposal("Building the route table less often cuts matching time", {
        "name": "route_table", "type": "categorical", "choices": ["per_request", "per_file", "once"],
        "default": "per_request"}),
    "H3": _proposal("A bigger output buffer cuts write syscalls", {
        "name": "buffer_kb", "type": "int", "low": 1, "high": 4096, "log": True, "default": 8}),
    "H4": _proposal("Parsing lines in batches amortises per-line overhead", {
        "name": "batch", "type": "int", "low": 1, "high": 64, "log": True, "default": 1}),
    "H5": _proposal("A higher GC threshold cuts collection pauses", {
        "name": "gc_scale", "type": "float", "low": 0.05, "high": 20.0, "log": True, "default": 1.0}),
    "H6": _proposal(
        "Prefetching the next chunk overlaps reading with parsing",
        {"name": "prefetch", "type": "float", "low": 0.0, "high": 1.0, "log": False, "default": 0.0},
        {"name": "prefetch_async", "type": "bool", "default": False},
    ),
}
DEFAULTS = {lever["name"]: lever["default"] for p in PROPOSALS.values() for lever in p["levers"]}

# What the code does at a commit; SCENARIO changes it.
WORLD = {"scale": 1.0, "route": {"per_request": 4.5, "per_file": 0.9, "once": 0.3}, "batch_opt": 16,
         "gc_opt": 4.0, "crash_above": None}

# What happens before each round: hypotheses activated (`add`) or removed, levers repaired after a `fix`
# decision, and `world` changes. A non-empty step changes the code, so the round gets a new commit.
SCENARIO = [
    {"add": ["H1", "H2", "H3"]},
    {"add": ["H4"], "world": {"crash_above": 16}},  # H4's first version crashes on batches over 16
    {"fix": ["H4"], "add": ["H5"], "world": {"crash_above": None, "scale": 1.1}},  # a uniform slowdown
    {"remove": ["H3"], "add": ["H6"], "world": {"batch_opt": 4}},  # reorders configs by batch
    {},  # R1 changed nothing: the same commit
    {"world": {"route": {"per_request": 4.5, "per_file": 0.2, "once": 0.6}}},  # per_file now beats once
]

NOISE = 0.02  # relative sd of the noise every config shares at a repeat, and of each config's own noise


def true_metric(config: dict, world: dict) -> tuple[float, bool]:
    """Noise-free seconds, and whether the guard (identical output) holds. Levers not defined yet are at default."""
    c = {**DEFAULTS, **config}
    seconds = (
        0.5  # interpreter start-up
        + (0.1 if c["dedup_set"] else 6.0)  # a list makes the duplicate check O(n²)
        + world["route"][c["route_table"]]
        + 1.0 + 0.1 * math.log2(c["batch"] / world["batch_opt"]) ** 2
        + 0.6 + 0.15 * math.log(c["gc_scale"] / world["gc_opt"]) ** 2
        + 1.5 - c["prefetch"] * (1.5 if c["prefetch_async"] else 1.0)
    )  # buffer_kb has no effect
    return world["scale"] * seconds, c["prefetch"] <= 0.8  # prefetching further reorders the output


def best_config(world: dict) -> dict:
    """The feasible config with the lowest true metric at `world`. Each lever acts alone, so each has a best value."""
    return {"dedup_set": True, "route_table": min(world["route"], key=world["route"].get), "buffer_kb": 8,
            "batch": world["batch_opt"], "gc_scale": world["gc_opt"], "prefetch": 0.8, "prefetch_async": True}


def _normal(*key) -> float:
    """A standard normal draw fixed by `key`, the same in every process."""
    return random.Random(json.dumps(key)).gauss(0.0, 1.0)


def measure(config: dict, world: dict, commit: str, seed: int, repeats: int) -> dict:
    """One trial as evalrun.run_trial records it, plus `runs`, the eval invocations it cost.

    Repeat k's noise comes from (split, k) like a #55 eval's seeds, so configs are compared paired and the same
    config measured again on the same commit gets the same values. A guard failure or a crash stops the repeats.
    """
    c = {**DEFAULTS, **config}
    if world["crash_above"] and c["batch"] > world["crash_above"]:
        return {"state": evalrun.FAILED, "metric": None, "repeats": [], "runs": 1}
    true, ok = true_metric(c, world)
    key = store.config_key(c, DEFAULTS)
    values = []
    for k in range(1, repeats + 1):
        values.append(true * (1 + NOISE * (_normal(seed, "dev", k) + _normal(seed, commit, key, "dev", k))))
        if not ok:
            break
    state = evalrun.COMPLETE if ok else evalrun.INFEASIBLE
    return {"state": state, "metric": stats.median(values), "repeats": values, "runs": len(values)}


# --- the variant interface ------------------------------------------------------------------------------------


class Harness:
    """BOAR's round as the harness runs it. A variant subclasses it and overrides what it changes."""

    def backend(self, space: dict, seed: int, n_startup: int):
        """The round's optimizer, with OptunaBackend's methods: multivariate TPE."""
        return optimizer.OptunaBackend.create(None, "round", DIRECTION, space, seed, n_startup)  # None: in memory

    def warm(self, trials: list[dict], hyps: list[dict], r: int) -> tuple[list[dict], dict, dict | None]:
        """Round r's warm-start set, how many trials each rule left out, and the incumbent: rules 1-3."""
        valid, excluded = warmstart.select(trials, hyps, r)
        return valid, excluded, warmstart.incumbent(trials, hyps, r, DIRECTION)

    def queue(self, hyps: list[dict], r: int, incumbent: dict | None) -> dict:
        """The partial configs queued ahead of sampling: the incumbent (or the baseline), then extra_queue's."""
        return {"incumbent_config": incumbent["config"] if incumbent else None, "queue": rounds.extra_queue(hyps, r)}


VARIANTS = {"harness": Harness()}

# --- a simulated run -------------------------------------------------------------------------------------------


def _change(hyps: list[dict], step: dict, r: int) -> None:
    """Round r's hypothesis changes, as round r-1's close left them."""
    for hid in step.get("add", []):
        levers = [schema.normalize_lever(lever) for lever in PROPOSALS[hid]["levers"]]
        hyps.append({**PROPOSALS[hid], "id": hid, "levers": levers, "status": store.ACTIVE, "activated_round": r,
                     "decisions": {}})
    for hid in step.get("remove", []):
        store.hypothesis(hyps, hid).update(status=store.REMOVED, removed_round=r - 1)
        store.hypothesis(hyps, hid)["decisions"][str(r - 1)] = {"decision": "remove"}
    for hid in step.get("fix", []):
        store.hypothesis(hyps, hid)["decisions"][str(r - 1)] = {"decision": "fix"}


def simulate(variant: Harness, seed: int, cfg: dict) -> list[dict]:
    """One run through `variant`, the same for the same seed: a record per round. `cfg` has the harness config's
    rounds, trials_per_round and repeats."""
    hyps: list[dict] = []
    trials: list[dict] = []
    world, commit, out, seen = dict(WORLD), "", [], set()
    for r in range(1, cfg["rounds"] + 1):
        step = SCENARIO[r - 1] if r <= len(SCENARIO) else {}
        _change(hyps, step, r)
        world.update(step.get("world", {}))
        if not commit or step:
            commit = f"c{r}"
        space, defaults = store.search_space(hyps), store.lever_defaults(hyps)
        backend = variant.backend(space, seed + r, cfg["trials_per_round"])
        valid, excluded, incumbent = variant.warm(trials, hyps, r)
        backend.add_warm(valid, defaults)
        warm = variant.queue(hyps, r, incumbent)
        measured = [t for t in trials if t["commit"] == commit and t["state"] != evalrun.FAILED]
        warm["same_as"] = rounds._same_as(rounds._filled(warm, space, defaults), measured, defaults)
        rounds._enqueue(backend, space, defaults, warm)
        new, remeasured = [], 0
        for _ in range(control.round_size(cfg, {"warm": warm})):
            n = len(trials) + 1
            params, queued = backend.ask(n)
            config = store.full_config(hyps, params)
            key = store.config_key(config, DEFAULTS)
            remeasured += key in seen
            seen.add(key)
            result = measure(config, world, commit, seed, cfg["repeats"])
            new.append({"trial": n, "round": r, "commit": commit, "config": config, **result, "queued": queued})
            trials.append(new[-1])
            backend.tell(result["state"], result["metric"])
        # Regret: the true metric at the round's commit of the incumbent after the round (the baseline if there is
        # none, as at finalize) minus that of the best config in the round's search space.
        _, _, after = variant.warm(trials, hyps, r + 1)
        best = {name: value for name, value in best_config(world).items() if name in space}
        regret = true_metric(after["config"] if after else {}, world)[0] - true_metric(best, world)[0]
        out.append({"round": r, "commit": commit, "regret": regret, "runs": sum(t["runs"] for t in new),
                    "remeasured": remeasured, "excluded": excluded, "same_as": warm["same_as"], "trials": new})
    return out


# --- the report ------------------------------------------------------------------------------------------------


def _spread(xs: list[float], fmt: str = ".3g") -> str:
    q1, median, q3 = np.percentile(xs, [25, 50, 75])
    return f"{median:{fmt}} [{q1:{fmt}}, {q3:{fmt}}]"


def report(runs: list[list[dict]]) -> str:
    """A table of each round's regret (to 3 significant figures) and cost (in full), median [IQR] over the runs, and a
    last row for the whole run: the mean regret over its rounds and its total cost."""
    rows = [[rec["round"], _spread([run[i]["regret"] for run in runs]),
             *(_spread([run[i][k] for run in runs], "g") for k in ("runs", "remeasured"))]
            for i, rec in enumerate(runs[0])]
    rows.append(["all", _spread([np.mean([rec["regret"] for rec in run]) for run in runs]),
                 *(_spread([sum(rec[k] for rec in run) for run in runs], "g") for k in ("runs", "remeasured"))])
    return rounds.md_table(["round", "regret (s)", "eval runs", "re-measured"], rows)


def _per_run(run: list[dict]) -> dict:
    """A run's `all` row: its mean regret over rounds and its total eval runs."""
    return {"regret (s)": np.mean([rec["regret"] for rec in run]), "eval runs": sum(rec["runs"] for rec in run)}


def compare(base: list[list[dict]], runs: list[list[dict]]) -> str:
    """A table of each seed's difference from the base variant's run on the same seed (runs minus base) in mean regret
    and in eval runs: the mean, its 95% bootstrap CI, and how many seeds came out lower, tied and higher."""
    rng = np.random.default_rng(0)
    rows = []
    for k, fmt in (("regret (s)", ".3g"), ("eval runs", ".1f")):
        d = np.array([_per_run(run)[k] - _per_run(b)[k] for b, run in zip(base, runs)])
        lo, hi = np.percentile(rng.choice(d, (2000, len(d))).mean(axis=1), [2.5, 97.5])
        rows.append([k, f"{d.mean():{fmt}}", f"[{lo:{fmt}}, {hi:{fmt}}]",
                     " / ".join(str(int(n)) for n in ((d < 0).sum(), (d == 0).sum(), (d > 0).sum()))])
    return rounds.md_table(["", "mean", "95% CI", "seeds lower / tied / higher"], rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Simulated multi-round BOAR runs on a synthetic objective.")
    ap.add_argument("--seeds", type=int, default=100, help="runs per variant, seeded 0, 1, … (default 100)")
    ap.add_argument("--rounds", type=int, default=len(SCENARIO), help=f"default {len(SCENARIO)}, the scenario's")
    ap.add_argument("--trials", type=int, default=6, help="trials per round (default 6)")
    ap.add_argument("--repeats", type=int, default=3, help="eval runs per trial (default 3)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                    help="repeat to compare with the first (default harness)")
    args = ap.parse_args(argv)
    cfg = {"rounds": args.rounds, "trials_per_round": args.trials, "repeats": args.repeats}
    names = args.variant or ["harness"]
    base = None
    for name in names:
        start = time.monotonic()
        runs = [simulate(VARIANTS[name], seed, cfg) for seed in range(args.seeds)]
        print(f"{name}: {args.seeds} seeds in {time.monotonic() - start:.0f} s; median [IQR] over seeds\n")
        print(report(runs) + "\n")
        if base is None:
            base = runs
        else:
            print(f"{name} − {names[0]}, per seed:\n\n{compare(base, runs)}\n")


if __name__ == "__main__":
    main()
