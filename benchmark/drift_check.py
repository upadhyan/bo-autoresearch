"""#46: does re-measuring the top copied configs on a new commit detect drift, and does dropping what it
contradicts help?

docs/experiments/46-drift-check.md has the results. Run from the repo root:

    uv run --frozen --project plugin/harness python benchmark/drift_check.py --scenario drift --variant harness …
    uv run --frozen --project plugin/harness python benchmark/drift_check.py --scenario drift --detect
"""

from __future__ import annotations

import argparse
import itertools

import bench
from boar import rounds, stats, store

# A shift at most new commits, and no removal: in bench.SCENARIO, H3's removal before round 4 leaves out nearly every
# copied trial (each sets buffer_kb), so the round 4 shift has little copied data to contradict.
DRIFT = [
    {"add": ["H1", "H2", "H3"]},
    {"add": ["H4"]},  # a new lever and nothing else
    {"add": ["H5"], "world": {"scale": 1.1}},  # a uniform slowdown: no reorder
    {"world": {"batch_opt": 4}},  # reorders configs by batch
    {},  # R1 changed nothing: the same commit
    {"add": ["H6"], "world": {"gc_opt": 1.0}},  # reorders configs by gc_scale
    {"world": {"route": {"per_request": 4.5, "per_file": 0.2, "once": 0.6}}},  # per_file now beats once
    {"world": {"scale": 1.0}},  # a uniform speed-up: no reorder
]
# The one real run's shape: 30 rounds, levers added early, no shift at all, and a new commit in about a third of the
# rounds (19 of its 29 incumbent re-measures were on an unchanged commit). Here 12 of 30.
STEADY = [{"add": ["H1", "H2", "H3"]}, {"add": ["H4"]}, {"add": ["H5"]}, {"add": ["H6"]},
          *([{}, {}, {"world": {}}] * 9)[:26]]
# Its late phase at worst: R1 changes nothing after round 4, so the incumbent is never measured again.
UNCHANGED = [*STEADY[:4], *[{}] * 26]
SCENARIOS = {"default": bench.SCENARIO, "drift": DRIFT, "steady": STEADY, "unchanged": UNCHANGED}


def worlds(scenario: list[dict], n_rounds: int) -> list[dict]:
    """The world at each round, as bench.simulate builds it from `scenario`."""
    world, out = dict(bench.WORLD), []
    for r in range(1, n_rounds + 1):
        world.update((scenario[r - 1] if r <= len(scenario) else {}).get("world", {}))
        out.append(dict(world))
    return out


def contradicted(
    earlier: list[dict], new: list[dict], floor: float | None, base: float | None
) -> list[tuple[str, str]]:
    """The pairs of configs measured again in `new` whose order there goes against their pooled order over `earlier`,
    by more than the noise floor (scaled to the pair's earlier metric) on both sides. With no floor, any reversal."""
    before = stats.pooled(earlier, bench.DEFAULTS)
    now = [(key, t["metric"], before[key]["metric"]) for t in new if t["state"] == "complete"
           and (key := store.config_key(t["config"], bench.DEFAULTS)) in before]
    out = []
    for (a, new_a, old_a), (b, new_b, old_b) in itertools.combinations(now, 2):
        limit = stats.floor_at(floor, base, (old_a + old_b) / 2) or 0.0
        if (old_a - old_b) * (new_a - new_b) < 0 and min(abs(old_a - old_b), abs(new_a - new_b)) > limit:
            out.append((a, b))
    return out


RULE4 = "rule4_dropped"


def rechecked(trials: list[dict], r: int) -> list[dict]:
    """Round r's re-measurements of copied configs: the incumbent and the rechecks."""
    return [t for t in trials if t["round"] == r and (t["queued"] or "").startswith(("incumbent", "recheck"))]


class Rule4(bench.Harness):
    """The harness with a fourth warm-start rule: the trials `dropped` names leave the warm start and the incumbent."""

    def dropped(self, trials: list[dict], r: int) -> set[int]:
        return set()

    def warm(self, trials, hyps, r):
        out = self.dropped(trials, r)
        valid, excluded, _ = super().warm(trials, hyps, r)
        kept, _, incumbent = super().warm([t for t in trials if t["trial"] not in out], hyps, r)
        return kept, {**excluded, RULE4: len(valid) - len(kept)}, incumbent


class Recheck(Rule4):
    """Also re-measures the `n` - 1 best copied configs after the incumbent, as `recheck i`, on top of the round's
    trials. Like any queued config, one already measured on the round's commit is not run again.

    With `drop`, each round's check of those re-measurements (`contradicted`) feeds rule 4 from the next round on:
    "trials" leaves out the earlier trials of both configs of a contradicted pair, "older" every earlier trial.
    """

    def __init__(self, n: int, drop: str | None = None) -> None:
        self.n, self.drop = n, drop

    def dropped(self, trials, r):
        out: set[int] = set()
        for s in range(2, r) if self.drop else ():
            before = [t for t in trials if t["round"] < s]
            earlier = [t for t in before if t["trial"] not in out]  # what the checks before this one kept
            floor, base = stats.noise_floor(before, bench.DEFAULTS), stats.noise_base(before, bench.DEFAULTS)
            pairs = contradicted(earlier, rechecked(trials, s), floor, base)
            keys = {key for pair in pairs for key in pair}
            out |= {t["trial"] for t in earlier
                    if pairs and (self.drop == "older" or store.config_key(t["config"], bench.DEFAULTS) in keys)}
        return out

    def warm(self, trials, hyps, r):
        valid, excluded, incumbent = super().warm(trials, hyps, r)
        defaults = store.lever_defaults(hyps)
        groups = [g for g in stats.pooled(valid, defaults).values() if not incumbent or g["key"] != incumbent["key"]]
        groups.sort(key=lambda g: (stats.better(bench.DIRECTION) * g["metric"], len(g["config"]), min(g["trials"])))
        space = store.search_space(hyps)
        self._top = [{name: g["config"].get(name, defaults[name]) for name in space} for g in groups[: self.n - 1]]
        return valid, excluded, incumbent

    def queue(self, hyps, r, incumbent):
        warm = super().queue(hyps, r, incumbent)
        warm["queue"] += [{"label": f"recheck {i}", "config": c} for i, c in enumerate(self._top, 1)]
        return warm


class Age(Rule4):
    """Rule 4 by age: only the last `rounds` rounds' trials stay in the warm start (TPE takes no per-trial weights)."""

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds

    def dropped(self, trials, r):
        return {t["trial"] for t in trials if t["round"] < r - self.rounds}


class AgeStudy(Age):
    """Rule 4 by age on the study's copy only: the incumbent is still picked over every round."""

    def warm(self, trials, hyps, r):
        kept, excluded, _ = super().warm(trials, hyps, r)
        return kept, excluded, bench.Harness().warm(trials, hyps, r)[2]


def detection(run: list[dict]) -> list[dict]:
    """Each check in a simulated run of bench.SCENARIO, one per round that re-measured two or more copied configs:
    whether it flagged a pair, and whether the copied data misorders a pair of those configs at the round's world
    (the same check on noise-free metrics)."""
    world = worlds(bench.SCENARIO, len(run))
    trials = [t for rec in run for t in rec["trials"]]

    def noise_free(ts: list[dict]) -> list[dict]:
        out = [{**t, "metric": bench.true_metric(t["config"], world[t["round"] - 1])[0]} for t in ts]
        return [{**t, "repeats": [t["metric"]] * len(t["repeats"])} for t in out]

    checks = []
    for s in range(2, len(run) + 1):
        before = [t for t in trials if t["round"] < s]
        pooled = stats.pooled(before, bench.DEFAULTS)
        new = [t for t in rechecked(trials, s)
               if t["state"] == "complete" and store.config_key(t["config"], bench.DEFAULTS) in pooled]
        if len(new) >= 2:
            floor, base = stats.noise_floor(before, bench.DEFAULTS), stats.noise_base(before, bench.DEFAULTS)
            checks.append({"round": s, "flagged": bool(contradicted(before, new, floor, base)),
                           "misordered": bool(contradicted(noise_free(before), noise_free(new), None, None))})
    return checks


VARIANTS = {"harness": bench.Harness(), "recheck": Recheck(3), "drop-trials": Recheck(3, "trials"),
            "drop-older": Recheck(3, "older"), "age-3": Age(3), "age-3-study": AgeStudy(3)}
NOISES = (0.0, 0.01, 0.02, 0.05, 0.1)


def moved(runs: list[list[dict]], lever: str) -> str:
    """How many sampled trials set `lever` away from its default, from the round bench.SCENARIO adds it in."""
    hid = next(h for h, p in bench.PROPOSALS.items() if any(lv["name"] == lever for lv in p["levers"]))
    first = next(r for r, step in enumerate(bench.SCENARIO, 1) if hid in step.get("add", []))
    sampled = [t for run in runs for rec in run[first - 1:] for t in rec["trials"] if not t["queued"]]
    off = sum(t["config"][lever] != bench.DEFAULTS[lever] for t in sampled)
    return f"{off}/{len(sampled)} sampled trials from round {first} set {lever} away from its default"


def _rates(checks: list[dict]) -> list[str]:
    def share(xs: list[dict], key: str) -> str:
        return f"{sum(c[key] for c in xs)}/{len(xs)}"

    misordered = [c for c in checks if c["misordered"]]
    in_order = [c for c in checks if not c["misordered"]]
    return [len(checks), share(checks, "misordered"), share(misordered, "flagged"), share(in_order, "flagged")]


def detect(seeds: int, cfg: dict) -> str:
    """Tables of the checks in Recheck runs: by noise level and configs re-measured, then by round."""
    headers = ["checks", "misordered", "flagged if misordered", "flagged if in order"]
    default, rows, by_round = bench.NOISE, [], {}
    for noise in NOISES:
        bench.NOISE = noise
        for n in (2, 3, 4):
            checks = [c for seed in range(seeds) for c in detection(bench.simulate(Recheck(n), seed, cfg))]
            rows.append([noise, n, *_rates(checks)])
            if (noise, n) == (default, 3):
                by_round = {r: [c for c in checks if c["round"] == r] for r in sorted({c["round"] for c in checks})}
    bench.NOISE = default
    return "\n\n".join([
        rounds.md_table(["noise", "re-measured", *headers], rows),
        f"By round, at noise {default} with 3 configs re-measured:",
        rounds.md_table(["round", *headers], [[r, *_rates(cs)] for r, cs in by_round.items()]),
    ])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="#46's drift check on the offline benchmark.")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    ap.add_argument("--seeds", type=int, default=100, help="runs per variant, seeded 0, 1, … (default 100)")
    ap.add_argument("--trials", type=int, default=6, help="trials per round (default 6)")
    ap.add_argument("--repeats", type=int, default=2, help="eval runs per trial (default 2)")
    ap.add_argument("--base-trials", type=int, help="trials per round of the first variant (default --trials)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                    help="repeat to compare with the first (default harness)")
    ap.add_argument("--detect", action="store_true", help="measure the check's detection accuracy instead")
    ap.add_argument("--lever", help="also count the sampled trials that set this lever away from its default")
    args = ap.parse_args(argv)
    bench.SCENARIO = SCENARIOS[args.scenario]
    cfg = {"rounds": len(bench.SCENARIO), "trials_per_round": args.trials, "repeats": args.repeats}
    if args.detect:
        print(f"{args.scenario}: {args.seeds} seeds, {args.trials} trials per round, {args.repeats} repeats; "
              f"each check, over seeds\n\n{detect(args.seeds, cfg)}\n")
        return
    names = args.variant or ["harness"]
    base = None
    for name in names:
        label, c = name, cfg
        if base is None and args.base_trials:
            label, c = f"{name} at {args.base_trials} trials per round", {**cfg, "trials_per_round": args.base_trials}
        runs = [bench.simulate(VARIANTS[name], seed, c) for seed in range(args.seeds)]
        print(f"{label}: {args.seeds} seeds; median [IQR] over seeds\n\n{bench.report(runs)}\n")
        if args.lever:
            print(f"{label}: {moved(runs, args.lever)}\n")
        if base is None:
            base, base_label = runs, label
        else:
            print(f"{name} − {base_label}, per seed:\n\n{bench.compare(base, runs)}\n")


if __name__ == "__main__":
    main()
