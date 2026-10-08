"""#44's look at the round each hypothesis starts in (docs/experiments/44-seeding.md): the regret of the incumbent
it starts from, and how much of the new levers' gain the round's sampled trials, and the incumbent after it, leave.
With --own, instead, which hypotheses' levers a paired comparison's regret difference comes from.

Run from the repo root:

    uv run --frozen --project plugin/harness python benchmark/first_round.py [--seeds 100] [--scenario default] …
"""

from __future__ import annotations

import argparse

import numpy as np

import bench
from boar import rounds, store


def own_regret(hids: list[str], config: dict, world: dict) -> float:
    """What `config`'s setting of the levers of `hids` loses against their best setting at `world`. true_metric adds
    up a term per hypothesis, so the hypotheses' own regrets add up to the regret."""
    names = [lever["name"] for hid in hids for lever in bench.PROPOSALS[hid]["levers"]]
    best = bench.best_config(world)
    return (bench.true_metric({name: config.get(name, bench.DEFAULTS[name]) for name in names}, world)[0]
            - bench.true_metric({name: best[name] for name in names}, world)[0])


def replay(variant: bench.Harness, seed: int, cfg: dict, scenario: list[dict]) -> list[dict]:
    """One run's rounds: the hypotheses each starts, how many earlier trials it copied in, the regret of the incumbent
    it starts from (the baseline in round 1), the new levers' own regret at default (all they can save), in each
    sampled trial and in the incumbent after the round, that incumbent's own regret per active hypothesis, and the
    round's eval runs."""
    hyps, world, trials, out = [], dict(bench.WORLD), [], []
    for r, rec in enumerate(bench.simulate(variant, seed, cfg, scenario), 1):
        step = scenario[r - 1] if r <= len(scenario) else {}
        bench._change(hyps, step, r)
        world.update(step.get("world", {}))
        _, _, before = variant.warm(trials, hyps, r)
        copied = len(trials) - sum(rec["excluded"].values())
        trials += rec["trials"]
        _, _, after = variant.warm(trials, hyps, r + 1)
        new, active = step.get("add", []), [h["id"] for h in store.active_hypotheses(hyps)]
        config = after["config"] if after else {}
        out.append({"round": r, "starts": new, "copied": copied, "runs": rec["runs"],
                    "before": own_regret(active, before["config"] if before else {}, world),
                    "default": own_regret(new, {}, world),
                    "sampled": [own_regret(new, t["config"], world) for t in rec["trials"] if not t["queued"]],
                    "incumbent": own_regret(new, config, world),
                    "own": {hid: own_regret([hid], config, world) for hid in active}})
    return out


def first_rounds(variant: bench.Harness, seed: int, cfg: dict, scenario: list[dict]) -> list[dict]:
    """`replay`'s rounds that start hypotheses."""
    return [row for row in replay(variant, seed, cfg, scenario) if row["starts"]]


def split(base: list[list[dict]], runs: list[list[dict]], hids: list[str]) -> str:
    """bench.compare's regret row for the own regret of `hids` (runs minus base), over the whole run, then over each
    round alone."""
    def part(rs: list[list[dict]], pick: slice) -> list[list[dict]]:
        return [[{"regret": sum(row["own"].get(h, 0) for h in hids), "runs": row["runs"]} for row in run[pick]]
                for run in rs]

    rows = []
    for label, pick in [("all", slice(None))] + [(i + 1, slice(i, i + 1)) for i in range(len(base[0]))]:
        cells = bench.compare(part(base, pick), part(runs, pick)).splitlines()[2].strip("| ").split(" | ")
        rows.append([label, *cells[1:]])
    return rounds.md_table(["round", "mean", "95% CI", "seeds lower / tied / higher"], rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The new levers' own regret in the round each hypothesis starts in.")
    ap.add_argument("--seeds", type=int, default=100, help="runs per variant, seeded 0, 1, … (default 100)")
    ap.add_argument("--trials", type=int, default=6, help="trials per round (default 6)")
    ap.add_argument("--repeats", type=int, default=3, help="eval runs per trial (default 3)")
    ap.add_argument("--scenario", default="default", choices=sorted(bench.SCENARIOS))
    ap.add_argument("--variant", action="append", choices=sorted(bench.VARIANTS), help="repeat (default harness)")
    ap.add_argument("--own", action="append", type=lambda s: s.split(","), metavar="H1,H2",
                    help="instead: each variant after the first against it, per seed, in these hypotheses' own regret, "
                         "over the run and per round; repeat")
    args = ap.parse_args(argv)
    scenario = bench.SCENARIOS[args.scenario]
    cfg = {"rounds": len(scenario), "trials_per_round": args.trials, "repeats": args.repeats}
    names, base = args.variant or ["harness"], None
    for name in names:
        if args.own:
            runs = [replay(bench.VARIANTS[name], seed, cfg, scenario) for seed in range(args.seeds)]
            for hids in args.own if base else []:
                print(f"{name} − {names[0]}, per seed: the own regret of {', '.join(hids)}\n")
                print(split(base, runs, hids) + "\n")
            base = base or runs
            continue
        runs = [first_rounds(bench.VARIANTS[name], seed, cfg, scenario) for seed in range(args.seeds)]
        rows = [[row["round"], ", ".join(row["starts"]), f"{np.mean([run[i]['copied'] for run in runs]):.3g}",
                 bench._spread([run[i]["before"] for run in runs]), f"{row['default']:.3g}",
                 bench._spread([x for run in runs for x in run[i]["sampled"]]),
                 bench._spread([run[i]["incumbent"] for run in runs])] for i, row in enumerate(runs[0])]
        print(f"{name}: the starting incumbent's regret and the new levers' own regret (s), median [IQR] over seeds "
              "and over sampled trials\n")
        print(rounds.md_table(["round", "starts", "copied (mean)", "incumbent before", "at default", "sampled trials",
                               "incumbent after"], rows) + "\n")


if __name__ == "__main__":
    main()
