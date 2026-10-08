"""#44's look at the round each hypothesis starts in (docs/experiments/44-seeding.md): how much of the new levers'
gain the round's sampled trials, and the incumbent after it, leave.

Run from the repo root:

    uv run --frozen --project plugin/harness python benchmark/first_round.py [--seeds 100] [--scenario default] …
"""

from __future__ import annotations

import argparse

import numpy as np

import bench
from boar import rounds


def own_regret(hids: list[str], config: dict, world: dict) -> float:
    """What `config`'s setting of the levers of `hids` loses against their best setting at `world`. true_metric adds
    up a term per hypothesis, so the hypotheses' own regrets add up to the regret."""
    names = [lever["name"] for hid in hids for lever in bench.PROPOSALS[hid]["levers"]]
    best = bench.best_config(world)
    return (bench.true_metric({name: config.get(name, bench.DEFAULTS[name]) for name in names}, world)[0]
            - bench.true_metric({name: best[name] for name in names}, world)[0])


def first_rounds(variant: bench.Harness, seed: int, cfg: dict, scenario: list[dict]) -> list[dict]:
    """One run's rounds that start hypotheses: which, how many earlier trials the round copied in, and the new levers'
    own regret in each sampled trial and in the incumbent after the round."""
    hyps, world, trials, out = [], dict(bench.WORLD), [], []
    for r, rec in enumerate(bench.simulate(variant, seed, cfg, scenario), 1):
        step = scenario[r - 1] if r <= len(scenario) else {}
        bench._change(hyps, step, r)
        world.update(step.get("world", {}))
        copied = len(trials) - sum(rec["excluded"].values())
        trials += rec["trials"]
        if step.get("add"):
            _, _, incumbent = variant.warm(trials, hyps, r + 1)
            out.append({"round": r, "starts": step["add"], "copied": copied,
                        "sampled": [own_regret(step["add"], t["config"], world) for t in rec["trials"]
                                    if not t["queued"]],
                        "incumbent": own_regret(step["add"], incumbent["config"] if incumbent else {}, world)})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="The new levers' own regret in the round each hypothesis starts in.")
    ap.add_argument("--seeds", type=int, default=100, help="runs per variant, seeded 0, 1, … (default 100)")
    ap.add_argument("--trials", type=int, default=6, help="trials per round (default 6)")
    ap.add_argument("--repeats", type=int, default=3, help="eval runs per trial (default 3)")
    ap.add_argument("--scenario", default="default", choices=sorted(bench.SCENARIOS))
    ap.add_argument("--variant", action="append", choices=sorted(bench.VARIANTS), help="repeat (default harness)")
    args = ap.parse_args(argv)
    scenario = bench.SCENARIOS[args.scenario]
    cfg = {"rounds": len(scenario), "trials_per_round": args.trials, "repeats": args.repeats}
    for name in args.variant or ["harness"]:
        runs = [first_rounds(bench.VARIANTS[name], seed, cfg, scenario) for seed in range(args.seeds)]
        rows = [[row["round"], ", ".join(row["starts"]), f"{np.mean([run[i]['copied'] for run in runs]):.3g}",
                 bench._spread([x for run in runs for x in run[i]["sampled"]]),
                 bench._spread([run[i]["incumbent"] for run in runs])] for i, row in enumerate(runs[0])]
        print(f"{name}: the new levers' own regret (s), median [IQR] over sampled trials and over seeds\n")
        print(rounds.md_table(["round", "starts", "copied (mean)", "sampled trials", "incumbent after"], rows) + "\n")


if __name__ == "__main__":
    main()
