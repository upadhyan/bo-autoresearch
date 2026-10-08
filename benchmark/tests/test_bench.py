"""The benchmark: a seeded simulated run, its regret and cost, and variants swapped in through Harness."""

from __future__ import annotations

import random

import pytest

import bench
from boar import schema

SMALL = {"rounds": 6, "trials_per_round": 4, "repeats": 2}


def run(seed: int = 0, variant: bench.Harness | None = None) -> list[dict]:
    return bench.simulate(variant or bench.Harness(), seed, SMALL)


class Fixed:
    """A backend that asks for the same params every time."""

    def __init__(self, params: dict) -> None:
        self.params = params

    def add_warm(self, valid, defaults):
        pass

    def enqueue(self, label, params):
        pass

    def ask(self, n):
        return self.params, None

    def tell(self, state, metric):
        pass


class Baseline(bench.Harness):
    """Measures the baseline in every trial."""

    def backend(self, space, seed, n_startup):
        return Fixed({name: lever["default"] for name, lever in space.items()})


class Best(bench.Harness):
    """Measures the best config at the first commit in every trial."""

    def backend(self, space, seed, n_startup):
        return Fixed({name: value for name, value in bench.best_config(bench.WORLD).items() if name in space})


def test_the_scenarios_proposals_pass_the_harness_schema():
    assert schema.validate(list(bench.PROPOSALS.values()), set(), set(), allow_supersedes=False) == []


def test_a_seed_reproduces_its_run():
    assert run(3) == run(3)
    assert [t["metric"] for r in run(3) for t in r["trials"]] != [t["metric"] for r in run(4) for t in r["trials"]]


def test_a_removed_hypothesis_puts_its_levers_back_at_default():
    rounds = run()
    before, after = ([t for rec in rounds[a:b] for t in rec["trials"]] for a, b in ((0, 3), (3, 6)))
    assert any(t["config"]["buffer_kb"] != 8 for t in before)
    assert all(t["config"]["buffer_kb"] == 8 for t in after), "H3 was removed before round 4"


def test_the_warm_start_leaves_out_trials_by_the_harness_rules():
    rounds = run(2)  # a seed whose round 2 has a crash and a batch that ran
    earlier = [t for rec in rounds[:2] for t in rec["trials"]]
    crashed = [t for t in earlier if t["state"] == "failed"]  # H4 crashes above batch 16 in round 2
    moved_b = [t for t in earlier if t["state"] != "failed" and t["config"].get("batch", 1) != 1]  # H4 marked fix then
    assert crashed and moved_b
    assert rounds[2]["excluded"] == {"rule1_state": len(crashed), "rule2_outside_space": 0, "rule3_fixed": len(moved_b)}
    kept = [t for rec in rounds[:3] for t in rec["trials"] if t["state"] != "failed"]
    assert rounds[3]["excluded"]["rule2_outside_space"] == sum(t["config"]["buffer_kb"] != 8 for t in kept)


class KeepsStudies(bench.Harness):
    """The harness, keeping each round's study."""

    def __init__(self) -> None:
        self.studies = []

    def backend(self, space, seed, n_startup):
        backend = super().backend(space, seed, n_startup)
        self.studies.append(backend._study)
        return backend


def test_each_round_warm_starts_from_the_trials_the_harness_rules_keep():
    variant = KeepsStudies()
    rounds = run(variant=variant)
    for rec, study in zip(rounds, variant.studies):
        earlier = [t["trial"] for prev in rounds[: rec["round"] - 1] for t in prev["trials"]]
        warm = [t.user_attrs["boar_trial"] for t in study.trials if t.user_attrs.get("warm")]
        assert set(warm) <= set(earlier) and len(warm) == len(earlier) - sum(rec["excluded"].values())


class Cold(KeepsStudies):
    """README.md's variant: no warm start."""

    def warm(self, trials, hyps, r):
        valid, excluded, incumbent = super().warm(trials, hyps, r)
        return [], excluded, incumbent


def test_a_variant_turns_off_the_warm_start_by_overriding_warm():
    variant = Cold()
    run(variant=variant)
    assert not any(t.user_attrs.get("warm") for study in variant.studies for t in study.trials)


def test_the_incumbent_is_measured_again_only_on_a_new_commit():
    rounds = run()
    r4, r5 = rounds[3], rounds[4]
    assert r4["trials"][0]["queued"] == "incumbent"
    assert r5["commit"] == r4["commit"], "R1 changed nothing before round 5"
    assert len(r5["trials"]) == SMALL["trials_per_round"] and not any(t["queued"] for t in r5["trials"])
    by_id = {t["trial"]: t for rec in rounds for t in rec["trials"]}
    assert by_id[r5["same_as"]["incumbent"]]["commit"] == r4["commit"]


class TryDedup(bench.Harness):
    """Also queues dedup_set on, with the other levers at the incumbent."""

    def queue(self, hyps, r, incumbent):
        warm = super().queue(hyps, r, incumbent)
        warm["queue"].append({"label": "try dedup", "config": {"dedup_set": True}})
        return warm


def test_a_variant_changes_a_policy_by_overriding_its_method():
    round1 = run(variant=TryDedup())[0]["trials"]
    assert [t["queued"] for t in round1[:2]] == ["baseline", "try dedup"] and round1[1]["config"]["dedup_set"]
    assert len(round1) == SMALL["trials_per_round"] + 1


def test_regret_and_cost_of_a_run_that_only_measures_the_baseline():
    rounds = run(variant=Baseline())
    # The seconds the baseline loses to the best config in each round's space: dedup_set 5.9 and route_table once 4.2;
    # from round 2 batch 16 1.6; from round 3 gc_scale 4 0.15 ln²4 and everything 1.1x slower; from round 4 batch 4
    # 0.4 instead of 1.6, and prefetch 1.2; in round 6 route_table per_file 4.3 instead of once's 4.2.
    gc = 0.15 * 1.921812
    assert [rec["regret"] for rec in rounds] == pytest.approx(
        [10.1, 11.7, *[1.1 * (11.7 + gc)] * 3, 1.1 * (11.8 + gc)], rel=1e-6)
    assert [rec["runs"] for rec in rounds] == [8] * 6  # 4 trials of 2 repeats
    assert [rec["remeasured"] for rec in rounds] == [3, 4, 4, 4, 4, 4]


def test_regret_scores_the_incumbent_after_the_round():
    assert run(variant=Best())[0]["regret"] == 0  # the incumbent before round 1, the baseline, loses 10.1


def test_a_crash_or_a_broken_guard_stops_the_repeats():
    world = {**bench.WORLD, "crash_above": 16}
    trials = [bench.measure(config, world, "c1", 0, 3) for config in ({"batch": 32}, {"prefetch": 0.9}, {})]
    assert [(t["state"], t["runs"]) for t in trials] == [("failed", 1), ("infeasible", 1), ("complete", 3)]


def test_a_config_measured_again_gets_the_same_values_on_its_commit_and_new_ones_on_the_next():
    first, again, new = (bench.measure({"batch": 4}, bench.WORLD, c, 0, 3)["repeats"] for c in ("c1", "c1", "c2"))
    assert again == first != new


def _draw(lever: dict, rng: random.Random):
    if lever["type"] in ("bool", "categorical"):
        return rng.choice(lever.get("choices", [False, True]))
    pick = rng.randint if lever["type"] == "int" else rng.uniform
    return pick(lever["low"], lever["high"])


def test_no_feasible_config_beats_the_regret_reference():
    rng = random.Random(0)
    levers = [lever for p in bench.PROPOSALS.values() for lever in p["levers"]]
    world = dict(bench.WORLD)
    for step in bench.SCENARIO:
        world.update(step.get("world", {}))
        best, ok = bench.true_metric(bench.best_config(world), world)
        assert ok
        for _ in range(2000):
            metric, ok = bench.true_metric({lever["name"]: _draw(lever, rng) for lever in levers}, world)
            assert metric >= best or not ok


def test_the_report_gives_the_median_and_iqr_over_seeds_per_round_and_overall():
    seeds = [[{"round": 1, "regret": x, "runs": 6, "remeasured": 0},
              {"round": 2, "regret": 2 * x, "runs": 8, "remeasured": 1}] for x in (1, 2, 3, 4, 5)]
    rows = [line.strip("| ").split(" | ") for line in bench.report(seeds).splitlines()[2:]]
    assert rows == [
        ["1", "3 [2, 4]", "6 [6, 6]", "0 [0, 0]"],
        ["2", "6 [4, 8]", "8 [8, 8]", "1 [1, 1]"],
        ["all", "4.5 [3, 6]", "14 [14, 14]", "1 [1, 1]"],  # regret: the mean over rounds; cost: the total
    ]


def test_seeds_rounds_and_trials_come_from_the_command_line(capsys):
    bench.main(["--seeds", "2", "--rounds", "2", "--trials", "3", "--repeats", "1"])
    out = capsys.readouterr().out
    assert "harness: 2 seeds" in out
    assert [line.split(" | ")[0].strip("| ") for line in out.splitlines() if line.startswith("| ")] == [
        "round", "1", "2", "all"]
    assert "| 3 [3, 3] |" in out  # round 1: 3 trials of 1 repeat


@pytest.mark.parametrize("flag", ["--seeds", "--rounds", "--trials", "--repeats"])
def test_the_command_line_refuses_a_count_below_one(flag):
    with pytest.raises(SystemExit):
        bench.main([flag, "0"])
