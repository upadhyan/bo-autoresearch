"""#46's drift check: re-measuring the top copied configs, the order check, and the warm-start rules it feeds."""

from __future__ import annotations

import itertools
import random

import bench
import drift_check
from boar import stats
from test_bench import _draw


def trial(n: int, r: int, config: dict, repeats: list[float], queued: str | None = None) -> dict:
    return {"trial": n, "round": r, "commit": f"c{r}", "config": {**bench.DEFAULTS, **config}, "state": "complete",
            "metric": stats.median(repeats), "repeats": repeats, "runs": len(repeats), "queued": queued}


def test_a_pair_is_contradicted_when_its_order_reverses_by_more_than_the_floor_on_both_sides():
    earlier = [trial(1, 1, {"batch": 2}, [10.0, 10.0]), trial(2, 1, {"batch": 8}, [11.0, 11.0]),
               trial(3, 1, {"batch": 32}, [10.9, 10.9])]
    new = [trial(4, 2, {"batch": 2}, [12.0, 12.0], "incumbent"), trial(5, 2, {"batch": 8}, [10.5, 10.5], "recheck 1"),
           trial(6, 2, {"batch": 32}, [12.1, 12.1], "recheck 2")]
    # A floor of 0.2 at a metric of 10 is about 0.21 at 10.5-11. batch 2 and 8 swap by 1 and 1.5; batch 8 and 32
    # swap by 0.1 earlier, within the floor; batch 2 and 32 keep their order.
    assert drift_check.contradicted(earlier, new, 0.2, 10.0) == [('{"batch": 2}', '{"batch": 8}')]
    # With no floor (repeats that never differ), any reversal counts.
    assert len(drift_check.contradicted(earlier, new, None, None)) == 2


def hypotheses(step: dict) -> list[dict]:
    hyps: list[dict] = []
    bench._change(hyps, step, 1)
    return hyps


def test_the_next_best_copied_configs_are_queued_after_the_incumbent_as_full_configs():
    hyps = hypotheses({"add": ["H1", "H2", "H3", "H4"]})
    trials = [trial(1, 1, {}, [13.0, 13.0], "baseline"), trial(2, 1, {"batch": 2, "dedup_set": True}, [10.0, 10.0]),
              trial(3, 1, {"batch": 32}, [12.0, 12.0]), trial(4, 1, {"batch": 8}, [11.0, 11.0])]
    variant = drift_check.Recheck(3)
    _, _, incumbent = variant.warm(trials, hyps, 2)
    warm = variant.queue(hyps, 2, incumbent)
    assert warm["incumbent_config"] == {"batch": 2, "dedup_set": True}
    at_default = {"dedup_set": False, "route_table": "per_request", "buffer_kb": 8}  # not the incumbent's dedup_set
    assert warm["queue"] == [{"label": "recheck 1", "config": {**at_default, "batch": 8}},
                             {"label": "recheck 2", "config": {**at_default, "batch": 32}}]


def test_a_new_commit_measures_the_rechecks_after_the_incumbent_on_top_of_the_rounds_trials():
    round2 = bench.simulate(drift_check.Recheck(3), 0, {"rounds": 2, "trials_per_round": 4, "repeats": 2})[1]
    assert [t["queued"] for t in round2["trials"][:3]] == ["incumbent", "recheck 1", "recheck 2"]
    assert len(round2["trials"]) == 4 + 2


def checked_round() -> tuple[list[dict], list[dict]]:
    """Round 2 re-measures batch 2, 8 and 32 on a new commit: batch 2 and 8 swap places, beyond the floor."""
    hyps = hypotheses({"add": ["H1", "H2", "H3", "H4"]})
    trials = [trial(1, 1, {}, [13.0, 13.2], "baseline"), trial(2, 1, {"batch": 2}, [9.0, 9.0]),
              trial(3, 1, {"batch": 8}, [11.0, 11.0]), trial(4, 1, {"batch": 32}, [12.0, 12.0]),
              trial(5, 2, {"batch": 2}, [12.0, 12.0], "incumbent"),
              trial(6, 2, {"batch": 8}, [10.5, 10.5], "recheck 1"),
              trial(7, 2, {"batch": 32}, [12.6, 12.6], "recheck 2"), trial(8, 2, {"batch": 4}, [11.5, 11.5])]
    return trials, hyps


def test_dropping_contradicted_trials_leaves_out_the_earlier_trials_of_both_configs():
    trials, hyps = checked_round()
    assert bench.Harness().warm(trials, hyps, 3)[2]["config"] == {"batch": 2}  # pooled over both commits: 10.5
    valid, _, incumbent = drift_check.Recheck(3, drop="trials").warm(trials, hyps, 3)
    assert [t["trial"] for t in valid] == [1, 4, 5, 6, 7, 8]
    assert incumbent["config"] == {"batch": 8}


def test_dropping_older_rounds_leaves_out_every_trial_from_before_a_failed_check():
    trials, hyps = checked_round()
    valid, _, incumbent = drift_check.Recheck(3, drop="older").warm(trials, hyps, 3)
    assert [t["trial"] for t in valid] == [5, 6, 7, 8] and incumbent["config"] == {"batch": 8}
    valid = drift_check.Recheck(3, drop="older").warm(trials, hyps, 2)[0]
    assert [t["trial"] for t in valid] == [1, 2, 3, 4]  # not checked yet


def test_ageing_leaves_out_trials_from_more_than_the_last_rounds():
    hyps = hypotheses({"add": ["H1", "H2", "H3", "H4"]})
    trials = [trial(1, 1, {}, [13.0, 13.2], "baseline"), trial(2, 1, {"batch": 2}, [9.0, 9.0]),
              trial(3, 2, {"batch": 8}, [11.0, 11.0]), trial(4, 3, {"batch": 32}, [12.0, 12.0]),
              trial(5, 4, {"batch": 4}, [11.5, 11.5])]
    valid, _, incumbent = drift_check.Age(3).warm(trials, hyps, 5)
    assert [t["trial"] for t in valid] == [3, 4, 5] and incumbent["config"] == {"batch": 8}
    assert [t["trial"] for t in drift_check.Age(3).warm(trials, hyps, 4)[0]] == [1, 2, 3, 4]
    # Ageing the study's copy only: the incumbent is still picked over every round.
    valid, _, incumbent = drift_check.VARIANTS["age-3-study"].warm(trials, hyps, 5)
    assert [t["trial"] for t in valid] == [3, 4, 5] and incumbent["config"] == {"batch": 2}
    # The baseline, the harness before rule 4, copies every trial rules 1-3 keep.
    assert [t["trial"] for t in drift_check.VARIANTS["copy-all"].warm(trials, hyps, 5)[0]] == [1, 2, 3, 4, 5]


def test_the_harness_now_runs_what_age_3_study_measured():
    cfg = {"rounds": 8, "trials_per_round": 3, "repeats": 2}
    for seed in range(2):
        harness, aged = (bench.simulate(v, seed, cfg) for v in (bench.Harness(), drift_check.VARIANTS["age-3-study"]))
        assert [(rec["regret"], rec["trials"]) for rec in harness] == [(rec["regret"], rec["trials"]) for rec in aged]
        assert harness[5]["excluded"]["rule4_old"] > 0  # rounds 1-2 left the study in round 6


def test_the_drift_scenario_reorders_configs_where_it_says():
    rng = random.Random(0)
    levers = [lever for p in bench.PROPOSALS.values() for lever in p["levers"]]
    configs = [{lever["name"]: _draw(lever, rng) for lever in levers} for _ in range(300)]
    worlds = drift_check.worlds(drift_check.DRIFT)
    reorders = set()
    for r, (before, after) in enumerate(zip(worlds, worlds[1:]), 2):
        old, new = ([bench.true_metric(c, world)[0] for c in configs] for world in (before, after))
        if any((old[i] - old[j]) * (new[i] - new[j]) < 0 for i, j in itertools.combinations(range(len(configs)), 2)):
            reorders.add(r)
    assert reorders == {4, 6, 7}  # rounds 2, 3 and 8 add a lever or scale every config; round 5 changes nothing


def test_without_noise_a_check_flags_exactly_when_the_copied_data_misorders_the_configs_it_measured(monkeypatch):
    monkeypatch.setattr(bench, "NOISE", 0.0)
    monkeypatch.setattr(bench, "SCENARIO", drift_check.DRIFT)
    cfg = {"rounds": 8, "trials_per_round": 4, "repeats": 2}
    checks = [c for seed in range(3) for c in drift_check.detection(bench.simulate(drift_check.Recheck(3), seed, cfg))]
    assert {c["round"] for c in checks} <= {2, 3, 4, 6, 7, 8}  # each new commit; round 5 is the same commit
    assert any(c["misordered"] for c in checks) and not all(c["misordered"] for c in checks)
    assert all(c["flagged"] == c["misordered"] for c in checks)


def table_rows(out: str, after: str) -> list[list[str]]:
    """The rows of the first table printed after the line `after`."""
    lines = out.split(after + "\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
    return [line.strip("| ").split(" | ") for line in lines[2:]]


def test_the_command_line_compares_variants_on_a_scenario_with_more_trials_for_the_first(monkeypatch, capsys):
    monkeypatch.setattr(bench, "SCENARIO", bench.SCENARIO)
    drift_check.main(["--scenario", "drift", "--seeds", "2", "--trials", "2", "--repeats", "1", "--base-trials", "3",
                      "--variant", "harness", "--variant", "drop-older", "--lever", "prefetch_async"])
    out = capsys.readouterr().out
    rows = table_rows(out, "harness at 3 trials per round: 2 seeds; median [IQR] over seeds")
    assert [row[0] for row in rows] == [*map(str, range(1, 9)), "all"] and rows[0][2] == "3 [3, 3]"
    assert table_rows(out, "drop-older: 2 seeds; median [IQR] over seeds")[0][2] == "2 [2, 2]"
    assert "drop-older − harness at 3 trials per round, per seed:" in out
    assert out.count("sampled trials from round 6 set prefetch_async away from its default") == 2  # H6's round


def test_the_command_line_measures_detection_over_noise_levels_and_configs_re_measured(monkeypatch, capsys):
    monkeypatch.setattr(bench, "SCENARIO", bench.SCENARIO)
    drift_check.main(["--detect", "--scenario", "drift", "--seeds", "1", "--trials", "2", "--repeats", "2"])
    out = capsys.readouterr().out
    rows = table_rows(out, "drift: 1 seeds, 2 trials per round, 2 repeats; each check, over seeds")
    assert [row[:2] for row in rows] == [[str(noise), str(n)] for noise in drift_check.NOISES for n in (2, 3, 4)]
    by_round = table_rows(out, "By round, at noise 0.02 with 3 configs re-measured:")
    assert by_round and {row[0] for row in by_round} <= {str(r) for r in range(2, 9)}
    assert bench.NOISE == 0.02


def test_the_steady_scenario_never_shifts_the_metric_and_commits_in_12_of_30_rounds():
    assert len(drift_check.STEADY) == 30 and sum(map(bool, drift_check.STEADY)) == 12  # a non-empty step commits
    assert drift_check.worlds(drift_check.STEADY) == [bench.WORLD] * 30
    assert [h for step in drift_check.STEADY for h in step.get("add", [])] == list(bench.PROPOSALS)


def test_the_unchanged_scenario_commits_nothing_after_round_4(monkeypatch):
    monkeypatch.setattr(bench, "SCENARIO", drift_check.UNCHANGED)
    run = bench.simulate(bench.Harness(), 0, {"rounds": 30, "trials_per_round": 1, "repeats": 1})
    assert [rec["commit"] for rec in run] == ["c1", "c2", "c3", *["c4"] * 27]
    assert drift_check.worlds(drift_check.UNCHANGED) == [bench.WORLD] * 30


def test_the_lever_count_takes_the_sampled_trials_from_the_round_that_adds_the_lever(monkeypatch):
    monkeypatch.setattr(bench, "SCENARIO", drift_check.STEADY)  # H6 in round 4
    on = {"prefetch_async": True}
    run = [{"trials": [trial(1, 1, on, [1.0])]}, {"trials": []}, {"trials": []},
           {"trials": [trial(2, 4, on, [1.0], "incumbent"), trial(3, 4, on, [1.0]), trial(4, 4, {}, [1.0])]}]
    assert drift_check.moved([run], "prefetch_async") == (
        "1/2 sampled trials from round 4 set prefetch_async away from its default")
