"""#46's drift check: re-measuring the top copied configs, the order check, and the warm-start rules it feeds."""

from __future__ import annotations

import itertools
import random
import re

import bench
import drift_check
from boar import stats


def trial(n: int, r: int, config: dict, repeats: list[float], queued: str | None = None) -> dict:
    return {"trial": n, "round": r, "commit": f"c{r}", "config": {**bench.DEFAULTS, **config}, "state": "complete",
            "metric": stats.median(repeats), "repeats": repeats, "runs": len(repeats), "queued": queued}


def draw(lever: dict, rng: random.Random):
    if lever["type"] in ("bool", "categorical"):
        return rng.choice(lever.get("choices", [False, True]))
    return (rng.randint if lever["type"] == "int" else rng.uniform)(lever["low"], lever["high"])


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
    valid, excluded, incumbent = drift_check.Recheck(3, drop="trials").warm(trials, hyps, 3)
    assert [t["trial"] for t in valid] == [1, 4, 5, 6, 7, 8]
    assert excluded[drift_check.RULE4] == 2 and sum(excluded.values()) == 2
    assert incumbent["config"] == {"batch": 8}


def test_dropping_older_rounds_leaves_out_every_trial_from_before_a_failed_check():
    trials, hyps = checked_round()
    valid, excluded, incumbent = drift_check.Recheck(3, drop="older").warm(trials, hyps, 3)
    assert [t["trial"] for t in valid] == [5, 6, 7, 8]
    assert excluded[drift_check.RULE4] == 4 and incumbent["config"] == {"batch": 8}
    assert drift_check.Recheck(3, drop="older").warm(trials, hyps, 2)[1][drift_check.RULE4] == 0  # not checked yet


def test_ageing_leaves_out_trials_from_more_than_the_last_rounds():
    hyps = hypotheses({"add": ["H1", "H2", "H3", "H4"]})
    trials = [trial(1, 1, {}, [13.0, 13.2], "baseline"), trial(2, 1, {"batch": 2}, [9.0, 9.0]),
              trial(3, 2, {"batch": 8}, [11.0, 11.0]), trial(4, 3, {"batch": 32}, [12.0, 12.0]),
              trial(5, 4, {"batch": 4}, [11.5, 11.5])]
    valid, excluded, incumbent = drift_check.Age(3).warm(trials, hyps, 5)
    assert [t["trial"] for t in valid] == [3, 4, 5]
    assert excluded[drift_check.RULE4] == 2 and incumbent["config"] == {"batch": 8}
    assert drift_check.Age(3).warm(trials, hyps, 4)[1][drift_check.RULE4] == 0
    # Ageing the study's copy only: the incumbent is still picked over every round.
    valid, excluded, incumbent = drift_check.VARIANTS["age-3-study"].warm(trials, hyps, 5)
    assert [t["trial"] for t in valid] == [3, 4, 5]
    assert excluded[drift_check.RULE4] == 2 and incumbent["config"] == {"batch": 2}


def test_the_drift_scenario_reorders_configs_where_it_says_and_keeps_the_regret_reference():
    rng = random.Random(0)
    levers = [lever for p in bench.PROPOSALS.values() for lever in p["levers"]]
    configs = [{lever["name"]: draw(lever, rng) for lever in levers} for _ in range(300)]
    worlds = drift_check.worlds(drift_check.DRIFT, len(drift_check.DRIFT))
    reorders = set()
    for r, (before, after) in enumerate(zip(worlds, worlds[1:]), 2):
        old, new = ([bench.true_metric(c, world)[0] for c in configs] for world in (before, after))
        if any((old[i] - old[j]) * (new[i] - new[j]) < 0 for i, j in itertools.combinations(range(len(configs)), 2)):
            reorders.add(r)
    assert reorders == {4, 6, 7}  # rounds 2, 3 and 8 add a lever or scale every config; round 5 changes nothing
    for world in worlds:
        best = bench.true_metric(bench.best_config(world), world)[0]
        assert all(m >= best for m, ok in (bench.true_metric(c, world) for c in configs) if ok)


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
                      "--variant", "harness", "--variant", "drop-older"])
    out = capsys.readouterr().out
    rows = table_rows(out, "harness at 3 trials per round: 2 seeds; median [IQR] over seeds")
    assert [row[0] for row in rows] == [*map(str, range(1, 9)), "all"] and rows[0][2] == "3 [3, 3]"
    assert table_rows(out, "drop-older: 2 seeds; median [IQR] over seeds")[0][2] == "2 [2, 2]"
    assert "drop-older − harness at 3 trials per round, per seed:" in out


def test_the_command_line_measures_detection_over_noise_levels_and_configs_re_measured(monkeypatch, capsys):
    monkeypatch.setattr(bench, "SCENARIO", bench.SCENARIO)
    drift_check.main(["--detect", "--scenario", "drift", "--seeds", "1", "--trials", "2", "--repeats", "2"])
    out = capsys.readouterr().out
    rows = table_rows(out, "drift: 1 seeds, 2 trials per round, 2 repeats; each check, over seeds")
    assert [row[:2] for row in rows] == [[str(noise), str(n)] for noise in drift_check.NOISES for n in (2, 3, 4)]
    assert table_rows(out, f"By round, at noise {bench.NOISE} with 3 configs re-measured:")
    assert bench.NOISE == 0.02


def test_the_steady_scenario_never_shifts_the_metric_and_commits_in_12_of_30_rounds():
    assert len(drift_check.STEADY) == 30 and sum(map(bool, drift_check.STEADY)) == 12  # a non-empty step commits
    assert drift_check.worlds(drift_check.STEADY, 30) == [bench.WORLD] * 30
    assert [h for step in drift_check.STEADY for h in step.get("add", [])] == list(bench.PROPOSALS)


def test_the_unchanged_scenario_is_steady_with_no_commit_after_round_4():
    assert drift_check.UNCHANGED[:4] == drift_check.STEADY[:4] and drift_check.UNCHANGED[4:] == [{}] * 26


class AsyncOn(bench.Harness):
    """Sets prefetch_async on in every sampled trial once H6 is active."""

    def backend(self, space, seed, n_startup):
        backend = super().backend(space, seed, n_startup)
        ask = backend.ask

        def ask_on(n):
            params, queued = ask(n)
            return ({**params, "prefetch_async": True} if "prefetch_async" in space and not queued else params), queued

        backend.ask = ask_on
        return backend


def test_the_command_line_counts_the_sampled_trials_that_move_a_lever_once_it_exists(monkeypatch, capsys):
    monkeypatch.setattr(bench, "SCENARIO", bench.SCENARIO)
    monkeypatch.setitem(drift_check.VARIANTS, "async-on", AsyncOn())
    drift_check.main(["--scenario", "steady", "--seeds", "1", "--trials", "2", "--repeats", "1",
                      "--variant", "harness", "--variant", "async-on", "--lever", "prefetch_async"])
    out = capsys.readouterr().out
    counts = dict(re.findall(r"^(\S+): (\d+/\d+) sampled trials from round 4 set prefetch_async away", out, re.M))
    on, sampled = counts["async-on"].split("/")
    assert on == sampled and int(sampled) > 30  # every sampled trial since H6 arrived, none from before
    assert set(counts) == {"harness", "async-on"}
