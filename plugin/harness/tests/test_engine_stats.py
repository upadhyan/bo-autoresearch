"""stats: medians, spreads, Spearman with ties, pooling, incumbent tie-breaks and the noise floor."""

from __future__ import annotations

import pytest

from boar import stats

DEFAULTS = {"a": False, "b": 1}


def trial(n, config, repeats, state="complete", round_=1):
    return {"trial": n, "round": round_, "config": config, "state": state, "repeats": repeats,
            "metric": stats.median(repeats)}


def test_median_and_spread():
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 2, 3]) == 2.5
    assert stats.median([]) is None
    assert stats.spread([5, 2, 9]) == 7
    assert stats.spread([]) is None


def test_spearman_perfect_and_inverse():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [9, 7, 5, 1]) == pytest.approx(-1.0)


def test_spearman_ties_use_average_ranks():
    # x ranks: 1, 2.5, 2.5, 4; y ranks: 1, 2, 3, 4 -> Pearson of ranks = 0.9486…
    assert stats.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)


def test_spearman_undefined_cases():
    assert stats.spearman([1, 2], [1, 2]) is None
    assert stats.spearman([1, 1, 1], [1, 2, 3]) is None
    assert stats.spearman([1, 2, 3], [5, 5, 5]) is None


def test_pooled_groups_by_non_default_config_and_skips_non_complete():
    trials = [
        trial(1, {"a": False, "b": 1}, [10, 11]),
        trial(2, {"a": False}, [12]),  # missing lever = default -> same key as trial 1
        trial(3, {"a": True, "b": 1}, [5, 6]),
        trial(4, {"a": True, "b": 1}, [50], state="infeasible"),
        trial(5, {"a": True, "b": 1}, [], state="failed"),
    ]
    groups = stats.pooled(trials, DEFAULTS)
    assert set(groups) == {stats.BASELINE_KEY, '{"a": true}'}
    base = groups[stats.BASELINE_KEY]
    assert base["trials"] == [1, 2] and base["repeats"] == [10, 11, 12] and base["metric"] == 11
    assert groups['{"a": true}']["trials"] == [3] and groups['{"a": true}']["config"] == {"a": True}


def test_incumbent_uses_pooled_median_not_best_single_trial():
    trials = [
        trial(1, {"a": True}, [1]),  # lucky once …
        trial(2, {"a": True}, [9, 9]),  # … contradicted by its other runs: pooled median 9
        trial(3, {"b": 2}, [5, 5]),
    ]
    assert stats.incumbent(trials, DEFAULTS, "min")["config"] == {"b": 2}
    assert stats.incumbent(trials, DEFAULTS, "max")["config"] == {"a": True}


def test_incumbent_ties_prefer_fewer_levers_then_earliest_trial():
    trials = [trial(1, {"a": True, "b": 2}, [5]), trial(2, {"b": 3}, [5]), trial(3, {"b": 2}, [5])]
    assert stats.incumbent(trials, DEFAULTS, "min")["trials"] == [2]
    assert stats.incumbent([trial(1, {"a": True}, [4]), trial(2, {}, [4])], DEFAULTS, "min")["config"] == {}


def test_incumbent_none_without_feasible_trials():
    assert stats.incumbent([trial(1, {}, [3], state="infeasible")], DEFAULTS, "min") is None


def test_incumbent_skips_a_config_whose_latest_trial_broke_a_guard():
    trials = [
        trial(1, {"a": True}, [1, 1]),  # best on its passing repeats …
        trial(2, {"a": True}, [1], state="infeasible"),  # … but it broke a guard when measured again
        trial(3, {"b": 2}, [5, 5]),
    ]
    assert stats.incumbent(trials, DEFAULTS, "min")["config"] == {"b": 2}
    assert stats.incumbent(trials[:2], DEFAULTS, "min") is None
    assert set(stats.pooled(trials, DEFAULTS)) == {'{"a": true}', '{"b": 2}'}  # pooling itself is unchanged
    passed_again = [*trials, trial(4, {"a": True}, [1, 1])]
    assert stats.incumbent(passed_again, DEFAULTS, "min")["config"] == {"a": True}, "its latest trial decides"


def test_a_baseline_guard_failure_does_not_hand_the_win_to_a_worse_config():
    trials = [trial(n, {}, [100, 100.2]) for n in range(1, 6)]
    trials += [trial(6, {}, [100.1], state="infeasible"), trial(7, {"a": True}, [110, 110.1])]
    inc = stats.incumbent(trials, DEFAULTS, "min")
    assert inc["config"] == {} and inc["trials"] == [1, 2, 3, 4, 5]


def test_noise_floor_is_median_within_trial_spread_of_baseline_trials():
    trials = [trial(1, {}, [10, 12]), trial(2, {"a": False}, [9]), trial(3, {"a": True}, [0, 100])]
    assert stats.noise_floor(trials, DEFAULTS) == 2
    assert stats.noise_floor([trial(1, {}, [10])], DEFAULTS) is None
    assert stats.noise_floor([trial(1, {"a": True}, [1, 2])], DEFAULTS) is None
    assert stats.noise_floor([trial(1, {}, [10, 12], state="infeasible")], DEFAULTS) is None


def test_noise_floor_ignores_drift_between_baseline_trials():
    # Each trial spreads by 0.2; the machine got 10% slower between them. Pooled max - min would be 10.2.
    trials = [trial(1, {}, [100.0, 100.2]), trial(2, {}, [110.0, 110.2], round_=2), trial(3, {}, [100.1, 100.4])]
    assert stats.noise_floor(trials, DEFAULTS) == pytest.approx(0.2)
    assert [t["trial"] for t in stats.noise_trials(trials, DEFAULTS)] == [1, 2, 3]
