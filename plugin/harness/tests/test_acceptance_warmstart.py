"""Acceptance: the four warm-start rules (design.md "BO and warm start"), on hand-built records.

An earlier trial joins round r's warm-start set when (1) its state is complete or infeasible,
(2) every lever outside round r's search space was at its default, and (3) none of the levers it
set away from default belongs to a hypothesis marked `fix` after the trial ran. The study copies
in (4) the set's trials from the last 3 rounds.
"""

from __future__ import annotations

import inspect

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState

from boar import store, warmstart

NO_EXCLUSIONS = {"rule1_state": 0, "rule2_outside_space": 0, "rule3_fixed": 0}


def hyp(hid: str, status: str, *levers: dict, decisions: dict | None = None, activated: int | None = 1) -> dict:
    return {
        "id": hid, "round": 0, "status": status,
        "statement": f"{hid} moves the metric", "mechanism": "toy", "source": "user", "citations": [],
        "levers": list(levers), "supersedes": None,
        "verdict": {"decision": "accept", "reason": "ok", "at": "2026-10-02T00:00:00+00:00"},
        "activated_round": activated if status in ("active", "removed") else None,
        "removed_round": None, "removed_reason": None,
        "decisions": {str(r): {"decision": d, "reason": "r", "trials": [1], "at": "2026-10-02T00:00:00+00:00"}
                      for r, d in (decisions or {}).items()},
    }


def bool_lever(name: str) -> dict:
    return {"name": name, "type": "bool", "default": False}


def int_lever(name: str, low: int = 1, high: int = 8, default: int = 1) -> dict:
    return {"name": name, "type": "int", "low": low, "high": high, "log": False, "default": default}


def trial(n: int, round_: int, config: dict, state: str = "complete", metric: float | None = 50.0) -> dict:
    return {
        "trial": n, "round": round_, "commit": "abc1234", "config": config, "state": state,
        "metric": metric, "repeats": [] if metric is None else [metric], "metrics": {},
        "duration_s": 0.1, "error": "exit 3" if state == "failed" else None, "queued": None,
    }


def ids(valid: list[dict]) -> list[int]:
    return sorted(t["trial"] for t in valid)


def frozen_trials(valid: list[dict], hyps: list[dict]) -> list:
    """build_frozen_trials over the hypotheses' search space; passes `direction` only if it is taken."""
    args = [valid, store.search_space(hyps), store.lever_defaults(hyps)]
    if len(inspect.signature(warmstart.build_frozen_trials).parameters) > 3:
        args.append("min")
    return warmstart.build_frozen_trials(*args)


def test_rule1_failed_trials_are_excluded_infeasible_ones_kept():
    hyps = [hyp("H1", "active", bool_lever("fast_a"))]
    trials = [
        trial(1, 1, {"fast_a": False}),
        trial(2, 1, {"fast_a": True}, state="infeasible", metric=30.0),
        trial(3, 1, {"fast_a": True}, state="failed", metric=None),
        trial(4, 1, {"fast_a": False}, state="failed", metric=None),
    ]
    valid, excluded = warmstart.select(trials, hyps, 2)
    assert ids(valid) == [1, 2]
    assert excluded == {**NO_EXCLUSIONS, "rule1_state": 2}


def test_rule1_infeasible_trial_without_a_metric_is_excluded():
    hyps = [hyp("H1", "active", bool_lever("fast_a"))]
    trials = [trial(1, 1, {"fast_a": False}), trial(2, 1, {"fast_a": True}, state="infeasible", metric=None)]
    valid, excluded = warmstart.select(trials, hyps, 2)
    assert ids(valid) == [1]
    assert excluded == {**NO_EXCLUSIONS, "rule1_state": 1}


def test_rule2_levers_outside_the_search_space_must_be_at_default():
    hyps = [
        hyp("H1", "active", bool_lever("fast_a")),
        hyp("H2", "removed", int_lever("fast_b")),
        hyp("H3", "pending", bool_lever("noop"), activated=None),
        hyp("H4", "rejected", bool_lever("cache_output"), activated=None),
    ]
    base = {"fast_a": False, "fast_b": 1, "noop": False, "cache_output": False}
    trials = [
        trial(1, 1, {**base, "fast_b": 4}),
        trial(2, 1, {**base, "fast_b": 1}),
        trial(3, 1, {**base, "noop": True}),
        trial(4, 1, {**base, "cache_output": True, "fast_a": True}),
        trial(5, 2, {**base, "fast_a": True}),
        trial(6, 1, {"fast_a": True}),
    ]
    valid, excluded = warmstart.select(trials, hyps, 3)
    assert ids(valid) == [2, 5, 6], "only trials with every pinned lever at default (missing = default) join"
    assert excluded == {**NO_EXCLUSIONS, "rule2_outside_space": 3}


def test_rule3_levers_of_a_hypothesis_fixed_after_the_trial_ran_drop_out():
    hyps = [
        hyp("H1", "active", bool_lever("fast_a"), decisions={1: "keep", 2: "fix", 3: "keep"}),
        hyp("H2", "active", int_lever("fast_b"), decisions={1: "keep", 2: "keep", 3: "keep"}),
    ]
    trials = [
        trial(1, 1, {"fast_a": True, "fast_b": 1}),
        trial(2, 2, {"fast_a": True, "fast_b": 1}),
        trial(3, 3, {"fast_a": True, "fast_b": 1}),
        trial(4, 1, {"fast_a": False, "fast_b": 5}),
        trial(5, 2, {"fast_a": False, "fast_b": 3}),
    ]
    valid, excluded = warmstart.select(trials, hyps, 4)
    assert ids(valid) == [3, 4, 5], (
        "fix in round 2 drops trials of rounds 1-2 that set fast_a; a round-3 trial ran after the repair"
    )
    assert excluded == {**NO_EXCLUSIONS, "rule3_fixed": 2}


def test_rule3_fix_in_a_later_round_reaches_back():
    hyps = [hyp("H1", "active", bool_lever("fast_a"), decisions={1: "keep", 3: "fix"})]
    trials = [trial(1, 1, {"fast_a": True}), trial(2, 2, {"fast_a": True}), trial(3, 3, {"fast_a": True})]
    valid, excluded = warmstart.select(trials, hyps, 4)
    assert ids(valid) == []
    assert excluded == {**NO_EXCLUSIONS, "rule3_fixed": 3}


def test_the_first_failing_rule_counts():
    hyps = [
        hyp("H1", "active", bool_lever("fast_a"), decisions={1: "fix"}),
        hyp("H2", "removed", int_lever("fast_b")),
    ]
    trials = [
        trial(1, 1, {"fast_a": True, "fast_b": 4}, state="failed", metric=None),
        trial(2, 1, {"fast_a": True, "fast_b": 4}),
        trial(3, 1, {"fast_a": True, "fast_b": 1}),
    ]
    valid, excluded = warmstart.select(trials, hyps, 2)
    assert valid == []
    assert excluded == {"rule1_state": 1, "rule2_outside_space": 1, "rule3_fixed": 1}


def test_only_earlier_rounds_are_considered():
    hyps = [hyp("H1", "active", bool_lever("fast_a"))]
    trials = [trial(1, 1, {"fast_a": True}), trial(2, 2, {"fast_a": True}), trial(3, 3, {"fast_a": False})]
    valid, excluded = warmstart.select(trials, hyps, 2)
    assert ids(valid) == [1]
    assert excluded == NO_EXCLUSIONS


def test_rule4_the_study_copies_the_warm_start_set_of_the_last_3_rounds():
    hyps = [hyp("H1", "active", bool_lever("fast_a"))]
    trials = [trial(n, n, {"fast_a": n % 2 == 0}) for n in range(1, 5)] + [trial(5, 1, {}, "failed", None)]
    copied, excluded = warmstart.copied(trials, hyps, 5)
    assert ids(copied) == [2, 3, 4]
    assert excluded == {**NO_EXCLUSIONS, "rule1_state": 1, "rule4_old": 1}, "the failed trial counts under rule 1"
    assert ids(warmstart.select(trials, hyps, 5)[0]) == [1, 2, 3, 4], "the warm-start set keeps every round"


def _round3_hyps() -> list[dict]:
    return [
        hyp("H1", "active", bool_lever("fast_a")),
        hyp("H2", "active", int_lever("fast_b")),
        hyp("H3", "active", {"name": "scale", "type": "float", "low": 0.5, "high": 2.0, "log": False, "default": 1.0},
            activated=3),
        hyp("H4", "active", {"name": "mode", "type": "categorical", "choices": ["x", "y", "z"], "default": "x"},
            activated=3),
    ]


def test_frozen_trials_fill_new_levers_at_default_and_carry_constraints():
    hyps = _round3_hyps()
    trials = [
        trial(1, 1, {"fast_a": False, "fast_b": 1}, metric=100.0),
        trial(2, 1, {"fast_a": True, "fast_b": 6}, state="infeasible", metric=70.0),
        trial(3, 2, {"fast_a": True, "fast_b": 3}, metric=74.0),
    ]
    valid, _ = warmstart.select(trials, hyps, 3)
    assert ids(valid) == [1, 2, 3]
    frozen = frozen_trials(valid, hyps)
    assert len(frozen) == 3
    by_trial = {ft.user_attrs["boar_trial"]: ft for ft in frozen}
    assert sorted(by_trial) == [1, 2, 3]

    for n, ft in by_trial.items():
        src = next(t for t in trials if t["trial"] == n)
        assert ft.state == TrialState.COMPLETE
        assert ft.value == pytest.approx(src["metric"])
        assert ft.user_attrs["warm"] is True
        assert ft.params == {"fast_a": src["config"]["fast_a"], "fast_b": src["config"]["fast_b"],
                             "scale": 1.0, "mode": "x"}, "levers new in round 3 are filled in at their default"
        assert ft.constraints == {"guards": 1.0 if src["state"] == "infeasible" else 0.0}
        dist = ft.distributions
        assert isinstance(dist["fast_a"], CategoricalDistribution) and set(dist["fast_a"].choices) == {False, True}
        assert isinstance(dist["fast_b"], IntDistribution) and (dist["fast_b"].low, dist["fast_b"].high) == (1, 8)
        assert isinstance(dist["scale"], FloatDistribution) and (dist["scale"].low, dist["scale"].high) == (0.5, 2.0)
        assert isinstance(dist["mode"], CategoricalDistribution) and list(dist["mode"].choices) == ["x", "y", "z"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=2, seed=0)
    )
    study.add_trials(frozen)
    assert len(study.trials) == 3
    assert {t.user_attrs["boar_trial"]: t.constraints["guards"] for t in study.trials} == {1: 0.0, 2: 1.0, 3: 0.0}
    t = study.ask({name: dist for name, dist in frozen[0].distributions.items()})
    assert set(t.params) == {"fast_a", "fast_b", "scale", "mode"}


def test_values_outside_the_current_distribution_never_reach_the_study():
    hyps = _round3_hyps()
    trials = [trial(1, 1, {"fast_a": False, "fast_b": 1}), trial(2, 2, {"fast_a": True, "fast_b": 12})]
    valid, excluded = warmstart.select(trials, hyps, 3)
    frozen = frozen_trials(valid, hyps)
    assert [ft.user_attrs["boar_trial"] for ft in frozen] == [1]
    assert excluded["rule2_outside_space"] + len(valid) - len(frozen) == 1, "the skipped trial counts under rule 2"
