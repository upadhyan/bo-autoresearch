"""warmstart: the three rules (each on its own), new levers at default, and what the sampler sees."""

from __future__ import annotations

import optuna
import pytest
from optuna.samplers._tpe import sampler as tpe_module
from optuna.trial import TrialState

from boar import store, warmstart


def hyp(hid, status, *levers, decisions=None):
    return {"id": hid, "status": status, "levers": list(levers), "decisions": decisions or {}}


A = {"name": "a", "type": "bool", "default": False}
B = {"name": "b", "type": "int", "low": 1, "high": 8, "log": False, "default": 1}
C = {"name": "c", "type": "bool", "default": False}
D = {"name": "d", "type": "float", "low": 0.1, "high": 1.0, "log": True, "default": 1.0}
E = {"name": "e", "type": "categorical", "choices": ["x", "y", "z"], "default": "x"}


def trial(n, round_, config, state="complete", metric=10.0):
    return {"trial": n, "round": round_, "config": config, "state": state, "metric": metric,
            "repeats": [] if metric is None else [metric]}


@pytest.fixture(autouse=True)
def _quiet():
    warmstart.quiet_optuna()


def test_rule1_only_complete_or_infeasible_trials_with_a_metric_join():
    hyps = [hyp("H1", "active", A)]
    trials = [
        trial(1, 1, {"a": False}),
        trial(2, 1, {"a": True}, state="infeasible", metric=3.0),
        trial(3, 1, {"a": True}, state="failed", metric=None),
        trial(4, 1, {"a": True}, state="failed", metric=2.0),  # partial repeats before the crash
        trial(5, 1, {"a": True}, state="infeasible", metric=None),
    ]
    valid, counts = warmstart.select(trials, hyps, 2)
    assert [t["trial"] for t in valid] == [1, 2]
    assert counts == {"rule1_state": 3, "rule2_outside_space": 0, "rule3_fixed": 0}


def test_rule2_levers_outside_the_space_must_be_at_default():
    hyps = [hyp("H1", "active", A), hyp("H2", "removed", C), hyp("H3", "pending", E)]
    trials = [
        trial(1, 1, {"a": True, "c": True, "e": "x"}),  # removed hypothesis's lever moved
        trial(2, 1, {"a": True, "c": False, "e": "x"}),
        trial(3, 1, {"a": True}),  # c and e missing -> default
        trial(4, 1, {"a": False, "c": False, "e": "y"}),  # pinned pending lever moved
    ]
    valid, counts = warmstart.select(trials, hyps, 2)
    assert [t["trial"] for t in valid] == [2, 3]
    assert counts == {"rule1_state": 0, "rule2_outside_space": 2, "rule3_fixed": 0}


def test_rule2_rejects_a_space_value_outside_the_lever_distribution():
    hyps = [hyp("H1", "active", B)]
    valid, counts = warmstart.select([trial(1, 1, {"b": 9}), trial(2, 1, {"b": 4})], hyps, 2)
    assert [t["trial"] for t in valid] == [2]
    assert counts["rule2_outside_space"] == 1


def test_rule3_levers_of_a_hypothesis_marked_fix_after_the_trial_ran():
    fix_r2 = {"2": {"decision": "fix", "reason": "crashed", "trials": [4]}, "1": {"decision": "keep"}}
    hyps = [hyp("H1", "active", A, decisions=fix_r2), hyp("H2", "active", B)]
    trials = [
        trial(1, 1, {"a": True, "b": 1}),  # ran in round 1, fix came in round 2 -> out
        trial(2, 2, {"a": True, "b": 3}),  # same round as the fix decision (decided after it ran) -> out
        trial(3, 3, {"a": True, "b": 3}),  # ran after the fix -> in
        trial(4, 2, {"a": False, "b": 5}),  # H1's levers at default -> in
    ]
    valid, counts = warmstart.select(trials, hyps, 4)
    assert [t["trial"] for t in valid] == [3, 4]
    assert counts == {"rule1_state": 0, "rule2_outside_space": 0, "rule3_fixed": 2}


def test_keep_and_remove_decisions_do_not_trigger_rule3():
    hyps = [hyp("H1", "active", A, decisions={"1": {"decision": "keep"}, "2": {"decision": "remove"}})]
    valid, counts = warmstart.select([trial(1, 1, {"a": True})], hyps, 3)
    assert len(valid) == 1 and counts["rule3_fixed"] == 0


def test_each_trial_counts_under_its_first_failing_rule_and_later_rounds_are_ignored():
    hyps = [hyp("H1", "active", A, decisions={"1": {"decision": "fix"}}), hyp("H2", "removed", C)]
    trials = [
        trial(1, 1, {"a": True, "c": True}, state="failed", metric=None),  # fails 1, 2 and 3 -> rule 1
        trial(2, 1, {"a": True, "c": True}),  # fails 2 and 3 -> rule 2
        trial(3, 1, {"a": True}),  # fails 3 only
        trial(4, 2, {"a": True}),  # round 2 is not earlier than round 2: not considered at all
    ]
    valid, counts = warmstart.select(trials, hyps, 2)
    assert valid == []
    assert counts == {"rule1_state": 1, "rule2_outside_space": 1, "rule3_fixed": 1}


def test_new_levers_are_filled_in_at_default():
    hyps = [hyp("H1", "active", A), hyp("H4", "active", D, E)]
    space, defaults = store.search_space(hyps), store.lever_defaults(hyps)
    old = trial(1, 1, {"a": True})  # ran before H4's levers existed
    valid, _ = warmstart.select([old], hyps, 2)
    [frozen] = warmstart.build_frozen_trials(valid, space, defaults)
    assert frozen.params == {"a": True, "d": 1.0, "e": "x"}
    assert frozen.user_attrs == {"boar_trial": 1, "warm": True}


def test_distributions_match_lever_types():
    dists = warmstart.distributions({"a": A, "b": B, "d": D, "e": E})
    assert dists["a"] == optuna.distributions.CategoricalDistribution([False, True])
    assert dists["b"] == optuna.distributions.IntDistribution(1, 8)
    assert dists["d"] == optuna.distributions.FloatDistribution(0.1, 1.0, log=True)
    assert dists["e"] == optuna.distributions.CategoricalDistribution(["x", "y", "z"])


def _warm_study(tmp_path, n_startup):
    hyps = [hyp("H1", "active", A), hyp("H2", "active", B)]
    space, defaults = store.search_space(hyps), store.lever_defaults(hyps)
    trials = [
        trial(1, 1, {"a": False, "b": 1}, metric=10.0),
        trial(2, 1, {"a": True, "b": 4}, metric=5.0),
        trial(3, 1, {"a": True, "b": 8}, state="infeasible", metric=1.0),  # best value, but a guard failed
        trial(4, 1, {"a": False, "b": 2}, metric=9.0),
    ]
    valid, _ = warmstart.select(trials, hyps, 2)
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=n_startup, seed=0)
    study = optuna.create_study(storage=f"sqlite:///{tmp_path}/study.db", study_name="round-2",
                                sampler=sampler, direction="minimize")
    study.add_trials(warmstart.build_frozen_trials(valid, space, defaults))
    return study, sampler, warmstart.distributions(space)


def test_copied_constraints_are_stored_where_tpe_reads_them(tmp_path):
    study, _, _ = _warm_study(tmp_path, n_startup=4)
    stored = study.get_trials()
    assert [t.constraints for t in stored] == [{"guards": 0.0}, {"guards": 0.0}, {"guards": 1.0}, {"guards": 0.0}]
    assert stored[2].system_attrs["constraints:guards"] == 1.0
    below, above = tpe_module._split_trials(study, stored, 1)
    assert [t.number for t in below] == [1]  # the feasible best, not the infeasible trial with value 1.0
    assert 2 in [t.number for t in above]


def _count_model_samples(sampler):
    calls = {"n": 0}
    original = sampler._sample

    def spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    sampler._sample = spy
    return calls


def test_enough_warm_trials_skip_random_startup(tmp_path):
    study, sampler, dists = _warm_study(tmp_path, n_startup=4)
    calls = _count_model_samples(sampler)
    t = study.ask(dists)
    assert calls["n"] == 1
    study.tell(t, state=TrialState.FAIL)


def test_too_few_warm_trials_sample_at_random(tmp_path):
    study, sampler, dists = _warm_study(tmp_path, n_startup=10)
    calls = _count_model_samples(sampler)
    t = study.ask(dists)
    assert calls["n"] == 0
    study.tell(t, state=TrialState.FAIL)


def test_enqueued_trial_carries_its_label_into_ask(tmp_path):
    study, _, dists = _warm_study(tmp_path, n_startup=4)
    study.enqueue_trial({"a": True, "b": 4}, user_attrs={"queued": "incumbent"})
    t = study.ask(dists)
    assert t.params == {"a": True, "b": 4} and t.user_attrs == {"queued": "incumbent"}
    t.set_constraint("guards", 0.0)
    study.tell(t, 4.0)
    assert study.get_trials()[-1].constraints == {"guards": 0.0}


def test_incumbent_dethrones_a_config_whose_latest_trial_failed():
    hyps = [hyp("H1", "active", A, B)]
    trials = [
        trial(1, 1, {"a": False}, metric=10.0),
        trial(2, 1, {"a": True}, metric=5.0),
        trial(3, 2, {"a": True}, state="failed", metric=None),  # crashed when measured again
        trial(4, 2, {"b": 3}, metric=8.0),
    ]
    assert warmstart.incumbent(trials, hyps, 2, "min")["config"] == {"a": True}
    assert warmstart.incumbent(trials, hyps, 3, "min")["config"] == {"b": 3}
    restored = [*trials, trial(5, 3, {"a": True}, metric=5.0)]
    assert warmstart.incumbent(restored, hyps, 4, "min")["config"] == {"a": True}, "a later complete trial restores it"


def test_incumbent_keeps_a_baseline_that_failed_and_ignores_failures_outside_the_space():
    hyps = [hyp("H1", "active", A), hyp("H2", "removed", C)]
    trials = [
        trial(1, 1, {"a": False}, metric=10.0),
        trial(2, 1, {"a": False}, state="failed", metric=None),
        trial(3, 1, {"a": True}, metric=5.0),
        trial(4, 1, {"a": True, "c": True}, state="failed", metric=None),  # a removed lever moved: not {"a": True}
    ]
    assert warmstart.incumbent(trials, hyps, 2, "min")["config"] == {"a": True}
    assert warmstart.incumbent(trials[:2], hyps, 2, "min")["config"] == {}
    fixed = [hyp("H1", "active", A, decisions={"1": {"decision": "fix"}})]
    assert warmstart.incumbent(trials[:3], fixed, 2, "min")["config"] == {}, "rule 3 still applies"
