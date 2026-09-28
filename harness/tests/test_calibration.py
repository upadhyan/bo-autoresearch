"""Calibration round (R0) and the budget ledger, against toys with planted noise, ranks and cost."""
import os

import pytest

from conftest import bo, events


def toy_env(**planted):
    return {**os.environ, **{f"TOY_{k.upper()}": str(v) for k, v in planted.items()}}


def only(run_dir, type):
    [e] = [e for e in events(run_dir) if e["type"] == type]
    return e


def trials_of(run_dir, kind):
    return [e["payload"] for e in events(run_dir)
            if e["type"] == "trial_started" and e["payload"]["kind"] == kind]


def test_r0_estimates_the_planted_noise_from_baseline_replicates(repo, init_run):
    run_dir = init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\nreplicates_k: 30\n")
    code, out = bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sigma=0.5))
    assert code == 0, out

    assert len(trials_of(run_dir, "smoke")) == 1
    baseline = trials_of(run_dir, "baseline")
    assert len(baseline) == 30
    assert all(t["levers"] == {"H1.scale": 2.0} and t["round"] == 0 for t in baseline)
    assert len({t["seed"] for t in baseline}) == 30  # a fresh seed per replicate

    noise = only(run_dir, "noise_estimate")
    assert noise["actor"] == "harness"
    p = noise["payload"]
    assert p["sigma"] == pytest.approx(0.5, rel=0.4)  # k=30: the sample sd is within 40% w.p. >0.999
    assert p["replication"] is True and p["deterministic"] is False
    [ref] = p["rungs"]
    assert ref["fidelity"] == {"epochs": 4} and ref["sigma"] == p["sigma"]
    assert ref["mean"] == pytest.approx(2.5, abs=0.5)
    assert ref["cost_s"] > 0
    assert only(run_dir, "round_ended")["payload"] == {"round": 0, "trigger": "calibrated"}
    assert out["noise_estimate"]["sigma"] == p["sigma"]


LADDER = "budget_s: 3600\nreference_fidelity: {epochs: 8}\nladder: [{epochs: 1}]\n"


def test_a_faithful_cheap_rung_is_chosen_when_it_is_cheaper_to_a_verdict(repo, init_run):
    run_dir = init_run(LADDER)
    code, out = bo(repo, "round-run", "--rationale", "calibrate",
                   env=toy_env(sigma=0.01, sleep_per_epoch=0.03))
    assert code == 0, out

    cal = only(run_dir, "fidelity_calibration")["payload"]
    configs = cal["configs"]
    assert len(configs) == 5 and len({c["H1.scale"] for c in configs}) == 5
    assert len(trials_of(run_dir, "calibration")) == 10  # 5 configs at the rung and the reference
    [rung] = cal["rungs"]
    assert rung["fidelity"] == {"epochs": 1}
    assert rung["rho"] == pytest.approx(1.0) and rung["passed"] is True
    assert rung["expected_cost_s"] < cal["reference"]["expected_cost_s"]
    assert cal["chosen"] == {"epochs": 1} and cal["fallback"] is False
    _, status = bo(repo, "status")
    assert status["fidelity"] == {"fidelity": {"epochs": 1}, "proxy": "validated", "fallback": False}


def test_a_scrambled_cheap_rung_fails_and_the_loop_falls_back_to_the_reference(repo, init_run):
    run_dir = init_run(LADDER)
    code, out = bo(repo, "round-run", "--rationale", "calibrate",
                   env=toy_env(sigma=0.01, sleep_per_epoch=0.03, scramble_below=2))
    assert code == 0, out

    cal = only(run_dir, "fidelity_calibration")["payload"]
    [rung] = cal["rungs"]
    assert rung["rho"] == pytest.approx(-1.0) and rung["passed"] is False
    assert cal["chosen"] == {"epochs": 8} and cal["fallback"] is True
    _, status = bo(repo, "status")
    assert status["fidelity"] == {"fidelity": {"epochs": 8}, "proxy": "reference", "fallback": True}
    assert not [e for e in events(run_dir) if e["type"] == "proxy_accepted_unvalidated"]

    code, out = bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}',
                   "--rationale", "the user accepts the unvalidated rung")
    assert code == 0, out
    accepted = only(run_dir, "proxy_accepted_unvalidated")
    assert accepted["actor"] == "orchestrator"
    assert accepted["payload"]["fidelity"] == {"epochs": 1}
    _, status = bo(repo, "status")
    assert status["fidelity"] == {"fidelity": {"epochs": 1}, "proxy": "unvalidated"}


def test_accept_proxy_is_refused_for_the_reference_a_passing_rung_or_before_r0(repo, init_run):
    run_dir = init_run(LADDER)
    code, out = bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "early")
    assert code != 0 and out["refused"] and "calibration" in out["reason"]
    bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sigma=0.01))
    for fid in ('{"epochs": 8}', '{"epochs": 1}', '{"epochs": 3}'):
        code, out = bo(repo, "accept-proxy", "--fidelity", fid, "--rationale", "why")
        assert code != 0 and out["refused"], fid
    assert not [e for e in events(run_dir) if e["type"] == "proxy_accepted_unvalidated"]


def test_a_deterministic_declaration_turns_off_replicates(repo, init_run):
    run_dir = init_run(LADDER + "deterministic: true\n")
    code, out = bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sigma=0.5))
    assert code == 0, out

    assert trials_of(run_dir, "baseline") == []
    noise = only(run_dir, "noise_estimate")["payload"]
    assert noise["sigma"] == 0 and noise["replication"] is False and noise["deterministic"] is True
    assert [r["sigma"] for r in noise["rungs"]] == [0, 0]
    _, status = bo(repo, "status")
    assert status["replication"] is False and status["r0_complete"] is True


def test_agreeing_replicates_give_zero_sigma_and_turn_off_replication(repo, init_run):
    run_dir = init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\n")
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0
    assert len(trials_of(run_dir, "baseline")) == 5  # the default k
    noise = only(run_dir, "noise_estimate")["payload"]
    assert noise["sigma"] == 0 and noise["replication"] is False and noise["deterministic"] is False


def test_round_run_after_r0_is_refused_until_a_hypothesis_is_registered(repo, init_run):
    run_dir = init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\n")
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0
    before = events(run_dir)
    code, out = bo(repo, "round-run", "--rationale", "again")
    assert code != 0 and out["refused"] and "hypothesis" in out["reason"]
    assert events(run_dir) == before


def test_a_trial_costing_more_than_the_remaining_budget_is_refused_and_logged(repo, init_run):
    run_dir = init_run("budget_s: 1.0\nreference_fidelity: {epochs: 4}\n")
    env = toy_env(sleep_per_epoch=0.15)  # ~0.6 s per trial: the first fits, a second can't
    assert bo(repo, "smoke", "--rationale", "first", env=env)[0] == 0
    code, out = bo(repo, "smoke", "--rationale", "second", env=env)
    assert code != 0 and out["refused"] and "budget" in out["reason"]

    refused = only(run_dir, "trial_refused")
    assert refused["actor"] == "harness"
    p = refused["payload"]
    assert p["kind"] == "smoke" and p["fidelity"] == {"epochs": 4}
    assert p["estimated_cost_s"] > p["remaining_s"] > 0
    assert len(trials_of(run_dir, "smoke")) == 1
    _, status = bo(repo, "status")
    assert status["trials"]["total"] == 1


def test_r0_stops_when_the_budget_runs_out(repo, init_run):
    run_dir = init_run("budget_s: 1.0\nreference_fidelity: {epochs: 4}\n")
    code, out = bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sleep_per_epoch=0.15))
    assert code != 0 and out["refused"] and "budget" in out["reason"]
    assert only(run_dir, "trial_refused")["payload"]["kind"] == "baseline"
    assert only(run_dir, "round_ended")["payload"] == {"round": 0, "trigger": "budget_spent"}
    assert not [e for e in events(run_dir) if e["type"] == "noise_estimate"]
    _, status = bo(repo, "status")
    assert status["r0_complete"] is False


@pytest.mark.parametrize("extra, field", [
    ("ladder: [{epochs: 4}]\n", "ladder"),
    ("ladder: {epochs: 1}\n", "ladder"),
    ("replicates_k: 1\n", "replicates_k"),
    ("deterministic: maybe\n", "deterministic"),
])
def test_init_refuses_an_invalid_calibration_setup(repo, tmp_path, project_python, extra, field):
    from conftest import write_run_yaml
    y = write_run_yaml(tmp_path / "run.yaml", project_python,
                       "budget_s: 10\nreference_fidelity: {epochs: 4}\n" + extra)
    code, out = bo(repo, "init", str(y), "--rationale", "go")
    assert code != 0 and out["refused"] and field in out["reason"]
