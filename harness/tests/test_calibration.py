"""Calibration round (R0) and the budget ledger, against toys with planted noise, ranks and cost."""
import json
import os
import random
import re
import statistics

import pytest

from conftest import bo, events, write_run_yaml

SHIFT = {  # a hypothesis shifting the toy's scale: its box is what the ladder is calibrated over
    "title": "Shift the scale", "rationale": "r", "mechanism": "m", "provenance": "novel",
    "lens": "l", "directives": [], "fidelity_sensitive": False,
    "levers": {"shift": {"kind": "float", "low": -1.0, "high": 1.0, "baseline": 0.0,
                         "predicted": "lower"}},
}


def toy_env(**planted):
    return {**os.environ, **{f"TOY_{k.upper()}": str(v) for k, v in planted.items()}}


def only(run_dir, type):
    [e] = [e for e in events(run_dir) if e["type"] == type]
    return e


def trials_of(run_dir, kind):
    return [e["payload"] for e in events(run_dir)
            if e["type"] == "trial_started" and e["payload"]["kind"] == kind]


def calibrate(repo, run_dir, env=None):
    """R0, then H2 (the shift) coded, then the round-run that calibrates the ladder over its box."""
    code, out = bo(repo, "round-run", "--rationale", "calibrate", env=env)
    assert code == 0, out
    f = run_dir / "shift.json"
    f.write_text(json.dumps(SHIFT))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    assert bo(repo, "register", "H2", "--rationale", "reviewed")[0] == 0
    train = run_dir / "worktree" / "train.py"
    anchor = '    scale, cheap = lever("H1.scale"), epochs < _env("CHEAP_BELOW")\n'
    train.write_text(train.read_text().replace(anchor, anchor + '    scale += lever("H2.shift")\n'))
    assert bo(repo, "smoke", "H2", "--rationale", "try", env=env)[1]["passed"]
    assert bo(repo, "commit-lever", "H2", "--rationale", "passed")[0] == 0
    return bo(repo, "round-run", "--rationale", "calibrate the ladder", env=env)


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
    planted = [0.5 * random.Random(t["seed"]).gauss(0, 1) for t in baseline]  # the toy's noise draws
    assert p["sigma"] == pytest.approx(statistics.stdev(planted))
    assert p["sigma"] == pytest.approx(0.5, rel=0.5)  # k=30: sd of the estimate is ~0.13σ
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
    code, out = calibrate(repo, run_dir, env=toy_env(sigma=0.01, sleep_per_epoch=0.03))
    assert code == 0, out

    cal = only(run_dir, "fidelity_calibration")["payload"]
    configs = cal["configs"]
    assert len(configs) == 5 and len({c["H2.shift"] for c in configs}) == 5
    assert all(c["H1.scale"] == 2.0 and -1 <= c["H2.shift"] <= 1 for c in configs)
    assert all(t["round"] == 0 for t in trials_of(run_dir, "calibration"))
    # the suggestion is 2σ of the toy's planted noise draws at the reference's baseline replicates
    planted = [0.01 * random.Random(t["seed"]).gauss(0, 1) for t in trials_of(run_dir, "baseline")
               if t["fidelity"] == {"epochs": 8}]
    [duty] = out["next"]
    assert duty.startswith("set-delta <value> (suggested 2σ = ")
    assert float(re.search(r"= ([0-9.e-]+)\)", duty)[1]) == pytest.approx(2 * statistics.stdev(planted),
                                                                          rel=1e-2)
    assert [e["payload"]["round"] for e in events(run_dir) if e["type"] == "round_started"] == [0]
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
    code, out = calibrate(repo, run_dir, env=toy_env(sigma=0.01, sleep_per_epoch=0.03, cheap_below=2, scramble=1))
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


def test_a_rung_whose_spread_is_under_3_sigma_fails_even_with_faithful_ranks(repo, init_run):
    run_dir = init_run(LADDER)
    code, out = calibrate(repo, run_dir, env=toy_env(sigma=0.01, cheap_below=2, cheap_replicate_sigma=5))
    assert code == 0, out
    cal = only(run_dir, "fidelity_calibration")["payload"]
    [rung] = cal["rungs"]
    assert rung["rho"] == pytest.approx(1.0)
    assert rung["spread"] < 3 * rung["sigma"] and rung["passed"] is False
    assert cal["chosen"] == {"epochs": 8} and cal["fallback"] is True


def test_a_passing_but_noisier_cheap_rung_loses_to_the_reference(repo, init_run):
    run_dir = init_run(LADDER)
    code, out = calibrate(repo, run_dir, env=toy_env(sigma=0.01, sleep_per_epoch=0.03,
                                                     cheap_below=2, cheap_replicate_sigma=0.3))
    assert code == 0, out
    cal = only(run_dir, "fidelity_calibration")["payload"]
    [rung] = cal["rungs"]
    assert rung["passed"] is True and rung["cost_s"] < cal["reference"]["cost_s"]
    assert cal["chosen"] == {"epochs": 8} and cal["fallback"] is False


def test_accept_proxy_is_refused_for_the_reference_a_passing_rung_or_before_r0(repo, init_run):
    run_dir = init_run(LADDER)
    code, out = bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "early")
    assert code != 0 and out["refused"] and "calibration" in out["reason"]
    calibrate(repo, run_dir, env=toy_env(sigma=0.01))
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
    run_dir = init_run("budget_s: 2.0\nreference_fidelity: {epochs: 4}\n")
    env = toy_env(sleep_per_epoch=0.3)  # ~1.3 s per trial: the first fits, a second can't
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
    run_dir = init_run("budget_s: 2.0\nreference_fidelity: {epochs: 4}\n")
    code, out = bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sleep_per_epoch=0.3))
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
    y = write_run_yaml(tmp_path / "run.yaml", project_python,
                       "budget_s: 10\nreference_fidelity: {epochs: 4}\n" + extra)
    code, out = bo(repo, "init", str(y), "--rationale", "go")
    assert code != 0 and out["refused"] and field in out["reason"]


def test_a_budget_stop_in_the_ladder_calibration_ends_the_run_and_keeps_r0(repo, init_run):
    run_dir = init_run("budget_s: 3.8\nreference_fidelity: {epochs: 8}\nladder: [{epochs: 1}]\n"
                       "replicates_k: 2\n")
    env = toy_env(sleep_per_epoch=0.1)  # R0 ~3 s, H2's smoke ~0.3 s: a reference config can't fit
    code, out = calibrate(repo, run_dir, env=env)
    assert code == 0, out
    assert out["trigger"] == "budget_spent" and out["run_ended"] == "budget_spent"
    assert only(run_dir, "trial_refused")["payload"]["kind"] == "calibration"
    assert [e["payload"] for e in events(run_dir) if e["type"] == "round_ended"] == [
        {"round": 0, "trigger": "calibrated"}]
    assert out["r0_complete"] is True and len(trials_of(run_dir, "baseline")) == 4
    code, out = bo(repo, "round-run", "--rationale", "again", env=env)
    assert code != 0 and "ended" in out["reason"]
    assert len([e for e in events(run_dir) if e["type"] == "noise_estimate"]) == 1
