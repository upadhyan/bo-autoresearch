"""Warm start, epochs and code changes (#28): the mapping rules, narrowing, commit-change,
add-dependency and the equivalence check, against the toy_bo trainer with lever code of our own."""
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import bo, expect_all, git, round_run, register
from test_rounds import BASE, BOWL, FLAT, coded, init, make_repo, of_type, toy_env
from test_verdicts import code_hypothesis, lever as vlever, spec as vspec, verdict_run

SIGMA = 0.05


def eligible(repo):
    """The next round's warm start as the harness sees it: {trial: levers}."""
    code, out = bo(repo, "trials", "--eligible")
    assert code == 0, out
    return {t["trial"]: t["levers"] for t in out["trials"]}


def finished(run_dir, epoch=None):
    """{trial: trial_started payload} of the finished trials (of `epoch` only, when given)."""
    ends = {p["trial"] for p in of_type(run_dir, "trial_finished")}
    return {p["trial"]: p for p in of_type(run_dir, "trial_started")
            if p["trial"] in ends and epoch in (None, p["epoch"])}


def current(run_dir):
    """The finished trials of the latest epoch (commit-lever's equivalence check fails a true no-op by
    chance, and its new epoch leaves the earlier trials behind)."""
    return finished(run_dir, max(p["epoch"] for p in of_type(run_dir, "trial_started")))


NOOP_BOWL = BOWL  # the planted bowl is 0 at the baseline (0.5, 0.5): a true no-op
OFF_BOWL = BOWL.replace(" - 0.08)", ")")  # 0.32 = 6σ at the baseline: not a no-op


def test_an_added_lever_keeps_prior_trials_eligible_backfilled_at_its_baseline(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.1\nseed: 1\n")
    coded(repo, run_dir, tmp_path, code=NOOP_BOWL, env=toy_env(sigma=SIGMA))
    [check] = of_type(run_dir, "equivalence_check")  # commit-lever's: the new code at baseline
    assert check["passed"] is True and check["change"] == "lever_committed"
    r0 = [t for t in finished(run_dir).values() if t.get("round") == 0]
    assert r0 and all(t["levers"] == {} for t in r0)  # R0 ran before H1's levers existed
    seen = eligible(repo)
    for t in r0:
        assert seen[t["trial"]] == {"H1.x": 0.5, "H1.y": 0.5}


def test_lever_code_whose_baseline_is_not_a_no_op_starts_a_new_epoch(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.1\nseed: 1\n")
    coded(repo, run_dir, tmp_path, code=OFF_BOWL, env=toy_env(sigma=SIGMA))
    [check] = of_type(run_dir, "equivalence_check")
    assert check["passed"] is False
    [started] = of_type(run_dir, "epoch_started")
    assert started["change"] == "lever_committed" and started["commit"] == check["commit"]
    assert all(t.get("epoch") == 1 for n, t in finished(run_dir).items() if n in eligible(repo))


def removal_run(d, python, seed, line):
    """H1 (lever x, baseline 0) with planted lever code, searched until rejected; then H2 registered."""
    run_dir, _ = verdict_run(d, python, seed, [(vspec({"x": vlever()}), line)])
    repo = run_dir.parents[1]
    f = d / "h2.json"
    f.write_text(json.dumps(vspec({"y": vlever()})))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "next idea")[0] == 0
    assert register(repo, "H2")[0] == 0
    return run_dir


def test_each_removal_reason_applies_its_own_rule(tmp_path, project_python):
    with ThreadPoolExecutor(2) as pool:
        irrelevant, no_gain = pool.map(lambda a: removal_run(tmp_path / a[0], project_python, 1, a[1]), [
            ("useless", '    term += 0 * lever("H1.x")'), ("harmful", '    term += 2 * lever("H1.x")')])
    assert [p["condition"] for p in of_type(irrelevant, "hypothesis_rejected")] == ["irrelevant"]
    assert [p["condition"] for p in of_type(no_gain, "hypothesis_rejected")] == ["no-improvement"]

    # credibly irrelevant: every trial kept, H1.x's key dropped, H2.y backfilled at its baseline
    trials = current(irrelevant)
    assert any(t["levers"].get("H1.x", 0.0) != 0.0 for t in trials.values())
    assert eligible(irrelevant.parents[1]) == {n: {"H2.y": 0.0} for n in trials}

    # no-improvement: only the trials that ran H1.x at its baseline
    trials = current(no_gain)
    at_baseline = {n for n, t in trials.items() if t["levers"].get("H1.x", 0.0) == 0.0}
    assert at_baseline != set(trials)
    assert eligible(no_gain.parents[1]) == {n: {"H1.x": 0.0, "H2.y": 0.0} for n in at_baseline}

    # the reports keep the two conditions apart (test_reports' planted run has no `irrelevant` reject)
    for run_dir, reason, words in ((irrelevant, "irrelevant", ": doesn't matter"),
                                   (no_gain, "no-improvement", ": doesn't help, and may matter")):
        with open(run_dir / "exports" / "hypotheses.csv") as f:
            assert {r["id"]: r["reason"] for r in csv.DictReader(f)}["H1.v1"] == reason
        summary = (run_dir / "SUMMARY.md").read_text()
        assert any(ln.startswith("- H1.v1 ") and ln.endswith(words) for ln in summary.splitlines()), summary


def calibrated(tmp_path, python, extra=""):
    """toy_bo after R0 (3 baseline replicates, σ = 0.05)."""
    repo = make_repo(tmp_path)
    run_dir = init(repo, python, BASE + "delta: 0.1\nseed: 1\n" + extra)
    assert round_run(repo, "--rationale", "calibrate", env=toy_env(sigma=SIGMA))[0] == 0
    return repo, run_dir


def change(repo, run_dir, old, new, *flags, env=None):
    train = run_dir / "worktree" / "train.py"
    train.write_text(train.read_text().replace(old, new))
    return bo(repo, "commit-change", "--reason", "the change under test", *flags,
              "--rationale", "a code change", env=env or toy_env(sigma=SIGMA))


def test_a_changed_objective_fails_the_equivalence_check_and_starts_a_new_epoch(tmp_path, project_python):
    repo, run_dir = calibrated(tmp_path, project_python)
    worktree = run_dir / "worktree"
    r0 = set(eligible(repo))

    # a true no-op passes: two baseline replicates on the new commit, within 2σ of the logged mean
    code, out = change(repo, run_dir, "    term = 0.0\n", "    term = 0.0  # the baseline term\n")
    assert code == 0, out
    head = git(worktree, "rev-parse", "HEAD")
    assert out["commit"] == head and git(worktree, "status", "--porcelain") == ""
    assert of_type(run_dir, "commit_change")[0]["reason"] == "the change under test"
    check = out["equivalence_check"]
    assert check["passed"] is True and [c["config"] for c in check["configs"]] == ["baseline"]
    reps = [t for t in finished(run_dir).values() if t["kind"] == "equivalence"]
    assert len(reps) == 2 and all(t["commit"] == head and t["levers"] == {} for t in reps)
    assert check["configs"][0]["trials"] == [t["trial"] for t in reps]
    assert not of_type(run_dir, "epoch_started") and out["epoch"] == 0
    assert set(eligible(repo)) == r0 | {t["trial"] for t in reps}  # still one epoch

    # every loss up by 1 (20σ): the first 2 replicates disagree beyond 2σ̂, so 4 more run, and all 6 still
    # disagree in the t-test; σ̂ is pooled over the epoch's baseline trials (R0's smoke and 3 replicates, the
    # no-op's 2: df 5), the tolerance t(1 − 0.05/2, 5)·σ̂·√(1/6 + 1/6) with t(0.975, 5) = 2.571 (one config)
    code, out = change(repo, run_dir, "loss = 1.0 +", "loss = 2.0 +")
    assert code == 0, out
    check = out["equivalence_check"]
    assert check["passed"] is False and out["epoch"] == 1
    assert check["df"] == 5 and check["alpha"] == 0.05
    [row] = check["configs"]
    assert row["stage1"]["passed"] is False and row["trials"][:2] == row["stage1"]["trials"]
    assert len(row["trials"]) == 6 and row["passed"] is False and row["logged_n"] == 6
    assert abs(row["mean"] - row["logged_mean"] - 1.0) < 0.2
    assert abs(row["tolerance"] - 2.571 * check["sigma"] * (1 / 3) ** 0.5) < 1e-3 * check["sigma"] + 1e-8
    [started] = of_type(run_dir, "epoch_started")
    assert started["epoch"] == 1 and started["breaking"] is False
    fresh = [t for t in finished(run_dir).values() if t.get("epoch") == 1]
    assert [t["kind"] for t in fresh] == ["baseline"] * 3
    [noise] = [p for p in of_type(run_dir, "noise_estimate") if p.get("epoch") == 1]
    assert noise["trials"] == [t["trial"] for t in fresh] and 0 < noise["sigma"] < 0.2
    status = bo(repo, "status")[1]
    assert status["epoch"] == 1 and status["sigma"] == noise["sigma"]
    assert set(eligible(repo)) == {t["trial"] for t in fresh}  # the earlier trials are telemetry


def test_a_first_look_failure_stands_when_the_budget_refuses_the_second(tmp_path, project_python):
    repo, run_dir = calibrated(tmp_path, project_python, "max_trials: 6\n")
    assert len(of_type(run_dir, "trial_started")) == 4  # R0: the smoke and 3 baseline replicates
    code, out = change(repo, run_dir, "loss = 1.0 +", "loss = 2.0 +")  # 20σ: the first look fails
    assert code == 0, out
    check = out["equivalence_check"]
    [row] = check["configs"]
    assert len(row["trials"]) == 2 and row["passed"] is False and row["stage1"] is None
    assert check["refused"] and check["passed"] is False  # the ceiling refused the 4 more replicates
    assert of_type(run_dir, "epoch_started")


def no_op_check(d, python, seed):
    d.mkdir()
    repo = make_repo(d)
    run_dir = init(repo, python, BASE.replace("replicates_k: 3", "replicates_k: 2") + f"delta: 0.1\nseed: {seed}\n")
    assert round_run(repo, "--rationale", "calibrate", env=toy_env(sigma=SIGMA))[0] == 0
    code, out = change(repo, run_dir, "    term = 0.0\n", "    term = 0.0  # the baseline term\n")
    assert code == 0, out
    return out["equivalence_check"]


@pytest.mark.slow
def test_a_true_no_op_rarely_starts_a_new_epoch(tmp_path, project_python):
    with ThreadPoolExecutor(20) as pool:
        checks = list(pool.map(lambda s: no_op_check(tmp_path / f"s{s}", project_python, s), range(40)))
    [rows] = zip(*(c["configs"] for c in checks))
    # σ̂ pooled over R0's smoke and 2 replicates (df 2): the first 2 replicates alone (the old rule) fail a
    # no-op P(|t₂| > 2/√(1/2 + 1/3)) ≈ 16%; the 6-replicate t-test at α = 5% keeps most of those.
    # Measured over seeds 0..39: first look failed 5, the check failed 2 (5%); at 5%, more than 4 of 40
    # fail has probability ~5%
    assert sum(c["passed"] is True for c in checks) >= 36, checks
    # the first look did fail some of them: the second is what kept them in the epoch (3 of 40 measured; at
    # 7.5%, none in 40 has probability ~4%)
    rescued = [r for r in rows if r["stage1"] and r["stage1"]["passed"] is False and r["passed"]]
    assert rescued and all(len(r["trials"]) == 6 for r in rescued), rows


def test_a_breaking_change_skips_the_check_and_restarts_burn_in(tmp_path, project_python):
    repo, run_dir = calibrated(tmp_path, project_python)
    coded(repo, run_dir, tmp_path, code=NOOP_BOWL, env=toy_env(sigma=SIGMA))
    code, out = round_run(repo, "--rationale", "search", env=toy_env(sigma=SIGMA))
    assert code == 0, out
    assert bo(repo, "verdict", "H1")[1]["burn_in"]["fresh"] >= 20
    inc = out["incumbent"]

    # a no-op after the round: the incumbent's logged mean leaves out the trial that made it the incumbent
    # (the best of many draws, biased) and keeps its replicates
    code, out = change(repo, run_dir, "import time\n", "import time  # the clock\n")
    assert code == 0, out
    row = next(r for r in out["equivalence_check"]["configs"] if r["config"] == "incumbent")
    ends = {p["trial"]: p["objective"] for p in of_type(run_dir, "trial_finished")}
    reps = [ends[t["trial"]] for t in finished(run_dir).values()
            if t.get("replicate_of") == inc["trial"] and t["trial"] not in row["trials"]]
    assert row["levers"] == inc["levers"] and row["logged_n"] == len(reps) >= 2
    assert abs(row["logged_mean"] - sum(reps) / len(reps)) < 1e-9
    checks = len(of_type(run_dir, "equivalence_check"))

    code, out = change(repo, run_dir, "loss = 1.0 +", "loss = 5.0 +", "--breaking")
    assert code == 0, out
    assert out["equivalence_check"] is None and len(of_type(run_dir, "equivalence_check")) == checks
    [started] = of_type(run_dir, "epoch_started")
    assert started == {"epoch": 1, "round": 2, "commit": out["commit"], "change": "commit_change",
                       "breaking": True}
    assert of_type(run_dir, "commit_change")[-1]["breaking"] is True
    fresh = {n for n, t in finished(run_dir).items() if t.get("epoch") == 1}
    assert len(fresh) == 3 and set(eligible(repo)) == fresh
    probe = bo(repo, "verdict", "H1")[1]
    assert probe["verdict"] == "burn-in" and probe["burn_in"]["fresh"] == 0  # burn-in restarts

    code, out = round_run(repo, "--rationale", "search the new epoch", env=toy_env(sigma=SIGMA))
    assert code == 0, out
    [r2] = [p for p in of_type(run_dir, "round_started") if p["round"] == 2]
    assert r2["epoch"] == 1 and set(r2["seeded"]) == fresh
    assert all(v["burn_in"]["fresh"] >= 20 for v in out["verdicts"])  # judged on this epoch alone


def epoch_noise(run_dir, epoch):
    return [p for p in of_type(run_dir, "noise_estimate") if p.get("epoch") == epoch]


def test_a_new_epoch_at_a_proxy_re_measures_the_reference_sigma_before_the_next_round(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    # one seed, the file's usual: nothing asserted here is a rate (σ's 20x jump is a sure thing, the
    # rest is structure)
    run_dir = init(repo, project_python, BASE + "ladder: [{epochs: 1}]\nseed: 1\n")
    env = toy_env(sigma=SIGMA, cheap_below=2)
    coded(repo, run_dir, tmp_path, code=FLAT, env=env)  # rejected soon: a short round
    code, out = round_run(repo, "--rationale", "calibrate the ladder", env=env)
    assert code == 0, out
    if out["fidelity_calibration"]["chosen"] != {"epochs": 1}:
        assert bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
    assert bo(repo, "set-delta", "0.1", "--rationale", "the user's effect")[0] == 0
    r0_ref = bo(repo, "status")[1]["sigma"]  # σ at the reference

    # a breaking change after which the reference is 20x noisier: σ there must be measured afresh
    noisy = toy_env(sigma=SIGMA, cheap_below=2, ref_sigma=1.0)
    code, out = change(repo, run_dir, "loss = 1.0 +", "loss = 1.5 +", "--breaking", env=noisy)
    assert code == 0, out
    epoch = out["epoch"]
    # the new epoch measures σ at once only where the run searches (the proxy)
    assert [p["fidelity"] for p in epoch_noise(run_dir, epoch)] == [{"epochs": 1}]

    code, out = round_run(repo, "--rationale", "search the new epoch", env=noisy)
    assert code == 0, out
    # the round's drift check runs at the reference: σ there is re-measured before the round starts
    [_, ref] = epoch_noise(run_dir, epoch)
    assert ref["fidelity"] == {"epochs": 4} and ref["n"] == 3
    started = {p["trial"]: p for p in of_type(run_dir, "trial_started")}
    reps = [started[t] for t in ref["trials"]]
    assert all(t["kind"] == "baseline" and t["fidelity"] == {"epochs": 4} and t["epoch"] == epoch
               and "round" not in t for t in reps)
    assert 0.3 < ref["sigma"] < 3  # the planted reference σ is 1 (R0's was 0.05); 3 replicates
    status = bo(repo, "status")[1]
    assert status["sigma"] == ref["sigma"] != r0_ref
    assert max(ref["trials"]) < min(t for t, p in started.items() if p.get("round") == out["round"])


def test_after_a_new_epoch_a_stalled_proxy_escalates_to_the_reference_not_a_stale_rung(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "ladder: [{epochs: 1}, {epochs: 2}]\ndeterministic: true\n")
    # no noise, so no chance: 1 epoch is the cheap rung, where the fidelity-sensitive lever is flat
    # (it fails calibration); 2 epochs rank exactly like the reference (4) and cost less (it passes)
    env = toy_env(cheap_below=2, sleep_per_epoch=0.05)
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    late = vspec({"x": vlever()}, fidelity_sensitive=True, fidelity_reason="pays off late")
    code_hypothesis(repo, run_dir, tmp_path, 1, late, '    term += 0 * lever("H1.x") if cheap else -lever("H1.x")', env)
    code, out = round_run(repo, "--rationale", "calibrate the ladder", env=env)
    assert code == 0, out
    [one, two] = out["fidelity_calibration"]["rungs"]
    assert two["passed"], two  # a validated rung the escalation would move up to in epoch 0
    if out["fidelity_calibration"]["chosen"] != {"epochs": 1}:
        assert bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
    assert bo(repo, "set-delta", "0.1", "--rationale", "the user's effect")[0] == 0
    code, out = change(repo, run_dir, "loss = 1.0 +", "loss = 1.5 +", "--breaking", env=env)
    assert code == 0, out
    code, out = bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap again")
    assert code == 1 and "earlier epoch" in out["reason"]  # the old calibration no longer holds

    for _ in range(4):  # stalled at the proxy: more replicates, then a rung up
        code, out = round_run(repo, "--rationale", "search", env=env)
        assert code == 0, out
        if any(e["step"] == "rung" for e in of_type(run_dir, "hypothesis_escalated")):
            break
    rung = [e for e in of_type(run_dir, "hypothesis_escalated") if e["step"] == "rung"]
    assert rung and rung[0]["fidelity"] == {"epochs": 4} and rung[0]["stale_ladder"] is True, rung
    assert bo(repo, "status")[1]["fidelity"]["fidelity"] == {"epochs": 4}


def test_code_changes_are_refused_mid_round_on_protected_paths_and_with_nothing_to_commit(
        tmp_path, project_python):
    repo, run_dir = calibrated(tmp_path, project_python)
    worktree = run_dir / "worktree"
    code, out = bo(repo, "commit-change", "--reason", "nothing", "--rationale", "empty")
    assert code == 1 and "no changes" in out["reason"]
    (worktree / "runner.py").write_text((worktree / "runner.py").read_text() + "# edited\n")
    code, out = bo(repo, "commit-change", "--reason", "edit the runner", "--rationale", "sneaky")
    assert code == 1 and "protected" in out["reason"] and "runner.py" in out["reason"]
    git(worktree, "checkout", "runner.py")

    coded(repo, run_dir, tmp_path, code=NOOP_BOWL, env=toy_env(sigma=SIGMA))
    code, out = narrow(repo, "H1", "H1.x", 0.2, 0.8)
    assert code == 1 and "registered" in out["reason"]  # not in the search yet: no verdict to rest on
    env = {**toy_env(sigma=SIGMA, sleep=60), "TOY_SLEEP_KIND": "sampler"}
    expect_all(repo)
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "go"],
                         cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not [s for s in of_type(run_dir, "round_started") if s["round"] == 1]:
            assert time.monotonic() < deadline and p.poll() is None
            time.sleep(0.1)
        (worktree / "train.py").write_text((worktree / "train.py").read_text() + "# mid-round\n")
        code, out = bo(repo, "commit-change", "--reason", "mid-round fix", "--rationale", "now")
        assert code == 1 and "round is running" in out["reason"]
        git(worktree, "checkout", "train.py")
        for args in (["add-dependency", "toydep"], ["narrow", "H1", "--lever", "H1.x", "--low", "0.2",
                                                    "--high", "0.8"]):
            code, out = bo(repo, *args, "--rationale", "now")
            assert code == 1 and "round is running" in out["reason"], args
    finally:
        p.kill()
        p.wait()
    assert not of_type(run_dir, "commit_change")


def narrow(repo, h, lever, low, high):
    return bo(repo, "narrow", h, "--lever", lever, "--low", str(low), "--high", str(high),
              "--rationale", "the posterior mass sits in the middle")


def test_narrowing_drops_out_of_range_trials_and_never_clips(tmp_path, project_python):
    useful = (vspec({"x": vlever(0.2)}), '    term += 4 * ((lever("H1.x") - 0.7) ** 2 - 0.25)')
    run_dir, [out] = verdict_run(tmp_path / "r", project_python, 1, [useful])
    repo = run_dir.parents[1]
    last = [v for v in of_type(run_dir, "verdict") if v["hypothesis"] == "H1.v1"][-1]
    assert all(g["passed"] for g in last["gates"].values()), last["gates"]
    code, out = narrow(repo, "H1", "H1.x", -0.1, 0.9)
    assert code == 1 and "inside" in out["reason"]  # a widening is not a narrowing
    code, out = narrow(repo, "H1", "H1.x", 0.3, 0.9)
    assert code == 1 and "baseline" in out["reason"]
    assert not of_type(run_dir, "narrowed")

    code, out = narrow(repo, "H1", "H1.x", 0.15, 0.85)
    assert code == 0, out
    assert of_type(run_dir, "narrowed") == [{"id": "H1.v1", "lever": "H1.x", "verdict": last["id"],
                                             "before": {"low": 0.0, "high": 1.0},
                                             "after": {"low": 0.15, "high": 0.85},
                                             "round": 2, "rationale": "the posterior mass sits in the middle"}]
    epoch = bo(repo, "status")[1]["epoch"]
    trials = {n: t for n, t in finished(run_dir).items() if t["epoch"] == epoch}
    inside = {n for n, t in trials.items() if 0.15 <= t["levers"].get("H1.x", 0.2) <= 0.85}
    assert inside != set(trials)
    seen = eligible(repo)
    assert set(seen) == inside
    assert all(seen[n]["H1.x"] == trials[n]["levers"].get("H1.x", 0.2) for n in seen)  # never clipped

    code, out = round_run(repo, "--rationale", "search the narrowed box", env=toy_env(sigma=SIGMA))
    assert code == 0, out
    [r2] = [p for p in of_type(run_dir, "round_started") if p["round"] == 2]
    assert r2["search_space"]["H1.x"]["low"] == 0.15 and r2["search_space"]["H1.x"]["high"] == 0.85
    assert set(r2["seeded"]) == inside
    r2_trials = [p for p in of_type(run_dir, "trial_started") if p.get("round") == 2]
    assert r2_trials and all(0.15 <= t["levers"]["H1.x"] <= 0.85 for t in r2_trials)


def test_a_merge_keeps_the_trials_its_mapping_expresses(tmp_path, project_python):
    mode = {"kind": "categorical", "options": ["x", "y", "z"], "baseline": "x"}
    useless = (vspec({"mode": mode}), '    term += 0 * (lever("H1.mode") == "y")')
    run_dir, _ = verdict_run(tmp_path / "r", project_python, 1, [useless])
    repo = run_dir.parents[1]
    assert [p["condition"] for p in of_type(run_dir, "hypothesis_rejected")] == ["irrelevant"]
    norm = {"kind": "categorical", "options": ["none", "bn", "ln"], "baseline": "none"}
    merged = vspec({"norm": norm}, merges={"from": ["H1.v1"], "mapping": {
        "norm": {"lever": "H1.mode", "values": [["x", "none"], ["y", "bn"]]}}})
    f = tmp_path / "h2.json"
    f.write_text(json.dumps(merged))
    code, out = bo(repo, "propose", "--file", str(f), "--rationale", "one norm lever")
    assert code == 0, out
    assert out["hypothesis"]["spec"]["merges"]["mapping"] == {"H2.norm": merged["merges"]["mapping"]["norm"]}
    assert register(repo, "H2")[0] == 0

    trials = finished(run_dir)
    ran = {n: t["levers"].get("H1.mode") for n, t in trials.items()}
    assert {"x", "y", "z", None} <= set(ran.values())
    to = {None: "none", "x": "none", "y": "bn"}  # R0 predates H1: H2.norm backfilled at baseline
    assert eligible(repo) == {n: {"H2.norm": to[m]} for n, m in ran.items() if m in to}


def wheel(d):
    """A tiny local wheel (no network, no build step): toydep 1.0, VALUE = 7."""
    path = d / "toydep-1.0-py3-none-any.whl"
    files = {"toydep/__init__.py": "VALUE = 7\n",
             "toydep-1.0.dist-info/METADATA": "Metadata-Version: 2.1\nName: toydep\nVersion: 1.0\n",
             "toydep-1.0.dist-info/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
                                           "Tag: py3-none-any\n"}
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
        z.writestr("toydep-1.0.dist-info/RECORD", "".join(f"{n},,\n" for n in files)
                   + "toydep-1.0.dist-info/RECORD,,\n")
    return path


def imports_toydep(python):
    return subprocess.run([str(python), "-c", "import toydep"], capture_output=True).returncode == 0


def pip_freeze(python):
    return subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True).stdout


PYPROJECT = '[project]\nname = "toy"\nversion = "0.1"\nrequires-python = ">=3.10"\ndependencies = []\n'


def locked(tmp_path, python, lock):
    """toy_bo as a project locked by `lock` (uv.lock or poetry.lock), after R0."""
    repo = make_repo(tmp_path)
    (repo / "pyproject.toml").write_text(PYPROJECT)
    if lock == "uv.lock":
        subprocess.run(["uv", "lock", "-q", "--offline"], cwd=repo, check=True)
    else:
        (repo / lock).write_text("# locked\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "locked")
    run_dir = init(repo, python, BASE + "delta: 0.1\nseed: 1\n")
    assert round_run(repo, "--rationale", "calibrate", env=toy_env(sigma=SIGMA))[0] == 0
    return repo, run_dir


def test_add_dependency_in_a_uv_project_updates_its_lock_file(tmp_path, project_python):
    repo, run_dir = locked(tmp_path, project_python, "uv.lock")
    worktree, whl = run_dir / "worktree", wheel(tmp_path)
    user_env = pip_freeze(project_python)
    code, out = bo(repo, "add-dependency", str(whl), "--rationale", "the trainer needs toydep",
                   env={**toy_env(sigma=SIGMA), "UV_OFFLINE": "1"})
    assert code == 0, out
    assert imports_toydep(run_dir / "venv" / "bin" / "python")
    assert not imports_toydep(project_python) and pip_freeze(project_python) == user_env
    head = git(worktree, "rev-parse", "HEAD")
    assert out["commit"] == head and git(worktree, "status", "--porcelain") == ""
    assert sorted(git(worktree, "diff-tree", "--no-commit-id", "--name-only", "-r", head).split()) == [
        "pyproject.toml", "uv.lock"]  # the project's own lock, not a freeze file of the harness's
    assert "toydep" in git(worktree, "show", "HEAD:pyproject.toml")
    assert 'name = "toydep"' in git(worktree, "show", "HEAD:uv.lock")
    [dep] = of_type(run_dir, "dependency_added")
    assert dep["lock_file"] == "uv.lock" and dep["commit"] == head
    assert any(r.startswith("toydep") for r in dep["freeze"])
    assert out["equivalence_check"]["passed"] is True
    code, out = bo(repo, "add-dependency", str(whl), "--rationale", "again", env={**os.environ, "UV_OFFLINE": "1"})
    assert code == 1 and "already" in out["reason"]


def fake_poetry(d, fail=False):
    """A `poetry` on PATH that adds the requirement to poetry.lock, or leaves a stray file and fails."""
    bin_ = d / "bin"
    bin_.mkdir()
    f = bin_ / "poetry"
    f.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                 + ("Path('stray.txt').write_text('half done')\nsys.exit('poetry: resolution failed')\n" if fail else
                    "Path('poetry.lock').write_text(Path('poetry.lock').read_text() + sys.argv[-1] + '\\n')\n"))
    f.chmod(0o755)
    return {**toy_env(sigma=SIGMA), "PATH": f"{bin_}:{os.environ['PATH']}"}


def test_add_dependency_in_a_poetry_project_updates_its_lock_file_without_installing_through_poetry(
        tmp_path, project_python):
    repo, run_dir = locked(tmp_path, project_python, "poetry.lock")
    worktree, whl = run_dir / "worktree", wheel(tmp_path)
    code, out = bo(repo, "add-dependency", str(whl), "--rationale", "the trainer needs toydep",
                   env=fake_poetry(tmp_path))
    assert code == 0, out
    # poetry locked it; the harness installed it into the run venv
    assert imports_toydep(run_dir / "venv" / "bin" / "python")
    assert git(worktree, "show", "HEAD:poetry.lock") == f"# locked\n{whl}"
    assert git(worktree, "status", "--porcelain") == ""
    assert of_type(run_dir, "dependency_added")[0]["lock_file"] == "poetry.lock"


def test_a_failed_lock_update_installs_nothing_and_leaves_the_worktree_as_it_was(tmp_path, project_python):
    repo, run_dir = locked(tmp_path, project_python, "poetry.lock")
    worktree, whl = run_dir / "worktree", wheel(tmp_path)
    head = git(worktree, "rev-parse", "HEAD")
    code, out = bo(repo, "add-dependency", str(whl), "--rationale", "the trainer needs toydep",
                   env=fake_poetry(tmp_path, fail=True))
    assert code == 1 and "poetry.lock" in out["reason"] and "resolution failed" in out["reason"]
    assert not imports_toydep(run_dir / "venv" / "bin" / "python")
    assert git(worktree, "rev-parse", "HEAD") == head and git(worktree, "status", "--porcelain") == ""
    assert not of_type(run_dir, "dependency_added")


def test_add_dependency_installs_only_into_the_run_venv_and_commits_its_freeze(tmp_path, project_python):
    repo, run_dir = calibrated(tmp_path, project_python)
    worktree, whl = run_dir / "worktree", wheel(tmp_path)
    user_env = pip_freeze(project_python)
    code, out = bo(repo, "add-dependency", "--rationale", "sneaky", "--", "--index-url=https://example.invalid")
    assert code == 1 and "not options" in out["reason"]

    code, out = bo(repo, "add-dependency", str(whl), "--rationale", "the trainer needs toydep",
                   env=toy_env(sigma=SIGMA))
    assert code == 0, out
    assert imports_toydep(run_dir / "venv" / "bin" / "python")
    assert not imports_toydep(project_python) and pip_freeze(project_python) == user_env
    assert not imports_toydep(sys.executable)  # nor the environment the harness was called from
    head = git(worktree, "rev-parse", "HEAD")
    assert out["commit"] == head and git(worktree, "status", "--porcelain") == ""
    freeze = git(worktree, "show", "HEAD:requirements-freeze.txt").splitlines()
    assert any(r.startswith("toydep") for r in freeze) and not any("boautoresearch" in r for r in freeze)
    [dep] = of_type(run_dir, "dependency_added")
    assert dep["requirement"] == str(whl) and dep["commit"] == head and dep["freeze"] == freeze
    check = out["equivalence_check"]
    assert check["change"] == "dependency_added" and check["commit"] == head and check["passed"] is True
    code, out = bo(repo, "add-dependency", str(whl), "--rationale", "again")
    assert code == 1 and "already" in out["reason"]
