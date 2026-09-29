"""Rounds where BO does the searching (#25), against a toy with a planted optimum."""
import json
import os
import random
import re
import statistics
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from conftest import bo, events, expect_all, git, ready, round_run, write_run_yaml

TOY = Path(__file__).parent / "toy_bo"
OPT = {"H1.x": 0.7, "H1.y": 0.3}  # the planted optimum
SPEC = {
    "title": "Two knobs have a sweet spot",
    "rationale": "The toy's loss is a bowl around an unknown point.",
    "mechanism": "Move both knobs towards the bottom of the bowl.",
    "provenance": "novel",
    "lens": "optimisation",
    "directives": [],
    "fidelity_sensitive": False,
    "levers": {n: {"kind": "float", "low": 0.0, "high": 1.0, "baseline": 0.5, "predicted": "higher"}
               for n in ("x", "y")},
}
# a true no-op at the baseline (0.5, 0.5): commit-lever's equivalence check holds it to that (#28)
BOWL = "    term = 4 * ((lever(\"H1.x\") - 0.7) ** 2 + (lever(\"H1.y\") - 0.3) ** 2 - 0.08)\n"
BEST = 1 - 0.32  # the loss at the optimum
FLAT = "    term = 0 * (lever(\"H1.x\") + lever(\"H1.y\"))\n"


def true_loss(levers):
    return 1 + 4 * (sum((levers[n] - v) ** 2 for n, v in OPT.items()) - 0.08)


def toy_env(**planted):
    return {**os.environ, **{f"TOY_{k.upper()}": str(v) for k, v in planted.items()}}


BASE = "budget_s: 3600\nreference_fidelity: {epochs: 4}\nreplicates_k: 3\n"


def make_repo(d):
    r = d / "project"
    shutil.copytree(TOY, r)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "toy")
    return r


def init(repo, python, extra=BASE):
    y = write_run_yaml(repo.parent / "run.yaml", python, extra)
    code, out = bo(repo, "init", str(y), "--rationale", "start")
    assert code == 0, out
    return Path(out["run_dir"])


@pytest.fixture
def bo_repo(tmp_path):
    return make_repo(tmp_path)


@pytest.fixture
def bo_run(bo_repo, project_python):
    return lambda extra=BASE: init(bo_repo, project_python, extra)


def coded(repo, run_dir, tmp_path, code=BOWL, env=None, spec=SPEC):
    """R0 done, then H1 proposed, registered, its lever code smoked and committed."""
    if not any(e["type"] == "round_ended" for e in events(run_dir)):
        assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    f = tmp_path / "spec.json"
    f.write_text(json.dumps(spec))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    assert bo(repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    train = run_dir / "worktree" / "train.py"
    src = train.read_text().replace("import fidelity, seed", "import fidelity, lever, seed")
    train.write_text(src.replace("    term = 0.0\n", code))
    code_, out = bo(repo, "smoke", "H1", "--rationale", "try", env=env)
    assert code_ == 0 and out["passed"], out
    code_, out = bo(repo, "commit-lever", "H1", "--rationale", "smoke passed", env=env)
    assert code_ == 0, out


def of_type(run_dir, type):
    return [e["payload"] for e in events(run_dir) if e["type"] == type]


def test_set_delta_is_set_once_after_r0(bo_repo, bo_run, tmp_path):
    run_dir = bo_run()
    code, out = bo(bo_repo, "set-delta", "0.1", "--rationale", "too early")
    assert code != 0 and "round-run" in out["reason"]
    coded(bo_repo, run_dir, tmp_path, env=toy_env(sigma=0.05))
    code, out = round_run(bo_repo, "--rationale", "no delta yet")
    assert code != 0 and "set-delta" in out["reason"]
    # the toy's noise draws at R0's baseline replicates: σ is their sd, the suggestion 2σ
    planted = [0.05 * random.Random(t["seed"]).gauss(0, 1) for t in of_type(run_dir, "trial_started")
               if t["kind"] == "baseline"]
    suggested = float(re.search(r"suggested 2σ = ([0-9.e-]+)", out["reason"])[1])
    assert suggested == pytest.approx(2 * statistics.stdev(planted), rel=1e-2)
    assert bo(bo_repo, "status")[1]["suggested_delta"] == pytest.approx(2 * statistics.stdev(planted))

    for bad in ("0", "-1", "nan", "x"):
        code, out = bo(bo_repo, "set-delta", bad, "--rationale", "bad")
        assert code != 0 and out["refused"], bad
    code, out = bo(bo_repo, "set-delta", "0.1", "--rationale", "the user's minimum effect")
    assert code == 0 and out["delta"] == 0.1, out
    code, out = bo(bo_repo, "set-delta", "0.2", "--rationale", "again")
    assert code != 0 and "already" in out["reason"]
    assert [p["delta"] for p in of_type(run_dir, "delta_set")] == [0.1]


def test_delta_from_run_yaml_cannot_be_changed(bo_repo, bo_run):
    bo_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\ndelta: 0.05\n")
    assert bo(bo_repo, "status")[1]["delta"] == 0.05
    code, out = bo(bo_repo, "set-delta", "0.1", "--rationale", "change it")
    assert code != 0 and "already" in out["reason"]


def test_a_deterministic_objective_suggests_no_delta_set_delta_would_refuse(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\ndeterministic: true\n")
    coded(bo_repo, run_dir, tmp_path)
    _, status = bo(bo_repo, "status")
    assert status["sigma"] == 0 and status["suggested_delta"] is None
    duty, _expected = status["next"]  # and H1's expected verdict
    assert duty.startswith("set-delta <value>") and "= 0)" not in duty and "σ = 0" in duty
    code, out = round_run(bo_repo, "--rationale", "no delta yet")
    assert code != 0 and "set-delta" in out["reason"] and "= 0)" not in out["reason"]


def started_in(run_dir, r):
    """The trials of round r as started, with their outcome from the log."""
    ends = {p["trial"]: {**p, "status": t[6:]} for t in ("trial_finished", "trial_failed",
                                                           "trial_abandoned")
            for p in of_type(run_dir, t)}
    return [{**p, **ends.get(p["trial"], {})} for p in of_type(run_dir, "trial_started")
            if p.get("round") == r]  # r None: the trials outside rounds (smoke <H>)


def search_once(d, python, sigma=0.01, seed=0):
    """One run on the bowl toy: R0, H1 coded, one BO round. Returns (run_dir, round-run JSON)."""
    repo = make_repo(d)
    run_dir = init(repo, python, BASE + f"delta: 0.02\nseed: {seed}\n")
    env = toy_env(sigma=sigma)
    coded(repo, run_dir, d, env=env)
    code, out = round_run(repo, "--rationale", "search H1", env=env)
    assert code == 0, out
    return run_dir, out


def random_search(n, seed, sigma=0.01):
    """Plain random search with n trials on the same toy: the true loss of its best observed point."""
    rng = random.Random(seed)
    points = [{n_: rng.random() for n_ in OPT} for _ in range(n)]
    observed = [true_loss(p) + sigma * rng.gauss(0, 1) for p in points]
    return true_loss(points[observed.index(min(observed))])


def test_bo_beats_random_search_at_equal_trial_count(tmp_path, project_python):
    seeds = [11, 22, 33, 44]
    for s in seeds:
        (tmp_path / f"s{s}").mkdir()
    with ThreadPoolExecutor(len(seeds)) as pool:  # independent runs, each with its own fixed seed
        runs = list(pool.map(lambda s: search_once(tmp_path / f"s{s}", project_python, seed=s), seeds))
    assert len({t["seed"] for run_dir, _ in runs for t in started_in(run_dir, 1)}) == sum(
        len(started_in(run_dir, 1)) for run_dir, _ in runs)  # no two runs share trial seeds
    bo_regret, rs_regret = [], []
    for seed, (run_dir, out) in zip(seeds, runs):
        assert out["trigger"] == "stall" and out["round"] == 1
        trials = started_in(run_dir, 1)
        bo_regret.append(true_loss(out["incumbent"]["levers"]) - BEST)
        rs_regret.append(random_search(len(trials), seed) - BEST)
    wins = sum(b < r for b, r in zip(bo_regret, rs_regret))
    assert wins >= 3, (bo_regret, rs_regret)  # BO's incumbent is better in at least 3 of 4 runs
    assert statistics.fmean(bo_regret) < statistics.fmean(rs_regret) / 2, (bo_regret, rs_regret)

    run_dir, out = runs[0]
    trials = started_in(run_dir, 1)
    [started] = [p for p in of_type(run_dir, "round_started") if p["round"] == 1]
    assert started["hypotheses"] == ["H1.v1"] and set(started["search_space"]) == set(OPT)
    assert of_type(run_dir, "hypothesis_activated") == [{"id": "H1.v1", "round": 1}]
    assert {t["kind"] for t in trials} <= {"sampler", "confirmation", "replicate"}
    assert all(0 <= t["levers"][n] <= 1 for t in trials for n in OPT)
    assert len({t["seed"] for t in trials}) == len(trials)
    # the stall: at least max(5·d, 20) fresh sampler trials, the run goes on
    assert sum(t["kind"] == "sampler" for t in trials) >= 20
    assert out["run_ended"] is None and ready(out)
    assert of_type(run_dir, "round_ended")[-1] == {"round": 1, "trigger": "stall",
                                                    "incumbent": out["incumbent"]}
    assert not of_type(run_dir, "run_ended")
    assert out["drift"] is None  # the round ran at the reference fidelity


def test_replicates_are_their_own_trials_and_every_new_incumbent_is_confirmed(tmp_path, project_python):
    run_dir, out = search_once(tmp_path, project_python, seed=1)
    trials = started_in(run_dir, 1)
    by_id = {t["trial"]: t for t in trials}
    replicates = [t for t in trials if "replicate_of" in t]
    assert len(replicates) >= 0.1 * len(trials)
    for t in replicates:  # same config, own seed and objective, the original untouched
        orig = by_id[t["replicate_of"]]
        assert t["levers"] == orig["levers"] and t["seed"] != orig["seed"]
        assert "replicate_of" not in orig
    # a sampler trial better than every trial seen before it is a new incumbent: each one got its 2
    # confirmations before the round ended (the stall never cuts one short)
    seen = list(eligible_now(run_dir, before=trials[0]["trial"]).values())  # the round's warm start
    new_best = []
    for t in trials:
        if t["kind"] == "sampler" and t["status"] == "finished" and t["objective"] < min(seen):
            new_best.append(t["trial"])
        if t["status"] == "finished":
            seen.append(t["objective"])
    assert len(new_best) >= 2  # (the warm start's baseline trials make the first incumbents)
    for root in new_best:
        assert sum(t.get("replicate_of") == root and t["kind"] == "confirmation" for t in trials) == 2
    inc = out["incumbent"]
    group = [t["objective"] for t in trials if inc["trial"] in (t["trial"], t.get("replicate_of"))]
    assert inc["confirmed"] and inc["replicates"] == len(group) - 1 >= 2
    assert inc["mean"] == pytest.approx(statistics.fmean(group))
    assert by_id[inc["trial"]]["objective"] != inc["mean"]  # replicates never averaged in
    # σ re-estimated from the round's replicates: the planted 0.01, no shift
    [noise] = [p for p in of_type(run_dir, "noise_estimate") if p["round"] == 1]
    assert noise["sigma"] == pytest.approx(0.01, rel=0.5) and noise["shift_flagged"] is False
    assert out["noise_estimate"] == noise


def test_a_noise_shift_of_more_than_2x_is_flagged(bo_repo, bo_run, tmp_path):
    run_dir = bo_run(BASE + "delta: 0.02\nseed: 5\n")
    coded(bo_repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    code, out = round_run(bo_repo, "--rationale", "search", env=toy_env(sigma=0.05))
    assert code == 0, out
    [noise] = [p for p in of_type(run_dir, "noise_estimate") if p.get("round") == 1]
    assert noise["sigma"] == pytest.approx(0.05, rel=0.5) and noise["shift_flagged"] is True
    # against the σ in force: R0's, or (R0's 3 replicates put it at 0.002) the re-estimate of the
    # epoch commit-lever's equivalence check started
    assert noise["r0_sigma"] == [p for p in of_type(run_dir, "noise_estimate") if p.get("round") != 1][-1]["sigma"]


def eligible_now(run_dir, before=None):
    """Independently: finished trials at the reference fidelity with the round's levers (a trial from
    before H1 backfilled at its baseline 0.5), in range."""
    ends = {p["trial"]: p for p in of_type(run_dir, "trial_finished")}
    out = {}
    for p in of_type(run_dir, "trial_started"):
        levers = {"H1.x": 0.5, "H1.y": 0.5, **p["levers"]}
        if (p["trial"] in ends and p["fidelity"] == {"epochs": 4} and set(levers) == set(OPT)
                and all(0 <= v <= 1 for v in levers.values()) and (before is None or p["trial"] < before)):
            out[p["trial"]] = ends[p["trial"]]["objective"]
    return out


def gone(pid, within=10.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_killed_round_is_resumed_from_the_log(bo_repo, bo_run, tmp_path):
    # δ out of reach: the bowl is `irrelevant` next to it, so R2 ends with its confirmed reject
    run_dir = bo_run(BASE + "delta: 10\nseed: 3\n")
    coded(bo_repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    n = len(of_type(run_dir, "trial_started"))
    # the first confirmation from the round's 12th trial on hangs
    env = {**toy_env(sigma=0.01, sleep=120, sleep_from=n + 12), "TOY_SLEEP_KIND": "confirmation"}
    expect_all(bo_repo)
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "go"],
                         cwd=bo_repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 90

    def hung():
        return [t for t in started_in(run_dir, 1) if t["kind"] == "confirmation" and t["trial"] >= n + 12]

    while not hung() or len([h for h in of_type(run_dir, "trial_heartbeat")
                             if h["trial"] == hung()[0]["trial"] and h["elapsed_s"] >= 0.5]) < 3:
        assert time.monotonic() < deadline and p.poll() is None
        time.sleep(0.1)
    p.kill()
    p.wait()
    [stuck] = hung()
    runner = {h["runner_pid"] for h in of_type(run_dir, "trial_heartbeat") if h["trial"] == stuck["trial"]}
    [runner_pid] = runner
    spent = bo(bo_repo, "status")[1]["budget"]["spent_s"]
    expected = eligible_now(run_dir)

    code, out = round_run(bo_repo, "--rationale", "resume", env=toy_env(sigma=0.01))
    assert code == 0, out
    assert gone(runner_pid)  # the orphaned runner is killed, not left running uncharged
    [abandoned] = of_type(run_dir, "trial_abandoned")
    beats = [h["elapsed_s"] for h in of_type(run_dir, "trial_heartbeat") if h["trial"] == stuck["trial"]
             and h["elapsed_s"] >= 0.5]
    assert abandoned["trial"] == stuck["trial"] and abandoned["wall_clock_s"] == beats[-1] >= 1
    assert all(b <= 2 * a + 0.5 for a, b in zip(beats, beats[1:]))  # a kill is charged ≥ half its time
    assert [(e["round"], e["trigger"]) for e in of_type(run_dir, "round_ended")][1:3] == [
        (1, "interrupted"), (2, "search_space")]
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert sorted(r2["seeded"]) == sorted(expected) and len(expected) >= 7
    assert stuck["trial"] not in [t["trial"] for t in started_in(run_dir, 2)]  # never retried
    # the confirmation still owed is paid first: the root ends with its 2 confirmations
    root, r2_trials = stuck["replicate_of"], started_in(run_dir, 2)
    assert r2_trials[0]["kind"] == "confirmation" and r2_trials[0]["replicate_of"] == root
    both = started_in(run_dir, 1) + r2_trials
    assert sum(t.get("replicate_of") == root and t["kind"] == "confirmation" for t in both) == 2
    # burn-in counts fresh sampler trials across the interrupted round: R2 needs fewer than 20
    samplers = [t for t in both if t["kind"] == "sampler" and t["status"] == "finished"]
    first = of_type(run_dir, "verdict")[0]
    r1 = sum(t["round"] == 1 for t in samplers)
    assert first["round"] == 2 and first["burn_in"]["fresh"] == 20 and 0 < r1
    assert sum(t["round"] == 2 and t["trial"] in first["trials"] for t in samplers) == 20 - r1
    _, status = bo(bo_repo, "status")
    assert status["trials"]["abandoned"] == 1
    r2_spent = sum(t["wall_clock_s"] for t in r2_trials)
    assert status["budget"]["spent_s"] == pytest.approx(spent + beats[-1] + r2_spent)


def test_the_round_cap_ends_a_round_at_a_quarter_of_the_remaining_budget(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 16\nreference_fidelity: {epochs: 4}\nreplicates_k: 3\ndelta: 0.02\n")
    env = toy_env(sigma=0.01, sleep=0.3)
    coded(bo_repo, run_dir, tmp_path, env=env)
    remaining = bo(bo_repo, "status")[1]["budget"]["remaining_s"]
    code, out = round_run(bo_repo, "--rationale", "search", env=env)
    assert code == 0, out
    assert out["trigger"] == "cap" and out["run_ended"] is None
    [started] = [s for s in of_type(run_dir, "round_started") if s["round"] == 1]
    assert started["cap_s"] == pytest.approx(0.25 * remaining)
    costs = [t["wall_clock_s"] for t in started_in(run_dir, 1)]
    assert started["cap_s"] - 2 * max(costs) < sum(costs) < started["cap_s"] + max(costs) / 2
    assert out["budget"]["remaining_s"] == pytest.approx(remaining - sum(costs))


def test_a_spent_budget_ends_the_round_and_the_run(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 3.8\nreference_fidelity: {epochs: 4}\ndeterministic: true\n"
                     "delta: 0.02\n")
    env = toy_env(sleep=1.0)  # smoke, then smoke H1's 2 trials: ~3.2 s; no third trial fits
    coded(bo_repo, run_dir, tmp_path, env=env)
    code, out = round_run(bo_repo, "--rationale", "search", env=env)
    assert code == 0, out
    assert out["trigger"] == "budget_spent" and out["run_ended"] == "budget_spent"
    assert out["next"] == []
    assert of_type(run_dir, "trial_refused")[-1]["kind"] == "sampler"
    assert of_type(run_dir, "run_ended") == [{"reason": "budget_spent"}]
    code, out = round_run(bo_repo, "--rationale", "again", env=env)
    assert code != 0 and "ended" in out["reason"]


def test_a_proxy_that_ranks_backwards_is_flagged_broken_by_the_drift_check(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\nladder: [{epochs: 1}]\n"
                     "deterministic: true\nseed: 2\n")  # unseeded, ~1 in 6 runs flaked
    env = toy_env(cheap_below=2, scramble=1)
    coded(bo_repo, run_dir, tmp_path, env=env)
    code, out = round_run(bo_repo, "--rationale", "calibrate the ladder", env=env)
    assert code == 0 and out["fidelity_calibration"]["fallback"] is True, out
    assert bo(bo_repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
    assert bo(bo_repo, "set-delta", "0.02", "--rationale", "the user's effect")[0] == 0
    code, out = round_run(bo_repo, "--rationale", "search at the proxy", env=env)
    assert code == 0, out
    trials = {t["trial"]: t for t in started_in(run_dir, 1)}
    [drift] = of_type(run_dir, "drift_check")
    assert out["drift"] == drift and drift["round"] == 1 and drift["broken"] is True
    inc, other = drift["configs"]
    assert inc == out["incumbent"]["trial"] and trials[other]["fidelity"] == {"epochs": 1}
    runs = [trials[t] for t in drift["trials"]]
    assert [t["kind"] for t in runs] == ["drift", "drift"]
    assert [t["fidelity"] for t in runs] == [{"epochs": 4}, {"epochs": 4}]
    assert [t["levers"] for t in runs] == [trials[inc]["levers"], trials[other]["levers"]]
    assert drift["reference"] == pytest.approx([true_loss(t["levers"]) for t in runs])


def test_no_stall_before_every_hypothesis_is_burned_in(bo_repo, bo_run, tmp_path):
    # 3 levers: the burn-in is max(10·3, 20) = 30 fresh sampler trials, the stall window only 20;
    # δ out of reach, so the stall window is complete 20 sampler trials after the first confirmation
    run_dir = bo_run(BASE + "delta: 10\nseed: 7\n")
    spec = {**SPEC, "levers": {**SPEC["levers"], "z": SPEC["levers"]["x"]}}
    code3 = BOWL.rstrip("\n") + " + 0 * lever(\"H1.z\")\n"
    coded(bo_repo, run_dir, tmp_path, code=code3, env=toy_env(sigma=0.01), spec=spec)
    code, out = round_run(bo_repo, "--rationale", "search", env=toy_env(sigma=0.01))
    assert code == 0, out
    # the stall window is complete long before 30; at burn-in the first verdict check finds the bowl
    # `irrelevant` next to δ = 10, and that pending reject holds the round open to its confirmation
    samplers = [t for t in started_in(run_dir, 1) if t["kind"] == "sampler"]
    [first, confirming] = of_type(run_dir, "verdict")
    assert first["burn_in"]["fresh"] == 30 and first["outcome"] == "pending-reject"
    assert confirming["outcome"] == "reject" and out["trigger"] == "search_space"
    assert len(samplers) == 30 + 15


def drift_run(d, python, seed):
    """A faithful, quiet cheap rung and a noisy reference (σ = 1): one BO round at the rung."""
    repo = make_repo(d)
    run_dir = init(repo, python, "budget_s: 3600\nreference_fidelity: {epochs: 4}\n"
                   f"ladder: [{{epochs: 1}}]\nreplicates_k: 3\nseed: {seed}\n")
    env = toy_env(cheap_below=2, sigma=0.01, ref_sigma=1.0, sleep_per_epoch=0.02)
    coded(repo, run_dir, d, env=env)
    code, out = round_run(repo, "--rationale", "calibrate the ladder", env=env)
    assert code == 0, out
    if out["fidelity_calibration"]["fallback"]:  # noise at the reference can fail the rung's ρ
        assert bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
    assert bo(repo, "set-delta", "0.02", "--rationale", "the user's effect")[0] == 0
    code, out = round_run(repo, "--rationale", "search at the rung", env=env)
    assert code == 0, out
    assert bo(repo, "status")[1]["fidelity"]["fidelity"] == {"epochs": 1}
    return run_dir, out


def test_reference_noise_alone_does_not_flag_a_faithful_proxy_broken(tmp_path, project_python):
    seeds = [1, 3, 4, 5]
    for s in seeds:
        (tmp_path / f"s{s}").mkdir()
    with ThreadPoolExecutor(len(seeds)) as pool:
        runs = list(pool.map(lambda s: drift_run(tmp_path / f"s{s}", project_python, s), seeds))
    drifts = [out["drift"] for _, out in runs]
    assert [d["broken"] for d in drifts] == [False] * len(seeds), drifts
    # the reference's noise did reverse the pair in some runs: the margin is what kept them unbroken
    flipped = [(d["proxy"][0] - d["proxy"][1]) * (d["reference"][0] - d["reference"][1]) < 0
               for d in drifts]
    assert any(flipped), drifts


def test_the_round_cap_covers_the_drift_check(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 16\nreference_fidelity: {epochs: 4}\nladder: [{epochs: 1}]\n"
                     "deterministic: true\n")
    env = toy_env(cheap_below=2, scramble=1, sleep_per_epoch=0.1)
    coded(bo_repo, run_dir, tmp_path, env=env)
    assert round_run(bo_repo, "--rationale", "calibrate the ladder", env=env)[0] == 0
    assert bo(bo_repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
    assert bo(bo_repo, "set-delta", "0.02", "--rationale", "the user's effect")[0] == 0
    code, out = round_run(bo_repo, "--rationale", "search", env=env)
    assert code == 0 and out["trigger"] == "cap", out
    [started] = [s for s in of_type(run_dir, "round_started") if s["round"] == 1]
    trials = started_in(run_dir, 1)
    assert [t["kind"] for t in trials][-2:] == ["drift", "drift"]
    proxy_costs = [t["wall_clock_s"] for t in trials if t["kind"] != "drift"]
    assert sum(t["wall_clock_s"] for t in trials) < started["cap_s"] + max(proxy_costs) / 2


def test_round_run_is_refused_until_the_selected_hypotheses_have_committed_code(bo_repo, bo_run,
                                                                                 tmp_path):
    run_dir = bo_run(BASE + "delta: 0.02\n")
    assert round_run(bo_repo, "--rationale", "calibrate")[0] == 0
    f = tmp_path / "spec.json"
    f.write_text(json.dumps(SPEC))
    assert bo(bo_repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    assert bo(bo_repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    before = events(run_dir)
    code, out = bo(bo_repo, "round-run", "--rationale", "too early")
    assert code != 0 and "H1.v1" in out["reason"] and "commit-lever" in out["reason"]
    assert events(run_dir) == before
    _, status = bo(bo_repo, "status")
    assert status["next"][0] == "write H1.v1's lever code, then smoke H1.v1 and commit-lever H1.v1"
