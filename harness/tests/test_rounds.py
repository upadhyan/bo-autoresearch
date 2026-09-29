"""Rounds where BO does the searching (#25), against a toy with a planted optimum."""
import json
import os
import random
import statistics
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from conftest import bo, events, git, write_run_yaml

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
BOWL = "    term = 4 * ((lever(\"H1.x\") - 0.7) ** 2 + (lever(\"H1.y\") - 0.3) ** 2)\n"
FLAT = "    term = 0 * (lever(\"H1.x\") + lever(\"H1.y\"))\n"


def true_loss(levers):
    return 1 + 4 * sum((levers[n] - v) ** 2 for n, v in OPT.items())


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


def coded(repo, run_dir, tmp_path, code=BOWL, env=None):
    """R0 done, then H1 proposed, registered, its lever code smoked and committed."""
    if not any(e["type"] == "round_ended" for e in events(run_dir)):
        assert bo(repo, "round-run", "--rationale", "calibrate", env=env)[0] == 0
    f = tmp_path / "spec.json"
    f.write_text(json.dumps(SPEC))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    assert bo(repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    train = run_dir / "worktree" / "train.py"
    src = train.read_text().replace("import fidelity, seed", "import fidelity, lever, seed")
    train.write_text(src.replace("    term = 0.0\n", code))
    code_, out = bo(repo, "smoke", "H1", "--rationale", "try", env=env)
    assert code_ == 0 and out["passed"], out
    code_, out = bo(repo, "commit-lever", "H1", "--rationale", "smoke passed")
    assert code_ == 0, out


def of_type(run_dir, type):
    return [e["payload"] for e in events(run_dir) if e["type"] == type]


def test_set_delta_is_set_once_after_r0(bo_repo, bo_run, tmp_path):
    run_dir = bo_run()
    code, out = bo(bo_repo, "set-delta", "0.1", "--rationale", "too early")
    assert code != 0 and "round-run" in out["reason"]
    coded(bo_repo, run_dir, tmp_path, env=toy_env(sigma=0.05))
    code, out = bo(bo_repo, "round-run", "--rationale", "no delta yet")
    assert code != 0 and "set-delta" in out["reason"]
    sigma = bo(bo_repo, "status")[1]["sigma"]
    assert f"{2 * sigma:.3g}" in out["reason"]  # suggested 2σ
    assert bo(bo_repo, "status")[1]["suggested_delta"] == pytest.approx(2 * sigma)

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




def started_in(run_dir, r):
    """The trials of round r as started, with their outcome from the log."""
    ends = {p["trial"]: {**p, "status": t[6:]} for t in ("trial_finished", "trial_failed",
                                                           "trial_abandoned")
            for p in of_type(run_dir, t)}
    return [{**p, **ends.get(p["trial"], {})} for p in of_type(run_dir, "trial_started")
            if p.get("round") == r]


def search_once(d, python, sigma=0.01):
    """One run on the bowl toy: R0, H1 coded, one BO round. Returns (run_dir, round-run JSON)."""
    repo = make_repo(d)
    run_dir = init(repo, python, BASE + "delta: 0.02\n")
    env = toy_env(sigma=sigma)
    coded(repo, run_dir, d, env=env)
    code, out = bo(repo, "round-run", "--rationale", "search H1", env=env)
    assert code == 0, out
    return run_dir, out


def random_search(n, seed, sigma=0.01):
    """Plain random search with n trials on the same toy: the true loss of its best observed point."""
    rng = random.Random(seed)
    points = [{n_: rng.random() for n_ in OPT} for _ in range(n)]
    observed = [true_loss(p) + sigma * rng.gauss(0, 1) for p in points]
    return true_loss(points[observed.index(min(observed))])


def test_bo_beats_random_search_at_equal_trial_count(tmp_path, project_python):
    dirs = [tmp_path / f"s{i}" for i in range(3)]
    for d in dirs:
        d.mkdir()
    with ThreadPoolExecutor(3) as pool:  # three independent runs (their own seeds), side by side
        runs = list(pool.map(lambda d: search_once(d, project_python), dirs))
    bo_regret, rs_regret = [], []
    for seed, (run_dir, out) in enumerate(runs):
        assert out["trigger"] == "stall" and out["round"] == 1
        trials = started_in(run_dir, 1)
        bo_regret.append(true_loss(out["incumbent"]["levers"]) - 1)
        rs_regret.append(random_search(len(trials), seed) - 1)
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
    assert out["run_ended"] is None and out["next"] == ["round-run"]
    assert of_type(run_dir, "round_ended")[-1] == {"round": 1, "trigger": "stall",
                                                    "incumbent": out["incumbent"]}
    assert not of_type(run_dir, "run_ended")
    assert out["drift"] is None  # the round ran at the reference fidelity


def test_replicates_are_their_own_trials_and_every_new_incumbent_is_confirmed(tmp_path, project_python):
    run_dir, out = search_once(tmp_path, project_python)
    trials = started_in(run_dir, 1)
    by_id = {t["trial"]: t for t in trials}
    replicates = [t for t in trials if "replicate_of" in t]
    assert len(replicates) >= 0.1 * len(trials)
    for t in replicates:  # same config, own seed and objective, the original untouched
        orig = by_id[t["replicate_of"]]
        assert t["levers"] == orig["levers"] and t["seed"] != orig["seed"]
        assert "replicate_of" not in orig
    # the best sampler trial was a new incumbent when it came in: it got its 2 confirmations
    samplers = [t for t in trials if t["kind"] == "sampler" and t["status"] == "finished"]
    best = min(samplers, key=lambda t: t["objective"])
    assert sum(t.get("replicate_of") == best["trial"] and t["kind"] == "confirmation"
               for t in trials) == 2
    inc = out["incumbent"]
    group = [t["objective"] for t in trials if inc["trial"] in (t["trial"], t.get("replicate_of"))]
    assert inc["confirmed"] and inc["replicates"] == len(group) - 1 >= 2
    assert inc["mean"] == pytest.approx(statistics.fmean(group))
    assert by_id[inc["trial"]]["objective"] != inc["mean"]  # replicates never averaged in


def eligible_now(run_dir):
    """Independently: finished trials at the reference fidelity with exactly the round's levers, in range."""
    ends = {p["trial"]: p for p in of_type(run_dir, "trial_finished")}
    return {p["trial"]: ends[p["trial"]]["objective"] for p in of_type(run_dir, "trial_started")
            if p["trial"] in ends and p["fidelity"] == {"epochs": 4}
            and set(p["levers"]) == set(OPT) and all(0 <= v <= 1 for v in p["levers"].values())}


def study_trials(run_dir):
    import optuna
    [s] = optuna.get_all_study_summaries(f"sqlite:///{run_dir / 'studies.db'}")
    study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{run_dir / 'studies.db'}")
    return {t.user_attrs["trial"]: t.value for t in study.get_trials(states=[optuna.trial.TrialState.COMPLETE])}


def test_a_killed_round_is_resumed_from_the_log(bo_repo, bo_run, tmp_path):
    run_dir = bo_run(BASE + "delta: 0.02\n")
    coded(bo_repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    n = len(of_type(run_dir, "trial_started"))
    env = toy_env(sigma=0.01, sleep=30, sleep_from=n + 8)  # the round's 8th trial hangs
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "go"],
                         cwd=bo_repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 60
    while not [h for h in of_type(run_dir, "trial_heartbeat") if h["trial"] == n + 8]:
        assert time.monotonic() < deadline and p.poll() is None
        time.sleep(0.2)
    p.kill()
    p.wait()
    spent = bo(bo_repo, "status")[1]["budget"]["spent_s"]

    (run_dir / "studies.db").unlink()  # disposable: the resumed round rebuilds it from the log
    expected = eligible_now(run_dir)
    code, out = bo(bo_repo, "round-run", "--rationale", "resume", env=toy_env(sigma=0.01))
    assert code == 0, out
    [abandoned] = of_type(run_dir, "trial_abandoned")
    beats = [h["elapsed_s"] for h in of_type(run_dir, "trial_heartbeat") if h["trial"] == n + 8]
    assert abandoned["trial"] == n + 8 and abandoned["wall_clock_s"] == beats[-1] >= 1
    assert [(e["round"], e["trigger"]) for e in of_type(run_dir, "round_ended")][1:3] == [
        (1, "interrupted"), (2, out["trigger"])]
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert sorted(r2["seeded"]) == sorted(expected) and len(expected) >= 7
    assert n + 8 not in [t["trial"] for t in started_in(run_dir, 2)]  # never retried
    assert study_trials(run_dir) == eligible_now(run_dir)  # the study is the log's eligible trials
    _, status = bo(bo_repo, "status")
    assert status["trials"]["abandoned"] == 1
    r2_spent = sum(t["wall_clock_s"] for t in started_in(run_dir, 2))
    assert status["budget"]["spent_s"] == pytest.approx(spent + beats[-1] + r2_spent)


def test_the_round_cap_ends_a_round_at_a_quarter_of_the_remaining_budget(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 16\nreference_fidelity: {epochs: 4}\nreplicates_k: 3\ndelta: 0.02\n")
    env = toy_env(sigma=0.01, sleep=0.3)
    coded(bo_repo, run_dir, tmp_path, env=env)
    remaining = bo(bo_repo, "status")[1]["budget"]["remaining_s"]
    code, out = bo(bo_repo, "round-run", "--rationale", "search", env=env)
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
    code, out = bo(bo_repo, "round-run", "--rationale", "search", env=env)
    assert code == 0, out
    assert out["trigger"] == "budget_spent" and out["run_ended"] == "budget_spent"
    assert out["next"] == []
    assert of_type(run_dir, "trial_refused")[-1]["kind"] == "sampler"
    assert of_type(run_dir, "run_ended") == [{"reason": "budget_spent"}]
    code, out = bo(bo_repo, "round-run", "--rationale", "again", env=env)
    assert code != 0 and "ended" in out["reason"]


def test_a_proxy_that_ranks_backwards_is_flagged_broken_by_the_drift_check(bo_repo, bo_run, tmp_path):
    run_dir = bo_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\nladder: [{epochs: 1}]\n"
                     "deterministic: true\n")
    env = toy_env(cheap_below=2, scramble=1)
    coded(bo_repo, run_dir, tmp_path, env=env)
    code, out = bo(bo_repo, "round-run", "--rationale", "calibrate the ladder", env=env)
    assert code == 0 and out["fidelity_calibration"]["fallback"] is True, out
    assert bo(bo_repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
    assert bo(bo_repo, "set-delta", "0.02", "--rationale", "the user's effect")[0] == 0
    code, out = bo(bo_repo, "round-run", "--rationale", "search at the proxy", env=env)
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


def test_round_run_is_refused_until_the_selected_hypotheses_have_committed_code(bo_repo, bo_run,
                                                                                 tmp_path):
    run_dir = bo_run(BASE + "delta: 0.02\n")
    assert bo(bo_repo, "round-run", "--rationale", "calibrate")[0] == 0
    f = tmp_path / "spec.json"
    f.write_text(json.dumps(SPEC))
    assert bo(bo_repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    assert bo(bo_repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    before = events(run_dir)
    code, out = bo(bo_repo, "round-run", "--rationale", "too early")
    assert code != 0 and "H1.v1" in out["reason"] and "commit-lever" in out["reason"]
    assert events(run_dir) == before
    _, status = bo(bo_repo, "status")
    assert status["next"] == ["write H1.v1's lever code, then smoke H1.v1 and commit-lever H1.v1"]
