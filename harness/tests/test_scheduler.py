"""The round scheduler and run control (#29): slots by priority within the dimension cap, park and
unpark, starvation, agent-chosen trials, expected verdicts, checkpoints, stop and the target."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import bo, events, expect_all, ready, round_run, write_run_yaml, register
from test_rounds import BASE, coded, init, make_repo, of_type, toy_env
from test_verdicts import code_hypothesis, lever, spec


def propose(repo, d, n, levers, **extra):
    f = d / f"spec{n}.json"
    f.write_text(json.dumps(spec({k: lever() for k in levers}, **extra)))
    code, out = bo(repo, "propose", "--file", str(f), "--rationale", "idea")
    assert code == 0, out
    assert register(repo, f"H{n}")[0] == 0


def schedule(repo):
    return bo(repo, "status")[1]["schedule"]


def test_slots_follow_priority_within_the_dimension_cap_and_never_co_select_exclusives(tmp_path,
                                                                                        project_python):
    # trials of ~1.1 s and ~110 s left after R0: a round (25%) affords ~25 trials, so the dimension
    # cap is 2 (a burn-in of max(10·d, 20) must fit)
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE.replace("budget_s: 3600", "budget_s: 115") + "delta: 0.1\n")
    assert bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sleep=1.0))[0] == 0
    propose(repo, tmp_path, 1, ["a"])
    propose(repo, tmp_path, 2, ["a", "b"])
    propose(repo, tmp_path, 3, ["a"])
    for h, p in (("H2", 5), ("H3", 3), ("H1", 0)):
        code, out = bo(repo, "prioritize", h, str(p), "--rationale", "the orchestrator's ranking")
        assert code == 0, out
    s = schedule(repo)
    assert s["dimension_cap"] == 2
    # H2 (2 levers) comes first and fills the cap: H3 would exceed it, and a lower priority never
    # jumps ahead of a higher one
    assert s["selected"] == ["H2.v1"] and s["queue"] == ["H2.v1", "H3.v1", "H1.v1"]
    _, status = bo(repo, "status")
    assert [d for d in status["next"] if "lever code" in d] == [
        "write H2.v1's lever code, then smoke H2.v1 and commit-lever H2.v1"]

    assert bo(repo, "prioritize", "H3", "9", "--rationale", "more promising")[0] == 0
    assert bo(repo, "prioritize", "H1", "9", "--rationale", "as promising")[0] == 0
    # equal priorities: registration order
    assert schedule(repo)["selected"] == ["H1.v1", "H3.v1"]
    assert [(p["id"], p["priority"]) for p in of_type(run_dir, "hypothesis_prioritized")][-2:] == [
        ("H3.v1", 9), ("H1.v1", 9)]

    # H4 excludes H1 and outranks everything: H1 is skipped (not a stop), H3 fills the second slot
    propose(repo, tmp_path, 4, ["a"], exclusive_with=["H1"])
    assert bo(repo, "prioritize", "H4", "10", "--rationale", "top")[0] == 0
    assert schedule(repo)["selected"] == ["H4.v1", "H3.v1"]

    code, out = bo(repo, "prioritize", "H9", "1", "--rationale", "no such")
    assert code != 0 and "H9" in out["reason"]


def expected(repo, d, h, verdict="undecided", reason="too early to say"):
    f = d / f"expected-{h}.json"
    f.write_text(json.dumps({"hypothesis": h, "verdict": verdict, "reason": reason}))
    return bo(repo, "record", "expected", "--file", str(f), "--rationale", "before the round")


def test_round_run_is_refused_until_every_selected_hypothesis_has_an_expected_verdict(tmp_path,
                                                                                      project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\n")
    coded(repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    _, status = bo(repo, "status")
    assert status["schedule"]["expected_missing"] == ["H1.v1"]
    assert status["next"] == ["record expected H1.v1 (retain, reject or undecided, with a one-line "
                              "reason) before the round"]
    before = events(run_dir)
    code, out = bo(repo, "round-run", "--rationale", "go")
    assert code != 0 and "H1.v1" in out["reason"] and "record expected" in out["reason"]
    assert events(run_dir) == before
    assert expected(repo, tmp_path, "H1.v1", "retain", "the bowl should pay")[0] == 0
    _, status = bo(repo, "status")
    assert status["schedule"]["expected_missing"] == [] and status["next"] == ["round-run"]
    code, out = bo(repo, "round-run", "--rationale", "go", env=toy_env(sigma=0.01))
    assert code == 0, out
    # an expectation is for one round: the next one needs its own
    assert out["schedule"]["expected_missing"] == ["H1.v1"]


def test_a_batch_of_expected_verdicts_is_one_record_call_all_or_nothing(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\n")
    assert bo(repo, "round-run", "--rationale", "calibrate", env=toy_env(sigma=0.01))[0] == 0
    propose(repo, tmp_path, 1, ["a"])
    propose(repo, tmp_path, 2, ["b"])
    assert schedule(repo)["expected_missing"] == ["H1.v1", "H2.v1"]
    f = tmp_path / "batch.json"
    items = [{"hypothesis": "H1.v1", "verdict": "retain", "reason": "the bowl should pay"},
             {"hypothesis": "H2.v1", "verdict": "reject", "reason": "no term reads it"}]
    before = events(run_dir)
    for bad, words in [([items[0], {**items[1], "verdict": "win"}], "expected[1]: verdict"),
                       ([items[0], items[0]], "expected[1].hypothesis: H1.v1 is already at expected[0]"),
                       ([], "one or more")]:
        f.write_text(json.dumps({"expected": bad}))
        code, out = bo(repo, "record", "expected", "--file", str(f), "--rationale", "before R1")
        assert code == 1 and words in out["reason"], out
    assert events(run_dir) == before  # the valid item of a refused batch isn't logged either
    f.write_text(json.dumps({"expected": items}))
    code, out = bo(repo, "record", "expected", "--file", str(f), "--rationale", "before R1")
    assert code == 0 and out == {"recorded": "expected", "agent_id": None, "items": 2}, out
    logged = [e for e in events(run_dir)[len(before):] if e["type"] == "record"]
    assert [(e["actor"], e["payload"]["kind"], e["payload"]["record"]) for e in logged] == [
        ("orchestrator", "expected", i) for i in items]
    assert schedule(repo)["expected_missing"] == []


def in_background(repo, run_dir, env, until):
    """round-run in the background, returned once `until(run_dir)` holds (the round still running)."""
    expect_all(repo)
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "go"],
                         cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    deadline = time.monotonic() + 60
    while not until(run_dir):
        assert time.monotonic() < deadline and p.poll() is None
        time.sleep(0.05)
    return p


def round_trials(run_dir, r):
    return [t for t in of_type(run_dir, "trial_started") if t.get("round") == r]


def test_a_user_stop_ends_the_running_round_and_the_run_and_newcomers_wait_for_a_boundary(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\n")
    env = toy_env(sigma=0.01, sleep=0.2)
    coded(repo, run_dir, tmp_path, env=env)
    p = in_background(repo, run_dir, env, lambda d: len(round_trials(d, 1)) >= 3)
    propose(repo, tmp_path, 2, ["a"])  # registered mid-round: it joins at a round boundary only
    code, out = bo(repo, "park", "H1", "--reason", "x", "--rationale", "mid-round")
    assert code != 0 and "round is running" in out["reason"]
    code, out = bo(repo, "stop", "--rationale", "the user stops the run")
    assert code == 0 and out["run_ended"] == "user_stop", out
    out = json.loads(p.communicate(timeout=60)[0])
    assert out["trigger"] == "user_stop" and out["run_ended"] == "user_stop"
    assert of_type(run_dir, "round_ended")[-1]["trigger"] == "user_stop"
    assert [e["reason"] for e in of_type(run_dir, "run_ended")] == ["user_stop"]
    [r1] = [s for s in of_type(run_dir, "round_started") if s["round"] == 1]
    assert r1["hypotheses"] == ["H1.v1"]
    assert [d.split(" ")[0] for d in out["next"]] == ["wrapup"]  # only the wrap-up is left
    for cmd in (["round-run"], ["stop"]):
        code, out = bo(repo, *cmd, "--rationale", "again")
        assert code != 0 and "ended" in out["reason"], cmd


def test_checkpoint_mode_pauses_at_each_round_boundary_and_a_user_pause_ends_the_running_round(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\ncheckpoint: true\n")
    env = toy_env(sigma=0.01)
    coded(repo, run_dir, tmp_path, env=env)
    code, out = round_run(repo, "--rationale", "R1", env=env)
    assert code == 0 and out["round"] == 1, out
    assert of_type(run_dir, "checkpoint") == [{"round": 1}]
    assert out["paused"] == {"by": "checkpoint", "round": 1}
    assert "checkpoint --resume" in out["next"][-1]
    code, out = round_run(repo, "--rationale", "R2 too early", env=env)
    assert code != 0 and "paused" in out["reason"]
    code, out = bo(repo, "checkpoint", "--resume", "--rationale", "the user reviewed R1")
    assert code == 0 and out["paused"] is None, out
    code, out = bo(repo, "checkpoint", "--resume", "--rationale", "twice")
    assert code != 0 and "not paused" in out["reason"]

    # the user pauses while R2 runs: the round ends at its next step, the run waits
    p = in_background(repo, run_dir, toy_env(sigma=0.01, sleep=0.2), lambda d: len(round_trials(d, 2)) >= 2)
    code, out = bo(repo, "checkpoint", "--rationale", "the user wants a look")
    assert code == 0 and out["paused"]["by"] == "user", out
    out = json.loads(p.communicate(timeout=60)[0])
    assert out["round"] == 2 and out["trigger"] == "checkpoint" and out["run_ended"] is None
    assert of_type(run_dir, "checkpoint") == [{"round": 1}]  # already paused: no second pause
    code, out = round_run(repo, "--rationale", "R3", env=env)
    assert code != 0 and "paused" in out["reason"]
    assert bo(repo, "checkpoint", "--resume", "--rationale", "go on")[0] == 0
    assert ready(bo(repo, "status")[1])


def two_coded(tmp_path, python, extra=""):
    """R0, then H1 (a useful lever) and H2 (a useless one) coded."""
    repo = make_repo(tmp_path)
    run_dir = init(repo, python, BASE + "delta: 0.02\n" + extra)
    env = toy_env(sigma=0.01)
    assert bo(repo, "round-run", "--rationale", "calibrate", env=env)[0] == 0
    code_hypothesis(repo, run_dir, tmp_path, 1, spec({"a": lever()}), '    term -= 0.3 * lever("H1.a")', env)
    code_hypothesis(repo, run_dir, tmp_path, 2, spec({"a": lever()}), '    term += 0 * lever("H2.a")', env)
    return repo, run_dir


def pause_after(repo, run_dir, r, n):
    """Round r, paused by the user once it has started n trials."""
    p = in_background(repo, run_dir, toy_env(sigma=0.01, sleep=0.1), lambda d: len(round_trials(d, r)) >= n)
    assert bo(repo, "checkpoint", "--rationale", "a look")[0] == 0
    out = json.loads(p.communicate(timeout=60)[0])
    assert out["trigger"] == "checkpoint"
    assert bo(repo, "checkpoint", "--resume", "--rationale", "go on")[0] == 0
    return out


def test_a_parked_hypothesis_leaves_the_search_without_evidence_and_returns_with_the_same_version(
        tmp_path, project_python):
    repo, run_dir = two_coded(tmp_path, project_python)
    pause_after(repo, run_dir, 1, 3)
    assert [s["hypotheses"] for s in of_type(run_dir, "round_started") if s["round"] == 1] == [["H1.v1", "H2.v1"]]
    code, out = bo(repo, "park", "H2", "--rationale", "a taste call")
    assert code != 0 and "--reason" in out["reason"]
    code, out = bo(repo, "park", "H2", "--reason", "off the brief's intent", "--rationale", "the review")
    assert code == 0, out
    assert [h["status"] for h in out["hypotheses"]] == ["active", "parked"]
    assert out["schedule"]["selected"] == ["H1.v1"]
    assert of_type(run_dir, "hypothesis_parked") == [
        {"id": "H2.v1", "reason": "off the brief's intent", "from": "active", "rationale": "the review"}]
    code, out = bo(repo, "park", "H2", "--reason", "again", "--rationale", "again")
    assert code != 0 and "parked" in out["reason"]

    pause_after(repo, run_dir, 2, 3)
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert r2["hypotheses"] == ["H1.v1"] and list(r2["search_space"]) == ["H1.a"]
    assert all(t["levers"]["H2.a"] == 0.0 for t in round_trials(run_dir, 2))  # at its baseline
    # parking is no evidence: no verdict, reject or inconclusive came of it
    assert not [v for v in of_type(run_dir, "verdict") if v["hypothesis"] == "H2.v1"]
    assert not of_type(run_dir, "hypothesis_rejected") + of_type(run_dir, "hypothesis_inconclusive")

    code, out = bo(repo, "unpark", "H1", "--reason", "x", "--rationale", "not parked")
    assert code != 0 and "not parked" in out["reason"]
    code, out = bo(repo, "unpark", "H2", "--reason", "the user wants it tested", "--rationale", "return")
    assert code == 0, out
    assert [(h["id"], h["status"]) for h in out["hypotheses"]] == [("H1.v1", "active"), ("H2.v1", "registered")]
    assert not [e for e in of_type(run_dir, "hypothesis_proposed") if e["id"] != "H1.v1" and e["id"] != "H2.v1"]
    # its code is still committed: it rejoins the next round as it was
    assert out["schedule"]["selected"] == ["H1.v1", "H2.v1"]
    assert not [d for d in out["next"] if "lever code" in d]
    pause_after(repo, run_dir, 3, 3)
    assert of_type(run_dir, "hypothesis_activated")[-1] == {"id": "H2.v1", "round": 3}


SIX = "seeds: [" + ", ".join(["{H1.x: 0.1}"] * 6) + "]\n"


@pytest.mark.parametrize("extra, field", [
    (SIX, "seeds"), ("seeds: [0.5]\n", "seeds"), ("seeds: {H1.x: 0.1}\n", "seeds"),
    ("target: low\n", "target"), ("max_trials: 0\n", "max_trials"), ("max_trials: 2.5\n", "max_trials"),
    ("checkpoint: yes please\n", "checkpoint"),
])
def test_init_refuses_invalid_run_control(repo, tmp_path, project_python, extra, field):
    y = write_run_yaml(tmp_path / "run.yaml", project_python, "budget_s: 10\nreference_fidelity: {epochs: 4}\n" + extra)
    code, out = bo(repo, "init", str(y), "--rationale", "go")
    assert code != 0 and field in out["reason"]


def enqueue(repo, config, expected="0.9"):
    args = ["enqueue", "--config", json.dumps(config) if not isinstance(config, str) else config]
    return bo(repo, *args, *(["--expected", expected] if expected is not None else []),
              "--rationale", "the orchestrator's pick")


def test_agent_chosen_trials_are_capped_per_round_by_a_shrinking_cap_and_need_an_expected_objective(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    # two user-named seeds: round-1 agent-chosen trials, the second outside H1.x's range
    run_dir = init(repo, project_python, BASE + "delta: 0.02\nseeds: [{H1.x: 0.7, H1.y: 0.3}, {H1.x: 5.0}]\n")
    code, out = enqueue(repo, {"H1.x": 0.6})
    assert code != 0 and "R1" in out["reason"]  # none before the calibration round
    env = toy_env(sigma=0.01)
    coded(repo, run_dir, tmp_path, env=env)
    assert bo(repo, "status")[1]["schedule"]["agent_trials"] == {"round": 1, "cap": 5, "used": 2}

    code, out = enqueue(repo, {"H1.x": 0.6}, expected=None)
    assert code != 0 and "--expected" in out["reason"]
    for bad, why in (({"H1.x": 1.5}, "range"), ({"H9.q": 0.1}, "H9.q"), ("[0.5]", "object"),
                     ("{nope", "JSON"), ({}, "lever")):
        code, out = enqueue(repo, bad)
        assert code != 0 and why in out["reason"], (bad, out)
    code, out = enqueue(repo, {"H1.x": 0.6}, expected="nan")
    assert code != 0 and "expected" in out["reason"]
    for x in (0.6, 0.65, 0.75):  # 5 − 2 seeds = 3 left in R1
        code, out = enqueue(repo, {"H1.x": x, "H1.y": 0.3}, expected="0.7")
        assert code == 0, out
    code, out = enqueue(repo, {"H1.x": 0.8})
    assert code != 0 and "cap" in out["reason"] and "5" in out["reason"]
    assert out == enqueue(repo, {"H1.x": 0.8})[1]  # refused, not queued

    code, out = round_run(repo, "--rationale", "R1", env=env)
    assert code == 0, out
    first = round_trials(run_dir, 1)[:4]
    assert [t["kind"] for t in first] == ["seed", "agent", "agent", "agent"]
    assert first[0]["levers"] == {"H1.x": 0.7, "H1.y": 0.3}
    assert [t["levers"]["H1.x"] for t in first[1:]] == [0.6, 0.65, 0.75]
    assert [t["expected"] for t in first[1:]] == [0.7] * 3 and "expected" not in first[0]
    assert [(d["kind"], d["config"], "range" in d["reason"]) for d in of_type(run_dir, "agent_trial_skipped")] == [
        ("seed", {"H1.x": 5.0}, True)]
    assert sum(t["kind"] in ("seed", "agent") for t in round_trials(run_dir, 1)) == 4
    # the cap shrinks: max(1, 6 − 2) = 4 in R2
    assert out["schedule"]["agent_trials"] == {"round": 2, "cap": 4, "used": 0}
    for x in (0.1, 0.2, 0.3, 0.4):
        assert enqueue(repo, {"H1.x": x})[0] == 0
    code, out = enqueue(repo, {"H1.x": 0.5})
    assert code != 0 and "cap" in out["reason"]


def test_a_reached_target_ends_the_run_only_once_confirmed(tmp_path, project_python):
    # the bowl's loss is 1.0 at the baseline and 0.68 at its bottom: 0.9 is well within reach
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\ntarget: 0.9\n")
    env = toy_env(sigma=0.01)
    coded(repo, run_dir, tmp_path, env=env)
    code, out = round_run(repo, "--rationale", "R1", env=env)
    assert code == 0, out
    assert out["trigger"] == "target" and out["run_ended"] == "target_reached"
    assert of_type(run_dir, "run_ended") == [{"reason": "target_reached", "incumbent": out["incumbent"]}]
    trials = [t for t in round_trials(run_dir, 1)]
    ends = {p["trial"]: p for p in of_type(run_dir, "trial_finished")}
    inc = out["incumbent"]
    group = [ends[t["trial"]]["objective"] for t in trials if inc["trial"] in (t["trial"], t.get("replicate_of"))]
    assert inc["confirmed"] and len(group) >= 3 and sum(group) / len(group) <= 0.9
    # reaching it was not enough: the round went on to confirm it
    first = next(t["trial"] for t in trials if t["trial"] in ends and ends[t["trial"]]["objective"] <= 0.9)
    assert sum(t["kind"] == "confirmation" and t["trial"] > first for t in trials) >= 2
    code, out = round_run(repo, "--rationale", "again", env=env)
    assert code != 0 and "ended" in out["reason"]


def test_the_trial_ceiling_ends_the_run(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\nmax_trials: 20\n")
    env = toy_env(sigma=0.01)
    coded(repo, run_dir, tmp_path, env=env)
    code, out = round_run(repo, "--rationale", "R1", env=env)
    assert code == 0, out
    assert out["trigger"] == "max_trials" and out["run_ended"] == "max_trials"
    assert out["trials"]["total"] == 20 and of_type(run_dir, "trial_refused")[-1]["max_trials"] == 20


def test_a_hypothesis_left_unselected_for_3_rounds_is_flagged(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\n")
    coded(repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    propose(repo, tmp_path, 2, ["a"], exclusive_with=["H1"])  # waits while H1 is in the search
    starved = []
    for r in (1, 2, 3):
        pause_after(repo, run_dir, r, 2)
        starved.append(schedule(repo)["starved"])
    assert starved == [[], [], [{"id": "H2.v1", "rounds": 3}]]
    assert all(s["hypotheses"] == ["H1.v1"] for s in of_type(run_dir, "round_started") if s["round"] >= 1)


def tight_cap_run(d, python, seed):
    """R0, then R1 searching H1 (a useful one-lever bowl, a no-op at its baseline 0.4, best at 0.7),
    then H2 (two do-nothing levers) queued behind it, and baseline smokes spending the budget until the
    dimension cap is 2: H1's lever and H2's two don't fit together. -> (repo, run_dir)"""
    d.mkdir(exist_ok=True)
    repo = make_repo(d)
    run_dir = init(repo, python, BASE.replace("budget_s: 3600", "budget_s: 150") + f"delta: 0.1\nseed: {seed}\n")
    env = toy_env(sigma=0.05, sleep=0.5)  # trials of ~0.6 s: R1 leaves a cap of ~4
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    code_hypothesis(repo, run_dir, d, 1, spec({"x": lever(0.4)}),
                    '    term += 4 * ((lever("H1.x") - 0.7) ** 2 - 0.09)', env)
    code, out = round_run(repo, "--rationale", "R1", env=env)
    assert code == 0, out
    assert out["schedule"]["narrowing"] == []  # nothing queued is crowded out: no narrowing
    code_hypothesis(repo, run_dir, d, 2, spec({"x": lever(), "y": lever()}),
                    '    term += 0 * (lever("H2.x") + lever("H2.y"))', env)
    # a 3 s smoke moves the cap's affordable trials by ~10%, under the 10 a step of the cap takes
    while (cap := schedule(repo)["dimension_cap"]) > 2:
        assert bo(repo, "smoke", "--rationale", "spend", env=toy_env(sigma=0.05, sleep=3))[0] == 0
    assert cap == 2
    return repo, run_dir


@pytest.mark.slow
def test_a_tight_dimension_cap_narrows_the_oldest_concentrated_retained_hypothesis_so_the_queue_joins(
        tmp_path, project_python):
    seeds = range(4)
    with ThreadPoolExecutor(len(seeds)) as pool:
        runs = list(pool.map(lambda s: tight_cap_run(tmp_path / f"s{s}", project_python, s), seeds))
    joined = 0
    for repo, run_dir in runs:
        _, status = bo(repo, "status")
        s, h1 = status["schedule"], status["hypotheses"][0]
        if not s["narrowing"]:  # nothing to narrow: H2 waits behind H1, which keeps its full range
            assert s["selected"] == ["H1.v1"]
            continue
        # H1, retained with its best trials and baseline in a window of at most half its range, is
        # narrowed to that window (never frozen), and stops counting against the cap: H2 joins
        [n] = s["narrowing"]
        assert h1["status"] == "retained" and (n["id"], n["lever"]) == ("H1.v1", "H1.x")
        lo, hi = n["after"]["low"], n["after"]["high"]
        assert 0 < lo <= 0.4 < hi <= 1 and hi - lo <= 0.5 and lo <= 0.7 <= hi
        assert s["selected"] == ["H1.v1", "H2.v1"] and s["context"] == ["H1.v1"]
        # an agent-chosen trial is judged against the narrowed range the round will search
        code, out = enqueue(repo, {"H1.x": lo / 2})
        assert code != 0 and "range" in out["reason"]
        pause_after(repo, run_dir, 2, 1)
        [e] = [e for e in events(run_dir) if e["type"] == "narrowed"]
        assert e["actor"] == "harness" and e["payload"]["after"] == n["after"] and e["payload"]["cap"]
        [r2] = [r for r in of_type(run_dir, "round_started") if r["round"] == 2]
        assert r2["hypotheses"] == ["H1.v1", "H2.v1"]
        assert {k: r2["search_space"]["H1.x"][k] for k in ("low", "high")} == n["after"]
        assert schedule(repo)["narrowing"] == []  # done once
        joined += 1
    # measured 5 of range(6) (H1 retained with a concentrated posterior in R1): at 5/6, fewer than 2
    # of 4 happens about 2% of the time
    assert joined >= 2, joined


def test_queued_agent_chosen_trials_an_interrupted_round_never_started_run_in_the_resuming_round(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\n")
    coded(repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    for x, want in ((0.6, "0.8"), (0.65, "0.75"), (0.75, "0.7")):
        assert enqueue(repo, {"H1.x": x, "H1.y": 0.3}, expected=want)[0] == 0
    n = len(of_type(run_dir, "trial_started"))
    # R1's first agent-chosen trial hangs, and the round is killed with it in flight
    env = {**toy_env(sigma=0.01, sleep=120, sleep_from=n + 1), "TOY_SLEEP_KIND": "agent"}
    expect_all(repo)
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "go"],
                         cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 90
    while not any(h["trial"] == n + 1 and h["elapsed_s"] >= 0.5 for h in of_type(run_dir, "trial_heartbeat")):
        assert time.monotonic() < deadline and p.poll() is None
        time.sleep(0.1)
    p.kill()
    p.wait()
    assert [t["queued"] for t in round_trials(run_dir, 1)] == [1]
    # the two never started are the resuming round's, and count against its cap of 4
    assert schedule(repo)["agent_trials"] == {"round": 2, "cap": 4, "used": 2}

    code, out = round_run(repo, "--rationale", "resume", env=toy_env(sigma=0.01))
    assert code == 0, out
    assert of_type(run_dir, "trial_abandoned")[0]["trial"] == n + 1
    r2 = round_trials(run_dir, 2)
    assert [(t["kind"], t["queued"], t["levers"]["H1.x"], t["expected"]) for t in r2[:2]] == [
        ("agent", 2, 0.65, 0.75), ("agent", 3, 0.75, 0.7)]
    # the abandoned one is never retried, and nothing runs twice
    assert sorted(t["queued"] for t in of_type(run_dir, "trial_started") if "queued" in t) == [1, 2, 3]
    assert out["schedule"]["agent_trials"] == {"round": 3, "cap": 3, "used": 0}
