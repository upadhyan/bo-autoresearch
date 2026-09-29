"""Parallel trials: run.yaml `workers` runs up to that many trials of a round at once."""
import subprocess
import sys
import time

import pytest

from conftest import bo, events, expect_all, round_run
from test_rounds import BASE, coded, eligible_now, gone, init, make_repo, of_type, started_in, toy_env

ENDS = ("trial_finished", "trial_failed", "trial_abandoned")


def flights(run_dir, r):
    """Replays round r from the log: the trials in flight at each trial start, as (trial, others)."""
    ids = {t["trial"]: t for t in started_in(run_dir, r)}
    live, out = {}, []
    for e in events(run_dir):
        n = e["payload"].get("trial")
        if n not in ids:
            continue
        if e["type"] == "trial_started":
            out.append((ids[n], list(live.values())))
            live[n] = ids[n]
        elif e["type"] in ENDS:
            live.pop(n)
    return out


def ended_before_round_end(run_dir, r):
    """Every trial of round r has its outcome logged before the round's round_ended."""
    evs = events(run_dir)
    [end] = [e["seq"] for e in evs if e["type"] == "round_ended" and e["payload"]["round"] == r]
    ids = {t["trial"] for t in started_in(run_dir, r)}
    done = {e["payload"]["trial"] for e in evs if e["type"] in ENDS and e["seq"] < end}
    return ids <= done


def test_three_workers_run_trials_concurrently_and_keep_the_serial_rounds_invariants(tmp_path, project_python):
    # a mechanics test: the premise (δ = 0.02 on a bowl with σ = 0.01 ends the round in a stall after
    # some new incumbents) holds on every seed (test_rounds: 8 of 8 serial runs stall); landing order
    # varies with timing, so seed 1 fixes only R0
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.02\nseed: 1\nworkers: 3\n")
    env = toy_env(sigma=0.01, sleep=0.3)
    coded(repo, run_dir, tmp_path, env=env)
    code, out = round_run(repo, "--rationale", "search H1", env=env)
    assert code == 0, out
    trials = started_in(run_dir, 1)
    # up to 3 trials at once, never more; no two sampler trials in flight ask the same point
    fl = flights(run_dir, 1)
    assert max(len(others) for _, others in fl) == 2
    assert sum(len(others) == 2 for _, others in fl) >= 5
    for t, others in fl:
        if t["kind"] == "sampler":
            assert all(o["levers"] != t["levers"] for o in others if o["kind"] == "sampler")
    # the stall is judged with nothing in flight: every trial of the round is recorded before it ends
    assert out["trigger"] == "stall" and ended_before_round_end(run_dir, 1)
    assert sum(t["kind"] == "sampler" for t in trials) >= 20
    assert all(0 <= t["levers"][n] <= 1 for t in trials for n in ("H1.x", "H1.y"))
    assert len({t["seed"] for t in trials}) == len(trials) and len({t["trial"] for t in trials}) == len(trials)
    # the replicate floor, and every new incumbent (better than every trial started before it) confirmed twice
    assert sum("replicate_of" in t for t in trials) >= 0.1 * len(trials)
    seen = list(eligible_now(run_dir, before=trials[0]["trial"]).values())
    new_best = []
    for t in trials:
        if t["kind"] == "sampler" and t["status"] == "finished" and t["objective"] < min(seen):
            new_best.append(t["trial"])
        if t["status"] == "finished":
            seen.append(t["objective"])
    assert new_best
    for root in new_best:
        assert sum(t.get("replicate_of") == root and t["kind"] == "confirmation" for t in trials) == 2
    inc = out["incumbent"]
    assert inc["confirmed"] and inc["replicates"] >= 2
    # the budget ledger is the summed wall-clock of the trials, whatever ran at the same time
    everything = [p["wall_clock_s"] for t in ENDS for p in of_type(run_dir, t)]
    assert out["budget"]["spent_s"] == pytest.approx(sum(everything))


def test_trials_in_flight_are_charged_up_front_so_parallel_trials_never_overdraw_the_budget(tmp_path,
                                                                                          project_python):
    # R0, smoke H1 and its equivalence check: 5 trials of ~2.05 s, so ~5.1 s (2.5 trials) remain for R1's
    # five seeds. Two fit; a third would start while two are in flight, each estimated at ~2.05 s (half a
    # trial of margin either way, so a loaded machine's slower trials don't change the count)
    repo = make_repo(tmp_path)
    seeds = ", ".join(f"{{H1.x: {x}, H1.y: 0.3}}" for x in (0.1, 0.3, 0.5, 0.7, 0.9))
    run_dir = init(repo, project_python, "budget_s: 15.4\nreference_fidelity: {epochs: 4}\ndeterministic: true\n"
                   f"delta: 0.02\nworkers: 3\nseeds: [{seeds}]\n")
    env = toy_env(sleep=2.0)
    coded(repo, run_dir, tmp_path, env=env)
    assert len(of_type(run_dir, "trial_started")) == 5
    code, out = round_run(repo, "--rationale", "search", env=env)
    assert code == 0, out
    assert out["trigger"] == "budget_spent" and out["run_ended"] == "budget_spent"
    trials = started_in(run_dir, 1)
    assert [t["kind"] for t in trials] == ["seed", "seed"]
    assert max(len(others) for _, others in flights(run_dir, 1)) == 1  # the two ran side by side
    # the third waited for the two in flight to land (an estimate is not a charge): only the ledger then
    # refused it, and the round and the run ended on the budget
    evs = events(run_dir)
    [refused] = [e for e in evs if e["type"] == "trial_refused"]
    landed = [e["seq"] for e in evs if e["type"] in ENDS and e["payload"]["trial"] in {t["trial"] for t in trials}]
    assert len(landed) == 2 and max(landed) < refused["seq"]
    p = refused["payload"]
    assert p["kind"] == "seed" and p["estimated_cost_s"] > p["remaining_s"] > 0
    assert p["remaining_s"] == pytest.approx(15.4 - sum(e["payload"]["wall_clock_s"] for e in evs
                                                       if e["type"] in ENDS and e["seq"] < refused["seq"]))
    assert ended_before_round_end(run_dir, 1) and all(t["status"] == "finished" for t in trials)
    assert out["budget"]["spent_s"] <= 15.4


def test_a_search_space_change_ends_a_parallel_round_once_the_trials_in_flight_are_recorded(tmp_path,
                                                                                          project_python):
    # δ out of reach: the bowl is rejectable next to it (Δ's bound is far below δ); its confirmed reject ends the round (a mechanics
    # test: with δ = 10 against a range of ~2 the reject is certain on any seed)
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 10\nseed: 3\nworkers: 3\n")
    env = toy_env(sigma=0.01, sleep=0.2)
    coded(repo, run_dir, tmp_path, env=env)
    code, out = round_run(repo, "--rationale", "search", env=env)
    assert code == 0, out
    assert out["trigger"] == "search_space"
    [first, confirming] = of_type(run_dir, "verdict")
    assert first["outcome"] == "pending-reject" and confirming["outcome"] == "reject"
    # checks fall due as trials land: one landing may bring up to 3 past the burn-in of 20
    assert 20 <= first["burn_in"]["fresh"] <= 22
    assert confirming["burn_in"]["fresh"] >= first["burn_in"]["fresh"] + 10
    assert [p["id"] for p in of_type(run_dir, "hypothesis_rejected")] == ["H1.v1"]
    assert max(len(others) for _, others in flights(run_dir, 1)) == 2
    # the trials in flight at the confirming check finished and belong to the round
    assert ended_before_round_end(run_dir, 1)
    assert all(t["status"] == "finished" for t in started_in(run_dir, 1))


def test_a_killed_parallel_round_resumes_with_every_trial_in_flight_abandoned_and_none_retried(
        tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 10\nseed: 3\nworkers: 3\n")
    coded(repo, run_dir, tmp_path, env=toy_env(sigma=0.01))
    n = len(of_type(run_dir, "trial_started"))
    # from the round's 7th trial on, every trial hangs: three of them end up in flight together
    env = toy_env(sigma=0.01, sleep=120, sleep_from=n + 7)
    expect_all(repo)
    p = subprocess.Popen([sys.executable, "-m", "boautoresearch", "round-run", "--rationale", "go"],
                         cwd=repo, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 90

    def hung():
        return [t for t in started_in(run_dir, 1) if t["trial"] >= n + 7]

    def beating(t):
        return any(h["trial"] == t["trial"] and h["elapsed_s"] >= 1 for h in of_type(run_dir, "trial_heartbeat"))

    while len(hung()) < 3 or not all(beating(t) for t in hung()):
        assert time.monotonic() < deadline and p.poll() is None
        time.sleep(0.1)
    p.kill()
    p.wait()
    stuck = hung()
    runners = {h["runner_pid"] for h in of_type(run_dir, "trial_heartbeat") if h["trial"] in {t["trial"] for t in stuck}}
    assert len(stuck) == 3 and len(runners) == 3
    spent = bo(repo, "status")[1]["budget"]["spent_s"]

    code, out = round_run(repo, "--rationale", "resume", env=toy_env(sigma=0.01))
    assert code == 0, out
    assert all(gone(pid) for pid in runners)  # no orphaned runner goes on uncharged
    abandoned = of_type(run_dir, "trial_abandoned")
    assert sorted(a["trial"] for a in abandoned) == sorted(t["trial"] for t in stuck)
    last_beat = {t["trial"]: max(h["elapsed_s"] for h in of_type(run_dir, "trial_heartbeat")
                                 if h["trial"] == t["trial"]) for t in stuck}
    assert all(a["wall_clock_s"] == last_beat[a["trial"]] for a in abandoned)
    assert [(e["round"], e["trigger"]) for e in of_type(run_dir, "round_ended")][1:3] == [
        (1, "interrupted"), (2, "search_space")]
    # nothing lost or run twice: every trial id once, the abandoned ones never retried
    r2 = started_in(run_dir, 2)
    ids = [t["trial"] for t in of_type(run_dir, "trial_started")]
    assert ids == list(range(1, len(ids) + 1))
    assert not {t["trial"] for t in stuck} & {t["trial"] for t in r2}
    assert ended_before_round_end(run_dir, 2)
    # R2 is charged exactly its trials on top of what R1 left, abandoned trials up to their last heartbeat
    _, status = bo(repo, "status")
    assert status["trials"]["abandoned"] == 3
    assert status["budget"]["spent_s"] == pytest.approx(
        spent + sum(last_beat.values()) + sum(t["wall_clock_s"] for t in r2))
