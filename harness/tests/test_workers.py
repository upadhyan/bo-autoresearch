"""Parallel trials: run.yaml `workers` runs up to that many trials of a round at once."""
import pytest

from conftest import bo, events, round_run
from test_rounds import BASE, coded, eligible_now, init, make_repo, of_type, started_in, toy_env

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
    # R0, smoke H1 and its equivalence check: 5 trials of ~1.04 s, so ~2.6 s remain for R1's five seeds.
    # Two fit; a third would start while two are in flight, each estimated at ~1.04 s
    repo = make_repo(tmp_path)
    seeds = ", ".join(f"{{H1.x: {x}, H1.y: 0.3}}" for x in (0.1, 0.3, 0.5, 0.7, 0.9))
    run_dir = init(repo, project_python, "budget_s: 7.8\nreference_fidelity: {epochs: 4}\ndeterministic: true\n"
                   f"delta: 0.02\nworkers: 3\nseeds: [{seeds}]\n")
    env = toy_env(sleep=1.0)
    coded(repo, run_dir, tmp_path, env=env)
    assert len(of_type(run_dir, "trial_started")) == 5
    code, out = round_run(repo, "--rationale", "search", env=env)
    assert code == 0, out
    assert out["trigger"] == "budget_spent" and out["run_ended"] == "budget_spent"
    trials = started_in(run_dir, 1)
    assert [t["kind"] for t in trials] == ["seed", "seed"]
    assert max(len(others) for _, others in flights(run_dir, 1)) == 1  # the two ran side by side
    [refused] = of_type(run_dir, "trial_refused")
    assert refused["kind"] == "seed" and refused["estimated_cost_s"] > refused["remaining_s"] > 0
    # the two in flight when the third was refused finished and were recorded in the round
    assert ended_before_round_end(run_dir, 1) and all(t["status"] == "finished" for t in trials)
    assert out["budget"]["spent_s"] <= 7.8
