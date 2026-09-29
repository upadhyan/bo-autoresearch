"""Removals, interplay reviews, revivals, generation passes and exhaustion (#30).

The mechanics run on parks (a removal with no statistics in it) with δ = 10, so no lever can matter;
the planted interactions at the end are statistical and asserted as rates over fixed seeds.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import yaml

from conftest import bo, events, round_run, write_run_yaml
from test_rounds import BASE, init, make_repo, of_type, toy_env
from test_scheduler import in_background, pause_after, round_trials
from test_verdicts import code_hypothesis, lever, spec

ENV = toy_env(sigma=0.01)


def flat(n, name="a"):
    return spec({name: lever()}), f'    term += 0 * lever("H{n}.{name}")'


def propose(repo, d, s, register=False):
    f = d / f"spec-{len(list(d.glob('spec-*')))}.json"
    f.write_text(json.dumps(s))
    code, out = bo(repo, "propose", "--file", str(f), "--rationale", "a new mechanism")
    assert code == 0, out
    if register:
        assert bo(repo, "register", out["hypothesis"]["id"], "--rationale", "reviewed")[0] == 0
    return out["hypothesis"]["id"]


def record(repo, d, kind, body, agent="ip"):
    f = d / f"{kind}-{len(list(d.glob(f'{kind}-*')))}.json"
    f.write_text(json.dumps(body))
    return bo(repo, "record", kind, "--file", str(f), "--agent-id", agent, "--actor", "interplay-reviewer",
              "--rationale", "the review")


def interplay(repo, d, flags=(), **who):
    """An interplay review of a removal (removed=...) or a newcomer (newcomer=...)."""
    code, out = record(repo, d, "interplay", {**who, "flags": list(flags)})
    assert code == 0, out
    return out


def flag(partner, cites, reason="their levers act on the same loss term"):
    return {"partner": partner, "reason": reason, "cites": cites}


def until_verdict(repo, run_dir):
    """A round run until its first verdict check (burn-in), then paused; -> the round's output."""
    p = in_background(repo, run_dir, ENV, lambda d: of_type(d, "verdict"))
    bo(repo, "checkpoint", "--rationale", "a look")  # refused when the check already ended the round
    out = json.loads(p.communicate(timeout=120)[0])
    bo(repo, "checkpoint", "--resume", "--rationale", "go on")
    return out


def setup(tmp_path, python, extra=""):
    repo = make_repo(tmp_path)
    run_dir = init(repo, python, BASE + "delta: 10\n" + extra)
    assert bo(repo, "round-run", "--rationale", "calibrate", env=ENV)[0] == 0
    return repo, run_dir


def test_every_removal_is_logged_with_its_evidence_and_round_run_waits_for_its_review(tmp_path,
                                                                                    project_python):
    repo, run_dir = setup(tmp_path, project_python)
    code_hypothesis(repo, run_dir, tmp_path, 1, *flat(1), ENV)
    untested = propose(repo, tmp_path, flat(2)[0])
    until_verdict(repo, run_dir)
    [first] = of_type(run_dir, "verdict")
    if bo(repo, "status")[1]["hypotheses"][0]["status"] == "active":  # else the check removed it
        assert bo(repo, "park", "H1", "--reason", "the user wants it out", "--rationale", "park")[0] == 0
    [removal] = of_type(run_dir, "removal")
    assert removal["id"] == "H1.v1" and removal["removal"] in ("parked", "inconclusive")
    assert first["id"] in removal["verdicts"] and removal["verdict"] == removal["verdicts"][-1]
    assert removal["context"] == {"retained": [], "co_active": []}
    assert removal["untested"] == [untested]

    _, status = bo(repo, "status")
    assert status["interplay_missing"] == [{"removed": "H1.v1"}]
    assert "record interplay: review H1.v1's removal against the untested list" in status["next"]
    register = bo(repo, "register", untested, "--rationale", "reviewed")
    assert register[0] == 0
    code, out = bo(repo, "round-run", "--rationale", "go", env=ENV)
    assert code != 0 and "interplay" in out["reason"] and "H1.v1" in out["reason"]

    interplay(repo, tmp_path, removed="H1.v1")
    _, status = bo(repo, "status")
    assert status["interplay_missing"] == []  # H2 was untested at the removal: its review covered it
    assert not [d for d in status["next"] if "interplay" in d]

    # a newcomer proposed after the removal is weighed against it at registration
    newcomer = propose(repo, tmp_path, flat(3)[0], register=True)
    _, status = bo(repo, "status")
    assert status["interplay_missing"] == [{"newcomer": newcomer}]
    assert f"record interplay: review newcomer {newcomer} against past removals" in status["next"]
    code, out = bo(repo, "round-run", "--rationale", "go", env=ENV)
    assert code != 0 and newcomer in out["reason"]
    interplay(repo, tmp_path, newcomer=newcomer)
    assert bo(repo, "status")[1]["interplay_missing"] == []


def write_code(repo, run_dir, hid, line):
    """Lever code for an already registered hypothesis: written, smoked and committed."""
    train = run_dir / "worktree" / "train.py"
    src = train.read_text().replace("import fidelity, seed", "import fidelity, lever, seed")
    train.write_text(src.replace("    term = 0.0\n", f"    term = 0.0\n{line}\n"))
    code_, out = bo(repo, "smoke", hid, "--rationale", "try", env=ENV)
    assert code_ == 0 and out["passed"], out
    assert bo(repo, "commit-lever", hid, "--rationale", "smoke passed", env=ENV)[0] == 0


def test_a_flagged_removal_is_revived_into_its_partners_round_once_per_pair(tmp_path, project_python):
    repo, run_dir = setup(tmp_path, project_python)
    code_hypothesis(repo, run_dir, tmp_path, 1, *flat(1), ENV)
    h2, h3 = propose(repo, tmp_path, flat(2)[0]), propose(repo, tmp_path, flat(3)[0])
    until_verdict(repo, run_dir)
    if bo(repo, "status")[1]["hypotheses"][0]["status"] == "active":
        assert bo(repo, "park", "H1", "--reason", "the user wants it out", "--rationale", "park")[0] == 0
    cite = [of_type(run_dir, "verdict")[-1]["id"]]
    interplay(repo, tmp_path, [flag(h2, cite)], removed="H1.v1")
    assert bo(repo, "register", h2, "--rationale", "reviewed")[0] == 0
    write_code(repo, run_dir, h2, '    term += 0 * lever("H2.a")')
    assert bo(repo, "generate", "--rationale", "the queue is low")[0] == 0

    # the partner is selected: H1 comes back as a new linked version, before the round runs
    [h1] = [h for h in of_type(run_dir, "hypothesis_proposed") if h["id"] == "H1.v1"]
    [committed] = [c["commit"] for c in of_type(run_dir, "lever_committed") if c["id"] == "H1.v1"]
    code_, out = bo(repo, "round-run", "--rationale", "R2", env=ENV)
    assert code_ == 0, out
    assert out["revived"] == [{"id": "H1.v2", "number": 1, "version": 2, "from": "H1.v1", "partner": h2,
                               "spec": h1["spec"], "commit": committed, "round": 2}]
    assert of_type(run_dir, "hypothesis_revived") == out["revived"]
    assert 2 not in [s["round"] for s in of_type(run_dir, "round_started")]
    assert out["schedule"]["selected"] == [h2, "H1.v2"]  # placed with its partner
    assert not [d for d in out["next"] if "lever code" in d]  # its code is still in place
    assert out["exhaustion"]["revivals_run"] is False  # it hasn't run yet
    if of_type(run_dir, "hypothesis_parked"):
        code_, refused = bo(repo, "unpark", "H1.v1", "--reason", "back", "--rationale", "return")
        assert code_ != 0 and "H1.v2" in refused["reason"]

    # H1.v2 removed before it ran and flagged with the same partner again: that pair is spent, but a
    # new partner still revives it
    assert bo(repo, "park", "H1.v2", "--reason", "not now", "--rationale", "park")[0] == 0
    interplay(repo, tmp_path, [flag(h2, cite), flag(h3, cite)], removed="H1.v2")
    pause_after(repo, run_dir, 2, 3)
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert r2["hypotheses"] == [h2] and len(of_type(run_dir, "hypothesis_revived")) == 1

    assert bo(repo, "register", h3, "--rationale", "reviewed")[0] == 0
    write_code(repo, run_dir, h3, '    term += 0 * lever("H3.a")')
    assert bo(repo, "generate", "--rationale", "the queue is low")[0] == 0
    code_, out = bo(repo, "round-run", "--rationale", "R3", env=ENV)
    assert code_ == 0 and [(r["id"], r["from"], r["partner"]) for r in out["revived"]] == [
        ("H1.v3", "H1.v2", h3)]
    pause_after(repo, run_dir, 3, 3)
    [r3] = [s for s in of_type(run_dir, "round_started") if s["round"] == 3]
    assert r3["hypotheses"] == [h2, h3, "H1.v3"]
    assert of_type(run_dir, "hypothesis_activated")[-2:] == [{"id": h3, "round": 3}, {"id": "H1.v3", "round": 3}]


def scripted(d, *passes):
    """run.yaml lines serving `passes` (lists of specs) as fixtures/pass1.yaml, pass2.yaml, ..."""
    fx = d / "fixtures"
    fx.mkdir()
    for i, specs in enumerate(passes, 1):
        (fx / f"pass{i}.yaml").write_text(yaml.safe_dump(specs))
    return "generation: scripted\nfixtures: fixtures\n"


def titled(title, name="a"):
    return {**spec({name: lever()}), "title": title}


def test_generation_is_due_at_run_start_when_the_queue_runs_low_and_on_an_analyst_flag(tmp_path,
                                                                                      project_python):
    first, second = [titled("first", "a"), titled("second", "b")], [titled("third", "c")]
    repo, run_dir = setup(tmp_path, project_python, scripted(tmp_path, first, second))
    _, status = bo(repo, "status")
    assert status["generation"] == {"mode": "scripted", "passes": 0, "due": ["run start"]}
    assert "generate (run start)" in status["next"]
    code_, out = bo(repo, "round-run", "--rationale", "go", env=ENV)
    assert code_ != 0 and "generate" in out["reason"] and "run start" in out["reason"]

    # the fixture passes, in order, then empty passes
    served = [bo(repo, "generate", "--rationale", "a pass")[1] for _ in range(3)]
    assert [(o["pass"], o["proposed"]) for o in served] == [(1, ["H1.v1", "H2.v1"]), (2, ["H3.v1"]), (3, [])]
    proposed = [e for e in events(run_dir) if e["type"] == "hypothesis_proposed"]
    assert [e["payload"]["spec"]["title"] for e in proposed] == ["first", "second", "third"]
    assert {e["actor"] for e in proposed} == {"harness"}
    assert [(p["pass"], p["mode"], p["triggers"]) for p in of_type(run_dir, "generation_pass")] == [
        (1, "scripted", ["run start"]), (2, "scripted", []), (3, "scripted", [])]
    assert served[-1]["generation"]["due"] == []

    assert bo(repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    write_code(repo, run_dir, "H1.v1", '    term += 0 * lever("H1.a")')
    pause_after(repo, run_dir, 1, 3)
    [due] = bo(repo, "status")[1]["generation"]["due"]
    assert due == "the queue holds 0, below 2x the 1 slot(s) per round"
    assert bo(repo, "generate", "--rationale", "the queue is low")[0] == 0
    assert bo(repo, "status")[1]["generation"]["due"] == []
    narrative = {"round": 1, "text": "Nothing decided yet.", "cites": [], "diagnostics": [],
                 "suggestions": ["look at the data pipeline"], "generation": True}
    assert record(repo, tmp_path, "narrative", narrative, agent="an")[0] == 0
    assert bo(repo, "status")[1]["generation"]["due"] == ["the round analyst flagged new ground"]


def test_generation_run_yaml_and_fixture_refusals(tmp_path, project_python, repo):
    def refused(extra):
        y = write_run_yaml(tmp_path / "run.yaml", project_python, BASE + extra)
        code_, out = bo(repo, "init", str(y), "--rationale", "go")
        return code_ != 0 and out["reason"]
    assert "generation" in refused("generation: freestyle\n")
    assert "fixtures" in refused("generation: scripted\n")
    assert "fixtures" in refused("generation: scripted\nfixtures: nowhere\n")
    assert "fixtures" in refused("fixtures: fixtures\n")
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "pass1.yaml").write_text(yaml.safe_dump([{"title": "no levers"}]))
    y = write_run_yaml(tmp_path / "run.yaml", project_python, BASE + "generation: scripted\nfixtures: fixtures\n")
    assert bo(repo, "init", str(y), "--rationale", "go")[0] == 0
    code_, out = bo(repo, "generate", "--rationale", "a pass")
    assert code_ != 0 and "pass1.yaml" in out["reason"]


def test_exhaustion_waits_for_every_condition_then_ends_the_run(tmp_path, project_python):
    repo, run_dir = setup(tmp_path, project_python, scripted(tmp_path, [titled("x")], [titled("y")]))

    def exhaustion():
        _, status = bo(repo, "status")
        assert status["run_ended"] is None
        return status["exhaustion"]
    assert exhaustion()["final_pass_empty"] is False  # no pass yet
    bo(repo, "generate", "--rationale", "run start")
    assert exhaustion()["final_pass_empty"] is False  # H1 is still undecided
    assert bo(repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    assert exhaustion()["queue_empty"] is False
    write_code(repo, run_dir, "H1.v1", '    term += 0 * lever("H1.a")')
    pause_after(repo, run_dir, 1, 3)
    assert exhaustion()["none_undecided"] is False  # H1 is active
    assert bo(repo, "park", "H1", "--reason", "the user wants it out", "--rationale", "park")[0] == 0
    assert exhaustion() == {"queue_empty": True, "none_undecided": True, "removals_reviewed": False,
                            "revivals_run": True, "final_pass_empty": False}  # and a round ran since
    interplay(repo, tmp_path, removed="H1.v1")
    assert exhaustion()["final_pass_empty"] is False
    bo(repo, "generate", "--rationale", "the queue is empty")
    assert exhaustion()["final_pass_empty"] is False  # H2 is undecided
    assert bo(repo, "park", "H2", "--reason", "failed its review", "--rationale", "park")[0] == 0
    assert exhaustion() == {"queue_empty": True, "none_undecided": True, "removals_reviewed": False,
                            "revivals_run": True, "final_pass_empty": True}
    interplay(repo, tmp_path, removed="H2.v1")
    assert all(exhaustion().values())  # judged at the next boundary
    assert bo(repo, "status")[1]["next"] == ["round-run (the hypothesis list is exhausted: it ends the run)"]
    code_, out = bo(repo, "round-run", "--rationale", "the next round", env=ENV)
    assert code_ == 0 and out["run_ended"] == "exhausted" and out["next"] == []
    assert of_type(run_dir, "run_ended") == [{"reason": "exhausted", "exhaustion": {
        "queue_empty": True, "none_undecided": True, "removals_reviewed": True, "revivals_run": True,
        "final_pass_empty": True}}]
    code_, out = bo(repo, "round-run", "--rationale", "more", env=ENV)
    assert code_ != 0 and "ended" in out["reason"]


# The planted interactions: A (C) does nothing alone and helps only with B (D). δ = 0.1, σ = 0.05;
# with both levers free the pair's loss term is −8a(b − 0.5), √V_T of each ≈ 1.3 ≫ δ. B's baseline sits
# mid-range: its warm start then covers neither end, so its fresh trials reach both (the coverage gate).
DELTA, SIGMA, SEEDS = 0.1, 0.05, (1, 2, 3)
ALONE = '    boost = 0.0\n    term += -4 * lever("H1.a") * boost'
PARTNER = ("    boost = 0.0\n", '    boost = 2 * (lever("H2.b") - 0.5)\n')


def toy(d, python, seed, passes):
    d.mkdir()
    repo = make_repo(d)
    run_dir = init(repo, python, BASE + f"delta: {DELTA}\nseed: {seed}\n" + scripted(d, *passes))
    env = toy_env(sigma=SIGMA)
    assert bo(repo, "round-run", "--rationale", "calibrate", env=env)[0] == 0
    return repo, run_dir, env


def partner_code(repo, run_dir, hid, env):
    train = run_dir / "worktree" / "train.py"
    train.write_text(train.read_text().replace(*PARTNER))
    code_, out = bo(repo, "smoke", hid, "--rationale", "try", env=env)
    assert code_ == 0 and out["passed"], out
    assert bo(repo, "commit-lever", hid, "--rationale", "smoke passed", env=env)[0] == 0


def statuses(repo):
    return {h["id"]: h["status"] for h in bo(repo, "status")[1]["hypotheses"]}


def revived_and_retained(repo, run_dir, env, partner):
    """H1 rejected and reviewed: the partner's round revives it, and the round tests them together.
    -> whether both ended retained."""
    code_, out = bo(repo, "round-run", "--rationale", "R2", env=env)
    assert code_ == 0 and [(r["id"], r["from"], r["partner"]) for r in out["revived"]] == [
        ("H1.v2", "H1.v1", partner)], out
    code_, out = round_run(repo, "--rationale", "R2", env=env)
    assert code_ == 0, out
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert r2["hypotheses"] == [partner, "H1.v2"]
    if out["trigger"] == "stall":
        assert out["run_ended"] is None  # a stall never ends the run
    both = [statuses(repo)[h] for h in (partner, "H1.v2")] == ["retained", "retained"]
    # the final pass is empty: the list is exhausted exactly when nothing is left undecided
    code_, out = bo(repo, "generate", "--rationale", "the queue is empty")
    assert code_ == 0 and all(out["exhaustion"].values()) == both, out["exhaustion"]
    if both:
        assert bo(repo, "round-run", "--rationale", "R3", env=env)[1]["run_ended"] == "exhausted"
    return both


def registration_toy(d, python, seed):
    a, b = spec({"a": lever()}), spec({"b": lever(0.5)})
    repo, run_dir, env = toy(d, python, seed, [[a], [b]])
    assert bo(repo, "generate", "--rationale", "run start")[1]["proposed"] == ["H1.v1"]
    assert bo(repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    write_code(repo, run_dir, "H1.v1", ALONE)
    assert round_run(repo, "--rationale", "R1", env=env)[0] == 0
    if statuses(repo)["H1.v1"] != "rejected":
        return False
    [removal] = of_type(run_dir, "removal")
    assert removal["removal"] == "rejected" and removal["untested"] == []
    interplay(repo, d, removed="H1.v1")
    assert bo(repo, "generate", "--rationale", "the queue is empty")[1]["proposed"] == ["H2.v1"]
    assert bo(repo, "register", "H2", "--rationale", "reviewed")[0] == 0
    assert bo(repo, "status")[1]["interplay_missing"] == [{"newcomer": "H2.v1"}]
    interplay(repo, d, [flag("H1.v1", [removal["verdict"]])], newcomer="H2.v1")
    partner_code(repo, run_dir, "H2.v1", env)
    return revived_and_retained(repo, run_dir, env, "H2.v1")


def review_toy(d, python, seed):
    c, dd = spec({"a": lever()}), spec({"b": lever(0.5)})
    repo, run_dir, env = toy(d, python, seed, [[c, dd]])
    assert bo(repo, "generate", "--rationale", "run start")[1]["proposed"] == ["H1.v1", "H2.v1"]
    assert bo(repo, "register", "H1", "--rationale", "reviewed")[0] == 0
    write_code(repo, run_dir, "H1.v1", ALONE)
    p = in_background(repo, run_dir, env, lambda r: round_trials(r, 1))
    assert bo(repo, "register", "H2", "--rationale", "reviewed")[0] == 0  # queued while C is tested
    assert p.communicate(timeout=600)[0]
    [r1] = [s for s in of_type(run_dir, "round_started") if s["round"] == 1]
    assert r1["hypotheses"] == ["H1.v1"]
    if statuses(repo)["H1.v1"] != "rejected":
        return False
    [removal] = of_type(run_dir, "removal")
    assert removal["untested"] == ["H2.v1"]
    assert bo(repo, "status")[1]["interplay_missing"] == [{"removed": "H1.v1"}]
    interplay(repo, d, [flag("H2.v1", [removal["verdict"]])], removed="H1.v1")
    partner_code(repo, run_dir, "H2.v1", env)
    assert bo(repo, "generate", "--rationale", "the queue is low")[0] == 0
    return revived_and_retained(repo, run_dir, env, "H2.v1")


def rate(toy_fn, tmp_path, python):
    with ThreadPoolExecutor(len(SEEDS)) as pool:
        return list(pool.map(lambda s: toy_fn(tmp_path / f"s{s}", python, s), SEEDS))


def test_a_rejected_hypothesis_flagged_at_a_newcomers_registration_is_revived_and_both_retained(
        tmp_path, project_python):
    outcomes = rate(registration_toy, tmp_path, project_python)
    assert sum(outcomes) >= 2, outcomes


def test_a_removal_flagged_by_its_interplay_review_is_revived_into_the_queued_partners_round(
        tmp_path, project_python):
    outcomes = rate(review_toy, tmp_path, project_python)
    assert sum(outcomes) >= 2, outcomes
