"""Pre-loop start (#35): the R1 gate as one refusal, headless run.yaml, the init draft, and the
probes the start skill and the read-only agents use (show, untested, sensitivity)."""
import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import bo, events, expect_all, git, register, write_run_yaml
from test_rounds import BASE, SPEC, coded, init, make_repo, of_type, toy_env
from test_verdicts import lever, spec, verdict_run

NO_BRIEF = BASE + "brief: null\n"


def started_rounds(run_dir):
    return [p["round"] for p in of_type(run_dir, "round_started")]


def ready_run(d, python, *, brief=True, register_h1=True, code=True, delta=True, expected=True,
              ladder=False):
    """R0 done; H1 proposed, registered, coded; δ set; expected verdicts recorded — each step
    skippable, so each R1 gate condition can fail alone. -> (repo, run_dir)"""
    repo = make_repo(d)
    extra = (BASE if brief else NO_BRIEF) + ("ladder: [{epochs: 1}]\n" if ladder else "")
    run_dir = init(repo, python, extra)
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0
    if code:
        coded(repo, run_dir, d)
    else:
        f = d / "spec.json"
        f.write_text(json.dumps(SPEC))
        assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
        if register_h1:
            assert register(repo, "H1")[0] == 0
    if delta:
        assert bo(repo, "set-delta", "0.1", "--rationale", "the user's minimum effect")[0] == 0
    if expected:
        expect_all(repo)
    return repo, run_dir


def test_r1_runs_once_every_gate_condition_holds(tmp_path, project_python):
    repo, run_dir = ready_run(tmp_path, project_python)
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code == 0, out
    assert 1 in started_rounds(run_dir)


def revise_brief(repo, run_dir):
    """A registry revision: every hypothesis owes a fresh registration review."""
    rev = run_dir.parent.parent.parent / "revision.yaml"
    rev.write_text("brief: {purpose: a new purpose, contribution: a lower loss, complexity: small, provenance: any}\n")
    for args in (["checkpoint"], ["checkpoint", "--revise", str(rev)], ["checkpoint", "--resume"]):
        code, out = bo(repo, *args, "--rationale", "the user revises the brief")
        assert code == 0, out


BREAK = {  # condition(s) failing -> how to break a ready run
    ("worktree",): lambda repo, run_dir: shutil.move(run_dir / "worktree", run_dir / "elsewhere"),
    ("venv",): lambda repo, run_dir: shutil.move(run_dir / "venv", run_dir / "venv-gone"),
    ("clean",): lambda repo, run_dir: (run_dir / "worktree" / "notes.txt").write_text("scratch\n"),
    ("head",): lambda repo, run_dir: git(run_dir / "worktree", "commit", "-q", "--allow-empty", "-m", "raw"),
    ("clean", "protected_paths"): lambda repo, run_dir: (run_dir / "worktree" / "runner.py").write_text("x = 1\n"),
    ("review",): revise_brief,
    ("hypothesis", "interplay"): lambda repo, run_dir: bo(repo, "park", "H1", "--reason", "set aside",
                                                            "--rationale", "the user parks it"),
}


@pytest.mark.parametrize("conditions", sorted(BREAK))
def test_r1_is_refused_naming_each_broken_condition(tmp_path, project_python, conditions):
    repo, run_dir = ready_run(tmp_path, project_python)
    BREAK[conditions](repo, run_dir)
    before = events(run_dir)
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code != 0 and out["refused"] and sorted(out["failing"]) == list(conditions), out
    assert events(run_dir) == before and 1 not in started_rounds(run_dir)


def test_r1_waits_for_a_due_generation_pass(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python, BASE + "delta: 0.1\n")
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code != 0 and sorted(out["failing"]) == ["generation", "hypothesis"], out
    assert "run start" in out["reason"]


@pytest.mark.parametrize("condition, skip, words", [
    ("registry", {"brief": False}, "brief"),
    ("hypothesis", {"code": False, "register_h1": False, "expected": False}, "register"),
    ("code", {"code": False}, "commit-lever"),
    ("expected", {"expected": False}, "record expected"),
    ("delta", {"delta": False}, "set-delta"),
])
def test_r1_is_refused_naming_the_one_condition_that_fails(tmp_path, project_python, condition, skip, words):
    repo, run_dir = ready_run(tmp_path, project_python, **skip)
    before = events(run_dir)
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code != 0 and out["failing"] == [condition] and words in out["reason"], out
    assert events(run_dir) == before and 1 not in started_rounds(run_dir)


def test_one_refusal_names_every_failing_condition(tmp_path, project_python):
    repo, run_dir = ready_run(tmp_path, project_python, brief=False, code=False, delta=False,
                              expected=False)
    (run_dir / "worktree" / "notes.txt").write_text("scratch\n")
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code != 0 and set(out["failing"]) == {"clean", "registry", "code", "expected", "delta"}, out
    for words in ("uncommitted", "brief", "commit-lever", "record expected", "set-delta"):
        assert words in out["reason"]


def test_ladder_calibration_that_cannot_run_yet_is_a_failing_condition(tmp_path, project_python):
    repo, run_dir = ready_run(tmp_path, project_python, ladder=True, code=False, delta=False)
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code != 0 and set(out["failing"]) == {"code", "calibration", "delta"}, out
    assert not of_type(run_dir, "fidelity_calibration")


def test_a_failed_smoke_keeps_the_run_in_the_calibration_round(repo, init_run):
    run_dir = init_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\n")
    env = {**os.environ, "TOY_RAISE": "1"}
    for _ in range(2):
        code, out = bo(repo, "round-run", "--rationale", "calibrate", env=env)
        assert code != 0 and "smoke" in out["reason"]
    assert started_rounds(run_dir) == [0, 0]
    assert bo(repo, "status")[1]["r0_complete"] is False


def headless(tmp_path, project_python, extra):
    repo = make_repo(tmp_path)
    y = write_run_yaml(tmp_path / "run.yaml", project_python, BASE + extra)
    return repo, bo(repo, "init", str(y), "--rationale", "headless start")


@pytest.mark.parametrize("extra, words", [
    ("go: true\nlenses: [data, optimisation]\n", "delta"),
    ("go: true\ndelta: 0.1\n", "lenses"),
])
def test_headless_init_is_refused_unless_run_yaml_is_complete(tmp_path, project_python, extra, words):
    _, (code, out) = headless(tmp_path, project_python, extra)
    assert code != 0 and "go: true" in out["reason"] and words in out["reason"], out


def test_a_complete_run_yaml_starts_headless(tmp_path, project_python):
    (tmp_path / "fixtures").mkdir()
    repo, (code, out) = headless(tmp_path, project_python,
                                 "go: true\ndelta: 0.1\ngeneration: scripted\nfixtures: fixtures\n")
    assert code == 0, out
    code, out = headless(tmp_path / "b", project_python,
                         "go: true\ndelta: 0.1\nlenses: [data, optimisation]\nproposals_per_lens: 3\n")[1]
    assert code == 0, out
    assert out["generation"]["lenses"] == ["data", "optimisation"] and out["generation"]["proposals_per_lens"] == 3


@pytest.mark.parametrize("extra", ["lenses: []\n", "lenses: [data, data]\n", "lenses: [3]\n",
                                   "proposals_per_lens: 1\n", "workers: 0\n", "workers: 2\n"])
def test_bad_generation_and_worker_settings_are_refused(tmp_path, project_python, extra):
    _, (code, out) = headless(tmp_path, project_python, extra)
    assert code != 0 and out["refused"], out


def test_init_ignores_its_own_uncommitted_draft_but_not_other_changes(tmp_path, project_python):
    repo = make_repo(tmp_path)
    draft = write_run_yaml(repo / "run.yaml", project_python, BASE)  # untracked, in the checkout
    code, out = bo(repo, "init", "run.yaml", "--rationale", "from the draft")
    assert code == 0, out
    repo2 = make_repo(tmp_path / "b")
    shutil.copy(draft, repo2 / "run.yaml")
    (repo2 / "notes.txt").write_text("uncommitted\n")
    code, out = bo(repo2, "init", "run.yaml", "--rationale", "dirty")
    assert code != 0 and "uncommitted" in out["reason"]


def test_show_and_untested_probes(tmp_path, project_python):
    repo, run_dir = ready_run(tmp_path, project_python, code=False, delta=False, expected=False)
    f = tmp_path / "spec2.json"
    f.write_text(json.dumps({**SPEC, "title": "Only proposed"}))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "another")[0] == 0
    before = events(run_dir)
    code, out = bo(repo, "show", "H1")
    assert code == 0 and out["id"] == "H1.v1" and out["status"] == "registered", out
    assert out["spec"]["title"] == SPEC["title"] and set(out["spec"]["levers"]) == {"H1.x", "H1.y"}
    assert out["verdicts"] == [] and out["review"]["directive_verdict"] == "allow"
    code, out = bo(repo, "untested")
    assert code == 0 and [(h["id"], h["status"]) for h in out["untested"]] == [
        ("H1.v1", "registered"), ("H2.v1", "proposed")], out
    assert bo(repo, "show", "H9")[0] != 0
    assert events(run_dir) == before  # probes are never logged


def test_sensitivity_reports_the_verdict_gp_telemetry(tmp_path, project_python):
    run_dir, _ = verdict_run(tmp_path, project_python, 1, [(spec({"x": lever()}), "    term = 0 * lever('H1.x')")])
    repo = tmp_path / "project"
    _, v = bo(repo, "verdict", "H1")
    code, out = bo(repo, "sensitivity", "H1")
    assert code == 0 and out["hypothesis"] == "H1.v1" and out["telemetry"] is True, out
    assert [r["id"] for r in out["records"]] == [r["id"] for r in v["records"]] != []
    last, rec = out["records"][-1], v["records"][-1]
    assert last["sqrt_vt"] == rec["sqrt_vt"] and last["levers"] == rec["levers"]
    assert last["sobol_index"] == rec["sobol_index"]


@pytest.mark.parametrize("probe", ["show H1", "untested", "sensitivity H1"])
def test_read_only_agents_may_run_the_new_probes(tmp_path, project_python, probe):
    repo, _ = ready_run(tmp_path, project_python, code=False, delta=False, expected=False)
    code, out = bo(repo, "check", "bash", "--agent", "boautoresearch:round-analyst", "--",
                   f"boautoresearch {probe}")
    assert code == 0 and out["allow"] is True, out


def test_delta_guide_and_an_infeasible_delta_needs_the_users_explicit_ok(tmp_path, project_python):
    repo = make_repo(tmp_path)
    run_dir = init(repo, project_python)
    env = toy_env(sigma=0.05, sleep_per_epoch=0.05)  # 4 epochs: each trial takes ~0.2 s
    assert bo(repo, "round-run", "--rationale", "calibrate", env=env)[0] == 0
    _, st = bo(repo, "status")
    guide = st["delta_guide"]
    # at δ = 2σ the noise needs ~4 trials, so burn-in (20 for one lever) decides: ~20 × 0.2 s
    assert guide["delta"] == st["suggested_delta"] and guide["trials"] == 20 and guide["feasible"] is True
    assert 4.0 <= guide["cost_s"] < 8.0
    tiny = st["sigma"] / 1000  # ~15.7 million trials: far past a 3600 s budget
    code, out = bo(repo, "set-delta", str(tiny), "--rationale", "too small")
    assert code != 0 and "budget" in out["reason"] and "--infeasible-ok" in out["reason"], out
    assert out["guide"]["trials"] > 15_000_000 and out["guide"]["feasible"] is False
    assert not of_type(run_dir, "delta_set")
    code, out = bo(repo, "set-delta", str(tiny), "--infeasible-ok", "--rationale", "the user insists")
    assert code == 0 and out["delta"] == tiny
    assert of_type(run_dir, "delta_set")[0]["guide"]["feasible"] is False
