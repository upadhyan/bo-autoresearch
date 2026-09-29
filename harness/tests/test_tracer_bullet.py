import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import bo, events, git

HARNESS = Path(__file__).resolve().parents[1]


def test_init_creates_worktree_venv_and_run_folder_and_logs_run_started(repo, run_dir):
    run_id = run_dir.name
    assert run_dir == repo / ".bo-research" / run_id
    worktree = run_dir / "worktree"
    assert git(worktree, "branch", "--show-current") == f"bo-research/{run_id}"
    assert git(worktree, "rev-parse", "HEAD") == git(repo, "rev-parse", "HEAD")
    venv_python = run_dir / "venv" / "bin" / "python"
    subprocess.run([venv_python, "-c", "import boautoresearch"], check=True)

    [started, registry] = events(run_dir)
    assert started["type"] == "run_started"
    # the directive registry (empty here) and the protected paths' hashes, written by the harness
    assert registry["type"] == "registry_revised" and registry["actor"] == "harness"
    assert registry["payload"]["version"] == 1 and registry["payload"]["directives"] == []
    assert sorted(registry["payload"]["manifest"]) == ["levers.json", "runner.py"]
    assert started["actor"] == "orchestrator"
    p = started["payload"]
    assert p["rationale"] == "start the toy run"
    assert (p["objective"], p["direction"], p["budget_s"]) == ("loss", "minimize", 3600)
    assert p["base_commit"] == git(repo, "rev-parse", "HEAD")
    assert p["venv_freeze"] == []  # the bare project venv has nothing installed
    assert (run_dir / "run.yaml").exists()
    assert git(repo, "status", "--porcelain") == ""  # the user's checkout is untouched


def test_init_refuses_a_dirty_user_tree(repo, run_yaml):
    (repo / "train.py").write_text("# uncommitted edit\n")
    code, out = bo(repo, "init", str(run_yaml), "--rationale", "go")
    assert code != 0
    assert out["refused"] is True
    assert "uncommitted" in out["reason"]
    assert not list((repo / ".bo-research").glob("*/log.db"))
    assert "bo-research/" not in git(repo, "branch", "--list")
    assert ".bo-research" not in (repo / ".git" / "info" / "exclude").read_text()


def test_init_refuses_an_invalid_run_yaml(repo, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("objective: loss\ndirection: sideways\nbudget_s: 10\n")
    code, out = bo(repo, "init", str(bad), "--rationale", "go")
    assert code != 0 and out["refused"] and "direction" in out["reason"]


def test_events_are_append_only(run_dir):
    con = sqlite3.connect(run_dir / "log.db")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE events SET actor = 'x'")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM events")
    con.close()
    assert len(events(run_dir)) == 2


def test_status_is_rebuilt_from_events_alone(repo, run_dir, tmp_path):
    assert bo(repo, "smoke", "--rationale", "check the runner")[0] == 0
    _, status = bo(repo, "status")

    # A bare repo holding nothing but a copy of log.db gives the same status.
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    (other / ".bo-research" / run_dir.name).mkdir(parents=True)
    shutil.copy(run_dir / "log.db", other / ".bo-research" / run_dir.name / "log.db")
    assert bo(other, "status") == (0, status)


def test_baseline_trial_runs_as_a_subprocess_in_the_run_venv(repo, run_dir):
    code, out = bo(repo, "smoke", "--rationale", "check the runner")
    assert code == 0, out

    evs = events(run_dir)
    # the heartbeat at spawn records the runner's pid, for recovery to kill if the harness dies
    assert [e["type"] for e in evs] == ["run_started", "registry_revised", "trial_started", "trial_heartbeat",
                                        "trial_finished"]
    started, finished = evs[2]["payload"], evs[4]["payload"]
    assert evs[2]["actor"] == "orchestrator" and started["rationale"] == "check the runner"
    assert evs[4]["actor"] == "harness"
    assert started["levers"] == {"H1.scale": 2.0}  # the committed baselines
    assert started["fidelity"] == {"epochs": 4}
    assert started["commit"] == git(run_dir / "worktree", "rev-parse", "HEAD")
    assert isinstance(started["seed"], int)

    assert finished["objective"] == pytest.approx(2.0 * (1 + 1 / 4))
    assert finished["constraints"] == {"runtime_s": 0.0}
    assert finished["telemetry"]["python"] == str(run_dir / "venv" / "bin" / "python")
    assert finished["telemetry"]["seed"] == started["seed"]
    assert finished["exit_code"] == 0
    assert finished["wall_clock_s"] > 0
    assert finished["peak_mem_mb"] > 0
    artifacts = run_dir / finished["artifact_dir"]
    assert (artifacts / "note.txt").exists()
    assert len(json.loads((artifacts / "curves.json").read_text())["train_loss"]) == 4

    assert out["trial"]["status"] == "finished"


def test_a_raising_runner_gives_trial_failed_with_no_objective(repo, run_dir):
    code, out = bo(repo, "smoke", "--rationale", "check", env={**os.environ, "TOY_RAISE": "1"})
    assert code == 0, out
    failed = events(run_dir)[-1]
    assert failed["type"] == "trial_failed"
    assert "objective" not in failed["payload"]
    assert failed["payload"]["exit_code"] != 0
    assert "toy trainer diverged" in failed["payload"]["error"]
    _, trials = bo(repo, "trials")
    assert trials["trials"][0]["status"] == "failed"


def test_runner_outside_the_harness_uses_levers_json_and_prints_the_result(repo):
    env = {**os.environ, "PYTHONPATH": str(HARNESS)}
    env.pop("BOAUTORESEARCH_TRIAL", None)
    p = subprocess.run([sys.executable, "runner.py"], cwd=repo, env=env,
                       capture_output=True, text=True, check=True)
    result = json.loads(p.stdout)
    assert result["objective"] == pytest.approx(2.0 * (1 + 1 / 10))
    assert git(repo, "status", "--porcelain") == ""  # no artifacts left in the checkout


def test_actions_require_a_rationale_and_log_their_actor(repo, run_dir):
    for args in (["smoke"], ["smoke", "--rationale", "  "]):
        code, out = bo(repo, *args)
        assert code != 0 and out["refused"] and "rationale" in out["reason"]
    code, out = bo(repo, "smoke", "--rationale", "why", "--actor", "harness")
    assert code != 0 and out["refused"] and "reserved" in out["reason"]
    assert len(events(run_dir)) == 2

    assert bo(repo, "smoke", "--rationale", "why", "--actor", "user")[0] == 0
    assert events(run_dir)[2]["actor"] == "user"


def test_probes_are_not_logged(repo, run_dir):
    bo(repo, "smoke", "--rationale", "why")
    before = events(run_dir)
    for probe in ("status", "trials"):
        assert bo(repo, probe)[0] == 0
    assert events(run_dir) == before


def test_status_and_trials_probes_and_trials_csv(repo, run_dir):
    bo(repo, "smoke", "--rationale", "why")
    bo(repo, "smoke", "--rationale", "why", env={**os.environ, "TOY_RAISE": "1"})

    _, status = bo(repo, "status")
    assert status["run_id"] == run_dir.name
    assert status["trials"] == {"total": 2, "running": 0, "finished": 1, "failed": 1, "abandoned": 0}
    assert 0 < status["budget"]["spent_s"] < 3600
    assert status["budget"]["remaining_s"] == pytest.approx(3600 - status["budget"]["spent_s"])

    _, trials = bo(repo, "trials")
    assert [t["trial"] for t in trials["trials"]] == [1, 2]

    rows = list(csv.DictReader((run_dir / "exports" / "trials.csv").open()))
    assert [r["status"] for r in rows] == ["finished", "failed"]
    assert [r["chosen_by"] for r in rows] == ["smoke", "smoke"]
    assert float(rows[0]["objective"]) == pytest.approx(2.5)
    assert rows[1]["objective"] == ""
    assert rows[0]["L:H1.scale"] == "2.0"
    assert rows[0]["c:runtime_s"] == "0.0"
    assert rows[0]["t:seed"] == str(trials["trials"][0]["seed"])


def test_refusals_are_json_with_a_reason(tmp_path, repo):
    git_less = tmp_path / "nogit"
    git_less.mkdir()
    for cwd, args in ((repo, ["status"]), (git_less, ["status"]), (repo, ["frobnicate"])):
        code, out = bo(cwd, *args)
        assert code != 0 and out["refused"] is True and out["reason"]


def test_the_runner_side_interface_is_six_names():
    import boautoresearch
    import boautoresearch.cli  # noqa: F401  importing the CLI must not shadow a runner-side name

    assert boautoresearch.__all__ == ["lever", "seed", "fidelity", "log", "artifact_dir", "run"]
    assert all(callable(getattr(boautoresearch, n)) for n in boautoresearch.__all__)
