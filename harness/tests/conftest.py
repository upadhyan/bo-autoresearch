"""Seam 1: drive the `boautoresearch` CLI as a scripted caller against a toy project."""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

TOY = Path(__file__).parent / "toy_project"


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def bo(cwd, *args, env=None):
    """Run the CLI; return (exit code, parsed JSON stdout)."""
    p = subprocess.run(
        [sys.executable, "-m", "boautoresearch", *args],
        cwd=cwd, capture_output=True, text=True, env=env,
    )
    try:
        return p.returncode, json.loads(p.stdout)
    except json.JSONDecodeError:
        raise AssertionError(f"non-JSON output:\n{p.stdout}\n{p.stderr}")


def events(run_dir):
    con = sqlite3.connect(run_dir / "log.db")
    rows = con.execute("SELECT seq, type, actor, payload FROM events ORDER BY seq").fetchall()
    con.close()
    return [{"seq": s, "type": t, "actor": a, "payload": json.loads(p)} for s, t, a, p in rows]


@pytest.fixture(scope="session")
def project_python(tmp_path_factory):
    """The user's project environment: a bare venv."""
    d = tmp_path_factory.mktemp("projenv")
    subprocess.run([sys.executable, "-m", "venv", d / "venv"], check=True)
    return d / "venv" / "bin" / "python"


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "project"
    shutil.copytree(TOY, r)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "toy")
    return r


@pytest.fixture
def run_yaml(tmp_path, project_python):
    p = tmp_path / "run.yaml"
    p.write_text(
        "objective: loss\n"
        "direction: minimize\n"
        "budget_s: 3600\n"
        "reference_fidelity: {epochs: 4}\n"
        f"python: {project_python}\n"
    )
    return p


@pytest.fixture
def run_dir(repo, run_yaml):
    code, out = bo(repo, "init", str(run_yaml), "--rationale", "start the toy run")
    assert code == 0, out
    return Path(out["run_dir"])
