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


def review(cwd, ref, **verdict):
    """A registration reviewer's `review` record for H (its latest version when ref is H<n>);
    allow / fits / no conflict unless `verdict` says otherwise."""
    hid = ref if ".v" in ref else max((h["id"] for h in bo(cwd, "status")[1]["hypotheses"]
                                       if h["id"].split(".")[0] == ref), key=lambda i: int(i.split(".v")[1]))
    body = {"hypothesis": hid, "directive_verdict": "allow", "intent": "fits", "conflict": "none",
            "conflict_with": [], "rationale": "nothing in the registry or the brief speaks against it",
            "strict": False, **verdict}
    f = Path(cwd).parent / f"review-{len(list(Path(cwd).parent.glob('review-*')))}.json"
    f.write_text(json.dumps(body))
    return bo(cwd, "record", "review", "--file", str(f), "--agent-id", "rr", "--actor",
              "registration-reviewer", "--rationale", "the registration review")


def register(cwd, ref, rationale="reviewed", **verdict):
    """`register`, its review recorded first; -> (code, out)."""
    code, out = review(cwd, ref, **verdict)
    assert code == 0, out
    return bo(cwd, "register", ref, "--rationale", rationale)


def expect_all(cwd):
    """Settle the routine duties before a round: an `allow` review of each hypothesis owed one, an
    interplay review flagging nothing for each one
    owed, a generation pass when one is due, and an `undecided` expected verdict for each hypothesis
    the next round still lacks one for."""
    for h in bo(cwd, "status")[1]["review_missing"]:
        code, out = review(cwd, h)
        assert code == 0, out
    status = bo(cwd, "status")[1]
    for m in status["interplay_missing"]:
        f = Path(cwd).parent / f"interplay-{len(list(Path(cwd).parent.glob('interplay-*')))}.json"
        f.write_text(json.dumps({**m, "flags": []}))
        code, out = bo(cwd, "record", "interplay", "--file", str(f), "--rationale", "nothing interacts")
        assert code == 0, out
    if bo(cwd, "status")[1]["generation"]["due"]:
        code, out = bo(cwd, "generate", "--rationale", "a pass is due")
        assert code == 0, out
    for h in bo(cwd, "status")[1]["schedule"]["expected_missing"]:
        f = Path(cwd).parent / f"expected-{h}.json"
        f.write_text(json.dumps({"hypothesis": h, "verdict": "undecided", "reason": "no view yet"}))
        code, out = bo(cwd, "record", "expected", "--file", str(f), "--rationale", "before the round")
        assert code == 0, out


def round_run(cwd, *args, env=None):
    """`round-run`, its expected verdicts recorded first."""
    expect_all(cwd)
    return bo(cwd, "round-run", *args, env=env)


SPARE = {"title": "A spare proposal", "rationale": "Any valid spec will do.",
         "mechanism": "None.", "provenance": "novel", "lens": "optimisation", "directives": [],
         "fidelity_sensitive": False,
         "levers": {"s": {"kind": "float", "low": 0.0, "high": 1.0, "baseline": 0.0, "predicted": "higher"}}}


def ready(out):
    """Nothing but routine duties (expected verdicts, interplay reviews, a generation pass) stands
    before round-run."""
    return out["next"] == ["round-run"] or all(d.startswith(("record expected", "record interplay", "record review", "generate"))
                                               for d in out["next"])


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


def write_run_yaml(path, python, extra=""):
    extra = extra or "budget_s: 3600\nreference_fidelity: {epochs: 4}\n"
    # a fixed seed unless the test sets its own: runs (and their equivalence checks, which fail by
    # chance at 2σ) replay the same way every time
    path.write_text("objective: loss\ndirection: minimize\n" f"python: {python}\n" + extra
                    + ("" if "seed:" in extra else "seed: 0\n"))
    return path


@pytest.fixture
def run_yaml(tmp_path, project_python):
    return write_run_yaml(tmp_path / "run.yaml", project_python)


@pytest.fixture
def init_run(repo, tmp_path, project_python):
    """init with run.yaml lines of your own (budget_s and reference_fidelity included)."""
    def _init(extra):
        y = write_run_yaml(tmp_path / "run.yaml", project_python, extra)
        code, out = bo(repo, "init", str(y), "--rationale", "start the toy run")
        assert code == 0, out
        return Path(out["run_dir"])
    return _init


@pytest.fixture
def run_dir(repo, run_yaml):
    code, out = bo(repo, "init", str(run_yaml), "--rationale", "start the toy run")
    assert code == 0, out
    return Path(out["run_dir"])
