"""Registering hypotheses and just-in-time lever code (#24), against a toy with no levers yet."""
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import bo, events, git, write_run_yaml

TOY = Path(__file__).parent / "toy_lever"
BEFORE = 3.0 * (1 + 1 / 4)  # the toy's planted loss at the reference fidelity, before any lever

SPEC = {
    "title": "A gain term lowers the loss",
    "rationale": "The toy leaves a constant on the table.",
    "mechanism": "Subtract a learned gain from the loss.",
    "provenance": "novel",
    "lens": "optimisation",
    "directives": [],
    "fidelity_sensitive": False,
    "levers": {"gain": {"kind": "float", "low": 0.0, "high": 1.0, "baseline": 0.0,
                        "predicted": "higher"}},
}
# Lever code for SPEC as proposed first (H1): at gain 0 the loss is untouched.
LEVER_CODE = ("    loss -= lever(\"H1.gain\")\n", "from boautoresearch import fidelity, lever")


@pytest.fixture
def lever_repo(tmp_path):
    r = tmp_path / "project"
    shutil.copytree(TOY, r)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "toy")
    return r


@pytest.fixture
def lever_run(lever_repo, tmp_path, project_python):
    def _init(extra="budget_s: 3600\nreference_fidelity: {epochs: 4}\ndeterministic: true\n"):
        y = write_run_yaml(tmp_path / "run.yaml", project_python, extra)
        code, out = bo(lever_repo, "init", str(y), "--rationale", "start")
        assert code == 0, out
        return Path(out["run_dir"])
    return _init


def propose(cwd, spec, tmp_path, **kw):
    f = tmp_path / f"spec-{len(list(tmp_path.glob('spec-*')))}.json"
    f.write_text(json.dumps(spec))
    return bo(cwd, "propose", "--file", str(f), "--rationale", "a new idea", **kw)


def with_lever(**fields):
    s = copy.deepcopy(SPEC)
    s["levers"]["gain"].update(fields)
    s["levers"]["gain"] = {k: v for k, v in s["levers"]["gain"].items() if v is not None}
    return s


def test_propose_assigns_an_id_and_prefixes_lever_names(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    code, out = propose(lever_repo, SPEC, tmp_path)
    assert code == 0, out
    assert out["hypothesis"]["id"] == "H1.v1"
    assert out["hypothesis"]["status"] == "proposed"
    assert list(out["hypothesis"]["spec"]["levers"]) == ["H1.gain"]
    code, out = propose(lever_repo, SPEC, tmp_path)
    assert out["hypothesis"]["id"] == "H2.v1"
    assert list(out["hypothesis"]["spec"]["levers"]) == ["H2.gain"]

    ev = [e for e in events(run_dir) if e["type"] == "hypothesis_proposed"]
    assert [e["payload"]["id"] for e in ev] == ["H1.v1", "H2.v1"]
    assert ev[0]["payload"]["rationale"] == "a new idea"
    assert ev[0]["payload"]["spec"]["title"] == SPEC["title"]


INVALID = [
    (with_lever(baseline=None), "baseline"),
    (with_lever(baseline=1.5), "baseline"),
    (with_lever(predicted=None), "predicted"),
    (with_lever(predicted="sideways"), "predicted"),
    (with_lever(kind="bool", low=None, high=None, predicted=None, baseline=False), "why_not_graded"),
    (with_lever(kind="int", low=0, high=5, baseline=0.5), "baseline"),
    (with_lever(kind="categorical", low=None, high=None, predicted=None, options=["a", "b"],
                baseline="c"), "baseline"),
    (with_lever(kind="categorical", low=None, high=None, predicted=None, options=[1, True],
                baseline=1), "options"),
    (with_lever(kind="categorical", low=None, high=None, predicted=None, options=["a", 1],
                baseline=True), "baseline"),
    (with_lever(low=2.0), "low"),
    (with_lever(log=True), "log"),
    ({**SPEC, "title": "x" * 81}, "title"),
    ({**SPEC, "provenance": "adapted"}, "source"),
    ({**SPEC, "fidelity_sensitive": True}, "fidelity_reason"),
    ({**SPEC, "levers": {}}, "levers"),
    ({**SPEC, "surprise": 1}, "surprise"),
    ({k: v for k, v in SPEC.items() if k != "mechanism"}, "mechanism"),
]


def test_invalid_specs_are_refused_naming_the_field(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    for spec, field in INVALID:
        code, out = propose(lever_repo, spec, tmp_path)
        assert code != 0 and out["refused"] and field in out["reason"], (field, out)
    assert not [e for e in events(run_dir) if e["type"] == "hypothesis_proposed"]


def test_valid_non_float_levers_are_accepted(lever_repo, lever_run, tmp_path):
    lever_run()
    spec = copy.deepcopy(SPEC)
    spec["levers"] = {
        "steps": {"kind": "int", "low": 1, "high": 64, "log": True, "baseline": 8, "predicted": "lower"},
        "norm": {"kind": "categorical", "options": ["none", "layer"], "baseline": "none"},
        "swap": {"kind": "bool", "baseline": False, "why_not_graded": "a code path, not a dose"},
    }
    code, out = propose(lever_repo, {**spec, "provenance": "adapted", "source": "a paper"}, tmp_path)
    assert code == 0, out


def write_lever_code(run_dir, code=LEVER_CODE[0], imports=LEVER_CODE[1]):
    train = run_dir / "worktree" / "train.py"
    src = train.read_text().replace("from boautoresearch import fidelity", imports)
    train.write_text(src.replace("    return loss\n", code + "    return loss\n"))


def registered(cwd, tmp_path, spec=SPEC):
    code, out = propose(cwd, spec, tmp_path)
    assert code == 0, out
    hid = out["hypothesis"]["id"]
    code, out = bo(cwd, "register", hid, "--rationale", "reviewed")
    assert code == 0, out
    return hid


def test_register_freezes_the_spec_once(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    propose(lever_repo, SPEC, tmp_path)
    code, out = bo(lever_repo, "register", "H1", "--rationale", "reviewed")
    assert code == 0, out
    assert out["hypothesis"]["status"] == "registered"
    code, out = bo(lever_repo, "register", "H1.v1", "--rationale", "again")
    assert code != 0 and "already registered" in out["reason"]
    code, out = bo(lever_repo, "register", "H9", "--rationale", "?")
    assert code != 0 and "no hypothesis H9" in out["reason"]
    [reg] = [e for e in events(run_dir) if e["type"] == "hypothesis_registered"]
    assert reg["payload"]["id"] == "H1.v1"
    _, status = bo(lever_repo, "status")
    assert status["hypotheses"] == [{"id": "H1.v1", "title": SPEC["title"], "status": "registered",
                                     "commit": None}]


def test_lever_code_is_smoked_then_committed_and_the_baseline_is_a_no_op(lever_repo, lever_run,
                                                                           tmp_path):
    run_dir = lever_run()
    worktree = run_dir / "worktree"
    base_head = git(worktree, "rev-parse", "HEAD")
    hid = registered(lever_repo, tmp_path)
    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "too early")
    assert code != 0 and "smoke" in out["reason"]

    write_lever_code(run_dir)
    code, out = bo(lever_repo, "smoke", hid, "--rationale", "try the gain")
    assert code == 0, out
    assert out["passed"] is True
    base, rand = out["trials"]
    assert (base["point"], rand["point"]) == ("baseline", "random")
    assert base["levers"] == {"H1.gain": 0.0}
    assert 0.0 <= rand["levers"]["H1.gain"] <= 1.0 and rand["levers"]["H1.gain"] != 0.0
    assert base["objective"] == BEFORE  # at baseline the objective is the pre-lever one
    assert rand["objective"] == pytest.approx(BEFORE - rand["levers"]["H1.gain"])
    assert base["fidelity"] == {"epochs": 4}  # no ladder: the reference is the cheapest rung
    assert base["commit"] == base_head  # uncommitted code: HEAD plus the hashed worktree
    assert base["tree"] == rand["tree"]

    code, out = bo(lever_repo, "round-run", "--rationale", "dirty")
    assert code != 0 and "uncommitted" in out["reason"]

    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "smoke passed",
                   "--actor", "lever-coder")
    assert code == 0, out
    sha = out["commit"]
    assert git(worktree, "rev-parse", "HEAD") == sha
    assert git(worktree, "rev-parse", "HEAD~1") == base_head
    assert git(worktree, "status", "--porcelain") == ""
    assert git(worktree, "log", "-1", "--format=%an %s") == f"boautoresearch H1.v1: {SPEC['title']}"
    assert json.loads((worktree / "levers.json").read_text()) == {"H1.gain": 0.0}
    [ev] = [e for e in events(run_dir) if e["type"] == "lever_committed"]
    assert (ev["payload"]["id"], ev["payload"]["commit"], ev["actor"]) == (hid, sha, "lever-coder")
    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "twice")
    assert code != 0 and "already committed" in out["reason"]

    # Committed baselines: the next trial runs on the new commit, and outside the harness too.
    code, out = bo(lever_repo, "round-run", "--rationale", "calibrate")
    assert code == 0, out
    [started] = [e["payload"] for e in events(run_dir) if e["type"] == "round_started"]
    assert started["commit"] == sha
    trial = [t for t in bo(lever_repo, "trials")[1]["trials"] if t.get("round") == 0][0]
    assert (trial["commit"], trial["levers"], trial["objective"]) == (sha, {"H1.gain": 0.0}, BEFORE)
    env = {k: v for k, v in os.environ.items() if k != "BOAUTORESEARCH_TRIAL"}
    p = subprocess.run([run_dir / "venv" / "bin" / "python", "runner.py"], cwd=worktree, env=env,
                       capture_output=True, text=True, check=True)
    assert json.loads(p.stdout)["objective"] == 3.0 * (1 + 1 / 10)  # the toy's own epochs


def test_commit_lever_needs_the_smoke_to_pass_on_the_same_contents(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    hid = registered(lever_repo, tmp_path)
    write_lever_code(run_dir, code="    if lever(\"H1.gain\") > 0:\n        raise RuntimeError('boom')\n")
    code, out = bo(lever_repo, "smoke", hid, "--rationale", "crashes off baseline")
    assert code == 0 and out["passed"] is False
    assert [t["status"] for t in out["trials"]] == ["finished", "failed"]
    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "failed smoke")
    assert code != 0 and "passing" in out["reason"]

    git(run_dir / "worktree", "checkout", "train.py")
    write_lever_code(run_dir)  # the fix: passes
    assert bo(lever_repo, "smoke", hid, "--rationale", "fixed")[1]["passed"] is True
    (run_dir / "worktree" / "notes.py").write_text("x = 1\n")  # an edit after the smoke
    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "edited")
    assert code != 0 and "changed since" in out["reason"]
    assert [e["payload"]["passed"] for e in events(run_dir) if e["type"] == "lever_smoke"] == [
        False, True]
    assert not [e for e in events(run_dir) if e["type"] == "lever_committed"]


SNEAKY = [  # each runs fine at both smoke points, so only the static check stands in the way
    ("    loss -= float(os.environ.get('GAIN', 0)) * lever(\"H1.gain\")\n", LEVER_CODE[1],
     "other than through lever()"),
    ("    loss -= float(g('GAIN', 0)) * lever(\"H1.gain\")\n",
     "from os import getenv as g\nfrom boautoresearch import fidelity, lever", "getenv"),
    ("    import argparse\n    loss -= lever(\"H1.gain\")\n", LEVER_CODE[1], "import argparse"),
    ("    loss -= json.load(open('levers.json')).get('H1.gain', 0)\n",
     "import json\nfrom boautoresearch import fidelity", "'levers.json'"),
    ("    loss -= 0\n", LEVER_CODE[1], "never reads ['H1.gain']"),
    ("    loss -= lever(\"H1.gain\")\n    if loss < -1:\n        loss = lever(\"H1.gian\")\n",
     LEVER_CODE[1], "no hypothesis declares"),
    ("    name = 'H1.gain'\n    loss -= lever(name)\n", LEVER_CODE[1], "literal"),
]


def test_lever_code_reading_config_without_lever_is_refused(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    worktree = run_dir / "worktree"
    hid = registered(lever_repo, tmp_path)
    for code, imports, reason in SNEAKY:
        git(worktree, "checkout", "train.py")
        write_lever_code(run_dir, code=code, imports=imports)
        assert bo(lever_repo, "smoke", hid, "--rationale", "try")[1]["passed"] is True, reason
        code_, out = bo(lever_repo, "commit-lever", hid, "--rationale", "sneaky")
        assert code_ != 0 and out["refused"] and reason in out["reason"], (reason, out)
    assert not [e for e in events(run_dir) if e["type"] == "lever_committed"]


def test_smoke_of_a_hypothesis_runs_at_the_cheapest_rung_once_r0_measured_it(lever_repo, lever_run,
                                                                               tmp_path):
    run_dir = lever_run("budget_s: 3600\nreference_fidelity: {epochs: 4}\ndeterministic: true\n"
                        "ladder: [{epochs: 1}]\n")
    env = {**os.environ, "TOY_SLEEP_PER_EPOCH": "0.05"}
    hid = registered(lever_repo, tmp_path)
    write_lever_code(run_dir)
    code, out = bo(lever_repo, "smoke", hid, "--rationale", "too early", env=env)
    assert code != 0 and "round-run" in out["reason"]

    git(run_dir / "worktree", "stash", "-q")
    assert bo(lever_repo, "round-run", "--rationale", "calibrate", env=env)[0] == 0
    git(run_dir / "worktree", "stash", "pop", "-q")
    code, out = bo(lever_repo, "smoke", hid, "--rationale", "now", env=env)
    assert code == 0 and out["passed"], out
    assert [t["fidelity"] for t in out["trials"]] == [{"epochs": 1}, {"epochs": 1}]
    assert out["trials"][0]["objective"] == 3.0 * (1 + 1 / 1)


def test_commit_lever_keeps_the_project_levers_and_numbers_past_them(lever_repo, lever_run, tmp_path):
    (lever_repo / "levers.json").write_text('{"H1.scale": 2.0}\n')  # a lever the project already has
    git(lever_repo, "commit", "-qam", "an existing lever")
    run_dir = lever_run()
    hid = registered(lever_repo, tmp_path)
    assert hid == "H2.v1"  # H1 is taken by levers.json
    write_lever_code(run_dir, code="    loss -= lever(\"H2.gain\")\n")
    code, out = bo(lever_repo, "smoke", hid, "--rationale", "try")
    assert code == 0 and out["passed"], out
    assert out["trials"][0]["levers"] == {"H1.scale": 2.0, "H2.gain": 0.0}
    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "smoke passed")
    assert code == 0, out
    assert json.loads((run_dir / "worktree" / "levers.json").read_text()) == {
        "H1.scale": 2.0, "H2.gain": 0.0}


def test_smoke_fails_when_its_trials_write_into_the_worktree(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    hid = registered(lever_repo, tmp_path)
    write_lever_code(run_dir, code="    open('out.txt', 'w').write('x')\n    loss -= lever(\"H1.gain\")\n")
    for _ in range(2):  # the second smoke already sees out.txt: the trials must still not touch it
        code, out = bo(lever_repo, "smoke", hid, "--rationale", "writes a file")
        assert code == 0 and out["passed"] is False and "worktree" in out["reason"], out
    code, out = bo(lever_repo, "commit-lever", hid, "--rationale", "failed smoke")
    assert code != 0 and "passing" in out["reason"]


def test_the_random_smoke_point_of_a_bool_lever_is_off_baseline(lever_repo, lever_run, tmp_path):
    run_dir = lever_run()
    spec = copy.deepcopy(SPEC)
    spec["levers"] = {"swap": {"kind": "bool", "baseline": False, "why_not_graded": "a code path"}}
    hid = registered(lever_repo, tmp_path, spec)
    write_lever_code(run_dir, code="    if lever(\"H1.swap\"):\n        loss -= 0.5\n")
    for _ in range(5):
        code, out = bo(lever_repo, "smoke", hid, "--rationale", "try")
        assert code == 0 and out["passed"], out
        assert out["trials"][1]["levers"] == {"H1.swap": True}
