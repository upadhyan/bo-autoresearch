"""Shared fixtures for the acceptance tests: a toy target repo, a scripted eval, and a run driver.

The tests drive the real CLI in-process (`boar.cli.main`) from inside a throwaway git repo, so a
whole run with real Optuna studies and real eval subprocesses finishes in seconds.
"""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

TARGET_PY = '''\
"""Toy target: a cost with two real slow paths and one knob that does nothing."""
import json
import os


def _diag(config):
    """Counters for BOAR_DIAG_DIR: the passes made and the batch size used."""
    where = os.environ.get("BOAR_DIAG_DIR")
    if where:
        batch = max(int(config.get("fast_b", 1)), int(config.get("fast_b_wide", 1)))
        with open(os.path.join(where, "diag.json"), "w") as f:
            json.dump({"passes": 2 if config.get("fast_a", False) else 3, "batch": batch}, f)


def cost(config):
    _diag(config)
    c = 100.0
    if config.get("fast_a", False):
        c -= 20.0
    c -= 3.0 * (max(int(config.get("fast_b", 1)), int(config.get("fast_b_wide", 1))) - 1)
    if config.get("cache_output", False):
        c -= 60.0
    return c
'''

SPEC_MD = """\
# Spec

## Goal
Make the toy target cheaper to run.

## Metric
Cost units reported by target.cost, median over repeats.
Direction: min

## Target population
Every config the toy target accepts.

## Guards
- The simulated memory ceiling holds.

## Scope
Changes to target.py only.

## Known cheats
- Caching the benchmark output.

## User ideas
1. Skip the redundant pass.
2. Batch the work.
3. Rename the loop variables.
4. Cache the benchmark output.

## Budget
rounds=3 trials_per_round=4
"""

EVAL_RUN = '''\
#!{python}
import hashlib, json, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.getcwd())
import target

COUNTER = {counter!r}
config = json.load(open(os.environ["BOAR_CONFIG"]))
split = os.environ.get("BOAR_SPLIT", "dev")
time.sleep({sleep!r})
if {crash_when}:
    print("boom", file=sys.stderr)
    sys.exit(3)
n = int(open(COUNTER).read()) if os.path.exists(COUNTER) else 0
open(COUNTER, "w").write(str(n + 1))
digest = hashlib.sha256((json.dumps(config, sort_keys=True) + split + str(n)).encode()).digest()
noise = (digest[0] / 255.0 - 0.5) * 0.4
metric = target.cost(config) + noise + (0.5 if split == "holdout" else 0.0)
guards_ok = not ({guard_fail})
print("toy eval: split=%s n=%d" % (split, n), file=sys.stderr)
print("progress line that is not the result")
print(json.dumps({{"metric": metric, "guards_ok": guards_ok, "metrics": {{"calls": n}}}}))
'''

# A fast, representative run for the acceptance tests.
RUN_CONFIG = {"rounds": 3, "trials_per_round": 4, "repeats": 2, "trial_target_s": 5, "holdout_repeats": 2}

REASON = (
    "Judged against spec.md: the change is in scope, tests a single mechanism, and its lever ranges are "
    "realistic for the target population."
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def lever(name: str, type_: str = "bool", **kw) -> dict:
    if type_ == "bool":
        return {"name": name, "type": "bool", "default": kw.get("default", False)}
    return {"name": name, "type": type_, **kw}


def proposal(statement: str, *levers: dict, source: str = "user", supersedes: str | None = None) -> dict:
    return {
        "statement": statement,
        "mechanism": f"mechanism behind: {statement.lower()}",
        "source": source,
        "citations": ["spec.md"],
        "levers": list(levers),
        "supersedes": supersedes,
    }


def prop_a() -> dict:
    return proposal("Skipping the redundant pass cuts cost", lever("fast_a"))


def prop_b() -> dict:
    return proposal("Batching the work cuts cost", lever("fast_b", "int", low=1, high=8, default=1))


def prop_noop() -> dict:
    return proposal("Renaming the loop variables cuts cost", lever("noop"))


def prop_cheat() -> dict:
    return proposal("Caching the benchmark output cuts cost", lever("cache_output"), source="research")


def default_proposals() -> list[dict]:
    return [prop_a(), prop_b(), prop_noop(), prop_cheat()]


TAG = re.compile(r"^\[([A-Z0-9_]+)\]")

# The tags the Stop hook lets the agent stop at, spelled out here so a change to nextstep's set is caught.
RELEASE_TAGS = {"NONE", "ASK_USER", "DONE", "ABORTED"}


def tag_of(text: str) -> str | None:
    m = TAG.match(text.strip())
    return m.group(1) if m else None


def write_eval(
    run_dir: Path,
    counter: Path,
    guard_fail: str = "False",
    sleep: float = 0.0,
    crash_when: str = "False",
) -> Path:
    """Write `eval/run`. `guard_fail` and `crash_when` are Python expressions over `config`."""
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    run = eval_dir / "run"
    run.write_text(
        EVAL_RUN.format(
            python=sys.executable, counter=str(counter), sleep=float(sleep), guard_fail=guard_fail, crash_when=crash_when
        )
    )
    run.chmod(0o755)
    return run


@pytest.fixture
def boar(capsys, monkeypatch):
    """Call the CLI in-process: `boar("next", "--hook", stdin="{}")` -> (exit code, stdout, stderr)."""
    from boar.cli import main

    monkeypatch.setenv("BOAR_DEBUG", "1")
    # The Stop hook falls back to the project dir; the session running the tests may have one set.
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)

    def call(*args, stdin: str = ""):
        capsys.readouterr()
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
        try:
            rc = main([str(a) for a in args])
        except SystemExit as e:
            rc = e.code
        out, err = capsys.readouterr()
        if rc is None:
            rc = 0
        elif not isinstance(rc, int):
            err, rc = f"{err}{rc}", 1
        return rc, out, err

    return call


@pytest.fixture
def target_repo(tmp_path, monkeypatch):
    """A committed toy target repo on `main`, isolated from the user's git config; cwd is set to it."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))  # init/next --session write a session registry
    repo = tmp_path / "target"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "BOAR Test")
    git(repo, "config", "user.email", "boar-test@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    (repo / "target.py").write_text(TARGET_PY)
    (repo / "spec.md").write_text(SPEC_MD)
    (repo / ".gitignore").write_text("__pycache__/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "toy target")
    monkeypatch.chdir(repo)
    return repo


@pytest.fixture
def make_eval(tmp_path):
    """`make_eval(run_dir, guard_fail=..., sleep=..., crash_when=...)` writes a fast eval into a run dir."""
    counter = tmp_path / "eval-counter"

    def make(run_dir: Path, **kw) -> Path:
        return write_eval(run_dir, counter, **kw)

    return make


class Driver:
    """Scripted agent + reviewer: one method per harness step, asserting each step succeeds.

    The module's helpers hang off the class too, so test modules never import conftest.
    """

    SMALL = {"rounds": 2, "trials_per_round": 2, "repeats": 1, "trial_target_s": 5, "holdout_repeats": 1}
    RUN_CONFIG = RUN_CONFIG
    lever = staticmethod(lever)
    proposal = staticmethod(proposal)
    prop_a = staticmethod(prop_a)
    prop_b = staticmethod(prop_b)
    prop_noop = staticmethod(prop_noop)
    prop_cheat = staticmethod(prop_cheat)
    tag_of = staticmethod(tag_of)

    def __init__(self, boar, repo: Path, tmp: Path, make_eval):
        self.boar = boar
        self.repo = repo
        self.tmp = tmp
        self.make_eval = make_eval
        self._files = 0

    # --- calling the CLI ---------------------------------------------------

    def ok(self, *args, stdin: str = "") -> str:
        rc, out, err = self.boar(*args, stdin=stdin)
        assert rc == 0, f"`boar {' '.join(map(str, args))}` exited {rc}\nstdout:\n{out}\nstderr:\n{err}"
        return out

    def refused(self, *args) -> str:
        rc, out, err = self.boar(*args)
        assert rc == 1, f"`boar {' '.join(map(str, args))}` should be refused, exited {rc}\nstdout:\n{out}\nstderr:\n{err}"
        assert "refused" in err.lower(), f"no refusal message on stderr:\n{err}"
        return err

    def next(self) -> str:
        """`boar next`, checking that the Stop hook holds the agent at every tag but the release tags."""
        text = self.ok("next")
        tag = tag_of(text)
        rc, err = self.hook()
        assert rc == (0 if tag in RELEASE_TAGS else 2), f"the Stop hook exited {rc} at [{tag}]\nstderr:\n{err}"
        return text

    def tag(self) -> str | None:
        return tag_of(self.next())

    def hook(self, stdin: str = "{}") -> tuple[int, str]:
        rc, out, err = self.boar("next", "--hook", stdin=stdin)
        return rc, err

    def git(self, *args: str) -> str:
        return git(self.repo, *args)

    # --- run files -----------------------------------------------------------

    @property
    def run_id(self) -> str:
        return (self.repo / ".boar" / "active").read_text().strip()

    @property
    def run_dir(self) -> Path:
        return self.repo / ".boar" / self.run_id

    def state(self) -> dict:
        return json.loads((self.run_dir / "state.json").read_text())

    def config(self) -> dict:
        return json.loads((self.run_dir / "config.json").read_text())

    def hyps(self) -> list[dict]:
        return json.loads((self.run_dir / "hypotheses.json").read_text())

    def hyp(self, hid: str) -> dict:
        return next(h for h in self.hyps() if h["id"] == hid)

    def hid(self, lever_name: str) -> str:
        """The id of the hypothesis that owns a lever."""
        for h in self.hyps():
            if any(lv["name"] == lever_name for lv in h["levers"]):
                return h["id"]
        raise KeyError(lever_name)

    def trials(self, round_: int | None = None) -> list[dict]:
        path = self.run_dir / "trials.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [t for t in rows if round_ is None or t["round"] == round_]

    def reviews(self) -> list[dict]:
        path = self.run_dir / "reviews.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # --- steps ---------------------------------------------------------------

    def init(self, session: str | None = None, **overrides) -> str:
        cfg = {**RUN_CONFIG, **overrides}
        args = ["init", "--spec", "spec.md"]
        if session is not None:
            args += ["--session", session]
        for key, value in cfg.items():
            args += [f"--{key.replace('_', '-')}", str(value)]
        return self.ok(*args)

    def write_proposals(self, proposals: list[dict]) -> Path:
        self._files += 1
        path = self.tmp / f"proposals-{self._files}.json"
        path.write_text(json.dumps(proposals, indent=2))
        return path

    def propose(self, *proposals: dict) -> list[str]:
        before = {h["id"] for h in self.hyps()}
        self.ok("propose", self.write_proposals(list(proposals)))
        new = [h["id"] for h in self.hyps() if h["id"] not in before]
        assert len(new) == len(proposals), f"expected {len(proposals)} new hypotheses, got {new}"
        return new

    def write_eval(self, **kw) -> Path:
        return self.make_eval(self.run_dir, **kw)

    def eval_check(self) -> str:
        return self.ok("eval", "check", "--foreground")

    def review(self, item: str, decision: str, reason: str = REASON) -> str:
        return self.ok("review", "record", item, decision, "--reason", reason)

    def decide(
        self, hid: str, decision: str, reason: str = "keep it: too little evidence yet", trials=None, key: str | None = "passes"
    ) -> str:
        """`key` is the observable key, passed with every decision but fix (the toy target measures `passes`)."""
        args = ["decide", hid, decision, "--reason", reason]
        if key and decision != "fix":
            args += ["--observable-key", key]
        if trials:
            args += ["--trials", ",".join(str(t) for t in trials)]
        return self.ok(*args)

    def round_run(self) -> str:
        return self.ok("round", "run", "--foreground")

    def close(self) -> str:
        return self.ok("round", "close")

    def start(
        self,
        until: str = "round1",
        proposals: list[dict] | None = None,
        accept: tuple[str, ...] = ("fast_a", "fast_b", "noop"),
        guard_fail: str = "False",
        session: str | None = None,
        **overrides,
    ) -> "Driver":
        """Drive a fresh run up to `until`: init, proposed, checked, round1 (setup reviewed), round1_run."""
        points = ["init", "proposed", "checked", "round1", "round1_run"]
        assert until in points, until
        stop = points.index(until)
        self.init(session=session, **overrides)
        if stop >= 1:
            self.propose(*(proposals if proposals is not None else default_proposals()))
        if stop >= 2:
            self.write_eval(guard_fail=guard_fail)
            self.eval_check()
        if stop >= 3:
            for h in self.hyps():
                accepted = any(lv["name"] in accept for lv in h["levers"])
                self.review(h["id"], "accept" if accepted else "reject",
                            REASON if accepted else "Rejected: this is a cheat listed in the spec's Known cheats.")
            self.review("eval", "accept", "Dev and holdout are disjoint and typical; guards cover the spec.")
            assert self.state()["phase"] == "round", self.next()
        if stop >= 4:
            self.round_run()
        return self


@pytest.fixture
def driver(boar, target_repo, tmp_path, make_eval):
    """A scripted agent in the toy repo. `driver.start(until=...)` drives a fresh run to a given point."""
    d = Driver(boar, target_repo, tmp_path, make_eval)
    yield d
    _kill_stray_worker(target_repo)


def _kill_stray_worker(repo: Path) -> None:
    """A test that fails mid-way must not leave a detached worker running."""
    for running in (repo / ".boar").glob("*/running.json"):
        try:
            pid = json.loads(running.read_text())["pid"]
            if isinstance(pid, int) and pid > 1 and os.getpgid(pid) not in (os.getpgrp(), os.getpid()):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, ValueError, KeyError, TypeError):
            pass
