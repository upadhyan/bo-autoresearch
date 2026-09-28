"""Agent-side CLI. Every response is JSON; refusals are {"refused": true, "reason": ...}.

Probes (status, trials) are free and never logged. Actions (init, smoke) need a
--rationale and log their actor.
"""
import argparse
import csv
import io
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import experiment_log as elog

HARNESS_SRC = Path(__file__).resolve().parents[1]  # holds pyproject.toml when run from the plugin
RUN_KEYS = {"objective", "direction", "budget_s", "runner", "reference_fidelity", "python"}


class Refused(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refused(message)


def _git(cwd: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def _repo_root() -> Path:
    """The user's main checkout, also when called from inside the run's worktree."""
    return Path(_git(Path.cwd(), "rev-parse", "--path-format=absolute", "--git-common-dir")).parent


def _open_run() -> tuple[Path, sqlite3.Connection]:
    runs = sorted((_repo_root() / ".bo-research").glob("*/log.db"))
    if not runs:
        raise Refused("no run: start one with `boautoresearch init run.yaml`")
    # ponytail: latest run wins; add --run when runs coexist
    return runs[-1].parent, elog.connect(runs[-1])


def _load_run_yaml(path: Path) -> dict:
    try:
        cfg = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise Refused(f"cannot read {path}: {e}")
    if not isinstance(cfg, dict):
        raise Refused(f"{path} must be a mapping")
    if unknown := set(cfg) - RUN_KEYS:
        raise Refused(f"unknown run.yaml keys: {sorted(unknown)}")
    if not isinstance(cfg.get("objective"), str) or not cfg["objective"]:
        raise Refused("run.yaml: objective must name the objective")
    if cfg.get("direction") not in ("minimize", "maximize"):
        raise Refused("run.yaml: direction must be minimize or maximize")
    budget = cfg.get("budget_s")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
        raise Refused("run.yaml: budget_s must be a positive number of seconds")
    cfg.setdefault("runner", "runner.py")
    cfg.setdefault("reference_fidelity", {})
    if not isinstance(cfg["reference_fidelity"], dict):
        raise Refused("run.yaml: reference_fidelity must be a mapping, e.g. {epochs: 20}")
    cfg["python"] = str(cfg.get("python") or sys.executable)
    return cfg


def _freeze(python: str) -> list[str]:
    # --exclude-editable: an editable install of the project would import the user's checkout
    # ponytail: needs pip in the project env; add a uv fallback when uv-only envs show up
    p = subprocess.run([python, "-m", "pip", "freeze", "--exclude-editable"],
                       capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"cannot freeze the project environment with {python}: {p.stderr.strip()}")
    return p.stdout.split()


def cmd_init(a) -> dict:
    cfg = _load_run_yaml(Path(a.run_yaml))
    root = _repo_root()
    runs = root / ".bo-research"
    exclude = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    if "/.bo-research/" not in (exclude.read_text() if exclude.exists() else "").split():
        with exclude.open("a") as f:
            f.write("\n/.bo-research/\n")
    if _git(root, "status", "--porcelain"):
        raise Refused("the working tree has uncommitted changes; commit or stash them first")
    if not (root / cfg["runner"]).is_file():
        raise Refused(f"runner {cfg['runner']} is not in the repository")
    if not (HARNESS_SRC / "pyproject.toml").exists():
        raise Refused(f"init must run from the plugin's harness source, not {HARNESS_SRC}")
    freeze = _freeze(cfg["python"])

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir, branch = runs / run_id, f"bo-research/{run_id}"
    if run_dir.exists():
        raise Refused(f"run {run_id} already exists")
    base_commit = _git(root, "rev-parse", "HEAD")
    run_dir.mkdir(parents=True)
    try:
        _git(root, "worktree", "add", "-q", "-b", branch, str(run_dir / "worktree"), base_commit)
        venv_python = run_dir / "venv" / "bin" / "python"
        subprocess.run([cfg["python"], "-m", "venv", run_dir / "venv"], check=True,
                       capture_output=True)
        (run_dir / "venv-freeze.txt").write_text("".join(f"{r}\n" for r in freeze))
        subprocess.run([venv_python, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                        "-r", run_dir / "venv-freeze.txt", HARNESS_SRC],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        subprocess.run(["git", "worktree", "remove", "--force", run_dir / "worktree"], cwd=root)
        subprocess.run(["git", "branch", "-D", branch], cwd=root, capture_output=True)
        shutil.rmtree(run_dir)
        raise Refused(f"creating the run venv failed: {e.stderr or e}")
    shutil.copy(a.run_yaml, run_dir / "run.yaml")

    con = elog.connect(run_dir / "log.db")
    elog.append(con, "run_started", a.actor, {
        "rationale": a.rationale, "run_id": run_id, "branch": branch, "base_commit": base_commit,
        "objective": cfg["objective"], "direction": cfg["direction"], "budget_s": cfg["budget_s"],
        "runner": cfg["runner"], "reference_fidelity": cfg["reference_fidelity"],
        "venv_freeze": freeze,
    })
    _regenerate(run_dir, con)
    return {"run_dir": str(run_dir), "worktree": str(run_dir / "worktree"),
            "venv": str(run_dir / "venv"), **_status(elog.state(elog.read(con)))}


def cmd_smoke(a) -> dict:
    """One trial with every lever at its committed baseline, at the reference fidelity."""
    run_dir, con = _open_run()
    st = elog.state(elog.read(con))
    run, n = st["run"], len(st["trials"]) + 1
    worktree = run_dir / "worktree"
    levers_file = worktree / "levers.json"
    levers = json.loads(levers_file.read_text()) if levers_file.exists() else {}
    trial = {"trial": n, "levers": levers, "fidelity": run["reference_fidelity"],
             "seed": random.Random(f"{run['run_id']}/{n}").getrandbits(31),
             "commit": _git(worktree, "rev-parse", "HEAD"),
             "artifact_dir": f"artifacts/trial-{n}"}
    elog.append(con, "trial_started", a.actor, {"rationale": a.rationale, **trial})
    outcome = _run_trial(run_dir, run["runner"], trial)
    elog.append(con, "trial_" + outcome.pop("status"), "harness", {"trial": n, **outcome})
    _regenerate(run_dir, con)
    return {"trial": elog.state(elog.read(con))["trials"][-1]}


def _run_trial(run_dir: Path, runner: str, trial: dict) -> dict:
    art = run_dir / trial["artifact_dir"]
    art.mkdir(parents=True)
    (art / "trial.json").write_text(json.dumps({**trial, "artifact_dir": str(art)}, indent=2))
    env = {**os.environ, "BOAUTORESEARCH_TRIAL": str(art / "trial.json")}
    t0 = time.monotonic()
    with open(art / "stdout.txt", "w") as out, open(art / "stderr.txt", "w") as err:
        p = subprocess.Popen([run_dir / "venv" / "bin" / "python", runner],
                             cwd=run_dir / "worktree", env=env, stdout=out, stderr=err)
        _, status, usage = os.wait4(p.pid, 0)  # per-child rusage gives this trial's peak memory
    p.returncode = os.waitstatus_to_exitcode(status)
    measured = {"exit_code": p.returncode, "wall_clock_s": time.monotonic() - t0,
                "peak_mem_mb": usage.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10),
                "artifact_dir": trial["artifact_dir"]}

    result_file = art / "result.json"
    if p.returncode:
        reason = f"runner exited with code {p.returncode}"
    elif not result_file.exists():
        reason = "runner produced no result; it must call boautoresearch.run(fn)"
    else:
        result = json.loads(result_file.read_text())
        if math.isfinite(result["objective"]):
            return {"status": "finished", **measured, **result}
        reason = "degenerate trial: non-finite objective"
    stderr = [ln for ln in (art / "stderr.txt").read_text().splitlines() if ln.strip()]
    return {"status": "failed", **measured, "reason": reason, "error": stderr[-1] if stderr else ""}


def _status(st: dict) -> dict:
    run, trials = st["run"], st["trials"]
    spent = sum(t["wall_clock_s"] for t in trials if t["status"] != "running")
    count = {s: sum(t["status"] == s for t in trials) for s in ("running", "finished", "failed")}
    return {"run_id": run["run_id"], "objective": run["objective"], "direction": run["direction"],
            "budget": {"total_s": run["budget_s"], "spent_s": spent,
                       "remaining_s": run["budget_s"] - spent},
            "trials": {"total": len(trials), **count}}


def cmd_status(a) -> dict:
    _, con = _open_run()
    return _status(elog.state(elog.read(con)))


def cmd_trials(a) -> dict:
    _, con = _open_run()
    return {"trials": elog.state(elog.read(con))["trials"]}


def _regenerate(run_dir: Path, con) -> None:
    """Rewrite every generated file from the events."""
    trials = elog.state(elog.read(con))["trials"]
    groups = {"L:": "levers", "c:": "constraints", "t:": "telemetry"}
    extra = sorted({p + k for t in trials for p, g in groups.items() for k in t.get(g, {})})
    meta = ["trial", "status", "objective", "seed", "fidelity", "commit", "wall_clock_s",
            "peak_mem_mb", "exit_code", "artifact_dir"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(meta + extra)
    for t in trials:
        row = [json.dumps(t["fidelity"], sort_keys=True) if k == "fidelity" else t.get(k, "")
               for k in meta]
        w.writerow(row + [t.get(groups[c[:2]], {}).get(c[2:], "") for c in extra])
    (run_dir / "exports").mkdir(exist_ok=True)
    (run_dir / "exports" / "trials.csv").write_text(buf.getvalue())


def _parser() -> argparse.ArgumentParser:
    p = _Parser(prog="boautoresearch")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    def action(name, fn):
        s = sub.add_parser(name)
        s.add_argument("--rationale", default="")
        s.add_argument("--actor", default="orchestrator")
        s.set_defaults(fn=fn, action=True)
        return s

    action("init", cmd_init).add_argument("run_yaml")
    action("smoke", cmd_smoke)
    sub.add_parser("status").set_defaults(fn=cmd_status, action=False)
    sub.add_parser("trials").set_defaults(fn=cmd_trials, action=False)
    return p


def main(argv=None) -> int:
    try:
        a = _parser().parse_args(argv)
        if a.action and not a.rationale.strip():
            raise Refused(f"{a.cmd} is an action: --rationale is required")
        out = a.fn(a)
    except Refused as e:
        print(json.dumps({"refused": True, "reason": str(e)}))
        return 1
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0
