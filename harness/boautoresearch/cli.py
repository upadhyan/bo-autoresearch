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
import statistics
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import calibration
from . import experiment_log as elog

HARNESS_SRC = Path(__file__).resolve().parents[1]  # holds pyproject.toml when run from the plugin
RUN_KEYS = {"objective", "direction", "budget_s", "runner", "reference_fidelity", "python",
            "ladder", "deterministic", "replicates_k"}


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
    ladder = cfg.setdefault("ladder", [])
    if not isinstance(ladder, list) or not all(isinstance(r, dict) and r for r in ladder):
        raise Refused("run.yaml: ladder must be a list of cheaper fidelities, e.g. [{epochs: 2}]")
    if cfg["reference_fidelity"] in ladder or any(ladder.count(r) > 1 for r in ladder):
        raise Refused("run.yaml: ladder rungs must differ from each other and from the reference")
    if not isinstance(cfg.setdefault("deterministic", False), bool):
        raise Refused("run.yaml: deterministic must be true or false")
    k = cfg.setdefault("replicates_k", 5)
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise Refused("run.yaml: replicates_k must be an integer of at least 2")
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
    if _git(root, "status", "--porcelain", "--", ".", ":!.bo-research"):
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
    exclude = root / ".git" / "info" / "exclude"  # keeps .bo-research/ out of the user's status
    exclude.parent.mkdir(exist_ok=True)
    if "/.bo-research/" not in (exclude.read_text() if exclude.exists() else "").split():
        with exclude.open("a") as f:
            f.write("\n/.bo-research/\n")
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
        "python": cfg["python"], "venv_freeze": freeze, "ladder": cfg["ladder"],
        "deterministic": cfg["deterministic"], "replicates_k": cfg["replicates_k"],
    })
    _regenerate(run_dir, con)
    return {"run_dir": str(run_dir), "worktree": str(run_dir / "worktree"),
            "venv": str(run_dir / "venv"), **_status(elog.state(elog.read(con)))}


def _baseline(run_dir: Path) -> dict:
    """The committed lever baselines: the worktree's levers.json."""
    levers_file = run_dir / "worktree" / "levers.json"
    try:
        return json.loads(levers_file.read_text()) if levers_file.exists() else {}
    except json.JSONDecodeError as e:
        raise Refused(f"levers.json is not valid JSON: {e}")


def _trial(run_dir: Path, con, a, kind: str, levers: dict, fidelity: dict, **extra) -> dict:
    """Run one trial, or log `trial_refused` and refuse when the budget can't cover it.

    `a` is the caller's action; None for a trial the harness schedules itself.
    """
    st = elog.state(elog.read(con))
    run, n = st["run"], len(st["trials"]) + 1
    remaining = _status(st)["budget"]["remaining_s"]
    estimate = _cost(st["trials"], fidelity)
    if remaining <= 0 or estimate > remaining:
        elog.append(con, "trial_refused", "harness", {
            "kind": kind, "fidelity": fidelity, "estimated_cost_s": estimate,
            "remaining_s": remaining, **extra})
        _regenerate(run_dir, con)
        raise Refused(f"budget: a {kind} trial at {json.dumps(fidelity)} is estimated at "
                      f"{estimate:.1f}s but {remaining:.1f}s of the budget remain")
    worktree = run_dir / "worktree"
    trial = {"trial": n, "kind": kind, "levers": levers, "fidelity": fidelity,
             "seed": random.Random(f"{run['run_id']}/{n}").getrandbits(31),
             "commit": _git(worktree, "rev-parse", "HEAD"),
             "artifact_dir": f"artifacts/trial-{n}", **extra}
    elog.append(con, "trial_started", *((a.actor, {"rationale": a.rationale, **trial}) if a
                                        else ("harness", trial)))
    outcome = _run_trial(run_dir, run["runner"], trial)
    elog.append(con, "trial_" + outcome.pop("status"), "harness", {"trial": n, **outcome})
    _regenerate(run_dir, con)
    return elog.state(elog.read(con))["trials"][-1]


def _cost(trials: list[dict], fidelity: dict) -> float:
    """Mean wall-clock of the completed trials at this fidelity; 0 while unknown."""
    done = [t["wall_clock_s"] for t in trials if t["fidelity"] == fidelity and "wall_clock_s" in t]
    return statistics.fmean(done) if done else 0.0


def cmd_smoke(a) -> dict:
    """One trial with every lever at its committed baseline, at the reference fidelity."""
    run_dir, con = _open_run()
    run = elog.state(elog.read(con))["run"]
    return {"trial": _trial(run_dir, con, a, "smoke", _baseline(run_dir), run["reference_fidelity"])}


def cmd_round_run(a) -> dict:
    run_dir, con = _open_run()
    st = elog.state(elog.read(con))
    if st["r0_complete"]:
        raise Refused("round-run for R1 is refused: no hypothesis is registered")
    return _run_r0(run_dir, con, a, st)


def _run_r0(run_dir: Path, con, a, st: dict) -> dict:
    """The calibration round: smoke, baseline replicates (σ), then ladder calibration."""
    run = st["run"]
    ref, ladder, k = run["reference_fidelity"], run["ladder"], run["replicates_k"]
    baseline = _baseline(run_dir)
    elog.append(con, "round_started", a.actor, {
        "rationale": a.rationale, "round": 0, "fidelity": ref, "pid": os.getpid(),
        "commit": _git(run_dir / "worktree", "rev-parse", "HEAD")})

    def end(trigger, reason=None):
        elog.append(con, "round_ended", "harness", {"round": 0, "trigger": trigger})
        _regenerate(run_dir, con)
        if reason:
            raise Refused(reason)

    try:
        if not any(t.get("kind") == "smoke" and t["status"] == "finished" for t in st["trials"]):
            smoke = _trial(run_dir, con, None, "smoke", baseline, ref, round=0)
            if smoke["status"] != "finished":
                end("smoke_failed", f"the smoke trial failed: {smoke['reason']}: {smoke['error']}")
        rows = []
        for f in (ref, *ladder):
            done = [] if run["deterministic"] else [
                _trial(run_dir, con, None, "baseline", baseline, f, round=0) for _ in range(k)]
            values = [t["objective"] for t in done if t["status"] == "finished"]
            if not run["deterministic"] and len(values) < 2:
                end("failed", f"fewer than 2 baseline replicates finished at {json.dumps(f)}")
            rows.append({"fidelity": f, "n": len(values), "trials": [t["trial"] for t in done],
                         "sigma": calibration.sigma(values) if values else 0.0,
                         "mean": statistics.fmean(values) if values else None,
                         "cost_s": _cost(elog.state(elog.read(con))["trials"], f)})
        elog.append(con, "noise_estimate", "harness", {
            "round": 0, "sigma": rows[0]["sigma"], "deterministic": run["deterministic"],
            "replication": any(r["sigma"] > 0 for r in rows), "rungs": rows})
        out = {"round": 0, "noise_estimate": elog.state(elog.read(con))["noise"]}
        if ladder:
            out["fidelity_calibration"] = _calibrate_ladder(run_dir, con, rows)
    except Refused:
        if elog.state(elog.read(con))["rounds"].get(0, {}).get("ended"):
            raise
        end("budget_spent")
        raise
    end("calibrated")
    return {**out, **_status(elog.state(elog.read(con)))}


def _calibrate_ladder(run_dir: Path, con, noise_rows: list[dict]) -> dict:
    run = elog.state(elog.read(con))["run"]
    cfgs = calibration.configs(_baseline(run_dir), random.Random(f"{run['run_id']}/calibration"))
    rows = [{"fidelity": r["fidelity"], "sigma": r["sigma"], "objectives": [], "trials": []}
            for r in noise_rows]
    for i, cfg in enumerate(cfgs):
        for r in rows:
            t = _trial(run_dir, con, None, "calibration", cfg, r["fidelity"], round=0, config=i)
            r["trials"].append(t["trial"])
            r["objectives"].append(t.get("objective"))
    trials = elog.state(elog.read(con))["trials"]
    for r in rows:
        r["cost_s"] = _cost(trials, r["fidelity"])
    chosen, fallback = calibration.choose(rows[0], rows[1:])
    elog.append(con, "fidelity_calibration", "harness", {
        "round": 0, "configs": cfgs, "reference": rows[0], "rungs": rows[1:],
        "chosen": chosen, "fallback": fallback})
    return elog.state(elog.read(con))["calibration"]


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
        if isinstance(result.get("objective"), float) and math.isfinite(result["objective"]):
            return {"status": "finished", **measured, **result}
        reason = "degenerate trial: objective is missing or non-finite"
    stderr = [ln for ln in (art / "stderr.txt").read_text().splitlines() if ln.strip()]
    return {"status": "failed", **measured, "reason": reason, "error": stderr[-1] if stderr else ""}


def _status(st: dict) -> dict:
    run, trials = st["run"], st["trials"]
    spent = sum(t["wall_clock_s"] for t in trials if t["status"] != "running")
    count = {s: sum(t["status"] == s for t in trials) for s in ("running", "finished", "failed")}
    return {"run_id": run["run_id"], "objective": run["objective"], "direction": run["direction"],
            "budget": {"total_s": run["budget_s"], "spent_s": spent,
                       "remaining_s": run["budget_s"] - spent},
            "trials": {"total": len(trials), **count},
            "r0_complete": st["r0_complete"], "fidelity": st["fidelity"],
            "sigma": st["noise"]["sigma"] if st["noise"] else None,
            "replication": st["noise"]["replication"] if st["noise"] else None}


def cmd_accept_proxy(a) -> dict:
    """The user's explicit acceptance of a ladder rung that failed calibration."""
    run_dir, con = _open_run()
    st = elog.state(elog.read(con))
    try:
        fidelity = json.loads(a.fidelity)
    except json.JSONDecodeError as e:
        raise Refused(f"--fidelity must be JSON, e.g. '{{\"epochs\": 2}}': {e}")
    if not st["r0_complete"] or not st["calibration"]:
        raise Refused("accept-proxy needs a completed ladder calibration in R0")
    rung = next((r for r in st["calibration"]["rungs"] if r["fidelity"] == fidelity), None)
    if rung is None:
        raise Refused(f"{a.fidelity} is not a rung of the ladder")
    if rung["passed"]:
        raise Refused(f"{a.fidelity} passed calibration: it is validated, not an unvalidated proxy")
    elog.append(con, "proxy_accepted_unvalidated", a.actor,
                {"rationale": a.rationale, "fidelity": fidelity, "rho": rung["rho"],
                 "spread": rung["spread"], "sigma": rung["sigma"]})
    _regenerate(run_dir, con)
    return _status(elog.state(elog.read(con)))


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
    meta = ["trial", "kind", "status", "objective", "seed", "fidelity", "commit", "wall_clock_s",
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
    action("round-run", cmd_round_run)
    action("accept-proxy", cmd_accept_proxy).add_argument("--fidelity", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status, action=False)
    sub.add_parser("trials").set_defaults(fn=cmd_trials, action=False)
    return p


def main(argv=None) -> int:
    try:
        a = _parser().parse_args(argv)
        if a.action and not a.rationale.strip():
            raise Refused(f"{a.cmd} is an action: --rationale is required")
        if a.action and a.actor.strip() in ("", "harness"):
            raise Refused("--actor must name the caller; 'harness' is reserved for the harness")
        out = a.fn(a)
    except Refused as e:
        print(json.dumps({"refused": True, "reason": str(e)}))
        return 1
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0
