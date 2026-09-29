"""Agent-side CLI. Every response is JSON; refusals are {"refused": true, "reason": ...}.

Probes (status, trials) are free and never logged. Actions (init, smoke) need a
--rationale and log their actor.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

from . import calibration, checks, directives, hypotheses, records, reports
from . import study as bo_study
from . import experiment_log as elog

HARNESS_SRC = Path(__file__).resolve().parents[1]  # holds pyproject.toml when run from the plugin
RUN_KEYS = {"objective", "direction", "budget_s", "runner", "reference_fidelity", "python",
            "ladder", "deterministic", "replicates_k", "delta", "seed", "checkpoint", "target",
            "max_trials", "seeds", "generation", "fixtures", "brief", "directives", "protected_paths", "go"}
CONFIRMATIONS = 2  # replicates of every new incumbent before it counts as the best
ROUND_CAP = 0.25  # a round may spend this share of the budget remaining at its start
REPLICATE_SHARE, ESCALATED_SHARE = 0.1, 0.3  # the replicate floor, raised for a stuck hypothesis
STARVED_AFTER = 3  # BO rounds a registered hypothesis waits unselected before the user is told


class Refused(Exception):
    pass


class TrialRefused(Refused):
    """No trial fits: the budget is spent, or (reason max_trials) the trial ceiling is reached."""
    def __init__(self, message: str, reason: str = "budget_spent"):
        super().__init__(message)
        self.reason = reason


class ProhibitedRefused(Refused):
    """The trial's resolved config breaks a prohibited directive's predicate: it never runs."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refused(message)


def _git(cwd: Path, *args: str, env: dict | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def _repo_root() -> Path:
    """The user's main checkout, also when called from inside the run's worktree."""
    return Path(_git(Path.cwd(), "rev-parse", "--path-format=absolute", "--git-common-dir")).parent


def _open_run(recover: bool = True) -> tuple[Path, sqlite3.Connection]:
    """The latest run. Actions first settle what a crashed harness process left running."""
    runs = sorted((_repo_root() / ".bo-research").glob("*/log.db"))
    if not runs:
        raise Refused("no run: start one with `boautoresearch init run.yaml`")
    # ponytail: latest run wins; add --run when runs coexist
    run_dir, con = runs[-1].parent, elog.connect(runs[-1])
    if recover:
        _recover(run_dir, con)
    return run_dir, con


def _alive(pid: int | None) -> bool:
    # ponytail: a recycled pid reads as alive; record the process start time if that bites
    if pid is None:  # logged before pids were: nothing can still be running it
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _recover(run_dir: Path, con) -> None:
    """Trials and rounds whose harness process died: abandoned (charged up to the last heartbeat,
    never retried) and interrupted."""
    st = elog.load(con)
    dead = [t for t in st["trials"] if t["status"] == "running" and not _alive(t.get("pid"))]
    for t in dead:
        if t.get("runner_pid"):  # the orphaned runner (its own process group) would run on uncharged
            try:
                os.killpg(t["runner_pid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        elog.append(con, "trial_abandoned", "harness", {
            "trial": t["trial"], "wall_clock_s": t.get("heartbeat_s", 0.0),
            "reason": f"the harness process {t.get('pid')} died while the trial ran"})
    crashed = [n for n, r in st["rounds"].items() if r["ended"] is None and not _alive(r.get("pid"))]
    for n in crashed:
        elog.append(con, "round_ended", "harness", {"round": n, "trigger": "interrupted"})
    if dead or crashed:
        _regenerate(run_dir, con)


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
    delta = cfg.setdefault("delta", None)
    if delta is not None and not _positive(delta):
        raise Refused("run.yaml: delta must be a positive number in objective units")
    seed = cfg.setdefault("seed", None)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise Refused("run.yaml: seed must be an integer")
    if not isinstance(cfg.setdefault("go", False), bool):
        raise Refused("run.yaml: go must be true (headless: no interviews, straight into the loop) or false")
    if not isinstance(cfg.setdefault("checkpoint", False), bool):
        raise Refused("run.yaml: checkpoint must be true (pause at each round boundary) or false")
    target = cfg.setdefault("target", None)
    if target is not None and (isinstance(target, bool) or not isinstance(target, (int, float))
                               or not math.isfinite(target)):
        raise Refused("run.yaml: target must be an objective value")
    cap = cfg.setdefault("max_trials", None)
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise Refused("run.yaml: max_trials must be a positive whole number of trials")
    seeds = cfg.setdefault("seeds", [])
    if (not isinstance(seeds, list) or len(seeds) > 5
            or not all(isinstance(s, dict) and s and all(isinstance(k, str) for k in s) for s in seeds)):
        raise Refused("run.yaml: seeds must list at most 5 configs, each mapping lever names to values")
    generation, fixtures = cfg.setdefault("generation", "llm"), cfg.setdefault("fixtures", None)
    if generation not in ("llm", "scripted"):
        raise Refused("run.yaml: generation must be llm (generator subagents) or scripted (fixture passes)")
    if generation == "scripted":
        if not isinstance(fixtures, str) or not (path.parent / fixtures).is_dir():
            raise Refused("run.yaml: generation: scripted needs fixtures: a directory of pass1.yaml, "
                          "pass2.yaml, ...")
        cfg["fixtures"] = str((path.parent / fixtures).resolve())
    elif fixtures is not None:
        raise Refused("run.yaml: fixtures are served only with generation: scripted")
    try:
        cfg["registry"] = directives.validate({k: cfg.pop(k, None) for k in directives.REGISTRY_KEYS})
    except ValueError as e:
        raise Refused(f"run.yaml: {e}")
    cfg["python"] = str(cfg.get("python") or sys.executable)
    return cfg


def _rng(run: dict, *path) -> random.Random:
    """The run's seeded stream for `path`: run.yaml `seed` fixes it, else it hangs off the run id."""
    root = run["run_id"] if run.get("seed") is None else run["seed"]
    return random.Random("/".join(map(str, (root, *path))))


def _positive(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def _freeze(python: str) -> list[str]:
    # --exclude-editable: an editable install of the project would import the user's checkout
    uv = shutil.which("uv")
    cmd = [uv, "pip", "freeze", "--python", python] if uv else [python, "-m", "pip", "freeze"]
    p = subprocess.run([*cmd, "--exclude-editable"], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"cannot freeze the project environment with {python}: {p.stderr.strip()}")
    # pip freeze leaves out the packaging tools; uv's doesn't
    return [r for r in p.stdout.split()
            if re.split(r"[=<>@ ]", r)[0].lower() not in ("pip", "setuptools", "wheel", "distribute")]


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
        (run_dir / "venv-freeze.txt").write_text("".join(f"{r}\n" for r in freeze))
        reqs: list = ["-r", run_dir / "venv-freeze.txt", HARNESS_SRC]
        if uv := shutil.which("uv"):  # uv's cache keeps inits fast with torch in the deps
            subprocess.run([uv, "venv", "-q", "--python", cfg["python"], run_dir / "venv"],
                           check=True, capture_output=True, text=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", venv_python, *reqs],
                           check=True, capture_output=True, text=True)
        else:
            subprocess.run([cfg["python"], "-m", "venv", run_dir / "venv"], check=True,
                           capture_output=True, text=True)
            subprocess.run([venv_python, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                            *reqs], check=True, capture_output=True, text=True)
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
        "delta": cfg["delta"], "seed": cfg["seed"], "checkpoint": cfg["checkpoint"],
        "target": cfg["target"], "max_trials": cfg["max_trials"], "seeds": cfg["seeds"],
        "generation": cfg["generation"], "fixtures": cfg["fixtures"], "go": cfg["go"],
    })
    # the directive registry, protected paths and brief, owned by the harness from here on
    elog.append(con, "registry_revised", "harness", {
        "version": 1, **cfg["registry"],
        "manifest": _manifest(run_dir, cfg["runner"], cfg["registry"]["protected_paths"])})
    _regenerate(run_dir, con)
    return {"run_dir": str(run_dir), "worktree": str(run_dir / "worktree"),
            "venv": str(run_dir / "venv"), **_status(elog.load(con))}


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
    st = elog.load(con)
    run, n = st["run"], len(st["trials"]) + 1
    levers, masking = _mask(st, levers)
    if d := directives.violated(st["registry"]["directives"], levers, elog.lever_paths(st["hypotheses"])):
        elog.append(con, "prohibited_check_refused", "harness", {
            "kind": kind, "levers": levers, "fidelity": fidelity, "directive": d["id"], **extra})
        _regenerate(run_dir, con)
        raise ProhibitedRefused(f"a {kind} trial at {json.dumps(levers, sort_keys=True)} breaks prohibited "
                                f"directive {d['id']} ({d['predicate']}): it never runs")
    remaining = _remaining(st)
    estimate = _cost(st["trials"], fidelity)
    if run["max_trials"] is not None and n > run["max_trials"]:
        elog.append(con, "trial_refused", "harness", {"kind": kind, "fidelity": fidelity,
                                                      "max_trials": run["max_trials"], **extra})
        _regenerate(run_dir, con)
        raise TrialRefused(f"the run's ceiling of {run['max_trials']} trials is reached", "max_trials")
    if remaining <= 0 or estimate > remaining:
        elog.append(con, "trial_refused", "harness", {
            "kind": kind, "fidelity": fidelity, "estimated_cost_s": estimate,
            "remaining_s": remaining, **extra})
        _regenerate(run_dir, con)
        raise TrialRefused(f"budget: a {kind} trial at {json.dumps(fidelity)} is estimated at "
                      f"{estimate:.1f}s but {remaining:.1f}s of the budget remain")
    worktree = run_dir / "worktree"
    trial = {"trial": n, "kind": kind, "levers": levers, "fidelity": fidelity,
             "seed": _rng(run, n).getrandbits(31), "epoch": st["epoch"]["epoch"],
             "commit": _git(worktree, "rev-parse", "HEAD"),
             "artifact_dir": f"artifacts/trial-{n}", **masking, **extra}
    started = {**trial, "pid": os.getpid()}  # a dead pid on a running trial means it was abandoned
    if a:
        elog.append(con, "trial_started", a.actor, {"rationale": a.rationale, **started})
    else:
        elog.append(con, "trial_started", "harness", started)
    outcome = _run_trial(run_dir, run["runner"], trial)
    elog.append(con, "trial_" + outcome.pop("status"), "harness", {"trial": n, **outcome})
    _regenerate(run_dir, con)
    return elog.load(con)["trials"][-1]


def _mask(st: dict, sampled: dict) -> tuple[dict, dict]:
    """Masking: while any lever of a masking hypothesis is away from its baseline (its mechanism is on),
    the masked hypothesis's levers run at their baselines. -> (the effective levers, what the trial
    logs beside them: the masked levers' `sampled` values and the hypotheses `masked` in it)."""
    # ponytail: judged on the sampled values, so a masker that is itself masked still masks (chains);
    # iterate to a fixed point on the effective values if chained masking is ever declared
    hyps, levers = st["hypotheses"].values(), dict(sampled)
    logged: dict = {}
    masked: list[str] = []
    for m in _masking(st):
        masked_levers = {n: lv["baseline"] for n, lv in st["hypotheses"][m["masked"]]["spec"]["levers"].items()
                if n in sampled}
        masker_levers = {n: lv["baseline"] for x in hyps if f"H{x['number']}" == m["by"]
                  for n, lv in x["spec"]["levers"].items()}
        logged.update({n: sampled[n] for n in masked_levers})
        if masked_levers and any(n in sampled and sampled[n] != b for n, b in masker_levers.items()):
            levers.update(masked_levers)
            masked.append(m["masked"])
    return levers, {"sampled": logged, "masked": masked} if logged else {}


def _cost(trials: list[dict], fidelity: dict) -> float:
    """Mean wall-clock of the completed trials at this fidelity; 0 while unknown."""
    done = [t["wall_clock_s"] for t in trials
            if t["fidelity"] == fidelity and t["status"] in ("finished", "failed")]
    return statistics.fmean(done) if done else 0.0


def cmd_smoke(a) -> dict:
    """No hypothesis: one trial with every lever at its committed baseline, at the reference fidelity.

    `smoke <H>`: the uncommitted lever code of H, one trial at baseline and one at a random
    in-range point, at the cheapest rung. A pass is what commit-lever needs.
    """
    run_dir, con = _open_run()
    st = elog.load(con)
    if a.hypothesis is None:
        return {"trial": _trial(run_dir, con, a, "smoke", _baseline(run_dir),
                                st["run"]["reference_fidelity"])}
    h = _uncommitted(st, a.hypothesis)
    tree, fidelity = _worktree_tree(run_dir / "worktree"), _cheapest(st)
    stamp = _worktree_stamp(run_dir / "worktree")
    levers = h["spec"]["levers"]
    base = {**_baseline(run_dir), **{n: lv["baseline"] for n, lv in levers.items()}}
    rng = _rng(st["run"], "smoke", h["id"], len(st["trials"]))
    # ponytail: 100 draws for a point inside the allowed region (the trial check refuses one outside);
    # sample inside the region directly if a narrow one ever defeats it
    for _ in range(100):
        rand = {**base, **hypotheses.random_point(levers, rng)}
        if not _prohibited(rand, st):
            break
    trials = []
    for point, cfg in (("baseline", base), ("random", rand)):
        trials.append(_trial(run_dir, con, a, "smoke", cfg, fidelity,
                             hypothesis=h["id"], point=point, tree=tree))
        if trials[-1]["status"] != "finished":
            break
    passed = len(trials) == 2 and trials[-1]["status"] == "finished"
    reason = None if passed else f"trial {trials[-1]['trial']} {trials[-1]['status']}"
    if _worktree_stamp(run_dir / "worktree") != stamp:  # would dirty rounds, or be committed as code
        passed, reason = False, "the trials wrote into the worktree: write outputs elsewhere"
    elog.append(con, "lever_smoke", "harness", {"id": h["id"], "tree": tree, "fidelity": fidelity,
                                                "trials": [t["trial"] for t in trials], "passed": passed,
                                                "reason": reason})
    _regenerate(run_dir, con)
    return {"hypothesis": h["id"], "passed": passed, "reason": reason, "trials": trials}


def _hypothesis(st: dict, ref: str) -> dict:
    """`H<n>.v<k>`, or `H<n>` for its latest version."""
    hyps = st["hypotheses"]
    if ref in hyps:
        return hyps[ref]
    versions = [h for h in hyps.values() if f"H{h['number']}" == ref]
    if not versions:
        raise Refused(f"no hypothesis {ref}")
    return max(versions, key=lambda h: h["version"])


def _uncommitted(st: dict, ref: str) -> dict:
    h = _hypothesis(st, ref)
    if h["status"] != "registered":
        raise Refused(f"{h['id']} is {h['status']}: register it before writing its lever code")
    if h["commit"]:
        raise Refused(f"{h['id']}'s lever code is already committed as {h['commit']}")
    return h


def _cheapest(st: dict) -> dict:
    """The fidelity with the lowest measured mean cost; the ladder's costs come from R0."""
    run = st["run"]
    if run["ladder"] and not st["r0_complete"]:
        raise Refused("smoke <H> runs at the cheapest rung, which the calibration round measures: "
                      "run `round-run` first")
    return min((run["reference_fidelity"], *run["ladder"]), key=lambda f: _cost(st["trials"], f))


def _worktree_tree(worktree: Path) -> str:
    """Hash of the worktree's full contents (untracked files included, ignored ones not)."""
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(d) / "index")}
        _git(worktree, "add", "-A", env=env)
        return _git(worktree, "write-tree", env=env)


def _worktree_stamp(worktree: Path) -> dict:
    """mtime of every non-ignored worktree file: a trial rewriting one, even identically, shows."""
    files = _git(worktree, "ls-files", "-z", "-co", "--exclude-standard").split("\0")
    return {f: (worktree / f).stat().st_mtime_ns for f in files if f and (worktree / f).exists()}


def _manifest(run_dir: Path, runner: str, globs: list[str]) -> dict:
    """sha256 of every worktree file the protected globs match (untracked included, ignored not)."""
    worktree = run_dir / "worktree"
    files = [f for f in _git(worktree, "ls-files", "-z", "-co", "--exclude-standard").split("\0")
             if f and (worktree / f).is_file()]
    return directives.manifest(worktree, directives.protected(worktree, files, runner, globs))


def _check_manifest(run_dir: Path, st: dict) -> None:
    """Before every round and every commit: the protected paths are as the harness last recorded
    them, whatever wrote to them (an Edit, a shell redirect, `python -c`, a commit)."""
    if (recorded := st["manifest"]) is None:
        return
    now = _manifest(run_dir, st["run"]["runner"], st["registry"]["protected_paths"])
    if changed := sorted(f for f in set(recorded) | set(now) if recorded.get(f) != now.get(f)):
        raise Refused(f"protected paths changed since the harness recorded their hashes: {changed}; "
                      "restore them (the research may never change how the score is measured)")


def _read_json(path: str, what: str, validate) -> dict:
    try:
        body = json.loads(sys.stdin.read() if path == "-" else Path(path).read_text())
        validate(body)
    except OSError as e:
        raise Refused(f"cannot read {path}: {e}")
    except ValueError as e:  # JSONDecodeError included
        raise Refused(f"{what}: {e}")
    return body


def _propose(run_dir: Path, con, spec: dict, actor: str, rationale: str) -> str:
    """Log a validated spec as the next hypothesis number's v1."""
    st = elog.load(con)
    # past every number in use, the project's own levers.json included (e.g. an existing H1.scale)
    taken = [h["number"] for h in st["hypotheses"].values()]
    taken += [int(m[1]) for k in _baseline(run_dir) if (m := re.match(r"H(\d+)\.", k))]
    n = max(taken, default=0) + 1
    spec = {**spec, "levers": {f"H{n}.{name}": lv for name, lv in spec["levers"].items()}}
    if "merges" in spec:
        spec["merges"] = {**spec["merges"],
                          "mapping": {f"H{n}.{k}": r for k, r in spec["merges"]["mapping"].items()}}
    hid = f"H{n}.v1"
    elog.append(con, "hypothesis_proposed", actor, {
        "rationale": rationale, "id": hid, "number": n, "version": 1, "spec": spec})
    return hid


def cmd_propose(a) -> dict:
    run_dir, con = _open_run()
    hid = _propose(run_dir, con, _read_json(a.file, "spec", hypotheses.validate), a.actor, a.rationale)
    _regenerate(run_dir, con)
    return {"hypothesis": elog.load(con)["hypotheses"][hid]}


def cmd_record(a) -> dict:
    """The inlet for subagent output: schema- and quote-checked, logged with its actor."""
    run_dir, con = _open_run()
    st = elog.load(con)
    body = _read_json(a.file, a.kind, lambda r: records.validate(a.kind, r, st))
    if a.kind == "review" and st["hypotheses"][body["hypothesis"]]["status"] in ("active", "retained"):
        _between_rounds(st, "a hypothesis in the search is reviewed")  # its verdict may park it
    ids = ([_propose(run_dir, con, s, a.actor, a.rationale) for s in body["hypotheses"]]
           if a.kind == "proposal" else [])
    elog.append(con, "record", a.actor, {"rationale": a.rationale, "kind": a.kind,
                                         "agent_id": a.agent_id, "record": body,
                                         **({"hypotheses": ids} if ids else {})})
    if a.kind == "review":
        _apply_review(con, body)
    _regenerate(run_dir, con)
    return {"recorded": a.kind, "agent_id": a.agent_id, **({"hypotheses": ids} if ids else {})}


def _check_run() -> tuple[Path, sqlite3.Connection, dict] | None:
    """The run the hooks enforce: the latest, until its wrap-up has finished (then, as with no run,
    nothing is enforced)."""
    try:
        run_dir, con = _open_run(recover=False)
    except Refused:
        return None
    st = elog.load(con)  # paths compare resolved (macOS's /var is /private/var)
    return None if st["wrapup_finished"] else (Path(os.path.realpath(run_dir)), con, st)


def _decide(con, check: str, agent: str, subject: str, reason: str | None, **extra) -> dict:
    """A check's answer; every block is logged as `hook_blocked`."""
    if reason is None:
        return {"allow": True, **extra}
    elog.append(con, "hook_blocked", "harness", {"check": check, "agent": agent, "subject": subject,
                                                 "reason": reason})
    return {"allow": False, "reason": reason, **extra}


def _live(st: dict) -> bool:
    """A harness process (a background round-run) is running a round or a trial."""
    return (any(r["ended"] is None and _alive(r.get("pid")) for r in st["rounds"].values())
            or any(t["status"] == "running" and _alive(t.get("pid")) for t in st["trials"]))


def _write_refusal(run_dir: Path, st: dict, path: Path) -> str | None:
    # the worktree freeze is the research loop's: wrap-up (after the run ended) writes the distilled one
    return checks.write_refusal(path, run_dir.parents[1], st["run"]["runner"], st["registry"]["protected_paths"],
                                _live(st) and not st["run_ended"])


def cmd_check_write(a) -> dict:
    """For PreToolUse on Edit, Write and NotebookEdit."""
    if not (run := _check_run()):
        return {"allow": True}
    run_dir, con, st = run
    p = checks.resolve(Path.cwd(), a.path)
    return _decide(con, "write", a.agent, str(p), _write_refusal(run_dir, st, p))


PROBES = {"status", "summary", "trials", "verdict", "next"}  # what read-only agents may run


def cmd_check_bash(a) -> dict:
    """For PreToolUse on Bash; a subagent's `record` comes back as `updated_command` carrying its
    agent id, so SubagentStop's `check recorded` finds it."""
    if not (run := _check_run()):
        return {"allow": True}
    run_dir, con, st = run
    why = checks.bash_refusal(a.command, Path.cwd(), a.agent, run_dir.parent, PROBES,
                              lambda p: _write_refusal(run_dir, st, p))
    new = checks.with_agent_id(a.command, a.agent_id, a.agent) if a.agent_id and not why else None
    return _decide(con, "bash", a.agent, a.command, why, **({"updated_command": new} if new else {}))


def cmd_check_read(a) -> dict:
    """For PreToolUse on Read, Grep and Glob: the orchestrator reads probes, not raw logs."""
    if not (run := _check_run()):
        return {"allow": True}
    run_dir, con, st = run
    p = checks.resolve(Path.cwd(), a.path)
    raw = checks.role(a.agent) == "orchestrator" and checks.raw_read(p, run_dir.parent)
    return _decide(con, "read", a.agent, str(p), checks.RAW_READ if raw else None)


def cmd_check_stop(a) -> dict:
    """For Stop: allowed with no run, while waiting on the user (before the loop starts, at a
    checkpoint or a user pause), while a background round-run is live, and once wrap-up finished."""
    if not (run := _check_run()):
        return {"allow": True}
    run_dir, con, st = run
    if _live(st):
        why = None  # the background round-run wakes the orchestrator when it returns
    elif st["run_ended"]:
        why = (f"the run has ended ({st['run_ended']}) but wrap-up hasn't finished: `boautoresearch wrapup` "
               "(distillation, verification and the report) comes first")
    elif st["paused"] or (not st["run"].get("go") and not any(r >= 1 for r in st["rounds"])):
        why = None  # a checkpoint, a user pause, or the pre-loop interviews: the user answers
    else:
        why = ("the research loop is autonomous: it ends only on the budget, the user, the target or "
               f"exhaustion. Next: {'; '.join(_next(st))}")
    return _decide(con, "stop", "orchestrator", "stop", why)


def cmd_check_recorded(a) -> dict:
    """For SubagentStop (and SubagentHandback in auto mode): has this agent left a valid record?
    Only the roles that record are held; --agent left out holds it."""
    if not (run := _check_run()):
        return {"allow": True, "recorded": False}
    _, con, st = run
    done = any(r["agent_id"] == a.agent_id for r in st["records"])
    held = a.agent is None or checks.role(a.agent) in checks.RECORDING
    why = None if done or not held else (
        f"no valid record from this agent yet: finish with `boautoresearch record <kind> --file -` (the "
        "harness checks the schema and quotes; fix what it refuses and record again)")
    return _decide(con, "recorded", a.agent or "subagent", a.agent_id, why, recorded=done)


def cmd_next(a) -> dict:
    """The duties outstanding before the next round-run (re-injected after every harness command)."""
    _, con = _open_run(recover=False)
    return {"next": _next(elog.load(con))}


def _apply_review(con, r: dict) -> None:
    """A review's verdict: prune prunes a hypothesis not yet tested (parks one in the search), and
    off-intent parks it. Taste never prunes; stretch and deprioritize only lower its priority."""
    h = elog.load(con)["hypotheses"][r["hypothesis"]]
    if h["status"] not in ("proposed", "registered", "active", "retained"):
        return
    if r["directive_verdict"] == "prune" and h["status"] in records.UNTESTED:
        elog.append(con, "hypothesis_pruned", "harness", {"id": h["id"], "directive": r["directive"],
                                                          "by": "review", "reason": r["rationale"]})
    elif r["directive_verdict"] == "prune" or r["intent"] == "off-intent":
        _park(con, h, (f"prohibited: {r['directive']}: " if r["directive_verdict"] == "prune"
                       else "off-intent: ") + r["rationale"])


def _park(con, h: dict, reason: str) -> None:
    """The harness parks H (a review's verdict, or a revision): a removal like any park."""
    elog.append(con, "hypothesis_parked", "harness", {"id": h["id"], "reason": reason, "from": h["status"]})
    _removal(con, h["id"], "parked", reason)


def _review(st: dict, h: dict) -> dict | None:
    """H's latest registration review since the registry's latest revision (a revision asks for a
    re-review of every hypothesis)."""
    return next((x["record"] for x in reversed(st["records"]) if x["kind"] == "review"
                 and x["record"]["hypothesis"] == h["id"] and x["seq"] > st["registry"]["seq"]), None)


def _review_missing(st: dict) -> list[str]:
    """The hypotheses in the loop (queued or in the search) with no review since the registry's
    latest version: revived versions, and every one after a revision."""
    return [h["id"] for h in st["hypotheses"].values()
            if h["status"] in ("registered", "active", "retained") and _review(st, h) is None]


def _priority(st: dict, h: dict) -> int:
    """H's priority: the orchestrator's (`prioritize`), less the harness's penalty."""
    return h["priority"] - _penalty(st, h)


def _penalty(st: dict, h: dict) -> int:
    """How far the harness lowers H's priority: a stretch intent verdict, and a deprioritize verdict
    or a discouraged directive declared (recomputed when the registry is revised)."""
    r = _review(st, h) or {}
    discouraged = {d["id"] for d in st["registry"]["directives"] if d["severity"] == "discouraged"}
    return (r.get("intent") == "stretch") + (r.get("directive_verdict") == "deprioritize"
                                             or bool(discouraged & set(h["spec"]["directives"])))


def _outside(run_dir: Path, st: dict, h: dict) -> tuple[dict, str] | None:
    """The prohibited directive whose allowed region H's lever box leaves, and why (None: it stays
    inside), checked on a grid with every other lever at its baseline."""
    levers = h["spec"]["levers"]
    base = {**_baseline(run_dir), **{n: lv["baseline"] for n, lv in levers.items()}}
    paths = elog.lever_paths(st["hypotheses"])
    for point in directives.grid(levers):
        if d := directives.violated(st["registry"]["directives"], {**base, **point}, paths):
            return d, (f"{h['id']}'s levers leave the allowed region of prohibited directive {d['id']} "
                       f"({d['statement']}: {d['predicate']}) at {json.dumps(point, sort_keys=True)}")
    return None


def cmd_register(a) -> dict:
    """Freeze a proposed spec, once the registration reviewer has reviewed it: from here on only its
    lever code may be written."""
    run_dir, con = _open_run()
    st = elog.load(con)
    h = _hypothesis(st, a.hypothesis)
    if h["status"] != "proposed":
        raise Refused(f"{h['id']} is already {h['status']}")
    if (r := _review(st, h)) is None:
        raise Refused(f"register {h['id']} needs a `review` record from the registration reviewer "
                      f"(directive, intent and conflict verdicts) against registry version "
                      f"{st['registry']['version']}")
    if out := _outside(run_dir, st, h):
        raise Refused(f"{out[1]}: propose a version whose ranges and options stay inside it")
    moving = _merged_sources(st, h)
    if why := _clash(st, h) or _unresolved(st, h, r):
        raise Refused(why)
    lowered = _penalty(st, h)
    elog.append(con, "hypothesis_registered", a.actor, {
        "rationale": a.rationale, "id": h["id"], "review": {k: r[k] for k in ("directive_verdict", "intent")},
        # a stretch or deprioritize verdict is tested only with a stated reason: the rationale
        **({"lowered": {"by": lowered, "reason": a.rationale}} if lowered else {})})
    for x in moving:  # carried on by the merged hypothesis: out of the loop, no verdict of their own
        elog.append(con, "hypothesis_merged", "harness", {"id": x["id"], "into": h["id"], "from": x["status"]})
    _regenerate(run_dir, con)
    return {"hypothesis": elog.load(con)["hypotheses"][h["id"]]}


LIVE = ("registered", "active", "retained")
IN_LOOP = ("proposed", *LIVE)


def _numbers(ids) -> set[int]:
    return {int(i[1:].split(".")[0]) for i in ids}  # H<n> or H<n>.v<k>


def _merged_sources(st: dict, h: dict) -> list[dict]:
    """The hypotheses H's merge takes out of the loop (those still in it; removed ones are lineage
    only), refused unless its mapping carries every one of their levers over their whole range."""
    m, hyps = h["spec"].get("merges"), st["hypotheses"]
    if not m:
        return []
    if bad := [i for i in m["from"] if i not in hyps or hyps[i]["number"] == h["number"]]:
        raise Refused(f"merges.from: {bad} are not other hypotheses")
    old = {n: lv for i in m["from"] for n, lv in hyps[i]["spec"]["levers"].items()}
    pairs = [(new, rule) for new, rs in m["mapping"].items() for rule in hypotheses.rules(rs)]
    if bad := [rule["lever"] for _, rule in pairs if rule["lever"] not in old]:
        raise Refused(f"merges.mapping: {bad} are not levers of {m['from']}")
    moving = [hyps[i] for i in m["from"] if hyps[i]["status"] in IN_LOOP]
    for x in moving:
        for n, lv in x["spec"]["levers"].items():
            mine = [(new, rule) for new, rule in pairs if rule["lever"] == n]
            if not mine:
                raise Refused(f"merges.mapping leaves {n} of {x['id']} out: the merged hypothesis covers both")
            for new, rule in mine:
                if why := hypotheses.uncovered(lv, h["spec"]["levers"][new], rule):
                    raise Refused(f"merges.mapping.{new} ({n}): {why}: the merged hypothesis covers both")
    if any(x["status"] in ("active", "retained") for x in moving):
        _between_rounds(st, "a merge takes hypotheses out of the search")
    return moving


def _clash(st: dict, h: dict) -> str | None:
    """A config path belongs to one live hypothesis: the one H's levers would share (unless H merges it)."""
    merging = set(h["spec"].get("merges", {}).get("from", []))
    paths = {lv["path"] for lv in h["spec"]["levers"].values() if "path" in lv}
    for x in st["hypotheses"].values():
        if x["status"] in LIVE and x["number"] != h["number"] and x["id"] not in merging:
            if shared := sorted(paths & {lv["path"] for lv in x["spec"]["levers"].values() if "path" in lv}):
                return (f"config path {', '.join(shared)} belongs to {x['id']}: propose one hypothesis that "
                        f"merges both (`merges` from {x['id']}, its mapping covering both ranges)")
    return None


def _unresolved(st: dict, h: dict, r: dict) -> str | None:
    """A shared-lever or exclusive conflict the review flagged with a live hypothesis and H's spec
    leaves unresolved: shared levers merge; exclusive mechanisms merge (one categorical lever with a
    "none" option), mask (`masked_by`, either way) or never share a round (`exclusive_with`)."""
    hyps = st["hypotheses"]
    if both := [x["id"] for x in hyps.values() if x["status"] in LIVE and _masks(x, h) and _masks(h, x)]:
        return f"{h['id']} and {both[0]} mask each other: masking declares one precedence"
    for i in r["conflict_with"] if r["conflict"] in ("shared-lever", "exclusive") else []:
        x = hyps[i]
        if x["status"] not in LIVE:
            continue
        merged = i in h["spec"].get("merges", {}).get("from", [])
        if r["conflict"] == "shared-lever" and not merged:
            return f"the review finds {h['id']} shares levers with {i}: propose one hypothesis that merges both"
        if r["conflict"] == "exclusive" and not (merged or _masks(h, x) or _masks(x, h) or _exclusive(h, x)):
            return (f"the review finds {h['id']} exclusive with {i}: merge them (one categorical lever with a "
                    "\"none\" option), declare masking (`masked_by`), or `exclusive_with` for code paths that "
                    "can't run together")
    return None


def _masks(masker: dict, masked: dict) -> bool:
    return masker["number"] in _numbers(masked["spec"].get("masked_by", {}))


def _masking(st: dict) -> list[dict]:
    """The registered masking declarations: {masked, by: H<m>, reason}."""
    return [{"masked": h["id"], "by": m, "reason": why} for h in st["hypotheses"].values()
            if h["status"] not in ("proposed", "pruned") for m, why in h["spec"].get("masked_by", {}).items()]


def _rivals(st: dict) -> list[list[str]]:
    """The rival links: pairs each hypothesis's latest review calls rivals. Both are tested (together
    where the cap allows), each with its own verdict."""
    latest = {x["record"]["hypothesis"]: x["record"] for x in st["records"] if x["kind"] == "review"}
    return [list(p) for p in sorted({tuple(sorted((h, i))) for h, r in latest.items()
                                     if r["conflict"] == "rival" for i in r["conflict_with"]})]



def cmd_prioritize(a) -> dict:
    """The orchestrator's priority for H (higher runs first); the harness fills the slots with it
    at the next round boundary."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    h = _hypothesis(st, a.hypothesis)
    if h["status"] in ("rejected", "inconclusive", "merged"):
        raise Refused(f"{h['id']} is {h['status']}: it has left the loop")
    elog.append(con, "hypothesis_prioritized", a.actor,
                {"rationale": a.rationale, "id": h["id"], "priority": a.priority})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


RESUME_HINT = "the user reviews the run, then `checkpoint --resume`"


def cmd_checkpoint(a) -> dict:
    """The user's pause (a running round ends at its next step), or --resume after a pause or a
    checkpoint-mode round boundary."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    if a.revise:
        if a.resume or not st["paused"]:
            raise Refused("the registry is revised only while the run is paused (a checkpoint or a user "
                          "pause): `checkpoint`, then `checkpoint --revise <file>`, then `checkpoint --resume`")
        return _revise(run_dir, con, a, st)
    if a.resume:
        if not st["paused"]:
            raise Refused("the run is not paused: nothing to resume")
        elog.append(con, "user_resume", a.actor, {"rationale": a.rationale})
    else:
        if st["paused"]:
            raise Refused("the run is already paused")
        elog.append(con, "user_pause", a.actor, {"rationale": a.rationale})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def _revise(run_dir: Path, con, a, st: dict) -> dict:
    """A new registry version (brief, directives and protected paths; keys left out stay). It takes
    effect before the next round: a prohibition breaking a hypothesis's lever box prunes it
    (registered) or parks it (in the search), violating trials leave the warm start, a pruned
    hypothesis whose directive changed goes back to proposed, compat flags and priorities are
    recomputed, and every hypothesis needs a fresh review."""
    _between_rounds(st, "the registry is revised")
    _check_manifest(run_dir, st)  # else re-hashing would launder a tampered file
    try:
        body = yaml.safe_load(Path(a.revise).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise Refused(f"cannot read {a.revise}: {e}")
    if not isinstance(body, dict) or not body or set(body) - directives.REGISTRY_KEYS:
        raise Refused(f"a revision states some of {sorted(directives.REGISTRY_KEYS)}")
    old = st["registry"]
    try:
        new = directives.validate({k: body.get(k, old[k]) for k in directives.REGISTRY_KEYS})
    except ValueError as e:
        raise Refused(f"revision: {e}")
    version = old["version"] + 1
    elog.append(con, "registry_revised", a.actor, {
        "rationale": a.rationale, "version": version, **new,
        "manifest": _manifest(run_dir, st["run"]["runner"], new["protected_paths"])})
    before = {d["id"]: d for d in old["directives"]}
    after = {d["id"]: d for d in new["directives"]}
    for h in [h for h in st["hypotheses"].values() if h["status"] == "pruned"]:
        if before.get(h["pruned_by"]) != after.get(h["pruned_by"]):  # relaxed (or reworded): re-review
            elog.append(con, "hypothesis_unpruned", "harness",
                        {"id": h["id"], "directive": h["pruned_by"], "version": version})
    st = elog.load(con)
    for h in [h for h in st["hypotheses"].values() if h["status"] in ("registered", "active", "retained")]:
        if (out := _outside(run_dir, st, h)) is None:
            continue
        d, why = out
        if h["status"] == "registered":
            elog.append(con, "hypothesis_pruned", "harness",
                        {"id": h["id"], "directive": d["id"], "by": "revision", "reason": why})
        else:
            _park(con, h, f"prohibited: {d['id']} (directives version {version}): {why}")
    _regenerate(run_dir, con)
    return {"registry": version, **_status(elog.load(con))}


def cmd_park(a) -> dict:
    """Withdraw H without a verdict (never evidence against it); its levers run at baseline."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    _between_rounds(st, "parking happens")
    h = _hypothesis(st, a.hypothesis)
    if h["status"] not in IN_LOOP:
        raise Refused(f"{h['id']} is {h['status']}: only a hypothesis still in the loop can be parked")
    if not a.reason.strip():
        raise Refused("park needs a --reason")
    elog.append(con, "hypothesis_parked", a.actor,
                {"rationale": a.rationale, "id": h["id"], "reason": a.reason, "from": h["status"]})
    _removal(con, h["id"], "parked", a.reason)  # parking is a removal: it owes an interplay review
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def cmd_unpark(a) -> dict:
    """A parked hypothesis returns unchanged (same version and reject conditions) to the queue; it
    rejoins the search at a round boundary, its verdict schedule restarting."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    h = _hypothesis(st, a.hypothesis)
    if h["status"] != "parked":
        raise Refused(f"{h['id']} is {h['status']}, not parked")
    if (latest := _hypothesis(st, f"H{h['number']}"))["id"] != h["id"]:
        raise Refused(f"{h['id']} was revived as {latest['id']}: that version carries it on")
    if not a.reason.strip():
        raise Refused("unpark needs a --reason")
    # the user's call overrides a taste verdict (off-intent), never a prohibition
    if out := _outside(run_dir, st, h):
        raise Refused(f"{out[1]}: it can't return while the prohibition stands")
    if why := _clash(st, h):
        raise Refused(f"{why}; it can't return while another hypothesis owns its config path")
    if (r := _review(st, h)) and (why := _unresolved(st, h, r)):
        raise Refused(f"{why}; it can't return until the conflict is resolved")
    if (r := _review(st, h)) and r["directive_verdict"] == "prune":
        raise Refused(f"{h['id']}'s review prunes it under {r['directive']}: it returns only after the "
                      "directives are revised and a new review allows it")
    # it keeps its review unless the registry was revised since (then round-run asks for a new one)
    elog.append(con, "hypothesis_unparked", a.actor, {"rationale": a.rationale, "id": h["id"], "reason": a.reason})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def cmd_stop(a) -> dict:
    """The user ends the run; a running round ends at its next step (running trials finish)."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    elog.append(con, "run_ended", a.actor, {"rationale": a.rationale, "reason": "user_stop"})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def cmd_commit_lever(a) -> dict:
    """The harness-made commit of H's lever code, after its smoke passed on these exact contents."""
    run_dir, con = _open_run()
    st = elog.load(con)
    h = _uncommitted(st, a.hypothesis)
    worktree = run_dir / "worktree"
    if not (h["smoke"] and h["smoke"]["passed"]):
        raise Refused(f"commit-lever {h['id']} needs a passing `smoke {h['id']}` first")
    tree = _worktree_tree(worktree)
    if tree != h["smoke"]["tree"]:
        raise Refused(f"the worktree changed since {h['id']}'s passing smoke: smoke it again")
    paths = _changed_paths(worktree, tree)
    _change_allowed(run_dir, st, paths, tree)

    def show(rev_path: str):
        p = subprocess.run(["git", "show", rev_path], cwd=worktree, capture_output=True, text=True)
        return p.stdout if p.returncode == 0 else None

    changed = {path: (show(f"HEAD:{path}"), show(f"{tree}:{path}"))
               for path in paths if path.endswith(".py")}
    coded = [x for x in st["hypotheses"].values() if x["commit"]] + [h]
    try:
        hypotheses.check_lever_code(changed, list(h["spec"]["levers"]),
                                    {n for x in coded for n in x["spec"]["levers"]}
                                    | set(_baseline(run_dir)))  # the project's own levers too
    except ValueError as e:
        raise Refused(f"lever code: {e}")
    # the levers.json H was smoked with (the project's own levers kept), plus H's baselines
    baselines = {**_baseline(run_dir), **{n: lv["baseline"] for n, lv in h["spec"]["levers"].items()}}
    (worktree / "levers.json").write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
    sha = _commit(worktree, f"{h['id']}: {h['spec']['title']}")
    elog.append(con, "lever_committed", a.actor,
                {"rationale": a.rationale, "id": h["id"], "commit": sha, "levers": baselines,
                 # the harness rewrote levers.json, a protected path: its hash moves on with it
                 "manifest": _manifest(run_dir, st["run"]["runner"], st["registry"]["protected_paths"])})
    # new lever code at its baseline should pass trivially: the check catches a baseline that
    # isn't really a no-op
    return {"hypothesis": h["id"], "commit": sha, **_after_change(run_dir, con, sha, "lever_committed")}


def _changed_paths(worktree: Path, tree: str) -> list[str]:
    """The files that differ between HEAD and the worktree tree `tree`."""
    return [p for p in _git(worktree, "diff-tree", "-r", "-z", "--name-only", "HEAD", tree).split("\0") if p]


def _commit(worktree: Path, message: str) -> str:
    """A harness-made commit of everything in the worktree."""
    _git(worktree, "add", "-A")
    _git(worktree, "-c", "user.name=boautoresearch", "-c", "user.email=harness@boautoresearch",
         "commit", "-q", "--no-verify", "-m", message)
    return _git(worktree, "rev-parse", "HEAD")


def _not_ended(st: dict) -> None:
    if st["run_ended"]:
        raise Refused(f"the run has ended ({st['run_ended']})")


def _between_rounds(st: dict, what: str) -> None:
    if any(r["ended"] is None for r in st["rounds"].values()):  # _open_run settled the dead ones
        raise Refused(f"a round is running: {what} only between rounds (a round runs on one commit "
                      "and one search space); do it once the round has ended")


def _change_allowed(run_dir: Path, st: dict, paths: list[str], tree: str | None = None) -> None:
    """Code changes only between rounds, never on protected paths, with the protected-path manifest
    intact, and (the diff to the worktree tree `tree`) with no prohibited directive's forbidden
    pattern in an added line."""
    _between_rounds(st, "code changes")
    _check_manifest(run_dir, st)
    worktree = run_dir / "worktree"
    if hit := directives.protected(worktree, paths, st["run"]["runner"], st["registry"]["protected_paths"]):
        raise Refused(f"the change touches protected paths {hit}: the research may never change them")
    if tree is None:
        return
    added, path = [], None
    for ln in _git(worktree, "diff-tree", "-p", "-r", "--no-color", "HEAD", tree).splitlines():
        if ln.startswith("+++ "):
            path = ln[len("+++ b/"):]
        elif ln.startswith("+") and path:
            added.append((path, ln[1:]))
    if found := directives.forbidden(st["registry"]["directives"], added):
        d, where, line = found
        raise Refused(f"{where} adds `{line.strip()}`, a forbidden pattern of prohibited directive "
                      f"{d['id']} ({d['statement']}): the commit is refused")


def cmd_commit_change(a) -> dict:
    """A change that isn't lever code (a refactor, a crash fix), committed with its reason."""
    run_dir, con = _open_run()
    st, worktree = elog.load(con), run_dir / "worktree"
    if not a.reason.strip():
        raise Refused("commit-change needs a --reason")
    tree = _worktree_tree(worktree)
    paths = _changed_paths(worktree, tree)
    if not paths:
        raise Refused("nothing to commit: the worktree has no changes")
    _change_allowed(run_dir, st, paths, tree)
    sha = _commit(worktree, f"commit-change: {a.reason}")
    elog.append(con, "commit_change", a.actor, {"rationale": a.rationale, "reason": a.reason,
                                                "commit": sha, "paths": paths, "breaking": a.breaking})
    return {"commit": sha, **_after_change(run_dir, con, sha, "commit_change", a.breaking)}


FREEZE_FILE = "requirements-freeze.txt"  # the run venv's exact freeze, committed in the worktree


def cmd_add_dependency(a) -> dict:
    """A new requirement, installed into the run venv only (never the user's environment); the
    venv's updated freeze is committed."""
    run_dir, con = _open_run()
    st, worktree = elog.load(con), run_dir / "worktree"
    if not a.requirement.strip() or a.requirement.startswith("-"):
        raise Refused("add-dependency takes one requirement (a name, specifier, path or URL), not options")
    if _git(worktree, "status", "--porcelain"):
        raise Refused("the worktree has uncommitted changes: commit them with commit-change first")
    _change_allowed(run_dir, st, [FREEZE_FILE])
    python = run_dir / "venv" / "bin" / "python"
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "-q", "--python", str(python), a.requirement] if uv else
           [str(python), "-m", "pip", "install", "-q", "--disable-pip-version-check", a.requirement])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:  # ponytail: trusts the installer to leave the venv as it was on failure
        raise Refused(f"installing {a.requirement} into the run venv failed: {p.stderr.strip()}")
    # ponytail: writes its own freeze file, not the project's lock file (uv.lock, poetry.lock);
    # update that one when a project needs it
    freeze = [r for r in _freeze(str(python)) if re.split(r"[=<>@ ]", r)[0].lower() != "boautoresearch"]
    (worktree / FREEZE_FILE).write_text("".join(f"{r}\n" for r in freeze))
    if not _git(worktree, "status", "--porcelain"):
        raise Refused(f"{a.requirement} is already in the run venv: nothing changed")
    sha = _commit(worktree, f"add-dependency: {a.requirement}")
    elog.append(con, "dependency_added", a.actor, {"rationale": a.rationale, "requirement": a.requirement,
                                                   "commit": sha, "freeze": freeze})
    return {"requirement": a.requirement, "commit": sha,
            **_after_change(run_dir, con, sha, "dependency_added")}


def cmd_narrow(a) -> dict:
    """Shrink a searched lever's range on evidence (its latest verdict record's gates passed). The
    hypothesis and its reject conditions stay; the next round searches the new range, and trials
    outside it drop out of the warm start (never clipped)."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _between_rounds(st, "narrowing happens")
    h = _hypothesis(st, a.hypothesis)
    if h["status"] not in ("active", "retained"):
        raise Refused(f"{h['id']} is {h['status']}: only a hypothesis in the search can be narrowed")
    if a.lever not in _group(h):
        raise Refused(f"{a.lever} is not one of {h['id']}'s searched levers {_group(h)}")
    # several levers may be narrowed at one round boundary on the same record, but a later narrowing
    # needs a record on the narrowed ranges
    last = _latest(h, st, narrowed=False)
    r = max(st["rounds"]) + 1
    need = f"narrowing {h['id']} needs its latest verdict record's gates to pass (the importance gates and the surrogate fit)"
    if last is None or last["round"] < h.get("narrowed_round", 0) < r:
        raise Refused(f"{need}: it has no verdict record " + ("on its narrowed ranges yet" if last else "yet"))
    if failed := [g for g, v in last["gates"].items() if not v["passed"]]:
        raise Refused(f"{need}: {last['id']} failed {failed}")
    lv = h["spec"]["levers"][a.lever]
    if lv["kind"] in hypotheses.GRADED:
        if a.low is None or a.high is None or a.options is not None:
            raise Refused(f"{a.lever} is {lv['kind']}: narrow it with --low and --high")
        low, high = a.low, a.high
        if lv["kind"] == "int":
            if not (low.is_integer() and high.is_integer()):
                raise Refused(f"{a.lever} is an int lever: --low and --high must be integers")
            low, high = int(low), int(high)
        before, after = {"low": lv["low"], "high": lv["high"]}, {"low": low, "high": high}
        if not lv["low"] <= low < high <= lv["high"] or after == before:
            raise Refused(f"[{low}, {high}] must be a smaller range inside {a.lever}'s [{lv['low']}, {lv['high']}]")
        if not low <= lv["baseline"] <= high:
            raise Refused(f"the baseline {lv['baseline']} must stay inside the narrowed range")
    elif lv["kind"] == "categorical":
        try:
            opts = json.loads(a.options) if a.options is not None else None
        except json.JSONDecodeError as e:
            raise Refused(f"--options must be a JSON list: {e}")
        if not isinstance(opts, list) or a.low is not None or a.high is not None:
            raise Refused(f"{a.lever} is categorical: narrow it with --options '[...]'")
        before, after = {"options": lv["options"]}, {"options": opts}
        if (len(opts) < 2 or len(opts) >= len(lv["options"])
                or not all(bo_study.in_range(o, lv) for o in opts) or len({json.dumps(o) for o in opts}) < len(opts)):
            raise Refused(f"--options must keep at least 2 but fewer than all of {lv['options']}")
        if not bo_study.in_range(lv["baseline"], {"kind": "categorical", "options": opts}):
            raise Refused(f"the baseline {lv['baseline']!r} must stay among the options")
    else:
        raise Refused(f"{a.lever} is a bool lever: it has nothing to narrow")
    elog.append(con, "narrowed", a.actor, {"rationale": a.rationale, "id": h["id"], "lever": a.lever,
                                           "before": before, "after": after, "verdict": last["id"],
                                           "round": r})
    _regenerate(run_dir, con)
    return {"hypothesis": h["id"], "lever": a.lever, "range": after, **_status(elog.load(con))}


def _after_change(run_dir: Path, con, sha: str, change: str, breaking: bool = False) -> dict:
    """Every code change: the equivalence check, and a new epoch when it fails or the user declared
    the change breaking. Before the calibration round has measured anything there is nothing to
    compare, so neither."""
    check = epoch = None
    if elog.load(con)["r0_complete"]:
        if not breaking:
            check = _equivalence_check(run_dir, con, sha, change)
        if breaking or (check and check["passed"] is False):  # None: the budget refused the check
            epoch = _new_epoch(run_dir, con, sha, change, breaking)
    _regenerate(run_dir, con)
    st = elog.load(con)
    return {"equivalence_check": check, "epoch_started": epoch, **_status(st)}


def _equivalence_check(run_dir: Path, con, sha: str, change: str) -> dict:
    """The incumbent and the baseline, 2 replicates each at the current fidelity on the new commit,
    against their logged means: within 2σ (a float tolerance when σ = 0)."""
    st = elog.load(con)
    fidelity, baseline = st["fidelity"]["fidelity"], _baseline(run_dir)
    sign = 1 if st["run"]["direction"] == "minimize" else -1
    trials = _eligibility(st, _space(_in_search(st)), baseline, fidelity)(st["trials"])
    inc = bo_study.incumbent(trials, sign, CONFIRMATIONS if st["noise"]["replication"] else 0)
    configs = [("baseline", baseline)]
    if inc and {**baseline, **inc["levers"]} != baseline:
        configs.append(("incumbent", {**baseline, **inc["levers"]}))
    sigma = _sigma(st, fidelity)
    logged = {name: [t["objective"] for t in trials if {**baseline, **t["levers"]} == levers]
              for name, levers in configs}
    rows, refused = [], None
    for name, levers in configs:
        runs = []
        try:
            for _ in range(2):
                runs.append(_trial(run_dir, con, None, "equivalence", levers, fidelity, equivalence_of=name,
                                   **({"replicate_of": inc["trial"]} if name == "incumbent" and inc else {})))
        except TrialRefused as e:  # the run can't go on anyway: no round fits either
            refused = str(e)
            break
        ys = [t["objective"] for t in runs if t["status"] == "finished"]
        before = statistics.fmean(logged[name]) if logged[name] else None
        mean = statistics.fmean(ys) if len(ys) == len(runs) else None
        tol = 2 * sigma + 1e-9 * max(1.0, abs(before or 0.0))
        # nothing logged at this config yet: nothing it could disagree with
        passed = mean is not None and (before is None or abs(mean - before) <= tol)
        rows.append({"config": name, "levers": levers, "logged_mean": before, "logged_n": len(logged[name]),
                     "trials": [t["trial"] for t in runs], "mean": mean, "tolerance": tol,
                     "passed": passed})
    payload = {"commit": sha, "change": change, "fidelity": fidelity, "sigma": sigma, "configs": rows,
               "passed": None if refused else all(r["passed"] for r in rows), "refused": refused}
    elog.append(con, "equivalence_check", "harness", payload)
    return payload


def _new_epoch(run_dir: Path, con, sha: str, change: str, breaking: bool) -> dict:
    """Earlier trials become telemetry; σ is re-estimated from baseline replicates in the new epoch
    (burn-in restarts, since only this epoch's trials are eligible)."""
    st = elog.load(con)
    run, fidelity, baseline = st["run"], st["fidelity"]["fidelity"], _baseline(run_dir)
    payload = {"epoch": st["epoch"]["epoch"] + 1, "round": max(st["rounds"]) + 1, "commit": sha,
               "change": change, "breaking": breaking}
    elog.append(con, "epoch_started", "harness", payload)
    # ponytail: re-measures σ at the current fidelity only; other rungs (and the ladder calibration)
    # keep the earlier epoch's until a code change's effect on them needs re-measuring too
    done = []
    try:
        for _ in range(1 if run["deterministic"] else run["replicates_k"]):
            done.append(_trial(run_dir, con, None, "baseline", baseline, fidelity))
    except TrialRefused:
        pass  # σ stays unmeasured in this epoch (None below); the budget ends the run
    ys = [t["objective"] for t in done if t["status"] == "finished"]
    sigma = 0.0 if run["deterministic"] else calibration.sigma(ys) if len(ys) >= 2 else None
    elog.append(con, "noise_estimate", "harness", {
        "epoch": payload["epoch"], "fidelity": fidelity, "sigma": sigma, "n": len(ys),
        "trials": [t["trial"] for t in done], "mean": statistics.fmean(ys) if ys else None,
        "previous_sigma": _sigma(st, fidelity)})
    return payload


def cmd_round_run(a) -> dict:
    run_dir, con = _open_run()
    st = elog.load(con)
    _check_manifest(run_dir, st)  # before every round, R0 included
    made = {st["run"]["base_commit"], *(e["commit"] for e in st["commits"])}
    if (head := _git(run_dir / "worktree", "rev-parse", "HEAD")) not in made:
        raise Refused(f"the worktree's HEAD {head[:12]} was not committed by the harness: commits go "
                      "through commit-lever, commit-change or add-dependency (their diffs are checked)")
    if _git(run_dir / "worktree", "status", "--porcelain"):
        raise Refused("the worktree has uncommitted changes: a round runs on one commit, so commit "
                      "lever code with `commit-lever <H>` first")
    _not_ended(st)
    if st["paused"]:
        raise Refused(f"the run is paused at a checkpoint: {RESUME_HINT}")
    if not st["r0_complete"]:
        return _run_r0(run_dir, con, a, st)
    def ready_selection(st: dict) -> list[dict]:
        """The selection, refused while its code or (once δ is set) its expected verdicts are missing."""
        selected = _selected(st)
        if not selected:
            raise Refused("no hypothesis is registered: propose and register one first")
        if uncoded := [h["id"] for h in selected if not h["commit"]]:
            raise Refused(f"the round selects {uncoded}, whose lever code is not committed: "
                          "write it, then `smoke <H>` and `commit-lever <H>`")
        if st["delta"] is not None and (missing := _expected_missing(st)):
            raise Refused(f"record expected {', '.join(missing)} first: before each round, an expected "
                          "verdict (retain, reject or undecided, one-line reason) for every hypothesis it tests")
        return selected

    if unreviewed := _review_missing(st):
        raise Refused(f"record review {', '.join(unreviewed)} first: the registration reviewer reviews every "
                      "hypothesis at registration, at revival, and after a registry revision")
    if missing := _interplay_missing(st):
        raise Refused(f"record interplay for {', '.join(_who(m) for m in missing)} first: every removal "
                      "is weighed against the untested list, and every newcomer against past removals")
    if due := _generation_due(st):
        raise Refused(f"`generate` first: a generation pass is due ({'; '.join(due)})")
    if all((conditions := _exhaustion(st)).values()):  # judged at the boundary, once a pass is in
        elog.append(con, "run_ended", "harness", {"reason": "exhausted", "exhaustion": conditions})
        _regenerate(run_dir, con)
        return _status(elog.load(con))
    if revived := _revive(run_dir, con, st):  # into the round the partner joins, shown before it runs
        _regenerate(run_dir, con)
        return {"revived": revived, **_status(elog.load(con))}
    selected = ready_selection(st)
    if st["run"]["ladder"] and not st["calibration"]:  # still part of R0; needs the lever boxes
        try:
            _calibrate_ladder(run_dir, con, st["noise"], _space(selected))
        except TrialRefused as e:
            # R0 itself has ended: only the run ends
            return _end_run(run_dir, con, {"round": 0, "trigger": e.reason}, log_round=False)
        st = elog.load(con)
        if st["delta"] is None:  # shown before δ is set
            return {"round": 0, "fidelity_calibration": st["calibration"], **_status(st)}
        ready_selection(st)  # the chosen rung's cost moves the dimension cap, so the selection too
    if st["delta"] is None:
        raise Refused(f"set δ first: `{_delta_duty(st)}`")
    return _run_round(run_dir, con, a, st)


def _run_r0(run_dir: Path, con, a, st: dict) -> dict:
    """The calibration round: smoke and baseline replicates (σ and each rung's cost)."""
    ref = st["run"]["reference_fidelity"]
    baseline = _baseline(run_dir)
    elog.append(con, "round_started", a.actor, {
        "rationale": a.rationale, "round": 0, "fidelity": ref, "pid": os.getpid(),
        "commit": _git(run_dir / "worktree", "rev-parse", "HEAD")})

    trigger = "failed"
    try:
        if not any(t.get("kind") == "smoke" and t["status"] == "finished" and "hypothesis" not in t
                   for t in st["trials"]):  # a smoke <H> ran uncommitted code
            smoke = _trial(run_dir, con, None, "smoke", baseline, ref, round=0)
            if smoke["status"] != "finished":
                trigger = "smoke_failed"
                raise Refused(f"the smoke trial failed: {smoke['reason']}: {smoke['error']}")
        if st["noise"] is None:  # a rerun after a stop keeps the σ it already paid for
            _estimate_noise(run_dir, con, baseline)
        out = {"round": 0, "noise_estimate": elog.load(con)["noise"]}
        trigger = "calibrated"  # the ladder is calibrated at the first R1 round-run (needs levers)
    except TrialRefused as e:
        trigger = e.reason
        raise
    finally:
        elog.append(con, "round_ended", "harness", {"round": 0, "trigger": trigger})
        _regenerate(run_dir, con)
    return {**out, **_status(elog.load(con))}


def _group(h: dict) -> list[str]:
    """The hypothesis's levers still in the search (frozen ones run at baseline)."""
    return [n for n in h["spec"]["levers"] if n not in h["frozen"]]


def _space(selected: list[dict]) -> dict:
    return {n: h["spec"]["levers"][n] for h in selected for n in _group(h)}


def _end_run(run_dir: Path, con, round_ended: dict, log_round: bool = True) -> dict:
    """No trial fits (the budget is spent, or the trial ceiling reached): the round and the run end
    together, unless a user stop ended the run first."""
    if log_round:
        elog.append(con, "round_ended", "harness", round_ended)
    if not elog.load(con)["run_ended"]:
        elog.append(con, "run_ended", "harness", {"reason": round_ended["trigger"]})
    _regenerate(run_dir, con)
    return {**round_ended, **_status(elog.load(con))}


def _run_round(run_dir: Path, con, a, st: dict) -> dict:
    """R1+: GPSampler searches the selected hypotheses' levers until a round-end trigger fires."""

    run, r = st["run"], max(st["rounds"]) + 1
    selected = _selected(st)
    space, baseline = _space(selected), _baseline(run_dir)
    fidelity, ref = st["fidelity"]["fidelity"], run["reference_fidelity"]
    sign = 1 if run["direction"] == "minimize" else -1
    confirmations = CONFIRMATIONS if st["noise"]["replication"] else 0
    cap = ROUND_CAP * _status(st)["budget"]["remaining_s"]
    for h in selected:
        if h["status"] == "registered":
            elog.append(con, "hypothesis_activated", "harness", {"id": h["id"], "round": r})

    eligible = _eligibility(st, space, baseline, fidelity)
    seeds = eligible(st["trials"])
    elog.append(con, "round_started", a.actor, {
        "rationale": a.rationale, "round": r, "commit": _git(run_dir / "worktree", "rev-parse", "HEAD"),
        "fidelity": fidelity, "epoch": st["epoch"]["epoch"], "search_space": space, "hypotheses": [h["id"] for h in selected],
        "pid": os.getpid(), "cap_s": cap, "seeded": [t["trial"] for t in seeds]})
    rng = _rng(run, f"R{r}")
    study = bo_study.Study(run_dir / "studies.db", run["direction"], space, seeds, rng.getrandbits(31))
    # ponytail: one trial at a time; run.yaml `workers` needs asks with pending trials (constant liar)

    def best(trials):
        return bo_study.incumbent(eligible(trials), sign, confirmations)

    def user_trigger(st: dict) -> str | None:
        """A `stop` (the only way the run ends from outside a round) or a `checkpoint` pause since
        the round started: it ends at its next step."""
        return "user_stop" if st["run_ended"] else "checkpoint" if st["paused"] else None

    inc, trigger, noise, drift, refusals = best(st["trials"]), None, None, None, 0
    stall_after = max(5 * len(space), 20)
    share = ESCALATED_SHARE if any(e["step"] == "replicates" for h in selected
                                   for e in h["escalations"]) else REPLICATE_SHARE
    # the agent-chosen trials first: R1's user-named seeds, then the ones enqueued for this round
    # ponytail: a queued trial whose round was interrupted is not carried to the resuming round
    chosen: list[tuple[str, dict, dict]] = [("seed", s, {}) for s in run["seeds"]] if r == 1 else []
    chosen += [("agent", q["config"], {"queued": q["queued"], "expected": q["expected"]})
               for q in st["queue"] if q["round"] == r]
    try:
        for kind, cfg, extra in chosen:
            if trigger := user_trigger(elog.load(con)):
                break
            if why := _config_error(cfg, space) or _prohibited({**baseline, **cfg}, st):  # checked again
                elog.append(con, "agent_trial_skipped", "harness",
                            {"round": r, "kind": kind, "config": cfg, "reason": why, **extra})
                continue
            t = _trial(run_dir, con, None, kind, {**baseline, **cfg}, fidelity, round=r, **extra)
            if t["status"] == "finished":
                study.add(t)
        while trigger is None:
            st = elog.load(con)
            inc = best(st["trials"])
            if trigger := user_trigger(st):
                break
            # the user's target, reached by the confirmed incumbent (≥ 2 replicates) at the reference
            # ponytail: never at a proxy fidelity; confirm a proxy's incumbent at the reference if
            # runs at a rung should end on target too
            if (run["target"] is not None and fidelity == ref and inc and inc["confirmed"]
                    and sign * (inc["mean"] - run["target"]) <= 0):
                trigger = "target"
                break
            checked = _verdict_checks(con, st, r, space, baseline, eligible, sign)
            if checked:  # SUMMARY.md changes after every verdict
                _regenerate(run_dir, con)
            if any(v["frozen"] or v["outcome"] in ("reject", "inconclusive") for v in checked):
                trigger = "search_space"  # a confirmed reject, a freeze or an inconclusive
                break
            if checked:  # a reject just gone pending must hold the round open below
                st = elog.load(con)
            pending, since = _progress(st, r, eligible, sign, confirmations)
            holding = any((v := _latest(h, st)) is not None and v["outcome"] == "pending-reject"
                          for h in _in_search(st))  # a pending reject holds the round open
            this = [t for t in st["trials"] if t.get("round") == r]
            spent = sum(t["wall_clock_s"] for t in this if t["status"] != "running")
            # the drift check's two reference trials come out of the cap too; the first trial always
            # runs (only the budget refuses it), so a round never ends empty
            reserve = 2 * _cost(st["trials"], ref) if fidelity != ref else 0.0
            if spent > 0 and spent + _cost(st["trials"], fidelity) + reserve > cap:
                trigger = "cap"
            elif (since >= stall_after and not pending and not holding
                  and _burned_in(st, selected, eligible)):
                trigger = "stall"
            elif pending or (confirmations
                             and sum("replicate_of" in t for t in this) < share * len(this)):
                if pending:
                    root, kind = pending[0], "confirmation"
                else:  # the replicate floor
                    # ponytail: the least-replicated of the top 3 configs (escalation raises only the
                    # share); pick by posterior overlap if replicates are wasted on settled configs
                    top = bo_study.ranked(eligible(st["trials"]), sign)[:3]
                    root, kind = min(top, key=lambda c: c["replicates"])["trial"], "replicate"
                levers = next(t["levers"] for t in eligible(st["trials"]) if t["trial"] == root)
                t = _trial(run_dir, con, None, kind, {**baseline, **levers}, fidelity, round=r,
                           replicate_of=root)
                if t["status"] == "finished":
                    study.add(t)
            else:
                asked = study.ask()
                try:
                    t = _trial(run_dir, con, None, "sampler", {**baseline, **asked.params}, fidelity, round=r)
                except ProhibitedRefused:  # a point between the grid registration checked
                    t, refusals = {"status": "refused"}, refusals + 1
                    # ponytail: ends the round after 10 refused asks (the sampler learns nothing from
                    # a FAIL); reparameterise the region into a box if a hypothesis keeps hitting it
                    if refusals >= 10:
                        trigger = "prohibited"
                study.tell(asked, t)
        st = elog.load(con)
        inc = best(st["trials"])
        noise = _reestimate_noise(con, st, r, eligible, fidelity) if confirmations else None
        if fidelity != ref and inc and not st["run_ended"]:  # a stopped run runs no more trials
            drift = _drift_check(run_dir, con, r, inc, eligible, sign)
    except TrialRefused as e:
        _finalise_rejects(con, r, None)
        return {**_end_run(run_dir, con, {"round": r, "trigger": e.reason, "incumbent": inc}),
                "verdicts": _round_verdicts(elog.load(con), r)}
    except BaseException:
        elog.append(con, "round_ended", "harness", {"round": r, "trigger": "failed", "incumbent": inc})
        _regenerate(run_dir, con)
        raise
    _finalise_rejects(con, r, drift)
    if trigger == "stall":
        _escalate(con, elog.load(con), r)
    elog.append(con, "round_ended", "harness", {"round": r, "trigger": trigger, "incumbent": inc})
    if trigger == "target" and not elog.load(con)["run_ended"]:
        elog.append(con, "run_ended", "harness", {"reason": "target_reached", "incumbent": inc})
    st = elog.load(con)
    if run["checkpoint"] and not st["paused"] and not st["run_ended"]:  # checkpoint mode
        elog.append(con, "checkpoint", "harness", {"round": r})
    _regenerate(run_dir, con)
    st = elog.load(con)
    return {"round": r, "trigger": trigger, "incumbent": inc, "drift": drift, "noise_estimate": noise,
            "verdicts": _round_verdicts(st, r), **_status(st)}


def _eligibility(st: dict, space: dict, baseline: dict, fidelity: dict):
    """The round's warm start: its eligible trials, rewritten into its levers (study.eligible)."""
    hyps = st["hypotheses"].values()
    latest = [h for h in hyps if h["version"] == max(x["version"] for x in hyps if x["number"] == h["number"])]
    irrelevant = frozenset(n for h in latest for n in h["frozen"]) | frozenset(
        n for h in latest if h["status"] == "rejected" and h["condition"] == "irrelevant"
        for n in h["spec"]["levers"])

    merges = {n: rule for h in hyps for n, rule in h["spec"].get("merges", {}).get("mapping", {}).items()
              if n in space}
    epoch = st["epoch"]["epoch"]

    def eligible(trials: list[dict]) -> list[dict]:
        # a trial breaking a prohibition (a revision tightened one since it ran) is out; a masked
        # lever the round searches enters with its sampled value (the box stays a box), one it
        # doesn't with the value it ran at
        return [m for t in trials if not _prohibited(t["levers"], st)
                and (m := bo_study.eligible(_sampled(t, space), space, baseline, fidelity, epoch,
                                            irrelevant, merges))]
    return eligible


def _sampled(t: dict, space: dict) -> dict:
    """The trial with its searched masked levers at their sampled values."""
    over = {n: v for n, v in t.get("sampled", {}).items() if n in space}
    return {**t, "levers": {**t["levers"], **over}} if over else t


def _prohibited(levers: dict, st: dict) -> str | None:
    """Why this resolved config lies outside the allowed region (None: it doesn't)."""
    d = directives.violated(st["registry"]["directives"], levers, elog.lever_paths(st["hypotheses"]))
    return f"outside the allowed region of prohibited directive {d['id']} ({d['predicate']})" if d else None


def _in_search(st: dict) -> list[dict]:
    return [h for h in st["hypotheses"].values() if h["status"] in ("active", "retained")]


def _latest(h: dict, st: dict, narrowed: bool = True) -> dict | None:
    """The hypothesis's last verdict record on its current group at the run's fidelity, in this
    epoch, since it (re-)entered the search (a return from parked starts afresh) and (unless
    `narrowed` is False) on its current, possibly narrowed, ranges."""
    start = max(st["epoch"]["round"], h.get("activated_round", 0),
                h.get("narrowed_round", 0) if narrowed else 0)
    past = [v for v in h["verdicts"] if v["fidelity"] == st["fidelity"]["fidelity"]
            and v["group"] == _group(h) and v["round"] >= start]
    return past[-1] if past else None


def _round_verdicts(st: dict, r: int) -> list[dict]:
    return [v for h in st["hypotheses"].values() for v in h["verdicts"] if v["round"] == r]


def _fresh(st: dict, h: dict, eligible) -> list[int]:
    """Fresh sampler trials: eligible sampler trials since the hypothesis entered the search, where
    it was effective (not masked)."""
    return [t["trial"] for t in eligible(st["trials"]) if t["kind"] == "sampler"
            and t["round"] >= h["activated_round"] and h["id"] not in t.get("masked", [])]


def _schedule(h: dict) -> dict:
    d = len(_group(h))
    return {"needed": max(10 * d, 20), "spacing": max(5 * d, 10), "cap": max(40 * d, 80)}


def _verdict_checks(con, st: dict, r: int, space: dict, baseline: dict, eligible,
                    sign: int) -> list[dict]:
    """Every verdict check now due: the first at burn-in, then every max(5·d, 10) fresh sampler
    trials. Logs and returns the verdict records."""
    run, delta, fid = st["run"], st["delta"], st["fidelity"]
    at_proxy = fid["fidelity"] != run["reference_fidelity"]
    sigma = _sigma(st, fid["fidelity"])
    trials = eligible(st["trials"])
    checked = []
    for h in _in_search(st):
        sched, group, last = _schedule(h), _group(h), _latest(h, st)
        fresh = _fresh(st, h, eligible)
        due = last["burn_in"]["fresh"] + sched["spacing"] if last else sched["needed"]
        if len(fresh) < due:
            continue
        check = sum(v["round"] == r for v in h["verdicts"]) + 1
        vid = f"V-R{r}-{h['id']}-{check}"
        from . import verdict  # numpy/scipy: imported here so the hooks' checks start fast
        stats = verdict.judge(space, baseline, group, trials, sign, sigma, fresh,
                              _rng(run, "verdict", vid).getrandbits(32))
        gates = {"trials": {"sampler": len(fresh), "agent": sum(t["kind"] in ("agent", "seed")
                                                                 for t in trials),
                            "passed": len(fresh) >= sched["needed"]}, **stats.pop("gates")}
        gated = all(g["passed"] for g in gates.values())
        vt, dl = stats["sqrt_vt"], stats["delta_stat"]
        condition = ("irrelevant" if vt["upper"] < delta else
                     "no-improvement" if dl["upper"] < delta else None)
        pending = last is not None and last["outcome"] == "pending-reject"
        held, reason = None, None
        if not gates["homogeneity"]["passed"]:  # v1 models constant noise only: never a reject
            outcome, reason = "inconclusive", "noise differs by region"
        elif gated and condition and h["spec"]["fidelity_sensitive"] and at_proxy:
            outcome, held = "active", "fidelity-sensitive: never rejected at a proxy fidelity"
        elif gated and condition:
            outcome = "reject" if pending else "pending-reject"
        elif gated and vt["lower"] > delta and dl["lower"] > delta:
            outcome = "retained"
        else:
            outcome = "active"
        if outcome == "active" and len(fresh) >= sched["cap"]:
            outcome, reason = "inconclusive", f"no verdict after {len(fresh)} fresh sampler trials"
        frozen = []
        if gated and outcome in ("active", "retained") and len(group) > 1:
            frozen = [n for n, b in stats["levers"].items() if b["upper"] < delta]
            frozen = frozen if len(frozen) < len(group) else []  # the whole group: that's a reject
        contradicted = [n for n in group if h["spec"]["levers"][n].get("predicted") and
                        _against(h["spec"]["levers"][n], stats["best_point"][n])]
        record = {
            "id": vid, "hypothesis": h["id"], "round": r, "check": check, "group": group,
            "outcome": outcome, "condition": condition, "held": held, "reason": reason,
            **stats, "delta": delta, "gates": gates,
            "burn_in": {"fresh": len(fresh), **sched, "passed": len(fresh) >= sched["needed"]},
            "confirmation": {"pending": outcome == "pending-reject",
                             "confirms": last["id"] if last and outcome == "reject" else None,
                             "due_at": len(fresh) + sched["spacing"] if outcome == "pending-reject"
                             else None},
            "fidelity": fid["fidelity"], "proxy": fid["proxy"],
            "prediction": {"flag": "retained-against-prediction"
                           if outcome == "retained" and contradicted else None,
                           "contradicted": contradicted},
            "frozen": frozen, "trials": [t["trial"] for t in trials],
            "context": {"retained": [x["id"] for x in _in_search(st)
                                     if x["status"] == "retained" and x["id"] != h["id"]],
                        "co_active": [x["id"] for x in _in_search(st) if x["id"] != h["id"]]},
        }
        elog.append(con, "verdict", "harness", record)
        for n in frozen:
            elog.append(con, "lever_frozen", "harness", {"id": h["id"], "lever": n, "verdict": vid})
        if outcome == "inconclusive":
            elog.append(con, "hypothesis_inconclusive", "harness",
                        {"id": h["id"], "verdict": vid, "reason": reason})
            _removal(con, h["id"], "inconclusive", reason)
        checked.append(record)
    return checked


def _sigma(st: dict, fidelity: dict) -> float:
    """σ at this fidelity: the calibration round's, or the current epoch's re-estimate."""
    return next((x["sigma"] for x in st["noise"]["rungs"] if x["fidelity"] == fidelity),
                st["noise"]["sigma"])


def _against(lv: dict, best) -> bool:
    """The best posterior point lies on the other side of the baseline from the prediction."""
    return (best < lv["baseline"]) if lv["predicted"] == "higher" else (best > lv["baseline"])


def _finalise_rejects(con, r: int, drift: dict | None) -> None:
    """The round's confirmed rejects. At a proxy fidelity they stand only once the drift check
    passed; a broken (or unchecked) proxy fidelity logs them as inconclusive instead."""
    st = elog.load(con)
    at_proxy = st["fidelity"]["fidelity"] != st["run"]["reference_fidelity"]
    for h in _in_search(st):
        v = h["verdicts"][-1] if h["verdicts"] else None
        if not v or v["round"] != r or v["outcome"] != "reject":
            continue
        if at_proxy and not (drift and drift["broken"] is False):
            reason = "broken proxy fidelity" if drift and drift["broken"] else "proxy fidelity unchecked"
            elog.append(con, "hypothesis_inconclusive", "harness",
                        {"id": h["id"], "verdict": v["id"], "reason": reason})
            _removal(con, h["id"], "inconclusive", reason)
        else:
            elog.append(con, "hypothesis_rejected", "harness", {
                "id": h["id"], "verdict": v["id"], "condition": v["condition"],
                "frozen": v["group"] if v["condition"] == "no-improvement" else []})
            _removal(con, h["id"], "rejected", v["condition"])


def _escalate(con, st: dict, r: int) -> None:
    """After a stalled round, each undecided hypothesis escalates: a larger replicate share, then
    the next fidelity rung up. The evidence cap (inconclusive) is applied at its verdict checks."""
    # ponytail: moves up a rung without asking calibration whether the effect is resolvable at
    # this one; add that test when rung choice gets per-hypothesis effect sizes
    ref = st["run"]["reference_fidelity"]
    up = _next_rung(st) if st["fidelity"]["fidelity"] != ref else None  # one rung per stall, run-wide
    for h in _in_search(st):
        last = _latest(h, st)
        if not last or last["outcome"] != "active":
            continue
        if "replicates" not in [e["step"] for e in h["escalations"]]:
            elog.append(con, "hypothesis_escalated", "harness", {
                "id": h["id"], "round": r, "step": "replicates", "replicate_share": ESCALATED_SHARE})
        elif up is not None:
            elog.append(con, "hypothesis_escalated", "harness", {
                "id": h["id"], "round": r, "step": "rung", "fidelity": up,
                "proxy": "reference" if up == ref else "validated"})


def _next_rung(st: dict) -> dict:
    """The cheapest validated rung dearer than the current fidelity, else the reference."""
    fid, ref = st["fidelity"]["fidelity"], st["run"]["reference_fidelity"]
    rungs = (st["calibration"] or {}).get("rungs", [])
    cost = [(x["fidelity"], x["cost_s"]) for x in st["noise"]["rungs"]]
    here = next(c for f, c in cost if f == fid)
    passed = [x["fidelity"] for x in rungs if x.get("passed")]
    up = [(c, f) for f, c in cost if c > here and f in passed]
    return min(up, key=lambda cf: cf[0])[1] if up else ref


def _progress(st: dict, r: int, eligible, sign: int, confirmations: int) -> tuple[list[int], int]:
    """The confirmation replicates still owed (by original trial) and the fresh sampler trials since
    the confirmed incumbent last gained ≥ δ, both replayed from the log so a crash loses neither.

    A sampler or agent-chosen trial is a new incumbent when it beats the confirmed incumbent's mean
    (or none is confirmed yet); it is owed 2 confirmations, and a confirmation counts once started (an abandoned
    one is never retried). The window opens at this round, or at the first interrupted round it resumes.
    """
    # ponytail: the stall window restarts every round (else a round after a stall would end at once
    # with the same selection); #26/#29 decide whether it should span rounds once verdicts exist
    # ponytail: replays the window's trials on every step, quadratic in a round's length; keep
    # running totals if rounds reach thousands of trials
    start = r
    while st["rounds"].get(start - 1, {}).get("ended") == "interrupted":
        start -= 1
    done = eligible(st["trials"])
    seen = [t for t in done if t.get("round", -1) < start]
    inc = bo_study.incumbent(seen, sign, confirmations)
    anchor = inc["mean"] if inc and inc["confirmed"] else None
    since, owed = 0, []
    for t in done:
        if t.get("round", -1) < start:
            continue
        if t["kind"] in ("sampler", "agent", "seed"):  # a new incumbent from any of them is confirmed
            since += t["kind"] == "sampler"
            if confirmations and (inc is None or not inc["confirmed"]
                                  or sign * (t["objective"] - inc["mean"]) < 0):
                owed.append(t["trial"])
        seen.append(t)
        inc = bo_study.incumbent(seen, sign, confirmations)
        if inc and inc["confirmed"] and (anchor is None or sign * (anchor - inc["mean"]) >= st["delta"]):
            anchor, since = inc["mean"], 0
    given = [t.get("replicate_of") for t in st["trials"] if t["kind"] == "confirmation"]
    return [root for root in owed for _ in range(confirmations - given.count(root))], since


def _reestimate_noise(con, st: dict, r: int, eligible, fidelity: dict) -> dict:
    """σ from this round's replicates (pooled within each replicated config's trials of the round),
    flagged when it moved more than 2× from the calibration round's σ at this fidelity."""
    ss, df = 0.0, 0
    for ts in bo_study.groups([t for t in eligible(st["trials"]) if t.get("round") == r]).values():
        ys = [t["objective"] for t in ts]
        if len(ys) > 1:
            ss += sum((y - statistics.fmean(ys)) ** 2 for y in ys)
            df += len(ys) - 1
    sigma = math.sqrt(ss / df) if df >= 2 else None
    r0 = next((x["sigma"] for x in st["noise"]["rungs"] if x["fidelity"] == fidelity), None)
    shift = sigma / r0 if sigma is not None and r0 else None
    payload = {"round": r, "fidelity": fidelity, "sigma": sigma, "df": df, "r0_sigma": r0,
               "shift": shift, "shift_flagged": shift is not None and not 0.5 <= shift <= 2}
    elog.append(con, "noise_estimate", "harness", payload)
    return payload


def _burned_in(st: dict, selected: list[dict], eligible) -> bool:
    """Every hypothesis in the search has max(10·d, 20) fresh sampler trials."""
    hyps = st["hypotheses"]
    return all(len(_fresh(st, hyps[h["id"]], eligible)) >= _schedule(hyps[h["id"]])["needed"]
               for h in selected)


def _drift_check(run_dir: Path, con, r: int, inc: dict, eligible, sign: int) -> dict | None:
    """The incumbent and one random config of the round, re-run at the reference fidelity."""
    st = elog.load(con)
    ref = st["run"]["reference_fidelity"]
    others = [t for t in eligible(st["trials"])
              if t.get("round") == r and "replicate_of" not in t and t["levers"] != inc["levers"]]
    if not others:
        return None
    other = _rng(st["run"], f"R{r}", "drift").choice(others)
    proxy = {c["trial"]: c["mean"] for c in bo_study.ranked(eligible(st["trials"]), sign)}
    pair = [(inc["trial"], inc["levers"]), (other["trial"], other["levers"])]
    baseline = _baseline(run_dir)
    runs = [_trial(run_dir, con, None, "drift", {**baseline, **levers}, ref, round=r, drift_of=root)
            for root, levers in pair]
    at_ref = [t.get("objective") for t in runs]
    at_proxy = [proxy[root] for root, _ in pair]
    # broken: the reference reverses the proxy's order by more than noise alone would, 2 sd of the
    # difference of two reference trials (0 for a deterministic objective)
    # ponytail: one pair; rank agreement over more configs (or against calibration ρ) if it misfires
    margin = 2 * math.sqrt(2) * st["noise"]["rungs"][0]["sigma"]
    broken = None  # unknown when a drift trial failed
    if at_ref[0] is not None and at_ref[1] is not None:
        d_proxy, d_ref = sign * (at_proxy[0] - at_proxy[1]), sign * (at_ref[0] - at_ref[1])
        broken = d_proxy * d_ref < 0 and abs(d_ref) > margin
    payload = {"round": r, "configs": [root for root, _ in pair], "trials": [t["trial"] for t in runs],
               "proxy": at_proxy, "reference": at_ref, "margin": margin, "broken": broken}
    elog.append(con, "drift_check", "harness", payload)
    return payload


def _estimate_noise(run_dir: Path, con, baseline: dict) -> None:
    """k baseline replicates at each fidelity give its σ; a deterministic objective skips them."""
    run = elog.load(con)["run"]
    rows = []
    for f in (run["reference_fidelity"], *run["ladder"]):
        done = [] if run["deterministic"] else [
            _trial(run_dir, con, None, "baseline", baseline, f, round=0)
            for _ in range(run["replicates_k"])]
        values = [t["objective"] for t in done if t["status"] == "finished"]
        if not run["deterministic"] and len(values) < 2:
            raise Refused(f"fewer than 2 baseline replicates finished at {json.dumps(f)}")
        rows.append({"fidelity": f, "n": len(values), "trials": [t["trial"] for t in done],
                     "sigma": calibration.sigma(values) if values else 0.0,
                     "mean": statistics.fmean(values) if values else None,
                     "cost_s": _cost(elog.load(con)["trials"], f)})
    elog.append(con, "noise_estimate", "harness", {
        "round": 0, "sigma": rows[0]["sigma"], "deterministic": run["deterministic"],
        "replication": any(r["sigma"] > 0 for r in rows), "rungs": rows})


def _calibrate_ladder(run_dir: Path, con, noise: dict, space: dict) -> dict:
    """Ladder calibration: ~5 diverse configs over the selected hypotheses' levers at every rung."""
    run = elog.load(con)["run"]
    # a quasi-random point can fall in a hole of a region that isn't box-shaped: it never runs
    cfgs = [c for c in calibration.configs(space, _baseline(run_dir), _rng(run, "calibration"))
            if not _prohibited(c, elog.load(con))]
    rows = [{"fidelity": r["fidelity"], "sigma": r["sigma"], "objectives": [], "trials": []}
            for r in noise["rungs"]]
    for i, cfg in enumerate(cfgs):
        for r in rows:
            t = _trial(run_dir, con, None, "calibration", cfg, r["fidelity"], round=0, config=i)
            r["trials"].append(t["trial"])
            r["objectives"].append(t.get("objective"))
    trials = elog.load(con)["trials"]
    for r in rows:
        r["cost_s"] = _cost(trials, r["fidelity"])
    chosen, fallback = calibration.choose(rows[0], rows[1:])
    elog.append(con, "fidelity_calibration", "harness", {
        "round": 0, "configs": cfgs, "reference": rows[0], "rungs": rows[1:],
        "chosen": chosen, "fallback": fallback})
    return elog.load(con)["calibration"]


def _run_trial(run_dir: Path, runner: str, trial: dict) -> dict:
    art = run_dir / trial["artifact_dir"]
    art.mkdir(parents=True)
    (art / "trial.json").write_text(json.dumps({**trial, "artifact_dir": str(art)}, indent=2))
    # no __pycache__ in the worktree: it would dirty the tree a round and commit-lever pin
    env = {**os.environ, "BOAUTORESEARCH_TRIAL": str(art / "trial.json"), "PYTHONDONTWRITEBYTECODE": "1"}
    t0, done = time.monotonic(), threading.Event()

    def heartbeat(pid: int):
        """What a crash is charged up to: at spawn (with the runner's pid, for recovery to kill),
        then at 1, 2, 4, 8 s and every 10 s on, so a kill is charged at least half its time."""
        con = elog.connect(run_dir / "log.db")
        elapsed = 0.0
        while True:
            elog.append(con, "trial_heartbeat", "harness",
                        {"trial": trial["trial"], "elapsed_s": elapsed, "runner_pid": pid})
            if done.wait(min(max(elapsed, 1.0), 10.0)):
                break
            elapsed = time.monotonic() - t0
        con.close()

    with open(art / "stdout.txt", "w") as out, open(art / "stderr.txt", "w") as err:
        # its own process group: recovery, or this process failing, kills the runner and its children
        p = subprocess.Popen([run_dir / "venv" / "bin" / "python", runner], start_new_session=True,
                             cwd=run_dir / "worktree", env=env, stdout=out, stderr=err)
        beat = threading.Thread(target=heartbeat, args=(p.pid,), daemon=True)
        beat.start()
        try:
            _, status, usage = os.wait4(p.pid, 0)  # per-child rusage gives this trial's peak memory
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL)
            raise
        finally:
            done.set()
            beat.join()
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


def _spent(st: dict) -> float:
    return sum(t["wall_clock_s"] for t in st["trials"] if t["status"] != "running")


def _remaining(st: dict) -> float:
    return st["run"]["budget_s"] - _spent(st)


def _status(st: dict) -> dict:
    run, trials = st["run"], st["trials"]
    spent = _spent(st)
    count = {s: sum(t["status"] == s for t in trials)
             for s in ("running", "finished", "failed", "abandoned")}
    return {"run_id": run["run_id"], "objective": run["objective"], "direction": run["direction"],
            "budget": {"total_s": run["budget_s"], "spent_s": spent,
                       "remaining_s": run["budget_s"] - spent},
            "trials": {"total": len(trials), **count},
            "r0_complete": st["r0_complete"], "fidelity": st["fidelity"],
            "sigma": st["noise"]["sigma"] if st["noise"] else None,
            "replication": st["noise"]["replication"] if st["noise"] else None,
            "delta": st["delta"], "suggested_delta": _suggested_delta(st), "run_ended": st["run_ended"],
            "epoch": st["epoch"]["epoch"],
            "hypotheses": [{"id": h["id"], "title": h["spec"]["title"], "status": h["status"],
                            "commit": h["commit"], "priority": _priority(st, h)}
                           for h in st["hypotheses"].values()],
            "schedule": _schedule_status(st), "paused": st["paused"],
            "interplay_missing": _interplay_missing(st), "review_missing": _review_missing(st),
            "generation": {"mode": run["generation"], "passes": len(st["passes"]),
                           "due": _generation_due(st)},
            "exhaustion": _exhaustion(st), "next": _next(st),
            "conflicts": {"rivals": _rivals(st), "masking": _masking(st),
                          "merged": [{"id": h["id"], "into": h["merged_into"]}
                                     for h in st["hypotheses"].values() if h["status"] == "merged"]}}


def _suggested_delta(st: dict) -> float | None:
    """2σ; none for a noiseless objective (set-delta refuses 0)."""
    return 2 * st["noise"]["sigma"] if st["noise"] and st["noise"]["sigma"] > 0 else None


def _delta_duty(st: dict) -> str:
    if (d := _suggested_delta(st)) is None:
        return "set-delta <value> (σ = 0: the smallest effect worth having, in objective units)"
    return f"set-delta <value> (suggested 2σ = {d:.3g})"


def _queue(st: dict) -> list[dict]:
    """The registered hypotheses, by priority (highest first), then registration order."""
    return sorted((h for h in st["hypotheses"].values() if h["status"] == "registered"),
                  key=lambda h: (-_priority(st, h), h["registered"]))


def _dimension_cap(st: dict) -> int | None:
    """The most lever dimensions d a round can take: its burn-in of max(10·d, 20) fresh trials must
    fit in the trials the round cap affords at the current fidelity (None while their cost is unknown)."""
    cost = _cost(st["trials"], st["fidelity"]["fidelity"])
    if cost <= 0:
        return None
    affordable = ROUND_CAP * _remaining(st) / cost
    return int(affordable // 10) if affordable >= 20 else 0


def _exclusive(a: dict, b: dict) -> bool:
    """Either declares the other (any version) in exclusive_with."""
    def declares(x, y):
        return any(int(re.match(r"H(\d+)", i)[1]) == y["number"] for i in x["spec"].get("exclusive_with", []))
    return declares(a, b) or declares(b, a)


def _selected(st: dict) -> list[dict]:
    """The hypotheses the next round searches, a deterministic function of the log: those in the
    search stay (retained ones included), then the queue joins in order while the lever dimensions
    fit the cap (a lower priority never jumps a higher one that doesn't fit; the first always fits an
    empty round). A hypothesis exclusive with one already chosen is skipped."""
    # ponytail: retained hypotheses keep their full ranges when the cap is tight; narrow the oldest
    # concentrated ones (spec: "narrowed, not frozen") if they crowd the queue out
    # ponytail: queued hypotheses flagged as interacting, and rivals, join together or not at all; a revived
    # version rides with its partner uncounted (spec: the same round), so it can take a round past the cap
    chosen, queue, hyps = _in_search(st), _queue(st), st["hypotheses"]
    linked = {frozenset((hyps[x]["number"], hyps[p]["number"])) for x, p in [*_links(st), *_rivals(st)]}
    cap, dims = _dimension_cap(st), sum(len(_group(h)) for h in chosen)
    for h in queue:
        if h in chosen or any(_exclusive(h, x) for x in chosen):
            continue
        unit = [h]
        for x in queue:
            if (x is not h and x not in chosen and frozenset((h["number"], x["number"])) in linked
                    and not any(_exclusive(x, y) for y in chosen + unit)):
                unit.append(x)
        d = sum(len(_group(u)) for u in unit if "revived_from" not in u or u is h)
        if chosen and cap is not None and dims + d > cap:
            break
        chosen += unit
        dims += d
    return chosen


def _upcoming(st: dict) -> int:
    """The next round to start."""
    return max(st["rounds"], default=-1) + 1


def _agent_trials(st: dict) -> dict:
    """The next round's agent-chosen trials: max(1, 6 − R) per round (so the loop is never worse
    than plain BO), the user's run.yaml seeds counting against R1's."""
    r = _upcoming(st)
    used = (len(st["run"]["seeds"]) if r == 1 else 0) + sum(q["round"] == r for q in st["queue"])
    return {"round": r, "cap": max(1, 6 - r), "used": used}


def _config_error(cfg, space: dict) -> str | None:
    """Why an agent-chosen config can't run in a round searching `space` (None: it can)."""
    if not isinstance(cfg, dict) or not cfg:
        return "a config must be a JSON object naming at least one lever"
    if unknown := sorted(set(cfg) - set(space)):
        return f"{unknown} are not levers of the round's search space {sorted(space)}"
    if bad := sorted(n for n, v in cfg.items() if not hypotheses.in_range(v, space[n])):
        return f"{bad} lie outside their range"
    return None


def cmd_enqueue(a) -> dict:
    """An agent-chosen trial for the next round, with the objective the agent expects of it."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    if not st["r0_complete"]:
        raise Refused("agent-chosen trials run in BO rounds, from R1: run the calibration round first")
    if not math.isfinite(a.expected):
        raise Refused("--expected must be the objective value the trial is expected to reach")
    try:
        cfg = json.loads(a.config)
    except json.JSONDecodeError as e:
        raise Refused(f"--config must be JSON, e.g. '{{\"H1.x\": 0.5}}': {e}")
    if why := _config_error(cfg, _space(_selected(st))):
        raise Refused(f"--config: {why}")
    q = _agent_trials(st)
    if q["used"] >= q["cap"]:
        raise Refused(f"R{q['round']} has used its cap of {q['cap']} agent-chosen trials "
                      "(max(1, 6 − R), run.yaml seeds included in R1)")
    elog.append(con, "trial_enqueued", a.actor, {"rationale": a.rationale, "queued": len(st["queue"]) + 1,
                                                 "round": q["round"], "config": cfg, "expected": a.expected})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def _expected_missing(st: dict) -> list[str]:
    """The next round's hypotheses with no expected verdict recorded for that round."""
    r = _upcoming(st)
    have = {x["record"]["hypothesis"] for x in st["records"] if x["kind"] == "expected" and x["round"] == r}
    return [h["id"] for h in _selected(st) if h["id"] not in have] if st["r0_complete"] else []


def _schedule_status(st: dict) -> dict:
    selected = _selected(st)
    return {"selected": [h["id"] for h in selected], "queue": [h["id"] for h in _queue(st)],
            "dimension_cap": _dimension_cap(st), "dimensions": len(_space(selected)),
            "expected_missing": _expected_missing(st), "agent_trials": _agent_trials(st),
            # flagged to the user, never starved silently
            "starved": [{"id": h["id"], "rounds": h["unselected"]} for h in _queue(st)
                        if h["unselected"] >= STARVED_AFTER]}


def _next(st: dict) -> list[str]:
    """The duties outstanding before the next round-run can go ahead."""
    if st["run_ended"]:
        return []
    if not st["r0_complete"]:
        return ["round-run (the calibration round)"]
    if all(_exhaustion(st).values()):
        return ["round-run (the hypothesis list is exhausted: it ends the run)"]
    duties = [f"write {h['id']}'s lever code, then smoke {h['id']} and commit-lever {h['id']}"
              for h in _selected(st) if not h["commit"]]
    if not _selected(st):
        duties.append("propose and register a hypothesis")
    if st["delta"] is None:
        duties.append(_delta_duty(st))
    duties += [f"record review {h} (the registration reviewer: directive, intent and conflict verdicts)"
               for h in _review_missing(st)]
    duties += [f"record interplay: review {m['removed']}'s removal against the untested list"
               if "removed" in m else f"record interplay: review newcomer {m['newcomer']} against past removals"
               for m in _interplay_missing(st)]
    if due := _generation_due(st):
        duties.append(f"generate ({'; '.join(due)})")
    duties += [f"record expected {h} (retain, reject or undecided, with a one-line reason) before the round"
               for h in _expected_missing(st)]
    if st["paused"]:
        duties.append(f"the run is paused: {RESUME_HINT}")
    return duties or ["round-run"]


def _removal(con, hid: str, removal: str, reason: str | None) -> None:
    """A reject, inconclusive or park: the evidence bundle its interplay review weighs against the
    untested list."""
    st = elog.load(con)
    h = st["hypotheses"][hid]
    last = h["verdicts"][-1] if h["verdicts"] else None
    elog.append(con, "removal", "harness", {
        "id": hid, "removal": removal, "reason": reason, "verdict": last["id"] if last else None,
        "verdicts": [v["id"] for v in h["verdicts"]], "context": last["context"] if last else None,
        "untested": [x["id"] for x in st["hypotheses"].values() if x["status"] in records.UNTESTED]})


def _interplay_missing(st: dict) -> list[dict]:
    """The interplay reviews owed: every removal since its last review, and every queued newcomer
    (a revival excepted) against the removals no review weighed it against."""
    reviews = [x for x in st["records"] if x["kind"] == "interplay"]

    def reviewed(key: str, hid: str, since: int) -> bool:
        return any(x["record"].get(key) == hid and x["seq"] > since for x in reviews)
    owed = [{"removed": r["id"]} for r in st["removals"] if not reviewed("removed", r["id"], r["seq"])]
    for h in _queue(st):
        past = [r["seq"] for r in st["removals"] if r["seq"] < h["queued"] and r["id"] != h["id"]
                and h["id"] not in r["untested"]]
        if past and "revived_from" not in h and not reviewed("newcomer", h["id"], max(past)):
            owed.append({"newcomer": h["id"]})
    return owed


def _who(m: dict) -> str:
    return f"{m['removed']}'s removal" if "removed" in m else f"newcomer {m['newcomer']}"


def _links(st: dict) -> list[tuple[str, str]]:
    """The pairs the interplay reviews flagged: (removed hypothesis, untested partner)."""
    out = []
    for x in st["records"]:
        if x["kind"] == "interplay":
            r = x["record"]
            out += [(r["removed"], f["partner"]) if r.get("removed") else (f["partner"], r["newcomer"])
                    for f in r["flags"]]
    return out


def _pending_links(st: dict) -> list[tuple[dict, str]]:
    """Flagged pairs still to revive (at most once per pair of hypothesis numbers): the removed
    hypothesis's latest version, and its partner, still untested."""
    hyps = st["hypotheses"]
    done = {(h["number"], hyps[h["partner"]]["number"]) for h in hyps.values() if "revived_from" in h}
    out = []
    for x, p in _links(st):
        pair = (hyps[x]["number"], hyps[p]["number"])
        last = _hypothesis(st, f"H{pair[0]}")
        if (pair not in done and last["status"] in records.REMOVED
                and hyps[p]["status"] in records.UNTESTED):
            out.append((last, p))
            done.add(pair)
    return out


def _revive(run_dir: Path, con, st: dict) -> list[dict]:
    """Each flagged removal whose partner the next round selects comes back as a new linked version,
    its code already in place (same lever names) and its verdict schedule fresh; the scheduler
    places it in its partner's round."""
    # the revived version registers at once; round-run then refuses until the registration reviewer
    # has reviewed it (_review_missing)
    selected = {h["id"] for h in _selected(st) if h["status"] == "registered"}
    revived: list[dict] = []
    for old, partner in _pending_links(st):
        if partner not in selected or any(r["from"] == old["id"] for r in revived):
            continue
        k = old["version"] + 1
        payload = {"id": f"H{old['number']}.v{k}", "number": old["number"], "version": k,
                   "from": old["id"], "partner": partner, "spec": old["origin"], "commit": old["commit"],
                   "round": _upcoming(st)}
        elog.append(con, "hypothesis_revived", "harness", payload)
        revived.append(payload)
        new = elog.load(con)["hypotheses"][payload["id"]]
        if out := _outside(run_dir, st, new):  # registered at once, so checked like a registration
            elog.append(con, "hypothesis_pruned", "harness",
                        {"id": new["id"], "directive": out[0]["id"], "by": "revival", "reason": out[1]})
        elif why := _clash(elog.load(con), new):  # a merge is the way to test it now
            _park(con, new, why)
    return revived


def _generation_due(st: dict) -> list[str]:
    """What calls for a generation pass (once R0 is done): run start, the queue below 2x the round's
    slots with a BO round run since the last pass, or a round analyst's flag since it."""
    if not st["r0_complete"] or st["run_ended"]:
        return []
    last = st["passes"][-1]["seq"] if st["passes"] else 0
    due = ["run start"] if not st["passes"] and not st["hypotheses"] else []
    queue = len(_queue(st))
    slots = max(1, sum(h["status"] == "registered" for h in _selected(st)))  # a round's newcomers
    if queue < 2 * slots and any(n >= 1 and r["seq"] > last for n, r in st["rounds"].items()):
        due.append(f"the queue holds {queue}, below 2x the {slots} slot(s) per round")
    if any(x["kind"] == "narrative" and x["record"]["generation"] and x["seq"] > last
           for x in st["records"]):
        due.append("the round analyst flagged new ground")
    return due


def _exhaustion(st: dict) -> dict:
    """The run-end conditions of exhaustion; the next round-run ends the run once all of them hold
    (a boundary: an llm pass opens before its generators record)."""
    hyps = st["hypotheses"].values()
    final = st["passes"][-1]["pass"] if st["passes"] else None
    return {
        "queue_empty": not _queue(st),
        "none_undecided": not any(h["status"] == "active" for h in hyps),
        "removals_reviewed": not _interplay_missing(st),
        "revivals_run": not _pending_links(st) and not any(
            "revived_from" in h and h["status"] == "registered" for h in hyps),
        # the latest pass, with no trigger since, left nothing registered (nor undecided); an llm
        # pass counts once its generators recorded proposals (each proposes at least 2)
        "final_pass_empty": final is not None and not _generation_due(st) and not any(
            h["status"] == "proposed" or (h["pass"] == final and "registered" in h) for h in hyps)
        and (st["run"]["generation"] == "scripted" or any(h["pass"] == final for h in hyps)),
    }


def cmd_generate(a) -> dict:
    """Open a generation pass. Generator subagents record its proposals; with `generation: scripted`
    the harness serves fixtures/pass<n>.yaml (a list of hypothesis specs) instead, and a pass past the
    last fixture is empty."""
    run_dir, con = _open_run()
    st = elog.load(con)
    _not_ended(st)
    run, n = st["run"], len(st["passes"]) + 1
    specs: list = []
    fixture = Path(run["fixtures"]) / f"pass{n}.yaml" if run["generation"] == "scripted" else None
    if fixture and fixture.exists():
        try:
            specs = yaml.safe_load(fixture.read_text()) or []
            if not isinstance(specs, list):
                raise ValueError("a pass lists hypothesis specs")
            for i, s in enumerate(specs):
                try:
                    hypotheses.validate(s)
                except ValueError as e:
                    raise ValueError(f"[{i}]: {e}")
        except (OSError, yaml.YAMLError, ValueError) as e:
            raise Refused(f"fixture {fixture}: {e}")
    elog.append(con, "generation_pass", a.actor, {"rationale": a.rationale, "pass": n, "mode": run["generation"],
                                                  "triggers": _generation_due(st)})
    ids = [_propose(run_dir, con, s, "harness", f"scripted generation pass {n}") for s in specs]
    _regenerate(run_dir, con)
    return {**_status(elog.load(con)), "pass": n, "mode": run["generation"], "proposed": ids}


def cmd_accept_proxy(a) -> dict:
    """The user's explicit acceptance of a ladder rung that failed calibration."""
    run_dir, con = _open_run()
    st = elog.load(con)
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
        raise Refused(f"{a.fidelity} passed calibration: the harness already chooses among validated rungs")
    elog.append(con, "proxy_accepted_unvalidated", a.actor,
                {"rationale": a.rationale, "fidelity": fidelity, "rho": rung["rho"],
                 "spread": rung["spread"], "sigma": rung["sigma"]})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def cmd_set_delta(a) -> dict:
    """The minimum meaningful effect δ, fixed for the whole run."""
    run_dir, con = _open_run()
    st = elog.load(con)
    if st["delta"] is not None:
        raise Refused(f"δ is already {st['delta']}: it is fixed for the whole run")
    if not st["r0_complete"]:
        raise Refused("δ is set from the calibration round's σ: run `round-run` first")
    if not _positive(a.value):
        raise Refused(f"δ must be a positive number in objective units, not {a.value}")
    elog.append(con, "delta_set", a.actor, {"rationale": a.rationale, "delta": a.value,
                                            "suggested": _suggested_delta(st)})
    _regenerate(run_dir, con)
    return _status(elog.load(con))


def cmd_status(a) -> dict:
    _, con = _open_run(recover=False)
    return _status(elog.load(con))


def cmd_verdict(a) -> dict:
    """A hypothesis's verdict records, latest last; `burn-in` until its first verdict check."""
    run_dir, con = _open_run(recover=False)
    st = elog.load(con)
    h = _hypothesis(st, a.hypothesis)
    out = {"hypothesis": h["id"], "status": h["status"], "records": h["verdicts"]}
    if h["status"] in ("active", "retained"):
        fresh = _fresh(st, h, _next_eligibility(run_dir, st))
        last = _latest(h, st)
        out["verdict"] = last["outcome"] if last else "burn-in"
        out["burn_in"] = {"fresh": len(fresh), **_schedule(h)}
    return out


def cmd_summary(a) -> dict:
    """The research summary (SUMMARY.md) as the log stands now."""
    _, con = _open_run(recover=False)
    return {"summary": _render(elog.load(con))["SUMMARY.md"]}


def cmd_rebuild(a) -> dict:
    """Rewrite every generated file from the log (nothing is logged: they are derived)."""
    run_dir, con = _open_run(recover=False)
    return {"rebuilt": _write(run_dir, elog.load(con))}


def _next_eligibility(run_dir: Path, st: dict):
    """The next round's warm start: its selected hypotheses' levers at the run's fidelity."""
    return _eligibility(st, _space(_selected(st)), _baseline(run_dir), st["fidelity"]["fidelity"])


def cmd_trials(a) -> dict:
    """Every logged trial; --eligible: the next round's warm start, rewritten into its levers."""
    run_dir, con = _open_run(recover=False)
    st = elog.load(con)
    return {"trials": _next_eligibility(run_dir, st)(st["trials"]) if a.eligible else st["trials"]}


def _render(st: dict) -> dict[str, str]:
    return reports.render(st, {
        "status": _status(st), "revivals": {old["id"]: p for old, p in _pending_links(st)},
        "testing": {h["id"]: {"record": _latest(h, st), **_schedule(h)} for h in _in_search(st)
                    if h["status"] == "active"}})


def _regenerate(run_dir: Path, con) -> list[str]:
    """After every state change: log the round summary of each round just ended (the analyst's
    narrative fills it later), then rewrite every generated file."""
    st = elog.load(con)
    if ended := [n for n, r in st["rounds"].items() if r["ended"] and not r.get("summarised")]:
        for n in ended:
            elog.append(con, "round_summary", "harness", {"round": n, "path": f"rounds/{n:03d}.md"})
        st = elog.load(con)
    return _write(run_dir, st)


def _write(run_dir: Path, st: dict) -> list[str]:
    """Rewrite every generated file from the state; -> their paths."""
    files = _render(st)
    for rel, text in files.items():
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")  # a concurrent reader never sees half
        tmp.write_text(text)
        os.replace(tmp, path)
    return sorted(files)


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
    action("smoke", cmd_smoke).add_argument("hypothesis", nargs="?")
    action("propose", cmd_propose).add_argument("--file", required=True)
    action("register", cmd_register).add_argument("hypothesis")
    action("commit-lever", cmd_commit_lever).add_argument("hypothesis")
    action("stop", cmd_stop)
    action("generate", cmd_generate)
    q = action("enqueue", cmd_enqueue)
    q.add_argument("--config", required=True)
    q.add_argument("--expected", required=True, type=float)
    for name, fn in (("park", cmd_park), ("unpark", cmd_unpark)):
        pk = action(name, fn)
        pk.add_argument("hypothesis")
        pk.add_argument("--reason", required=True)
    cp = action("checkpoint", cmd_checkpoint)
    cp.add_argument("--resume", action="store_true")
    cp.add_argument("--revise")  # a YAML revision of the registry: brief, directives, protected_paths
    pr = action("prioritize", cmd_prioritize)
    pr.add_argument("hypothesis")
    pr.add_argument("priority", type=int)
    action("round-run", cmd_round_run)
    c = action("commit-change", cmd_commit_change)
    c.add_argument("--reason", required=True)
    c.add_argument("--breaking", action="store_true")
    action("add-dependency", cmd_add_dependency).add_argument("requirement")
    n = action("narrow", cmd_narrow)
    n.add_argument("hypothesis")
    n.add_argument("--lever", required=True)
    n.add_argument("--low", type=float)
    n.add_argument("--high", type=float)
    n.add_argument("--options")
    action("accept-proxy", cmd_accept_proxy).add_argument("--fidelity", required=True)
    action("set-delta", cmd_set_delta).add_argument("value", type=float)
    r = action("record", cmd_record)
    r.add_argument("kind", choices=sorted(records.KINDS))
    r.add_argument("--file", required=True)
    r.add_argument("--agent-id")  # None for the orchestrator's own records (expected verdicts)
    check = sub.add_parser("check").add_subparsers(dest="check", required=True, parser_class=_Parser)
    c = check.add_parser("recorded")
    c.add_argument("agent_id")
    c.add_argument("--agent")  # the hook's agent_type
    c.set_defaults(fn=cmd_check_recorded, action=False)
    for name, fn, arg in (("write", cmd_check_write, "path"), ("read", cmd_check_read, "path"),
                          ("bash", cmd_check_bash, "command")):
        c = check.add_parser(name)
        c.add_argument(arg)  # after `--`: it may start with a dash
        c.add_argument("--agent", default="orchestrator")  # the hook's agent_type; none: the main session
        if name == "bash":
            c.add_argument("--agent-id")
        c.set_defaults(fn=fn, action=False)
    check.add_parser("stop").set_defaults(fn=cmd_check_stop, action=False)
    sub.add_parser("next").set_defaults(fn=cmd_next, action=False)
    sub.add_parser("status").set_defaults(fn=cmd_status, action=False)
    sub.add_parser("summary").set_defaults(fn=cmd_summary, action=False)
    sub.add_parser("rebuild").set_defaults(fn=cmd_rebuild, action=False)
    t = sub.add_parser("trials")
    t.add_argument("--eligible", action="store_true")
    t.set_defaults(fn=cmd_trials, action=False)
    v = sub.add_parser("verdict")
    v.add_argument("hypothesis")
    v.set_defaults(fn=cmd_verdict, action=False)
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
