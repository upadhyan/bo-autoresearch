"""The state machine: every `boar` command that changes run state, except the engine's trial loops.

Each command checks its preconditions first and raises Refused before touching
anything, so a refused command leaves the run exactly as it was.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
import shutil
import signal
import stat
import sys
from pathlib import Path
from typing import Any, Iterator

from boar import config as cfgmod
from boar import gitops, schema, store
from boar.errors import Refused
from boar.store import ABORTED, ACTIVE, DONE, FINALIZE, PENDING, REJECTED, REMOVED, ROUND, SETUP, Run

REVIEW_HINT = 'boar review record <id> accept|reject --reason "<one paragraph>"'


# --- shared predicates (nextstep uses these too) ----------------------------------


def fresh_round() -> dict:
    return {
        "commit": None,
        "run_status": "not_started",
        "warm": None,
        "incumbent_trial": None,
        "new_trials": [],
        "propose_none": None,
        "closed": False,
    }


def awaiting_verdict(h: dict) -> bool:
    return h["status"] == PENDING and h.get("verdict") is None


def accepted(h: dict) -> bool:
    """Accepted by the reviewer but not yet active."""
    return h["status"] == PENDING and (h.get("verdict") or {}).get("decision") == "accept"


def pending_items(state: dict, hyps: list[dict]) -> list[str]:
    """Every item still waiting for a reviewer verdict."""
    ids = [h["id"] for h in hyps if awaiting_verdict(h)]
    if state["eval"]["status"] == "pending":
        ids.append("eval")
    ids += [rm["id"] for rm in state["removals"] if rm.get("verdict") is None]
    return ids


def eval_changes(run: Run, manifest: dict[str, str] | None, expected_hash: str | None) -> list[str]:
    """What differs in eval/ from a recorded version: eval_diff lines, plus "unreadable: …" lines; [] if nothing.

    With the version's manifest only entries whose stat changed are re-read, so `boar next` and the Stop
    hook stay cheap with large data in eval/. A read error is reported, never raised: the hook must not
    release the session because eval/ can't be read.
    """
    from boar import evalrun

    try:
        if manifest is not None:
            return evalrun.eval_unchanged(run, manifest)
        return [] if evalrun.eval_hash(run) == expected_hash else ["changed: eval/ (no file list was kept)"]
    except OSError as e:
        return [f"unreadable: {e.filename or 'eval/'}: {e.strerror or e}"]


def unreadable_only(changes: list[str]) -> bool:
    return bool(changes) and all(c.startswith("unreadable: ") for c in changes)


def eval_stale(run: Run, state: dict) -> list[str]:
    """What changed in eval/ since its passing check while the eval awaits review ([] if nothing).

    A stale eval can't be accepted: it goes back to the check.
    """
    ev = state["eval"]
    if ev["status"] != "pending":
        return []
    check = store.read_json(run.eval_check_path) or {}
    manifest = check.get("manifest") if check.get("hash") == ev["checked_hash"] else None
    return eval_changes(run, manifest, ev["checked_hash"])


def reviewable_items(run: Run, state: dict, hyps: list[dict]) -> list[str]:
    """pending_items, less a stale eval: it goes back to `boar eval check`, not to the reviewer."""
    items = pending_items(state, hyps)
    return [i for i in items if i != "eval"] if "eval" in items and eval_stale(run, state) else items


def session_id(value: Any) -> str | None:
    """A usable session id, or None; `$…` is an unsubstituted `${CLAUDE_SESSION_ID}`."""
    if not isinstance(value, str) or not value.strip() or value.startswith("$"):
        return None
    return value.strip()


_SESSION_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _session_file(sid: str | None) -> Path | None:
    """${XDG_STATE_HOME:-~/.local/state}/boar/sessions/<sid>, or None for an id that isn't a safe file name."""
    if not sid or not _SESSION_FILE.fullmatch(sid):
        return None
    base = os.environ.get("XDG_STATE_HOME", "")
    # The XDG spec says to ignore a relative path.
    return (Path(base) if os.path.isabs(base) else Path.home() / ".local" / "state") / "boar" / "sessions" / sid


def register_session(sid: str | None, root: Path) -> None:
    """Remember the repo a session drives, so its Stop hook finds the run from any cwd. Best effort."""
    with contextlib.suppress(OSError, RuntimeError):
        path = _session_file(sid)
        if path is not None:
            store.write_text(path, f"{root}\n")


def registered_root(sid: str | None) -> Path | None:
    """The repo `register_session` recorded for `sid`, or None."""
    try:
        path = _session_file(sid)
        text = path.read_text().strip() if path is not None else ""
    except (OSError, RuntimeError, ValueError):
        return None
    return Path(text) if os.path.isabs(text) else None


def undecided(hyps: list[dict], r: int) -> list[str]:
    return [h["id"] for h in store.active_hypotheses(hyps) if str(r) not in h.get("decisions", {})]


def plan_close(state: dict, hyps: list[dict], r: int) -> dict:
    """What `round close` would do: removals and supersedes applied, accepted proposals activated."""
    active = [h["id"] for h in store.active_hypotheses(hyps)]
    removed: dict[str, str] = {}
    for rm in state["removals"]:
        if rm["round"] == r and (rm.get("verdict") or {}).get("decision") == "accept" and rm["hypothesis"] in active:
            removed[rm["hypothesis"]] = f"removal {rm['id']} accepted: {rm['reason']}"
    incoming = [h["id"] for h in hyps if h["round"] == r and accepted(h)]
    for h in hyps:
        if h["id"] in incoming and h.get("supersedes") in active and h["supersedes"] not in removed:
            removed[h["supersedes"]] = f"superseded by {h['id']}"
    after = [hid for hid in active if hid not in removed] + incoming
    return {"removed": removed, "activated": incoming, "active_after": after}


def incumbent_after_close(
    run: Run, state: dict, hyps: list[dict], r: int, removing: set[str] | frozenset[str] = frozenset()
) -> tuple[dict | None, list[str]]:
    """The incumbent once round `r` closes as planned, with `removing` removed as well, and who stays active.

    An accepted removal or a supersede pins the levers at default, so the trials that set them leave the
    warm-start set the incumbent is taken from (rule 2); after the last round nothing searches again.
    """
    from boar import warmstart

    plan = plan_close(state, hyps, r)
    removed = set(plan["removed"]) | set(removing)
    after = [hid for hid in plan["active_after"] if hid not in removed]
    hyps = [
        {**h, "status": REMOVED if h["id"] in removed else ACTIVE if h["id"] in plan["activated"] else h["status"]}
        for h in hyps
    ]
    return warmstart.incumbent(run.trials(), hyps, r + 1, run.config()["direction"]), after


def incumbent_label(inc: dict | None) -> str:
    if not inc:
        return "none"
    cfg = json.dumps(inc["config"], sort_keys=True) if inc["config"] else "the baseline"
    return f"{cfg} metric {inc['metric']:.6g} (trials {', '.join(map(str, inc['trials']))})"


def incumbent_change(before: dict | None, after: dict | None) -> str | None:
    """A warning when closing moves the incumbent off its config, else None."""
    if (before or {}).get("config") == (after or {}).get("config"):
        return None
    return (
        f"closing changes the incumbent from {incumbent_label(before)} to {incumbent_label(after)}: the trials "
        "that set the levers of removed or superseded hypotheses leave the warm-start set"
    )


def decision_text(state: dict, h: dict, r: str) -> str:
    """`h`'s round-r decision; a removal says how its review went, since a rejected one left `h` active."""
    decision = h["decisions"][r]["decision"]
    if decision != "remove":
        return decision
    rm = next((x for x in state.get("removals", []) if x["id"] == f"rm-{h['id']}-r{r}"), None)
    verdict = ((rm or {}).get("verdict") or {}).get("decision")
    return f"remove ({ {'accept': 'accepted', 'reject': 'rejected'}.get(verdict, 'not reviewed')})"


def _reason(reason: str | None) -> str:
    if not reason or not reason.strip():
        raise Refused("--reason must not be empty")
    return reason.strip()


@contextlib.contextmanager
def _editing(run: Run) -> Iterator[tuple[dict, list[dict]]]:
    """Read state and hypotheses under the lock and save both on a clean exit; a Refused saves nothing."""
    with run.locked():
        state, hyps = run.state(), run.hypotheses()
        yield state, hyps
        run.save_hypotheses(hyps)
        run.save_state(state)


def _join(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


# --- init ---------------------------------------------------------------------------


def init(spec: str | Path, session: str | None, overrides: dict, start: Path | None = None) -> str:
    start = (start or Path.cwd()).resolve()
    if not gitops.is_repo(start):
        raise Refused("not inside a git repository; run `boar init` in the target repo (the run's code goes on a branch)")
    root = Path(gitops.git(start, "rev-parse", "--show-toplevel"))
    existing = store.active_run(root)
    if existing is not None:
        phase = existing.state()["phase"]
        if phase not in (DONE, ABORTED):
            raise Refused(
                f"run {existing.id} is still active (phase {phase}); finish it, or end it with `boar abort --reason …`"
            )
    if not gitops.has_commits(root):
        raise Refused("the repo has no commits yet; commit the target code first")
    base_branch = gitops.current_branch(root)
    if not base_branch:
        raise Refused("HEAD is detached; check out the branch the run should start from")
    if gitops.tracked_changes(root):
        raise Refused("tracked files have uncommitted changes; commit or stash them first (the run branches from HEAD)")
    untracked = _untracked_files(root)
    if untracked:
        shown = ", ".join(untracked[:10]) + (f" … and {len(untracked) - 10} more" if len(untracked) > 10 else "")
        spec_abs = Path(spec).resolve()
        spec_rel = spec_abs.relative_to(root).as_posix() if spec_abs.is_relative_to(root) else None
        hint = (
            f" (a spec draft belongs under {store.BOAR_DIR}/, e.g. `boar init --spec {store.BOAR_DIR}/spec-draft.md`)"
            if spec_rel in untracked else ""
        )
        raise Refused(
            f"untracked files would be committed to the run branch by the first `boar round run` (and then vanish "
            f"from disk when you check out your own branch): {shown}; commit them, add them to .gitignore, or move "
            f"them out of the repo first{hint}"
        )
    if not gitops.git(root, "var", "GIT_COMMITTER_IDENT", check=False) or not gitops.git(
        root, "var", "GIT_AUTHOR_IDENT", check=False
    ):
        raise Refused(
            "git can't commit here (no user.name/user.email); set an identity for this repo "
            "(`git config user.name …` and `git config user.email …`) before `boar init`: round commits happen unattended"
        )
    spec = Path(spec)
    try:
        spec_text = spec.read_text()
    except OSError as e:
        raise Refused(f"can't read spec {spec}: {e.strerror or e}") from None
    direction = cfgmod.direction_from_spec(spec_text)
    config = cfgmod.build_config(overrides, direction)

    run = Run(root=root, id=_new_run_id(root))
    branch = f"boar/{run.id}"
    state = {
        "run_id": run.id,
        "phase": SETUP,
        "round": 0,
        "rounds_total": config["rounds"],
        "session_id": session_id(session),
        "branch": branch,
        "base_branch": base_branch,
        "base_commit": gitops.head_commit(root),
        "created": store.now_iso(),
        "next_hypothesis": 1,
        "eval": {"status": "none", "checked_hash": None, "accepted_hash": None, "rejections": 0},
        "setup_failed_batches": 0,
        "rounds": {},
        "removals": [],
        "finalize": {"status": "not_started", "commit": None, "incumbent_config": None},
        "abort_reason": None,
    }
    gitops.exclude_boar_dir(root)
    run.dir.mkdir(parents=True)
    try:
        store.write_text(run.spec_path, spec_text)
        store.write_json(run.config_path, config)
        run.save_hypotheses([])
        run.save_state(state)
        gitops.create_branch(root, branch)
    except BaseException:
        shutil.rmtree(run.dir, ignore_errors=True)
        raise
    store.write_text(root / store.BOAR_DIR / "active", run.id + "\n")
    register_session(state["session_id"], root)
    settings = " ".join(f"{k}={v}" for k, v in config.items())
    lines = [
        f"run {run.id} started",
        f"branch: {branch} (from {base_branch} at {state['base_commit'][:12]})",
        f"run dir: {run.dir}",
        f"config: {settings}",
    ]
    if base_branch.startswith("boar/"):
        lines.append(
            f"warning: {base_branch} is an earlier BOAR run's branch, so this run builds on that run's code. "
            "To start from your own branch instead: `boar abort --reason …`, `git checkout <your branch>`, then init again"
        )
    return "\n".join(lines)


def _untracked_files(root: Path) -> list[str]:
    """Untracked, non-ignored files outside .boar/; `round run` commits everything in the worktree."""
    out = gitops.git(root, "ls-files", "--others", "--exclude-standard", "-z")
    paths = [p for p in out.split("\0") if p]
    return [p for p in paths if not p.startswith(f"{store.BOAR_DIR}/") and "__pycache__/" not in p and not p.endswith(".pyc")]


def _new_run_id(root: Path) -> str:
    base = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        cand = base if n == 1 else f"{base}-{n}"
        taken = (root / store.BOAR_DIR / cand).exists() or gitops.git(
            root, "rev-parse", "--verify", "--quiet", f"refs/heads/boar/{cand}", check=False
        )
        if not taken:
            return cand
    raise Refused(f"can't find a free run id for {base}")


# --- propose ------------------------------------------------------------------------


def _propose_round(state: dict, hyps: list[dict]) -> int:
    """0 in setup, r in R4 of round r; refuses anywhere else."""
    phase = state["phase"]
    if phase == SETUP:
        return 0
    if phase == ROUND:
        r = state["round"]
        status = state["rounds"][str(r)]["run_status"]
        if status == "done":
            missing = undecided(hyps, r)
            if not missing:
                return r
            now = f"R3 (decide on {_join(missing)} first)"
        elif status == "running":
            now = f"R2 (round {r} was interrupted; resume it with `boar round run`)"
        else:
            now = f"R1 (round {r} has not run yet)"
    else:
        now = phase
    raise Refused(f"propose is allowed in setup or R4; now in {now}")


def propose(path: str | Path, start: Path | None = None) -> str:
    run = store.require_live_run(start)
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except OSError as e:
        raise Refused(f"can't read {path}: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise Refused(f"{path} is not valid JSON: {e}") from None
    with _editing(run) as (state, hyps):
        r = _propose_round(state, hyps)
        active = store.active_hypotheses(hyps)
        blocked = {h["id"] for h in active if (h["decisions"].get(str(r)) or {}).get("decision") == "blocked"}
        errors = schema.validate(
            data, set(store.all_levers(hyps)), {h["id"] for h in active}, allow_supersedes=r > 0, blocked_ids=blocked,
            space=store.search_space(hyps),
        )
        if errors:
            lines = "\n".join(f"  - {e}" for e in errors)
            raise Refused(f"{path} has {len(errors)} problem(s); nothing was added:\n{lines}")
        ids = []
        for p in data:
            hid = f"H{state['next_hypothesis']}"
            state["next_hypothesis"] += 1
            ids.append(hid)
            hyps.append(
                {
                    "id": hid,
                    "round": r,
                    "status": PENDING,
                    "statement": p["statement"].strip(),
                    "mechanism": p["mechanism"].strip(),
                    "source": p["source"],
                    "citations": list(p.get("citations", [])),
                    "levers": [schema.normalize_lever(lever) for lever in p["levers"]],
                    "supersedes": p.get("supersedes"),
                    "enables": p.get("enables"),
                    "joint_config": p.get("joint_config"),
                    "verdict": None,
                    "activated_round": None,
                    "removed_round": None,
                    "removed_reason": None,
                    "observable_key": None,
                    "decisions": {},
                }
            )
    where = "setup" if r == 0 else f"round {r}"
    return f"proposed {', '.join(ids)} ({where}); each is pending review"


def propose_none(reason: str | None, start: Path | None = None) -> str:
    run = store.require_live_run(start)
    reason = _reason(reason)
    with _editing(run) as (state, hyps):
        r = _propose_round(state, hyps)
        if r == 0:
            raise Refused("`propose --none` is only for R4; setup needs at least one hypothesis")
        state["rounds"][str(r)]["propose_none"] = reason
    return f"round {r}: no new proposals ({reason})"


# --- eval check ---------------------------------------------------------------------


def eval_check(foreground: bool = True, start: Path | None = None) -> tuple[str, bool]:
    """Run the baseline check inline, or start it in a detached worker (a passing trial may outlast the shell tool)."""
    run = store.require_live_run(start)
    from boar import worker

    if foreground:
        # Holding the worker slot keeps a second check or round from measuring alongside this one.
        with worker.claim(run, worker.EVAL, None):
            return eval_check_body(run)
    rec = worker.spawn(run, worker.EVAL, None)
    return (
        f"eval check is running in the background (pid {rec['pid']}, log {rec['log']}). "
        "Run `boar wait` until it finishes.",
        True,
    )


def eval_check_body(run: Run) -> tuple[str, bool]:
    """The check itself: one baseline trial, then the eval's review status moves accordingly."""
    from boar import evalrun

    result = evalrun.eval_check(run)
    ok = bool(result.get("ok"))
    new_hash = result.get("hash")

    def apply(state: dict) -> dict:
        ev = state["eval"]
        if ev["status"] == "rejected":
            ev["checked_since_reject"] = True
        if ev["status"] != "accepted":
            if ok:
                ev["status"], ev["checked_hash"] = "pending", new_hash
            else:
                ev["checked_hash"] = None
                if ev["status"] != "rejected":
                    ev["status"] = "none"
        return dict(ev)

    ev = run.update_state(apply)
    lines = [f"eval check: {'PASS' if ok else 'FAIL'}", f"eval hash: {new_hash}"]
    trial = result.get("trial") or {}
    if trial:
        metric = trial.get("metric")
        reps = ", ".join(f"{x:g}" for x in trial.get("repeats") or [] if isinstance(x, (int, float)))
        dur = trial.get("duration_s")
        lines.append(
            f"baseline trial: {trial.get('state')}, metric {metric if metric is not None else 'n/a'}"
            + (f" (repeats {reps})" if reps else "")
            + (f", {dur:.1f} s for all repeats" if isinstance(dur, (int, float)) else "")
            + (f" (limit {result['limit_s']:g} s)" if isinstance(result.get("limit_s"), (int, float)) else "")
        )
        if trial.get("error"):
            lines.append(f"error: {trial['error']}")
    problems = result.get("problems") or []
    if problems:
        lines.append("problems:")
        lines += [f"  - {p}" for p in problems]
    lines += [f"warning: {w}" for w in result.get("warnings") or []]
    if ev["status"] == "accepted":
        if new_hash != ev["accepted_hash"]:
            lines.append(
                "WARNING: eval/ has changed since the reviewer accepted it (accepted hash "
                f"{ev['accepted_hash']}). The eval is frozen: `boar round run` and `boar finalize` refuse until "
                "eval/ matches the accepted version again. "
                + (
                    "Run `boar eval restore` to put the accepted version back."
                    if ev.get("accepted_manifest") is not None
                    else "Restore it, or abort and start a new run."
                )
            )
        else:
            lines.append("eval status: accepted (unchanged)")
    elif ok:
        lines.append("eval status: pending review; send `eval` to the boar-reviewer")
    else:
        lines.append(f"eval status: {ev['status']}; fix the problems and run `boar eval check` again")
    return "\n".join(lines), ok


# --- review -------------------------------------------------------------------------


def _removal(state: dict, rid: str) -> dict:
    for rm in state["removals"]:
        if rm["id"] == rid:
            return rm
    raise Refused(f"no removal {rid!r} in this run")


def review_record(item: str, decision: str, reason: str | None, start: Path | None = None) -> str:
    run = store.require_live_run(start)
    reason = _reason(reason)
    if decision not in ("accept", "reject"):
        raise Refused(f"decision must be accept or reject, got {decision!r}")
    verdict = {"decision": decision, "reason": reason, "at": store.now_iso()}
    with _editing(run) as (state, hyps):
        waiting_before = sum(awaiting_verdict(h) for h in hyps if h["round"] == 0)
        if item == "eval":
            kind, rnd = "eval", state["round"]
            _record_eval(run, state["eval"], decision)
        elif item.startswith("rm-"):
            rm = _removal(state, item)
            if rm.get("verdict") is not None:
                raise Refused(f"{item} is not pending: it already has a verdict ({rm['verdict']['decision']})")
            kind, rnd = "removal", rm["round"]
            rm["verdict"] = verdict
        else:
            h = store.hypothesis(hyps, item)
            if not awaiting_verdict(h):
                got = f"status {h['status']}" + (f", verdict {h['verdict']['decision']}" if h.get("verdict") else "")
                raise Refused(f"{item} is not pending review ({got})")
            kind, rnd = "proposal", h["round"]
            h["verdict"] = verdict
            if decision == "reject":
                h["status"] = REJECTED
        notes = _setup_bookkeeping(state, hyps, waiting_before) if state["phase"] == SETUP else []
    run.append_review(
        {"item": item, "kind": kind, "decision": decision, "reason": reason, "round": rnd, "at": verdict["at"]}
    )
    return "\n".join([f"recorded: {item} {decision}", *notes])


def _record_eval(run: Run, ev: dict, decision: str) -> None:
    if ev["status"] != "pending":
        raise Refused(f"the eval is not pending review (status {ev['status']}); it needs a passing `boar eval check` first")
    if decision == "reject":
        ev["status"] = "rejected"
        ev["rejections"] += 1
        ev["checked_since_reject"] = False
        return
    check = store.read_json(run.eval_check_path) or {}
    if not check.get("ok") or check.get("hash") != ev["checked_hash"]:
        raise Refused("the last `boar eval check` did not pass on this eval; run it again before review")
    from boar import evalrun

    manifest = check.get("manifest")
    if eval_changes(run, manifest, ev["checked_hash"]):
        raise Refused("eval/ changed since its last passing `boar eval check`; run the check again, then review it")
    # The freeze needs more than a hash: a file list to name what changed, and a copy to restore from.
    ev["accepted_manifest"] = manifest if manifest is not None else evalrun.eval_manifest(run)
    _copy_eval(run.eval_dir, run.eval_accepted_dir)
    ev["accepted_hash"] = ev["checked_hash"]
    ev["status"] = "accepted"


def _copy_eval(src: Path, dst: Path) -> None:
    """Replace `dst` with a copy of eval/ `src`: symlinks kept as links, bytecode left out (as eval_hash does)."""
    if os.path.lexists(dst):
        _rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _setup_bookkeeping(state: dict, hyps: list[dict], waiting_before: int) -> list[str]:
    """Count a failed proposal batch, and start round 1 once setup's done-when holds."""
    notes = []
    setup = [h for h in hyps if h["round"] == 0]
    waiting = sum(awaiting_verdict(h) for h in setup)
    acc = [h for h in setup if accepted(h)]
    if waiting_before and not waiting:
        if setup and not acc:
            state["setup_failed_batches"] += 1
            notes.append(f"no setup proposal was accepted (failed attempt {state['setup_failed_batches']}/3)")
    if state["eval"]["status"] == "accepted" and not waiting and acc:
        for h in acc:
            h["status"] = ACTIVE
            h["activated_round"] = 1
        state["phase"], state["round"] = ROUND, 1
        state["rounds"]["1"] = fresh_round()
        notes.append(f"setup complete: {_join([h['id'] for h in acc])} active; round 1 begins")
    return notes


# --- review material (read-only) ------------------------------------------------------


def _proposal_view(h: dict) -> dict:
    keys = ("id", "round", "statement", "mechanism", "source", "citations", "levers", "supersedes", "enables", "joint_config")
    return {k: h.get(k) for k in keys}


def _block(obj: Any) -> list[str]:
    return ["  " + line for line in json.dumps(obj, indent=2).splitlines()]


def _material(run: Run, state: dict, hyps: list[dict], item: str) -> list[str]:
    if item == "eval":
        return _eval_material(run, state)
    if item.startswith("rm-"):
        rm = _removal(state, item)
        h = store.hypothesis(hyps, rm["hypothesis"])
        view = {**_proposal_view(h), "activated_round": h.get("activated_round")}
        by_id = {t["trial"]: t for t in run.trials()}
        lines = [f"== {item}: removal of {rm['hypothesis']} proposed in round {rm['round']} =="]
        if rm.get("verdict"):
            lines.append(f"verdict: {rm['verdict']['decision']}: {rm['verdict']['reason']}")
        lines += ["hypothesis:", *_block(view), f"stated reason: {rm['reason']}", "cited trials (from trials.jsonl):"]
        lines += [f"  {json.dumps(by_id[t]) if t in by_id else f'trial {t}: not found'}" for t in rm["trials"]]
        lines.append(f"round summary (incumbent, noise floor, lever effects): {run.summary_path(rm['round'])}")
        lines.append(f"observable key: {h.get('observable_key') or 'none'} (the observable itself is stated in {run.research_path})")
        for rnd, dec in sorted(h["decisions"].items(), key=lambda kv: int(kv[0])):
            if dec["decision"] == "investigate":
                lines.append(f"investigated in round {rnd}: cause: {dec['cause']}; measured: {', '.join(dec['measure'])}")
        dirs = [str(p) for t in rm["trials"] for p in sorted(run.trial_dir(t).glob("diag-*"))]
        lines += ["diagnostics directories of the cited trials:", *(f"  {p}" for p in dirs or ["(none)"])]
        return lines
    h = store.hypothesis(hyps, item)
    where = "setup" if h["round"] == 0 else f"round {h['round']}"
    lines = [f"== {item}: proposal ({where}) =="]
    if h["status"] != PENDING or h.get("verdict"):
        verdict = h.get("verdict")
        lines.append(f"status: {h['status']}" + (f"; verdict {verdict['decision']}: {verdict['reason']}" if verdict else ""))
    lines += _block(_proposal_view(h))
    if h.get("supersedes"):
        lines.append(f"supersedes {h['supersedes']}; see it with `boar review show {h['supersedes']}`")
    if h.get("enables"):
        blocked = store.hypothesis(hyps, h["enables"])
        dec = (blocked.get("decisions") or {}).get(str(h["round"])) or {}
        by_id = {t["trial"]: t for t in run.trials()}
        lines += [f"enables {blocked['id']}, decided blocked in round {h['round']}:", *_block(_proposal_view(blocked)),
                  f"blocked because: {dec.get('reason')}", "trials that reason cites (from trials.jsonl):"]
        lines += [f"  {json.dumps(by_id[t]) if t in by_id else f'trial {t}: not found'}" for t in dec.get("trials") or []]
    return lines


def _eval_material(run: Run, state: dict) -> list[str]:
    ev = state["eval"]
    lines = [f"== eval (status {ev['status']}) ==", f"eval dir: {run.eval_dir}"]
    files = []
    if run.eval_dir.is_dir():
        for p in sorted(run.eval_dir.rglob("*")):
            rel = p.relative_to(run.eval_dir)
            if p.is_file() and "__pycache__" not in rel.parts and p.suffix != ".pyc":
                files.append(f"  {rel} ({p.stat().st_size} bytes{', executable' if os.access(p, os.X_OK) else ''})")
    lines += [
        "files (read every source file; sample each data file enough to judge it, compressed ones with "
        "`zcat <f> | head`):",
        *(files or ["  (none)"]),
    ]
    check = store.read_json(run.eval_check_path)
    if check:
        trial = check.get("trial") or {}
        lines.append(
            f"last `boar eval check`: {'PASS' if check.get('ok') else 'FAIL'} at {check.get('at')}; "
            f"baseline {trial.get('state')}, metric {trial.get('metric')}, "
            f"{trial.get('duration_s')} s for all repeats (limit {check.get('limit_s')} s)"
        )
        lines += [f"  problem: {p}" for p in check.get("problems") or []]
        lines += [f"  warning: {w}" for w in check.get("warnings") or []]
        with contextlib.suppress(Exception):
            same = not eval_changes(run, check.get("manifest"), check.get("hash"))
            lines.append(f"eval/ now: {'unchanged since that check' if same else 'CHANGED since that check'}")
    else:
        lines.append("last `boar eval check`: never run")
    return lines


def review_pending(start: Path | None = None) -> str:
    run = store.require_run(start)
    state, hyps = run.state(), run.hypotheses()
    items = reviewable_items(run, state, hyps)
    lines = [f"spec: {run.spec_path}  (judge every item against the spec, and nothing else)"]
    if "eval" in pending_items(state, hyps) and "eval" not in items:
        lines.append("eval: not ready for review (eval/ changed since its passing check); record no verdict on it")
    if not items:
        return "\n".join([*lines, "nothing is pending review"])
    lines.append(f"pending: {', '.join(items)}")
    for item in items:
        lines += ["", *_material(run, state, hyps, item)]
    return "\n".join([*lines, "", f"record each verdict with: {REVIEW_HINT}"])


def review_show(item: str, start: Path | None = None) -> str:
    run = store.require_run(start)
    state, hyps = run.state(), run.hypotheses()
    return "\n".join([f"spec: {run.spec_path}", *_material(run, state, hyps, item)])


# --- R3 decisions ---------------------------------------------------------------------


def parse_trials(tokens: list[str] | None) -> list[int]:
    """Trial ids given space- and/or comma-separated, deduplicated in order."""
    out: list[int] = []
    for tok in tokens or []:
        for part in tok.replace(",", " ").split():
            try:
                n = int(part)
            except ValueError:
                raise Refused(f"--trials takes trial ids (integers), got {part!r}") from None
            if n not in out:
                out.append(n)
    return out


DECISIONS = ("keep", "fix", "investigate", "blocked", "remove")
MAX_QUEUE = 2


def round_size(cfg: dict, rec: dict) -> int:
    """N trials, plus one for each extra config the round queued and did not find measured already."""
    warm = rec.get("warm") or {}
    return cfg["trials_per_round"] + sum(q["label"] not in (warm.get("same_as") or {}) for q in warm.get("queue") or [])


def investigation(h: dict, r: int) -> dict | None:
    """`h`'s round-`r` decision if it was investigate, else None."""
    dec = (h.get("decisions") or {}).get(str(r)) or {}
    return dec if dec.get("decision") == "investigate" else None


def parse_names(tokens: list[str] | None) -> list[str]:
    """Names given space- and/or comma-separated, deduplicated in order."""
    out: list[str] = []
    for tok in tokens or []:
        for part in tok.replace(",", " ").split():
            if part not in out:
                out.append(part)
    return out


def _queue(path: str, h: dict, hyps: list[dict]) -> list[dict]:
    """The --queue file's partial configs, checked against the current search space; refuses on any problem."""
    from boar import warmstart

    try:
        data = json.loads(Path(path).read_text())
    except OSError as e:
        raise Refused(f"can't read {path}: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise Refused(f"{path} is not valid JSON: {e}") from None
    if not isinstance(data, list) or not data or not all(isinstance(c, dict) for c in data):
        raise Refused(f"{path} must hold a JSON list of partial lever configs (objects)")
    if len(data) > MAX_QUEUE:
        raise Refused(f"{path} queues {len(data)} configs; an investigation queues at most {MAX_QUEUE}")
    space = store.search_space(hyps)
    problems = []
    for i, config in enumerate(data):
        for name, value in config.items():
            if name not in space:
                problems.append(f"config[{i}]: {name!r} is not a lever of an active hypothesis")
            elif not warmstart.allowed(space[name], value):
                problems.append(f"config[{i}]: {json.dumps(value)} is not a valid value of {name!r}")
    if problems:
        raise Refused(f"{path} has {len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems))
    own = {lever["name"]: lever["default"] for lever in h["levers"]}
    if not any(all(n in c and schema._same(c[n], v) for n, v in own.items()) for c in data):
        listed = json.dumps(own, sort_keys=True)
        raise Refused(f"{path} needs the off-state among its configs: one that sets {h['id']}'s levers to their defaults, {listed}")
    return data


def r4_owed(state: dict, hyps: list[dict], r: int) -> list[str]:
    """What R4 of round `r` still owes: a proposal superseding each investigated hypothesis resolved keep, and one
    enabling each hypothesis decided blocked. `propose --none` settles both."""
    if state["rounds"][str(r)].get("propose_none"):
        return []
    owed = []
    for h in store.active_hypotheses(hyps):
        decision = (h["decisions"].get(str(r)) or {}).get("decision")
        if decision == "keep" and investigation(h, r - 1):
            if not any(x["round"] == r and x.get("supersedes") == h["id"] for x in hyps):
                owed.append(f"a proposal that supersedes {h['id']} (its investigation resolved keep)")
        elif decision == "blocked":
            if not any(x["round"] == r and x.get("enables") == h["id"] for x in hyps):
                owed.append(f"a proposal that enables {h['id']} (decided blocked)")
    return owed


def diag_keys(trials: list[dict]) -> list[str]:
    """Every diagnostic key some trial measured, sorted."""
    return sorted({k for t in trials for k in (t.get("diagnostics") or {})})


def decide(
    hid: str, decision: str, reason: str | None, trials: list[int], start: Path | None = None,
    observable_key: str | None = None, cause: str | None = None, measure: list[str] | None = None,
    queue: str | None = None,
) -> str:
    run = store.require_live_run(start)
    if decision not in DECISIONS:
        raise Refused(f"decision must be one of {', '.join(DECISIONS)}, got {decision!r}")
    measured_now = parse_names(measure)
    if decision != "investigate" and (cause or measured_now or queue):
        raise Refused("--cause, --measure and --queue go with investigate only")
    if decision == "investigate":
        cause = (cause or "").strip()
        if not cause:
            raise Refused("investigate needs --cause: the suspected cause the next round measures")
        if not measured_now:
            raise Refused("investigate needs --measure: the diagnostic keys the next R1 adds")
        if not (reason and reason.strip()):
            reason = cause
    reason = _reason(reason)
    with _editing(run) as (state, hyps):
        if state["phase"] != ROUND:
            raise Refused(f"decide is for R3 of a round; now in {state['phase']}")
        r = state["round"]
        if state["rounds"][str(r)]["run_status"] != "done":
            raise Refused(f"round {r} has not finished running; decide after `boar round run` completes")
        h = store.hypothesis(hyps, hid)
        if h["status"] != ACTIVE:
            raise Refused(f"{hid} is not active (status {h['status']}); only active hypotheses get decisions")
        n_rounds = run.config()["rounds"]
        if decision == "fix" and r >= n_rounds:
            raise Refused(
                f"round {r} is the last round ({r}/{n_rounds}): no R1 follows to repair {hid}, and a fix would drop the "
                "trials that set its levers from the final incumbent; decide keep and leave the bug for Caveats"
            )
        if decision == "investigate":
            if r >= n_rounds - 1:
                raise Refused(
                    f"round {r} is one of the last two rounds ({r}/{n_rounds}): an investigation needs a round to measure "
                    "and a round after it to act on what it finds"
                )
            earlier = [s for s in (h.get("decisions") or {}) if int(s) < r and investigation(h, int(s))]
            if earlier:
                raise Refused(f"{hid} was investigated in round {earlier[0]}; a hypothesis gets one investigation")
        probe = investigation(h, r - 1)
        if decision == "blocked" and not probe:
            raise Refused(
                f"blocked resolves an investigation, and {hid} was not investigated in round {r - 1}; "
                "investigate it first, or decide keep, fix or remove"
            )
        if probe and decision in ("keep", "blocked", "remove") and not trials:
            raise Refused(
                f"{hid} was investigated in round {r - 1}: cite the trials whose diagnostics resolve it "
                f"({', '.join(probe['measure'])}) with --trials"
            )
        if decision in ("fix", "remove") and not trials:
            raise Refused(f"{decision} needs --trials: cite the trial ids that show it")
        all_trials = run.trials()
        known = {t["trial"] for t in all_trials}
        unknown = [t for t in trials if t not in known]
        if unknown:
            raise Refused(f"trial(s) {', '.join(map(str, unknown))} are not in this run's trials.jsonl")
        if probe and decision in ("keep", "blocked", "remove"):
            cited = diag_keys([t for t in all_trials if t["trial"] in trials])
            missing = [k for k in probe["measure"] if k not in cited]
            if missing:
                raise Refused(
                    f"no cited trial measured {', '.join(missing)}, which the round {r - 1} investigation of {hid} "
                    f"named; cite trials that did, or decide fix if the instrumentation never wrote it"
                )
        measured = diag_keys(all_trials)
        if observable_key is not None and observable_key not in measured:
            raise Refused(
                f"no trial measured diagnostic {observable_key!r} (measured: {_join(measured)}); the observable's "
                "instrumentation must write it to diag.json in $BOAR_DIAG_DIR"
            )
        if observable_key is None and decision != "fix" and not h.get("observable_key"):
            raise Refused(
                f"{hid} has no observable key yet: its first decision other than fix binds the observable stated in "
                f"research.md to the diagnostic that measures it, with --observable-key <key> (measured: {_join(measured)}). "
                "If the instrumentation never wrote it, decide fix"
            )
        rid = f"rm-{hid}-r{r}"
        old = next((rm for rm in state["removals"] if rm["id"] == rid), None)
        judged = ((old or {}).get("verdict") or {}).get("decision")
        if judged == "accept":
            raise Refused(f"{rid} was accepted by the reviewer; {hid}'s round {r} decision is final")
        if judged == "reject" and decision == "remove":
            raise Refused(
                f"the reviewer rejected {rid} ({old['verdict']['reason']}); {hid} can't be proposed for removal "
                f"again this round. Decide keep{'' if r >= n_rounds else ', or fix if the reason points to a bug'}"
                f"{'' if r >= n_rounds - 1 else ', or investigate what the reason leaves open'}"
            )
        if decision == "remove":
            _refuse_unpaired_removal(hyps, h, [t for t in all_trials if t["trial"] in trials])
            _refuse_final_removal(run, state, hyps, r, hid)
        # A judged removal stays on record (the report lists rejections); only a pending one is replaced.
        if judged is None:
            state["removals"] = [rm for rm in state["removals"] if rm["id"] != rid]
        else:
            old = None
        if decision == "remove":
            state["removals"].append(
                {"id": rid, "hypothesis": hid, "round": r, "reason": reason, "trials": trials, "verdict": None}
            )
        record = {"decision": decision, "reason": reason, "trials": trials, "at": store.now_iso()}
        if decision == "investigate":
            record.update(cause=cause, measure=measured_now, queue=_queue(queue, h, hyps) if queue else [])
        h["decisions"][str(r)] = record
        if observable_key is not None:
            h["observable_key"] = observable_key
    tail = {
        "keep": "stays active",
        "fix": "stays active; R1 of the next round repairs it",
        "investigate": "stays active; R1 of the next round adds the measurements, and R2 runs any queued configs",
        "blocked": "stays active; R4 proposes a hypothesis that enables it",
        "remove": f"pending removal {rid} needs a reviewer verdict",
    }[decision]
    note = "; replaced the pending removal" if old is not None and decision == "remove" else ""
    dropped = "; dropped the pending removal" if old is not None and decision != "remove" else ""
    return f"{hid}: {decision} (round {r}): {tail}{note}{dropped}"


def _refuse_unpaired_removal(hyps: list[dict], h: dict, cited: list[dict]) -> None:
    """An active enabler and the hypothesis it enables go only on a cited trial that moved both."""
    partners = [x for x in store.active_hypotheses(hyps) if x.get("enables") == h["id"]]
    if h.get("enables") and store.hypothesis(hyps, h["enables"])["status"] == ACTIVE:
        partners.append(store.hypothesis(hyps, h["enables"]))
    defaults = store.lever_defaults(hyps)

    def moved(t: dict, x: dict) -> bool:
        return any(lv["name"] in store.non_default(t["config"], defaults) for lv in x["levers"])

    for x in partners:
        if not any(moved(t, h) and moved(t, x) for t in cited):
            raise Refused(
                f"{h['id']} and {x['id']} are a blocked hypothesis and its enabler: cite a trial that set levers of "
                f"both away from default (the joint trial, say) before removing either"
            )


def _refuse_final_removal(run: Run, state: dict, hyps: list[dict], r: int, hid: str) -> None:
    """Refuse a removal that would change the final incumbent: in the last round no round follows to search again.

    The other removals of round `r` still awaiting or holding an accepting verdict count as if accepted.
    """
    n_rounds = run.config()["rounds"]
    if r < n_rounds:
        return  # a later round searches the smaller space again (R4's proposals still count; R6 warns on a move)
    others = {
        rm["hypothesis"] for rm in state["removals"]
        if rm["round"] == r and rm["hypothesis"] != hid and (rm.get("verdict") or {}).get("decision") != "reject"
    }
    after, _ = incumbent_after_close(run, state, hyps, r, others | {hid})
    before, _ = incumbent_after_close(run, state, hyps, r, others)
    if (before or {}).get("config") == (after or {}).get("config"):
        return
    raise Refused(
        f"round {r} is the last round ({r}/{n_rounds}): an accepted removal of {hid} pins its levers at default, so the trials that set them drop out of "
        f"the final warm-start set and the final incumbent would change from {incumbent_label(before)} to "
        f"{incumbent_label(after)}, with no round left to search again. Decide keep and argue the lack of effect "
        "in the report's What didn't"
    )


# --- round run / close, finalize --------------------------------------------------------


def _require_branch(run: Run, state: dict) -> str | None:
    current = gitops.current_branch(run.root)
    if current != state["branch"]:
        return f"the run's code lives on {state['branch']} but {current or 'a detached HEAD'} is checked out; `git checkout {state['branch']}` first"
    return None


def eval_frozen(run: Run, state: dict) -> str | None:
    """Why the accepted eval can't be used as it stands (naming what changed, or what can't be read), or None."""
    ev = state["eval"]
    if ev["status"] != "accepted":
        return f"the eval is not accepted (status {ev['status']})"
    from boar import evalrun

    manifest = ev.get("accepted_manifest")
    changes = eval_changes(run, manifest, ev["accepted_hash"])
    if not changes:
        return None
    what = "eval/ could not be read" if unreadable_only(changes) else "eval/ has changed since the reviewer accepted it"
    fix = (
        "run `boar eval restore` to put the accepted version back" if manifest is not None
        else "restore the accepted version, or abort and start a new run"
    )
    return f"{what} ({evalrun.diff_text(changes)}); the eval is frozen: {fix}"


# shutil.rmtree's error hook: `onerror` is deprecated from 3.12 in favour of `onexc`.
_ONEXC = "onexc" if sys.version_info >= (3, 12) else "onerror"


def _rmtree(path: Path) -> None:
    """Remove a tree, read-only directories in it too (a data dir copied with `cp -r` keeps its mode).

    A symlink or a file is unlinked, never followed. Raises OSError if something still can't be removed.
    """
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
        return
    top, seen = str(path), set()

    def retry(func, p, exc):
        if p != top:  # first: an entry of a directory without x can't even be looked at
            with contextlib.suppress(OSError):
                os.chmod(os.path.dirname(p), stat.S_IRWXU)
        try:
            os.lstat(p)
        except FileNotFoundError:
            return
        except OSError:
            pass  # still unreachable: removing it below raises
        if p in seen:
            raise exc if isinstance(exc, BaseException) else exc[1]
        seen.add(p)
        if func is not os.rmdir and os.path.isdir(p) and not os.path.islink(p):
            os.chmod(p, stat.S_IRWXU)
            shutil.rmtree(p, **{_ONEXC: retry})
        else:
            func(p)

    shutil.rmtree(path, **{_ONEXC: retry})


def eval_restore(start: Path | None = None) -> str:
    """Put the eval snapshot taken at acceptance back in eval/, and prove it by the accepted hash."""
    run = store.require_live_run(start)
    from boar import evalrun

    # The launch lock keeps a worker from starting (and reading eval/) while it is swapped.
    with run.launch_locked():
        rec = run.running()
        if rec is not None:
            raise Refused(f"`boar {rec.get('command')}` is still running (pid {rec['pid']}); wait for it with `boar wait`")
        ev = run.state()["eval"]
        if ev["status"] != "accepted":
            raise Refused(f"the eval is not accepted (status {ev['status']}); there is no accepted version to restore")
        snap = run.eval_accepted_dir
        if ev.get("accepted_manifest") is None or not snap.is_dir():
            raise Refused(
                f"no copy of the accepted eval was kept at {snap}; restore eval/ by hand, or abort and start a new run"
            )
        if run.eval_dir.is_symlink():
            raise Refused(
                f"eval/ is a symlink (to {os.readlink(run.eval_dir)}), not the run's own copy; remove the link "
                f"(`rm {run.eval_dir}`; what it points at is left alone) and run `boar eval restore` again"
            )
        changes = eval_changes(run, ev["accepted_manifest"], ev["accepted_hash"])
        if not changes:
            return "eval/ already matches the accepted version; nothing to restore"
        undone = evalrun.diff_text(changes)
        aside = run.path("eval.replaced")
        try:
            # Left over if an earlier restore was killed midway; eval/ itself was put back or copied by then.
            _rmtree(aside)
        except OSError as e:
            raise Refused(f"can't remove {aside} left by an earlier restore ({e}); remove it by hand, then restore again") from None
        if os.path.lexists(run.eval_dir):
            run.eval_dir.rename(aside)
        restored, failure = False, None
        try:
            shutil.copytree(snap, run.eval_dir, symlinks=True)
            restored = evalrun.eval_hash(run) == ev["accepted_hash"]
        except OSError as e:
            failure = e
        finally:
            if not restored:
                with contextlib.suppress(OSError):
                    _rmtree(run.eval_dir)
                if os.path.lexists(aside) and not os.path.lexists(run.eval_dir):
                    aside.rename(run.eval_dir)
        if not restored:
            why = f"copying it back failed ({failure})" if failure else "it does not match the accepted hash"
            raise Refused(
                f"the copy at {snap} could not be restored: {why}, so eval/ was left as it was; restore it by "
                "hand, or abort and start a new run"
            )
        try:
            _rmtree(aside)
        except OSError as e:
            return f"restored eval/ to the accepted version (undid: {undone}); remove {aside} by hand ({e})"
    return f"restored eval/ to the accepted version (undid: {undone})"


def round_run(foreground: bool, start: Path | None = None) -> str:
    run = store.require_live_run(start)
    state, hyps, config = run.state(), run.hypotheses(), run.config()
    if state["phase"] == SETUP:
        raise Refused("setup isn't done; `boar next` says what is missing")
    if state["phase"] != ROUND:
        raise Refused("all rounds are closed; run `boar finalize`")
    r = state["round"]
    if state["rounds"][str(r)]["run_status"] == "done":
        raise Refused(f"round {r} has already run; decide, propose and review, then `boar round close`")
    problems = []
    pend = pending_items(state, hyps)
    if pend:
        problems.append(f"verdicts are pending for {', '.join(pend)}; the boar-reviewer must judge them first")
    frozen = eval_frozen(run, state)
    if frozen:
        problems.append(frozen)
    if not store.active_hypotheses(hyps):
        problems.append("no hypothesis is active")
    wrong = _require_branch(run, state)
    if wrong:
        problems.append(wrong)
    if problems:
        raise Refused(f"round {r} can't run:\n" + "\n".join(f"  - {p}" for p in problems))
    from boar import rounds

    return rounds.start_round(run, foreground)


def close_problems(state: dict, hyps: list[dict], r: int) -> list[str]:
    """Which of round `r`'s R2–R5 done-when conditions don't hold yet ([] once R6 may run)."""
    rec = state["rounds"][str(r)]
    if rec["run_status"] != "done":
        return [f"R2: round {r} has not run; `boar round run`"]
    problems = []
    missing = undecided(hyps, r)
    if missing:
        problems.append(f"R3: no round {r} decision on {_join(missing)}; `boar decide`")
    if not any(h["round"] == r for h in hyps) and not rec.get("propose_none"):
        problems.append("R4: no proposal this round; `boar propose <file>` or `boar propose --none --reason …`")
    else:
        problems += [f"R4: {o}, or `boar propose --none --reason …`" for o in r4_owed(state, hyps, r)]
    waiting = [h["id"] for h in hyps if h["round"] == r and awaiting_verdict(h)]
    waiting += [rm["id"] for rm in state["removals"] if rm["round"] == r and rm.get("verdict") is None]
    if waiting:
        problems.append(f"R5: no verdict yet on {_join(waiting)}; send them to the boar-reviewer")
    return problems


def diag_prune(start: Path | None = None) -> str:
    """R6: delete all but diag.json from the diagnostics directories of trials nothing protects."""
    run = store.require_live_run(start)
    with run.locked():
        state, hyps = run.state(), run.hypotheses()
        if state["phase"] != ROUND:
            raise Refused(f"diag prune is for R6 of a round; now in {state['phase']}")
        r = state["round"]
        problems = close_problems(state, hyps, r)
        if problems:
            raise Refused(
                "diag prune is for R6, once the reviewer has seen everything this round cites:\n"
                + "\n".join(f"  - {p}" for p in problems)
            )
        trials = run.trials()
        inc, _ = incumbent_after_close(run, state, hyps, r)
        protected = {t["trial"] for t in trials if t["round"] == r} | set((inc or {}).get("trials") or [])
        protected |= {n for h in hyps for dec in (h.get("decisions") or {}).values() for n in dec.get("trials") or []}
        protected |= {n for rm in state["removals"] for n in rm["trials"]}
        pruned, freed = 0, 0
        for t in trials:
            if t["trial"] in protected:
                continue
            gone = False
            for diag in run.trial_dir(t["trial"]).glob("diag-*"):
                for entry in diag.iterdir():
                    if entry.name == "diag.json" and entry.is_file() and not entry.is_symlink():
                        continue
                    freed += _tree_size(entry)
                    _rmtree(entry)
                    gone = True
            pruned += gone
    return (
        f"pruned {pruned} trial(s)' diagnostics directories, freeing {freed / 2**20:.1f} MB; diag.json is kept "
        f"everywhere, and {len(protected)} protected trials (this round's, the incumbent's, every cited one) are untouched"
    )


def _tree_size(path: Path) -> int:
    if path.is_symlink() or not path.is_dir():
        return path.lstat().st_size
    return sum(p.lstat().st_size for p in path.rglob("*") if not p.is_dir() or p.is_symlink())


def round_close(start: Path | None = None) -> str:
    run = store.require_live_run(start)
    config = run.config()
    with _editing(run) as (state, hyps):
        if state["phase"] != ROUND:
            raise Refused(f"round close is R6 of a round; now in {state['phase']}")
        r = state["round"]
        rec = state["rounds"][str(r)]
        problems = close_problems(state, hyps, r)
        if problems:
            raise Refused(f"round {r} can't close yet:\n" + "\n".join(f"  - {p}" for p in problems))
        plan = plan_close(state, hyps, r)
        change = None
        if plan["removed"]:
            from boar import warmstart

            before = warmstart.incumbent(run.trials(), hyps, r + 1, config["direction"])
            change = incumbent_change(before, incumbent_after_close(run, state, hyps, r)[0])
        for hid, why in plan["removed"].items():
            h = store.hypothesis(hyps, hid)
            h.update(status=REMOVED, removed_round=r, removed_reason=why)
        for hid in plan["activated"]:
            store.hypothesis(hyps, hid).update(status=ACTIVE, activated_round=r + 1)
        rec["closed"] = True
        if r < config["rounds"] and plan["active_after"]:
            state["round"] = r + 1
            state["rounds"][str(r + 1)] = fresh_round()
            after = f"round {r + 1} begins"
        else:
            state["phase"] = FINALIZE
            after = "no rounds remain" if r >= config["rounds"] else "no hypothesis is active"
            after += "; the run moves to finalize"
    lines = [f"round {r} closed"]
    lines += [f"removed {hid}: {why}" for hid, why in plan["removed"].items()]
    if plan["activated"]:
        lines.append(f"activated {_join(plan['activated'])}")
    lines += [f"active now: {_join(plan['active_after'])}", after]
    if change:
        lines.append(f"warning: {change}")
    return "\n".join(lines)


def finalize(foreground: bool, report: bool = False, start: Path | None = None) -> str:
    run = store.require_live_run(start)
    state = run.state()
    if state["phase"] == SETUP:
        raise Refused("setup isn't done; `boar next` says what is missing")
    if state["phase"] == ROUND:
        raise Refused(
            f"rounds remain and a hypothesis is active: round {state['round']} is in progress; "
            "finish it and `boar round close`"
        )
    from boar import finalize as fin

    status = state["finalize"]["status"]
    if report:
        if status != "done":
            raise Refused(
                f"`--report` rewrites the report of a finished finalize, and finalize is {status}; run `boar finalize`"
            )
        # The measurements are all on disk, so the skeleton can be rebuilt without re-running the holdout.
        fin.write_report(run)
        return (
            f"rewrote {run.report_path} from the run's records (the holdout was not re-run; any narrative in it "
            f"is gone). Write the sections marked {store.TODO_MARKER}"
        )
    if status == "done":
        damaged = report_problem(run)
        if damaged is None:
            raise Refused(f"already finalized; write the narrative sections marked {store.TODO_MARKER} in {run.report_path}")
        raise Refused(f"already finalized, but {damaged}; {report_fix(damaged)}")
    problems = [p for p in (eval_frozen(run, state), _require_branch(run, state)) if p]
    if problems:
        raise Refused("finalize can't run:\n" + "\n".join(f"  - {p}" for p in problems))
    return fin.start_finalize(run, foreground)


# Headings write_report generates, each on a line of its own; a report without them is not the run's report.
REPORT_SECTIONS = (
    "## 1. Result", "## 2. What worked", "## 3. What didn't", "## 4. Rejected by review", "## 5. Caveats",
    "## 6. Experimental log", "## 7. Investigations",
)
REPORT_MISSING = "report.md is missing"


def report_problem(run: Run) -> str | None:
    """Why report.md can't be the finished report whatever its markers say (missing, unreadable, headings gone), or None."""
    try:
        text = run.report_path.read_text()
    except FileNotFoundError:
        return REPORT_MISSING
    except (OSError, ValueError) as e:
        return f"report.md can't be read ({getattr(e, 'strerror', None) or e})"
    lines = {line.rstrip() for line in text.splitlines()}
    lost = [h for h in REPORT_SECTIONS if h not in lines]
    return f"report.md has lost generated headings: {'; '.join(lost)}" if lost else None


def report_fix(problem: str) -> str:
    """How to repair what report_problem found, without throwing narrative away unasked."""
    if problem == REPORT_MISSING:
        return "`boar finalize --report` rewrites it from the stored results (the holdout is not re-run)"
    if problem.startswith("report.md can't be read"):
        return "make report.md readable again"
    return (
        "put those heading lines back exactly as named, keeping the narrative; `boar finalize --report` restores "
        "them too, but discards the narrative"
    )


# --- abort --------------------------------------------------------------------------


def abort(reason: str | None, start: Path | None = None) -> str:
    run = store.require_run(start)
    reason = _reason(reason)
    phase = run.state()["phase"]
    if phase in (DONE, ABORTED):
        raise Refused(f"run {run.id} is already {phase}")
    from boar import worker

    lines = []
    rec = run.running()
    if rec is not None:
        _terminate(rec["pid"])
        lines.append(f"sent SIGTERM to the running `boar {rec.get('command')}` (pid {rec['pid']})")
    # A worker killed without cleanup leaves its eval running; nothing else would ever stop it.
    killed = worker.kill_orphans(run)
    if killed:
        lines.append(f"killed the eval process group {', '.join(map(str, killed))}")
    run.update_state(lambda s: s.update(phase=ABORTED, abort_reason=reason))
    lines.append(f"run {run.id} aborted: {reason}; the Stop hook is released")
    return "\n".join(lines)


def _terminate(pid: int) -> None:
    """SIGTERM a verified worker: its whole group if it leads one (a detached worker), else just the process.

    A CLI running work inline shares its shell's group, which must not be signalled.
    """
    try:
        if os.getpgid(pid) == pid and pid != os.getpgrp():
            os.killpg(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
