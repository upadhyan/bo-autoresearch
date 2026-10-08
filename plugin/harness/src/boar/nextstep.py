"""`boar next`, the Stop hook and `boar status`: the single source of "what now".

The agent never decides its next step from memory; it asks here. Everything in
this module is read-only except two documented writes: `next --session` binds
the run to a session (and records it in the per-user session registry), and
`next` marks a finalized run with a complete report done. Checking eval/ may also
refresh its digest cache (eval.fingerprint.json), which holds no run state.
"""

from __future__ import annotations

import json
import os
import select
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from boar import control, store
from boar.store import ABORTED, DONE, FINALIZE, ROUND, SETUP, TODO_MARKER, Run

# Tags that leave the agent free to stop; every other tag means work remains.
RELEASE_TAGS = {"NONE", "ASK_USER", "DONE", "ABORTED"}


@dataclass
class Step:
    tag: str
    line: str
    details: list[str] = field(default_factory=list)
    finish: bool = False  # finalized and the report is complete: `boar next` marks the run done

    @property
    def first_line(self) -> str:
        return f"[{self.tag}] {self.line}"

    @property
    def text(self) -> str:
        return "\n".join([self.first_line, *(f"  {d}" for d in self.details)])


def compute(run: Run | None) -> Step:
    """The next thing the agent must do. Pure: reads the run, never writes it."""
    if run is None:
        return Step("NONE", "No active BOAR run. Start one with /boar <request>.")
    state = run.state()
    phase = state["phase"]
    if phase == ABORTED:
        return Step("ABORTED", f"Run {run.id} was aborted: {state.get('abort_reason')}")
    if phase == DONE:
        return Step("DONE", f"Run {run.id} is finished. Report: {run.report_path}")
    rec = run.running()
    if rec is not None:
        return _wait_step(run, state, rec)
    hyps, config = run.hypotheses(), run.config()
    if phase == SETUP:
        return _setup_step(run, state, hyps, config)
    if phase == ROUND:
        return _round_step(run, state, hyps, config)
    return _finalize_step(run, state)


def next_line(run: Run | None) -> str:
    """First line of `boar next`, without its side effects (for other commands' "next:" hints)."""
    return compute(run).first_line


def _wait_step(run: Run, state: dict, rec: dict) -> Step:
    config = run.config()
    if rec.get("command") == "eval check":
        progress = f"one baseline trial of {config['repeats']} repeats on dev"
    elif rec.get("command") == "finalize":
        done = len(store.read_jsonl(run.holdout_path))
        progress = f"finalize, holdout {done}/{2 * config['holdout_repeats']} runs done"
    else:
        r = rec.get("round", state["round"])
        done = sum(1 for t in run.trials() if t.get("round") == r)
        progress = f"round {r}, {done}/{control.round_size(config, state['rounds'].get(str(r)) or {})} trials done"
    if not detached(rec):
        return Step(
            "WAIT",
            f"boar {rec.get('command')} is running ({progress}). Run `boar wait`.",
            [f"it runs inline (--foreground, pid {rec['pid']}) and prints to the shell that started it"],
        )
    return Step(
        "WAIT",
        f"boar {rec.get('command')} is running ({progress}). Run `boar wait`.",
        [f"log: {rec.get('log')}"] if rec.get("log") else [],
    )


def detached(rec: dict) -> bool:
    """The worker slot is held by a detached worker (it leads its own session), not a CLI running inline."""
    try:
        return os.getsid(rec["pid"]) == rec["pid"]
    except (OSError, KeyError, TypeError):
        return False


def _reject_reasons(run: Run, kind: str, limit: int = 3) -> list[str]:
    rejects = [r for r in run.reviews() if r.get("kind") == kind and r.get("decision") == "reject"]
    return [f"{r['item']}: {r['reason']}" for r in rejects[-limit:]]


def _setup_step(run: Run, state: dict, hyps: list[dict], config: dict) -> Step:
    ev = state["eval"]
    setup = [h for h in hyps if h["round"] == 0]
    waiting = [h["id"] for h in setup if control.awaiting_verdict(h)]
    acc = [h["id"] for h in setup if control.accepted(h)]
    failed = state.get("setup_failed_batches", 0)
    # Ask once per rejection past the limit; once the agent is rebuilding (a check has run), S4 holds it to the work.
    if ev["status"] == "rejected" and ev["rejections"] >= 3 and not ev.get("checked_since_reject"):
        return Step(
            "ASK_USER",
            f"The eval was rejected {ev['rejections']} times. Stop and ask the user how to proceed.",
            ["latest reviewer reasons:", *_reject_reasons(run, "eval")],
        )
    if not acc and not waiting and failed >= 3:
        return Step(
            "ASK_USER",
            f"No hypothesis was accepted in {failed} attempts. Stop and ask the user how to proceed.",
            ["latest reviewer reasons:", *_reject_reasons(run, "proposal")],
        )
    if not setup:
        return Step(
            "S2",
            f"Research the problem into {run.research_path}; then [S3] submit hypotheses with `boar propose <file.json>`.",
            [
                "Collect candidate mechanisms from the target system (profile it, read its code and logs) and from "
                "outside sources (docs, papers, issue trackers, the web).",
                f"Every user idea in {run.spec_path} must appear in research.md as a candidate or as a rejection "
                "with a reason; every external candidate needs a citation.",
                f"S4 can go in parallel: build the eval in {run.eval_dir}/ and pass `boar eval check`.",
            ],
        )
    if not waiting and not acc:
        attempt = f" (failed attempt {failed}/3)" if failed else ""
        return Step(
            "S3",
            f"No hypothesis accepted{attempt}; propose new ones with `boar propose <file.json>`.",
            ["reviewer reasons:", *_reject_reasons(run, "proposal")],
        )
    if ev["status"] == "none":
        details = [
            "Contract: `eval/run` is executable, runs from the repo root, passes $BOAR_CONFIG (the lever config "
            "JSON's path) through to the target code untouched, never reading it in the measuring code, reads the "
            "split from $BOAR_SPLIT (dev|holdout), and prints one JSON object as its last stdout line: "
            '{"metric": <number>, "guards_ok": <bool>, "metrics": {…}}. One invocation is one repeat.',
            f"A baseline trial ({config['repeats']} repeats) should take about trial_target_s = "
            f"{config['trial_target_s']} s and must not exceed {2 * config['trial_target_s']} s.",
            "`boar eval check` runs that trial in the background; run `boar wait` until it prints `finished: eval check …`.",
        ]
        check = store.read_json(run.eval_check_path)
        if check and not check.get("ok"):
            details += ["last check failed:", *(f"  - {p}" for p in check.get("problems") or [])]
        return Step("S4", f"Build the eval in {run.eval_dir}/ and pass `boar eval check`.", details)
    if ev["status"] == "rejected":
        reasons = _reject_reasons(run, "eval", limit=1)
        reason = reasons[0].split(": ", 1)[1] if reasons else "see reviews.jsonl"
        details = []
        check = store.read_json(run.eval_check_path)
        if ev.get("checked_since_reject") and check and not check.get("ok"):
            details = ["last check failed:", *(f"  - {p}" for p in check.get("problems") or [])]
        return Step(
            "S4",
            f"Eval rejected (failed attempt {ev['rejections']}/3): {reason}; rebuild it and re-run `boar eval check`.",
            details,
        )
    # An eval edited after its passing check can't be accepted; it goes back to the check, not to the reviewer.
    items = control.pending_items(state, hyps)
    stale = control.eval_stale(run, state) if "eval" in items else []
    if stale:
        items.remove("eval")
    recheck = "eval/ changed since its passing `boar eval check`; re-run the check before the eval goes to review."
    if control.unreadable_only(stale):
        recheck = (
            f"eval/ could not be read ({'; '.join(stale[:5])}); make it readable and re-run `boar eval check` "
            "before the eval goes to review."
        )
    if items:
        return Step(
            "S5",
            f"Send the boar-reviewer agent these items: {', '.join(items)}.",
            ["It reads them with `boar review pending` and records each verdict with `boar review record`.",
             *([recheck] if stale else [])],
        )
    if stale:
        return Step("S4", recheck)
    return Step(
        "ASK_USER",
        "Setup looks complete but round 1 has not started; the run state is inconsistent. Ask the user.",
        ["`boar status` shows the run state."],
    )


def _round_step(run: Run, state: dict, hyps: list[dict], config: dict) -> Step:
    r = state["round"]
    rec = state["rounds"][str(r)]
    active = store.active_hypotheses(hyps)
    if rec["run_status"] in ("not_started", "running"):
        frozen = control.eval_frozen(run, state)
        if frozen:
            return Step("R1", f"{frozen}; then `boar round run`.")
    if rec["run_status"] == "not_started":
        new = [h for h in active if h.get("activated_round") == r]
        repair = [h for h in active if (h.get("decisions", {}).get(str(r - 1)) or {}).get("decision") == "fix"]
        measure = [(h, control.investigation(h, r - 1)) for h in active if control.investigation(h, r - 1)]
        details = [f"new: {h['id']} ({', '.join(lv['name'] for lv in h['levers'])})" for h in new]
        details += [f"repair: {h['id']}: {h['decisions'][str(r - 1)]['reason']}" for h in repair]
        details += [f"measure: {h['id']}: {', '.join(inv['measure'])} (cause: {inv['cause']})" for h, inv in measure]
        if new:
            details.append(
                "Instrument each new hypothesis's observable (research.md): the target code writes it to diag.json in "
                "$BOAR_DIAG_DIR, only when that variable is set."
            )
        details += [
            "Every lever must read its value from the BOAR_CONFIG JSON the eval passes through; its default must "
            "reproduce baseline.",
            f"All changes go on branch {state['branch']}. `boar round run` commits them and starts "
            f"{config['trials_per_round']} trials in the background; then run `boar wait` until it finishes.",
        ]
        if new or repair or measure:
            what = f"Implement levers for: {', '.join(h['id'] for h in new) or 'none'}; repair: "
            what += f"{', '.join(h['id'] for h in repair) or 'none'}"
            what += f"; measure: {', '.join(h['id'] for h, _ in measure)}" if measure else ""
            what += "; then `boar round run`."
        else:
            what = "Nothing new to implement or repair; run `boar round run`."
        return Step("R1", what, details)
    if rec["run_status"] == "running":
        log = run.round_dir(r) / "worker.log"  # only a detached worker writes one
        return Step(
            "R1",
            f"Round {r} was interrupted; resume with `boar round run`.",
            [f"worker log: {log}" if log.exists() else
             "it ran inline (--foreground), so its output went to the shell that started it"],
        )
    missing = control.undecided(hyps, r)
    if missing:
        last = r >= config["rounds"]
        fix = (
            f"fix: not in the last round ({r}/{config['rounds']}): no R1 follows to repair a bug, and a fix would drop "
            "the trials that set its levers from the final incumbent; decide keep and leave the bug for Caveats."
            if last else "fix: a lever crashed or misbehaved through an implementation bug (R1 repairs it)."
        )
        probes = [(h, control.investigation(h, r - 1)) for h in active if control.investigation(h, r - 1)]
        choices = ["keep", *([] if last else ["fix"]), *(["investigate"] if r < config["rounds"] - 1 else []),
                   *(["blocked"] if probes else []), "remove"]
        unbound = [h["id"] for h in active if h["id"] in missing and not h.get("observable_key")]
        diag = []
        if unbound:
            diag.append(
                f"observable key: {', '.join(unbound)} have none yet; a first decision other than fix binds the "
                "observable stated in research.md with --observable-key <diag key> (summary: Diagnostics)."
            )
        diag += [f"resolve: {h['id']} (investigated in round {r - 1}: {inv['cause']}; measured {', '.join(inv['measure'])})"
                 for h, inv in probes if h["id"] in missing]
        if probes:
            diag.append(
                "Resolve by what the measurements show: observable did not move -> fix; moved weakly or only at some "
                "values -> keep, then supersede it in R4; moved but another quantity got worse -> blocked (R4 then "
                "proposes an enabler); moved, nothing else changed, metric flat -> remove. Except for fix, cite trials "
                "whose diagnostics hold every measured key."
            )
        if "investigate" in choices:
            diag.append(
                "investigate: the metric alone can't say why; --cause \"…\" --measure k1,k2 [--queue <json list of at "
                "most 2 partial configs; the harness adds the off-state>] keeps it active one more round (once per "
                "hypothesis)."
            )
        return Step(
            "R3",
            f"Decide on: {', '.join(missing)} — `boar decide <id> {'|'.join(choices)} --reason … --trials …`.",
            [
                f"Round summary: {run.summary_path(r)}",
                *diag,
                f"keep: it helps, or the evidence is not in yet. {fix} remove: trials that varied its levers show no "
                "effect beyond the noise floor, or consistent harm (takes effect only if the reviewer accepts it)."
                + (" In the last round a removal that would change the final incumbent is refused: the trials that "
                   "set its levers would drop out of it; decide keep and argue the lack of effect in What didn't."
                   if last else " An accepted removal drops the trials that set its levers from the warm start, "
                   "so the incumbent may move to one with its levers at default."),
            ],
        )
    owed = control.r4_owed(state, hyps, r)
    if (not any(h["round"] == r for h in hyps) and not rec.get("propose_none")) or owed:
        details = [f"owed: {o}." for o in owed]
        details += [
            'To retune a lever, propose a new hypothesis with "supersedes": "<id>" '
            "(for example, widen a range when the incumbent sits at its edge).",
        ]
        if any("enables" in o for o in owed):
            details.append(
                'An enabler carries "enables": "<id>" and a "joint_config": a partial config (filled from the incumbent) '
                "that sets at least one of its own levers away from default; the harness runs it when the enabler activates."
            )
        if r >= config["rounds"]:
            details.append(
                f"This is the last round ({r}/{config['rounds']}): new hypotheses would never run, and an accepted "
                "supersede pins the superseded levers at default for the final incumbent. Normally run "
                '`boar propose --none --reason "last round"`.'
            )
        return Step(
            "R4",
            f"Propose new hypotheses from round {r}'s results and the spec with `boar propose <file.json>`, "
            "or run `boar propose --none --reason …`.",
            details,
        )
    items = control.pending_items(state, hyps)
    if items:
        return Step(
            "R5",
            f"Send the boar-reviewer agent: {', '.join(items)}.",
            ["It reads them with `boar review pending` and records each verdict with `boar review record`."],
        )
    plan = control.plan_close(state, hyps, r)
    details = [f"removes {hid}: {why}" for hid, why in plan["removed"].items()]
    # A rejected removal leaves the hypothesis active and unrepaired unless the agent re-decides it.
    for rm in state["removals"]:
        verdict = rm.get("verdict") or {}
        h = next((h for h in hyps if h["id"] == rm["hypothesis"]), None)
        latest = ((h or {}).get("decisions", {}).get(str(r)) or {}).get("decision")
        if rm["round"] != r or verdict.get("decision") != "reject" or latest != "remove":
            continue
        rejected = f"{rm['id']} was rejected ({verdict['reason']})"
        if h["id"] in plan["removed"]:
            details.append(f"{rejected}, but {h['id']} goes anyway: {plan['removed'][h['id']]}.")
        elif r >= config["rounds"]:
            # No R1 follows, and a fix decision drops the hypothesis's trials from the final incumbent.
            details.append(
                f"{rejected}; {h['id']} stays active. This is the last round: no R1 follows, so leave the bug for "
                f"Caveats (`boar decide {h['id']} keep` if it helps); fix is refused, since it would drop its trials "
                "from the final incumbent."
            )
        else:
            details.append(
                f"{rejected}; {h['id']} stays active. If the reason points to a bug, "
                f"`boar decide {h['id']} fix --reason … --trials …` so R1 repairs it."
            )
    if plan["activated"]:
        details.append(f"activates {', '.join(plan['activated'])}")
    if plan["removed"]:
        from boar import warmstart

        before = warmstart.incumbent(run.trials(), hyps, r + 1, config["direction"])
        change = control.incumbent_change(before, control.incumbent_after_close(run, state, hyps, r)[0])
        if change:
            details.append(f"Warning: {change}.")
    return Step("R6", "Run `boar round close`.", details)


def _finalize_step(run: Run, state: dict) -> Step:
    status = state["finalize"]["status"]
    if status != "done":
        frozen = control.eval_frozen(run, state)
        if frozen:
            return Step("FINALIZE", f"{frozen}; then `boar finalize`.")
        what = "Run `boar finalize`." if status == "not_started" else "Finalize was interrupted; resume with `boar finalize`."
        return Step(
            "FINALIZE",
            what,
            ["It runs the holdout check in the background and writes the report skeleton; then run `boar wait`."],
        )
    damaged = control.report_problem(run)
    if damaged == control.REPORT_MISSING:
        return Step(
            "REPORT",
            f"{damaged}; run `boar finalize --report` to rewrite it from the stored results (the holdout is not "
            f"re-run), then write its sections marked {TODO_MARKER}.",
        )
    if damaged:
        return Step("REPORT", f"{damaged}; {control.report_fix(damaged)}.")
    markers = run.report_path.read_text().count(TODO_MARKER)
    if markers:
        return Step(
            "REPORT",
            f"Write the narrative sections marked {TODO_MARKER} in {run.report_path}.",
            [
                f"{markers} marker(s) left. Replace each with the narrative; keep the generated tables as they are.",
                "What worked: the incumbent's hypotheses and the evidence for each. Caveats: dev vs holdout "
                "disagreement, noise, drift flags, untested lever interactions. Keep the narrative to one page.",
            ],
        )
    return Step("DONE", f"Run {run.id} is finished. Report: {run.report_path}", finish=True)


# --- commands -----------------------------------------------------------------------


def _finish(run: Run) -> None:
    def mark(state: dict) -> None:
        if state["phase"] == FINALIZE and state["finalize"]["status"] == "done":
            state["phase"] = DONE

    run.update_state(mark)


def run_next(session: str | None = None, start: Path | None = None) -> str:
    run = store.active_run(start)
    sid = control.session_id(session)
    before = None
    if run is not None and sid and run.state()["phase"] not in (DONE, ABORTED):

        def bind(state: dict) -> str | None:
            old = control.session_id(state.get("session_id"))
            state["session_id"] = sid
            return old

        before = run.update_state(bind)
        control.register_session(sid, run.root)
    step = compute(run)
    if step.finish:
        _finish(run)
        step = compute(run)
    if before and before != sid:
        # Moving the binding releases the other session's Stop hook; say so, in case it is still working.
        return f"{step.text}\nnote: the Stop hook now holds session {sid} to this run (was {before})"
    return step.text


def _read_stdin(timeout_s: float = 1.0, limit: int = 1 << 20) -> str:
    """The hook's JSON input, if any. Never blocks past `timeout_s`, even on a pipe nobody closes."""
    stream = sys.stdin
    chunks: list[bytes] = []
    try:
        if stream is None or stream.isatty():
            return ""
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            return stream.read(limit) or ""
        size = 0
        deadline = time.monotonic() + timeout_s
        while size < limit:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                break
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    except (OSError, ValueError):
        pass
    return b"".join(chunks).decode("utf-8", "replace")


def hook(start: Path | None = None) -> int:
    """`boar next --hook`: exit 2 with the next step on stderr while work remains, else 0. Never fails the hook."""
    try:
        try:
            data = json.loads(_read_stdin() or "{}")
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        theirs = control.session_id(data.get("session_id"))
        run = _hook_run(start, data, theirs)
        if run is None:
            return 0
        ours = control.session_id(run.state().get("session_id"))
        if ours and theirs and ours != theirs:
            return 0
        step = compute(run)
        if step.finish:
            _finish(run)
            step = compute(run)
        if step.tag in RELEASE_TAGS:
            return 0
        sys.stderr.write(f"BOAR run {run.id} is not finished. {step.text}\n")
        sys.stderr.flush()
        return 2
    except BaseException:
        return 0


def _hook_run(start: Path | None, data: dict, sid: str | None) -> Run | None:
    """The run the hook guards: the first live run, not bound to another session, found from the hook's cwd,
    the project dir or this process's cwd; else the run bound to this session in the repo the session
    registered with `init`/`next --session`.

    The session may have cd'd out of the target repo (into a vendor clone, say), or Claude may have been
    started in a parent directory, while the run is live.
    """
    if start is not None:
        return store.active_run(start)
    candidates = [data.get("cwd"), os.environ.get("CLAUDE_PROJECT_DIR"), os.getcwd()]
    for cand in candidates:
        if isinstance(cand, str) and cand and os.path.isdir(cand):
            run = store.active_run(Path(cand))
            if run is not None and _holds(run, sid):
                return run
    root = control.registered_root(sid)
    # The entry leads only to the run bound to this session; a stale one (no run there any more, or one now
    # finished, unbound or bound elsewhere) releases the session.
    run = store.active_run(root) if root is not None and root.is_dir() else None
    return run if run is not None and sid and control.session_id(run.state().get("session_id")) == sid else None


def _holds(run: Run, sid: str | None) -> bool:
    state = run.state()
    ours = control.session_id(state.get("session_id"))
    return state["phase"] not in (DONE, ABORTED) and not (ours and sid and ours != sid)


def status(start: Path | None = None) -> str:
    run = store.require_run(start)
    state, hyps, config = run.state(), run.hypotheses(), run.config()
    trials = run.trials()
    rnd = f"round {state['round']}/{config['rounds']}" if state["round"] else "setup"
    lines = [
        f"run {run.id}  phase {state['phase']}  {rnd}  direction {config['direction']}",
        f"branch {state['branch']} (from {state['base_branch']} at {state['base_commit'][:12]})",
        f"run dir {run.dir}",
        f"eval: {state['eval']['status']}"
        + (f" (hash {state['eval']['accepted_hash'][:12]})" if state["eval"].get("accepted_hash") else ""),
    ]
    if hyps:
        rows = [("id", "status", "levers", "latest decision")]
        for h in hyps:
            status_ = h["status"] + (" (accepted)" if control.accepted(h) else "")
            decs = h.get("decisions") or {}
            latest = ""
            if decs:
                last = max(decs, key=int)
                latest = f"r{last} {control.decision_text(state, h, last)}"
            rows.append((h["id"], status_, ", ".join(lv["name"] for lv in h["levers"]), latest))
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        lines.append("hypotheses:")
        lines += ["  " + "  ".join(row[i].ljust(widths[i]) for i in range(3)) + "  " + row[3] for row in rows]
    else:
        lines.append("hypotheses: none yet")
    counts = {s: sum(1 for t in trials if t.get("state") == s) for s in ("complete", "infeasible", "failed")}
    lines.append(f"trials: {len(trials)} ({', '.join(f'{k} {v}' for k, v in counts.items())})")
    lines.append(f"incumbent: {_incumbent_text(state, hyps, config, trials)}")
    lines.append(f"pending review: {', '.join(control.pending_items(state, hyps)) or 'none'}")
    rec = run.running()
    where = f"log {rec.get('log')}" if rec and detached(rec) else "inline in another shell"
    lines.append(
        f"worker: `boar {rec.get('command')}` running, pid {rec['pid']}, since {rec.get('started')}, {where}"
        if rec
        else "worker: none"
    )
    lines.append(f"next: {next_line(run)}")
    return "\n".join(lines)


def _incumbent_text(state: dict, hyps: list[dict], config: dict, trials: list[dict]) -> str:
    if not trials:
        return "none yet"
    try:
        from boar import warmstart

        inc = warmstart.incumbent(trials, hyps, state["round"] + 1, config["direction"])
    except Exception as e:
        return f"unavailable ({type(e).__name__}: {e})"
    if not inc:
        return "none (no feasible trial in the current search space)"
    reps = inc.get("repeats") or []
    ids = ", ".join(str(t) for t in inc.get("trials") or [])
    return f"{json.dumps(inc.get('config', {}), sort_keys=True)} metric {inc.get('metric')} over {len(reps)} repeats (trials {ids})"
