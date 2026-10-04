"""The `boar` command line. Parses arguments and maps refusals to exit status 1."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from boar import config as cfgmod
from boar import control, nextstep, store
from boar.errors import Refused


def _with_next(text: str) -> str:
    """Append the next step, so every state change tells the agent where it now stands."""
    try:
        return f"{text}\nnext: {nextstep.next_line(store.active_run())}"
    except Exception:
        return text


def _init(args: argparse.Namespace) -> int:
    overrides = cfgmod.parse_overrides(args.extra)
    print(_with_next(control.init(args.spec, args.session, overrides)))
    return 0


def _propose(args: argparse.Namespace) -> int:
    if args.none:
        if args.file:
            raise Refused("give either a proposal file or --none --reason …, not both")
        text = control.propose_none(args.reason)
    else:
        if not args.file:
            raise Refused("give a proposal file (`boar propose <file.json>`) or `--none --reason …`")
        if args.reason:
            raise Refused("--reason goes with --none; a proposal file carries its own statements")
        text = control.propose(args.file)
    print(_with_next(text))
    return 0


def _eval_check(args: argparse.Namespace) -> int:
    text, ok = control.eval_check(args.foreground)
    print(_with_next(text))
    return 0 if ok else 1


def _eval_restore(args: argparse.Namespace) -> int:
    print(_with_next(control.eval_restore()))
    return 0


def _review_pending(args: argparse.Namespace) -> int:
    print(control.review_pending())
    return 0


def _review_show(args: argparse.Namespace) -> int:
    print(control.review_show(args.id))
    return 0


def _review_record(args: argparse.Namespace) -> int:
    print(control.review_record(args.id, args.decision, args.reason))
    return 0


def _decide(args: argparse.Namespace) -> int:
    trials = control.parse_trials(args.trials)
    print(_with_next(control.decide(
        args.id, args.decision, args.reason, trials, observable_key=args.observable_key, cause=args.cause,
        measure=args.measure, queue=args.queue,
    )))
    return 0


def _diag_prune(args: argparse.Namespace) -> int:
    print(_with_next(control.diag_prune()))
    return 0


def _withdraw(args: argparse.Namespace) -> int:
    print(_with_next(control.withdraw(args.id, args.reason)))
    return 0


def _round_run(args: argparse.Namespace) -> int:
    print(control.round_run(args.foreground))
    return 0


def _round_close(args: argparse.Namespace) -> int:
    print(_with_next(control.round_close()))
    return 0


def _wait(args: argparse.Namespace) -> int:
    run = store.require_run()
    from boar import worker

    print(worker.wait(run, args.timeout))
    return 0


def _finalize(args: argparse.Namespace) -> int:
    text = control.finalize(args.foreground, args.report)
    print(_with_next(text) if args.report else text)
    return 0


def _next(args: argparse.Namespace) -> int:
    print(nextstep.run_next(args.session))
    return 0


def _status(args: argparse.Namespace) -> int:
    print(nextstep.status())
    return 0


def _abort(args: argparse.Namespace) -> int:
    print(control.abort(args.reason))
    return 0


def _worker(args: argparse.Namespace) -> int:
    from boar import worker

    return worker.worker_main(args.what)


# The commands that run trials inline with --foreground, by handler.
_INLINE = {_eval_check: "eval check", _round_run: "round run", _finalize: "finalize"}


def _interrupted(e: Exception) -> bool:
    """`e` is evalrun.Interrupted; only code that imported evalrun can raise it."""
    evalrun = sys.modules.get("boar.evalrun")
    return isinstance(e, getattr(evalrun, "Interrupted", ()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="boar",
        description="BOAR harness: owns the run state, runs the trials, and refuses any step taken out of order. "
        "Run `boar next` to see what to do now.",
    )
    sub = p.add_subparsers(dest="command", metavar="<command>", required=True)

    s = sub.add_parser(
        "init",
        help="start a run: run directory, branch boar/<id>, frozen config",
        allow_abbrev=False,
        description="Extra `--key value` pairs override the config: "
        + ", ".join(f"{k} (default {v})" for k, v in cfgmod.DEFAULTS.items()),
    )
    s.add_argument("--spec", required=True, help="the approved spec.md (needs a 'Direction: min|max' line)")
    s.add_argument("--session", help="Claude session id the Stop hook should hold to the run")
    s.set_defaults(func=_init, takes_extra=True)

    s = sub.add_parser("propose", help="submit a proposal file, or --none --reason … in R4")
    s.add_argument("file", nargs="?", help="JSON list of proposals")
    s.add_argument("--none", action="store_true", help="R4: no new proposals this round")
    s.add_argument("--reason")
    s.set_defaults(func=_propose)

    s = sub.add_parser(
        "eval", help="`eval check`: one baseline trial on dev, checks contract and timing; `eval restore`: undo edits"
    )
    es = s.add_subparsers(dest="eval_command", metavar="check|restore", required=True)
    t = es.add_parser("check", help="run the check (in the background; then `boar wait`)")
    t.add_argument("--foreground", action="store_true", help="run inline instead of in a detached worker")
    t.set_defaults(func=_eval_check)
    es.add_parser(
        "restore", help="put back the eval the reviewer accepted (from the copy kept at acceptance)"
    ).set_defaults(func=_eval_restore)

    s = sub.add_parser("review", help="reviewer: pending | show <id> | record <id> accept|reject --reason …")
    rs = s.add_subparsers(dest="review_command", metavar="pending|show|record", required=True)
    rs.add_parser("pending", help="every item awaiting a verdict, with its material").set_defaults(func=_review_pending)
    t = rs.add_parser("show", help="one item's material, any status")
    t.add_argument("id", help="H<n>, eval, or rm-H<n>-r<r>")
    t.set_defaults(func=_review_show)
    t = rs.add_parser("record", help="store a verdict on a pending item")
    t.add_argument("id")
    t.add_argument("decision", choices=["accept", "reject"])
    t.add_argument("--reason", required=True)
    t.set_defaults(func=_review_record)

    s = sub.add_parser("decide", help="R3: keep|fix|investigate|blocked|remove an active hypothesis")
    s.add_argument("id")
    s.add_argument("decision", choices=list(control.DECISIONS))
    s.add_argument("--reason", help="required, except that investigate falls back to its --cause")
    s.add_argument("--trials", nargs="+", default=[], help="cited trial ids, space- or comma-separated")
    s.add_argument(
        "--observable-key", help="the diagnostic key that measures the hypothesis's observable (required on its first "
        "decision other than fix)"
    )
    s.add_argument("--cause", help="investigate: the suspected cause the next round measures")
    s.add_argument("--measure", nargs="+", default=[], help="investigate: diagnostic keys R1 adds, comma-separated")
    s.add_argument("--queue", help="investigate: JSON list of at most 2 partial configs, one the hypothesis's off-state")
    s.set_defaults(func=_decide)

    s = sub.add_parser("diag", help="`diag prune` (R6): delete all but diag.json from unprotected diagnostics dirs")
    ds = s.add_subparsers(dest="diag_command", metavar="prune", required=True)
    ds.add_parser(
        "prune", help="keep diag.json; spare this round's trials, the incumbent's and every cited one"
    ).set_defaults(func=_diag_prune)

    s = sub.add_parser("withdraw", help="drop a pending or accepted-but-inactive proposal")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    s.set_defaults(func=_withdraw)

    s = sub.add_parser("round", help="`round run` (R2) or `round close` (R6)")
    rs = s.add_subparsers(dest="round_command", metavar="run|close", required=True)
    t = rs.add_parser("run", help="commit the code and run the round's trials (in the background)")
    t.add_argument("--foreground", action="store_true", help="run inline instead of in a detached worker")
    t.set_defaults(func=_round_run)
    rs.add_parser("close", help="apply removals and activations; start the next round").set_defaults(func=_round_close)

    s = sub.add_parser("wait", help="block until the background worker finishes (or the timeout passes)")
    s.add_argument("--timeout", type=float, default=540.0, help="seconds (default 540)")
    s.set_defaults(func=_wait)

    s = sub.add_parser("finalize", help="holdout check and report skeleton")
    s.add_argument("--foreground", action="store_true", help="run inline instead of in a detached worker")
    s.add_argument(
        "--report", action="store_true",
        help="once finalized: rewrite report.md from the stored results (no holdout re-run; discards its narrative)",
    )
    s.set_defaults(func=_finalize)

    s = sub.add_parser("next", help="what to do now; --hook for the Stop hook")
    s.add_argument("--hook", action="store_true", help="exit 2 with the next step on stderr while work remains")
    s.add_argument("--session", help="bind the run to this Claude session id")
    s.set_defaults(func=_next)

    sub.add_parser("status", help="run summary").set_defaults(func=_status)

    s = sub.add_parser("abort", help="end the run and release the Stop hook")
    s.add_argument("--reason", required=True)
    s.set_defaults(func=_abort)

    s = sub.add_parser("_worker")
    s.add_argument("what", choices=["round", "finalize", "eval"])
    s.set_defaults(func=_worker)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["next"] and "--hook" in argv:
        return nextstep.hook()
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and not getattr(args, "takes_extra", False):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    try:
        return args.func(args)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if _interrupted(e):
            # SIGTERM (from `boar abort`, say) stopped work this CLI ran inline; that is not a crash.
            print(f"interrupted: `boar {_INLINE.get(args.func, args.command)}` was stopped by SIGTERM; "
                  "`boar next` says what now", file=sys.stderr)
            return 143
        if os.environ.get("BOAR_DEBUG") == "1":
            traceback.print_exc()
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
