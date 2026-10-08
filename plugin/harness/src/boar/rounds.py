"""R2: a round's study over the current search space, warm-started, its N trials, and summary.md.

`start_round` does the steps that can refuse (frozen eval, commit or resume check) in the
foreground, then runs the body inline or in a detached worker. The body is resumable: it stops
once N trials of the round are in trials.jsonl, however many runs it took to get there.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from boar import control, evalrun, gitops, stats, store, warmstart, worker
from boar.errors import Refused
from boar.store import Run

NOT_STARTED, RUNNING, DONE = "not_started", "running", "done"
_DIRECTION = {"min": "minimize", "max": "maximize"}


def log(msg: str) -> None:
    print(f"[{store.now_iso()}] {msg}", flush=True)


# --- formatting shared with finalize -------------------------------------------


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return json.dumps(x)
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def config_text(non_default: dict) -> str:
    return f"`{json.dumps(non_default, sort_keys=True)}`" if non_default else "baseline (all levers at default)"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(x: Any) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def id_list(ids: list[int]) -> str:
    return ", ".join(str(i) for i in ids) if ids else "none"


# --- preconditions the engine re-checks -------------------------------------------


def _eval_changes(ev: dict, run: Run) -> list[str]:
    """How eval/ differs from the accepted version ([] if it doesn't); cheap when a manifest was kept."""
    manifest = ev.get("accepted_manifest")
    if manifest is not None:
        return evalrun.eval_unchanged(run, manifest)
    return [] if evalrun.eval_hash(run) == ev.get("accepted_hash") else ["eval/ differs from the accepted hash"]


def check_eval_frozen(run: Run) -> None:
    ev = run.state()["eval"]
    accepted = ev.get("accepted_hash")
    if ev.get("status") != "accepted" or not accepted:
        raise Refused("the eval has not been accepted by the reviewer yet; see `boar next`")
    diff = _eval_changes(ev, run)
    if diff:
        if ev.get("accepted_manifest") is not None:
            raise Refused(
                f"eval/ changed since the reviewer accepted it ({evalrun.diff_text(diff)}); the eval is frozen for "
                "the whole run. Run `boar eval restore` to put the accepted version back"
            )
        raise Refused(
            "eval/ no longer matches the version the reviewer accepted; the eval is frozen for the whole run, "
            "so undo the change to eval/ (changing the eval needs a new run)"
        )


def _listed(paths: list[str]) -> str:
    return ", ".join(paths[:5]) + (" …" if len(paths) > 5 else "")


def require_code(root: Path, commit: str, lead: str, consequence: str, command: str) -> None:
    """Refuse unless HEAD is `commit` and the repo status is clean; the refusal reads "<lead> at commit X but …".

    Untracked files count: what trials wrote into the repo was moved away after each one, so anything
    left was put there by someone else, and the next round commit would sweep it in.
    """
    head = gitops.head_commit(root)
    short = commit[:12]
    if head != commit:
        raise Refused(
            f"{lead} at commit {short} but HEAD is now {head[:12]}; {consequence}. Move the branch back to {short} "
            f"(e.g. `git reset --keep {short}`) and re-run `{command}`"
        )
    dirty = sorted(gitops.status(root))
    if dirty:
        raise Refused(
            f"{lead} at commit {short} but the working tree has changed ({_listed(dirty)}); {consequence}. "
            f"Restore it (e.g. `git stash -u`) and re-run `{command}`"
        )


def check_unchanged(run: Run, commit: str, step: str, command: str) -> None:
    """Before each trial: the code is still `commit`, the repo as clean as the step left it, eval/ the accepted one.

    A change here happened between trials, so it is someone else's: the run stops for it to be undone.
    """
    require_code(
        run.root, commit, f"{step} started", "it changed between trials, and the next trial would measure code "
        "the step didn't commit, so the run stopped before it", command,
    )
    check_eval_frozen(run)


def undo_trial_writes(
    run: Run, commit: str, dirs: list[str], keep: Path, step: str, command: str
) -> tuple[str | None, str | None]:
    """Right after a trial: put back what it wrote into the repo or eval/ (its eval's or its levers' doing).

    Returns (the trial's error, or None if it wrote nothing; why the writes could not all be undone, or None).
    Tracked files get their content from `commit` again; new files are moved under `keep` (repo-writes/,
    eval-writes/), never deleted; `dirs` is `gitops.untracked_dirs` before the trial. A moved HEAD is refused outright: that is a commit, not a trial's output.
    """
    head = gitops.head_commit(run.root)
    if head != commit:
        raise Refused(
            f"{step} started at commit {commit[:12]} but HEAD is now {head[:12]}; the trial that just ran can't be "
            f"tied to one commit, so it was not recorded. Move the branch back (e.g. `git reset --keep {commit[:12]}`) "
            f"and re-run `{command}`"
        )
    errors, stuck = [], []
    wrote = sorted(gitops.status(run.root))
    if wrote:
        errors.append(f"the trial wrote into the repo: {_listed(wrote)}")
        left = gitops.undo_writes(run.root, commit, {}, keep / "repo-writes", dirs)
        if left:
            stuck.append(f"the repo still has changes the trial made ({_listed(left)}); restore them (e.g. `git stash -u`)")
    ev = run.state()["eval"]
    diff = _eval_changes(ev, run)
    if diff:
        errors.append(f"the trial wrote into eval/: {evalrun.diff_text(diff)}")
        manifest = ev.get("accepted_manifest")
        why = evalrun.restore_eval(run, manifest, keep / "eval-writes") if manifest is not None else "no file list was kept"
        if why:
            stuck.append(f"eval/ could not be put back ({why}); run `boar eval restore`")
    return "; ".join(errors) or None, (
        f"{'; '.join(stuck)}, then re-run `{command}`" if stuck else None
    )


def ensure_commit(run: Run, recorded: str | None, message: str, step: str, command: str) -> str:
    """Commit the worktree for a fresh step; on resume, insist the code is still the recorded commit."""
    if not recorded:
        sha = gitops.commit_worktree(run.root, message)
        left = sorted(gitops.status(run.root))
        if left:
            # `git add -A` records a nested repo or submodule as its HEAD only; its uncommitted edits stay out.
            raise Refused(
                f"these paths could not be committed ({_listed(left)}); a nested git repo or submodule has "
                "uncommitted changes. Commit inside it (its new HEAD will be recorded) or vendor it as plain files, "
                f"then re-run `{command}`"
            )
        return sha
    require_code(run.root, recorded, f"{step} started", "resuming would measure different code", command)
    return recorded


def _prepare(run: Run, r: int) -> str:
    rec = run.state()["rounds"][str(r)]
    if rec.get("run_status") == DONE:
        raise Refused(f"round {r} has already run; see `boar next`")
    check_eval_frozen(run)
    commit = ensure_commit(run, rec.get("commit"), f"boar: round {r}", f"round {r}", "boar round run")

    def mark(state: dict) -> None:
        rr = state["rounds"][str(r)]
        rr["commit"] = commit
        rr["run_status"] = RUNNING

    run.update_state(mark)
    return commit


def start_round(run: Run, foreground: bool) -> str:
    r = run.state()["round"]
    with worker.claim(run, worker.ROUND, r) as c:
        _prepare(run, r)
        if foreground:
            run_round_worker(run)
            return f"round {r} finished; summary: {run.summary_path(r)}"
        rec = c.spawn()
    return (
        f"round {r} is running in the background (pid {rec['pid']}, log {rec['log']}). "
        "Run `boar wait` until it finishes."
    )


# --- the round body ---------------------------------------------------------------


def extra_queue(hyps: list[dict], r: int) -> list[dict]:
    """The configs round `r` runs on top of its N trials: what last round's investigations queued, and the joint
    trial of each enabler that activates this round.

    Each is {"label", "config"}, `config` a partial config that is filled from the incumbent when queued.
    """
    out = []
    for h in store.active_hypotheses(hyps):
        queue = (control.investigation(h, r - 1) or {}).get("queue") or []
        out += [{"label": f"investigate {h['id']} ({i}/{len(queue)})", "config": c} for i, c in enumerate(queue, 1)]
    for h in store.active_hypotheses(hyps):
        if h.get("enables") and h.get("activated_round") == r:
            out.append({"label": f"joint {h['id']}+{h['enables']}", "config": h["joint_config"]})
    return out


def _labels(warm: dict) -> list[str]:
    first = "baseline" if warm.get("incumbent_config") is None else "incumbent"
    return [first, *(q["label"] for q in warm.get("queue") or [])]


def _enqueue(study: optuna.Study, space: dict, defaults: dict, warm: dict, skip: set[str] = frozenset()) -> None:
    """Queue the incumbent (or the baseline), so it is measured again on this round's commit, then the extra configs.

    Each extra config is filled from the incumbent; a lever it names that is pinned now stays at its default.
    """
    base = warm.get("incumbent_config") or {}
    partials = [{}] + [q["config"] for q in warm.get("queue") or []]
    for label, partial in zip(_labels(warm), partials):
        if label in skip:
            continue
        params = {name: partial.get(name, base.get(name, defaults[name])) for name in space}
        study.enqueue_trial(params, user_attrs={"queued": label})


def _sampler(cfg: dict, seed: int) -> optuna.samplers.TPESampler:
    # numpy takes seeds in 0 .. 2**32 - 1 only; `seed` may be any integer.
    return optuna.samplers.TPESampler(multivariate=True, n_startup_trials=cfg["trials_per_round"], seed=seed % 2**32)


def _open_study(run: Run, r: int, cfg: dict, hyps: list[dict]) -> optuna.Study:
    space = store.search_space(hyps)
    defaults = store.lever_defaults(hyps)
    name = f"round-{r}"
    sampler = _sampler(cfg, cfg["seed"] + r)
    warm = run.state()["rounds"][str(r)].get("warm")
    if warm is None:
        # Nothing ran yet, so a half-built study from an interrupted start is discarded.
        with contextlib.suppress(KeyError):
            optuna.delete_study(study_name=name, storage=run.storage_url)
        study = optuna.create_study(
            study_name=name, storage=run.storage_url, sampler=sampler, direction=_DIRECTION[cfg["direction"]]
        )
        trials = run.trials()
        valid, excluded = warmstart.select(trials, hyps, r)
        study.add_trials(warmstart.build_frozen_trials(valid, space, defaults))
        inc = warmstart.incumbent(trials, hyps, r, cfg["direction"]) if r > 1 else None
        warm = {
            "copied": len(valid), "excluded": excluded, "incumbent_config": inc["config"] if inc else None,
            "queue": extra_queue(hyps, r),
        }
        _enqueue(study, space, defaults, warm)
        run.update_state(lambda s: s["rounds"][str(r)].__setitem__("warm", warm))
        log(f"round {r}: copied {len(valid)} warm trials, left out {excluded}; queued {config_text(warm['incumbent_config'] or {})}")
        return study
    try:
        study = optuna.load_study(study_name=name, storage=run.storage_url, sampler=sampler)
    except KeyError:
        raise Refused(f"study {name} is missing from {run.study_db}; the round can't be resumed") from None
    # Re-seeded from progress: the same seed would replay the random startup draws the round already ran.
    # Every attempt adds trials (an interrupted one stays as FAIL), so each resume gets a new seed.
    study.sampler = _sampler(cfg, cfg["seed"] + r + 1_000_003 * len(study.trials))
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)
    done = {t.get("queued") for t in run.trials() if t["round"] == r}
    waiting = {t.user_attrs.get("queued") for t in study.get_trials(deepcopy=False, states=(TrialState.WAITING,))}
    _enqueue(study, space, defaults, warm, skip=done | waiting)
    log(f"round {r}: resuming")
    return study


def run_round_worker(run: Run) -> None:
    """R2 steps 3-6 (and 1-2 again, harmlessly): warm start, queue, N trials, summary."""
    warmstart.quiet_optuna()
    r = run.state()["round"]
    commit = _prepare(run, r)
    cfg = run.config()
    hyps = run.hypotheses()
    dists = warmstart.distributions(store.search_space(hyps))
    defaults = store.lever_defaults(hyps)
    study = _open_study(run, r, cfg, hyps)
    n_target = control.round_size(cfg, run.state()["rounds"][str(r)])
    step, command = f"round {r}", "boar round run"
    while True:
        trials = run.trials()
        done = [t["trial"] for t in trials if t["round"] == r]
        if len(done) >= n_target:
            break
        check_unchanged(run, commit, step, command)
        dirs = gitops.untracked_dirs(run.root)
        n = 1 + max((t["trial"] for t in trials), default=0)
        trial = study.ask(dists)
        trial.set_user_attr("boar_trial", n)
        queued = trial.user_attrs.get("queued")
        config = store.full_config(hyps, trial.params)
        log(f"trial {n} ({len(done) + 1}/{n_target}{', ' + queued if queued else ''}): {config_text(store.non_default(config, defaults))}")
        try:
            result = evalrun.run_trial(
                run, config, "dev", cfg["repeats"], run.trial_dir(n), 3 * cfg["trial_target_s"]
            )
            wrote, stuck = undo_trial_writes(run, commit, dirs, run.trial_dir(n), step, command)
        except BaseException:
            with contextlib.suppress(Exception):
                study.tell(trial, state=TrialState.FAIL)
            raise
        if wrote:
            # Recorded, not discarded: the agent sees why in the summary and can mark the lever `fix`.
            result = {**result, "state": evalrun.FAILED, "error": "; ".join(e for e in (wrote, result["error"]) if e)}
        run.append_trial({"trial": n, "round": r, "commit": commit, "config": config, **result, "queued": queued})
        if result["state"] == evalrun.FAILED:
            study.tell(trial, state=TrialState.FAIL)
        else:
            trial.set_constraint(warmstart.GUARDS, 0.0 if result["state"] == evalrun.COMPLETE else 1.0)
            study.tell(trial, result["metric"])
        ids = [*done, n]
        run.update_state(lambda s: s["rounds"][str(r)].__setitem__("new_trials", ids))
        log(f"trial {n}: {result['state']} metric {fmt(result['metric'])} in {result['duration_s']:.1f}s"
            + (f" ({result['error']})" if result["error"] else ""))
        if stuck:
            raise Refused(f"trial {n} was recorded as failed, but {stuck}")
    facts = _facts(run, r)
    write_summary(run, r, facts)

    def finish(state: dict) -> None:
        rr = state["rounds"][str(r)]
        rr["new_trials"] = [t["trial"] for t in facts["round_trials"]]
        rr["incumbent_trial"] = facts["incumbent_trial"]
        # Stored now: later decisions change which trials the warm-start rules keep.
        inc = facts["incumbent"]
        rr["incumbent_metric"] = inc["metric"] if inc else None
        rr["incumbent_n"] = len(inc["repeats"]) if inc else 0
        rr["drift"] = facts["drift"]["flag"]
        rr["run_status"] = DONE

    run.update_state(finish)
    log(f"round {r} done; summary: {run.summary_path(r)}")


# --- summary --------------------------------------------------------------------


def _drift(
    queued: dict | None, before: list[dict], defaults: dict, floor: float | None, base: float | None = None
) -> dict:
    """Did the re-measured config move by more than the noise floor against its earlier repeats?

    The floor is scaled from the baseline's metric `base` to the config's earlier metric (`stats.floor_at`).
    """
    if queued is None:
        return {"flag": "n/a", "why": "nothing was re-measured this round"}
    earlier = stats.pooled(before, defaults).get(store.config_key(queued["config"], defaults))
    out: dict[str, Any] = {"flag": "n/a", "new": queued.get("metric"), "earlier": None, "earlier_n": 0}
    if earlier is not None:
        out.update(earlier=earlier["metric"], earlier_n=len(earlier["repeats"]))
    error = f" ({queued['error']})" if queued.get("error") else ""
    if earlier is None:
        failed = f"; this trial was {queued['state']}{error}" if queued["state"] != "complete" else ""
        return {**out, "why": f"no earlier repeats of this config{failed}"}
    if queued["state"] != "complete":
        # A config that no longer passes on this commit has moved, whatever its metric says.
        return {**out, "flag": "yes", "why": f"the re-measurement was {queued['state']}{error}"}
    if floor is None:
        return {**out, "why": "no noise floor yet"}
    delta = queued["metric"] - earlier["metric"]
    limit = stats.floor_at(floor, base, earlier["metric"])
    flag = "yes" if abs(delta) > limit else "no"
    why = f"|Δ| = {fmt(abs(delta))} {'>' if flag == 'yes' else '≤'} noise floor {fmt(limit)}"
    if limit != floor:
        why += f", scaled to this metric from {fmt(floor)} at the baseline's {fmt(base)}"
    return {**out, "flag": flag, "why": why}


def _lever_effects(trials: list[dict], space: dict, defaults: dict, key: str | None = None) -> list[list[str]]:
    """Each lever's marginal effect on the metric, or with `key` on that diagnostic, over the complete trials."""
    complete = [t for t in trials if t["state"] == "complete" and (key is None or key in (t.get("diagnostics") or {}))]
    rows = []
    for name, lever in space.items():
        pts = [(t["config"].get(name, defaults[name]), t["metric"] if key is None else t["diagnostics"][key])
               for t in complete]
        distinct = {json.dumps(v) for v, _ in pts}
        if not pts:
            text = "no complete trials"
        elif len(distinct) == 1:
            text = f"never varied (all {len(pts)} trials at `{json.dumps(pts[0][0])}`)"
        elif lever["type"] in ("bool", "categorical"):
            choices = [False, True] if lever["type"] == "bool" else lever["choices"]
            parts = []
            for c in choices:
                ms = [m for v, m in pts if type(v) is type(c) and v == c]
                if ms:
                    parts.append(f"`{json.dumps(c)}`: median {fmt(stats.median(ms))} (n={len(ms)})")
            text = "; ".join(parts)
        else:
            xs, ys = [v for v, _ in pts], [m for _, m in pts]
            text = (
                f"Spearman ρ = {fmt(stats.spearman(xs, ys), 3)} (n={len(pts)}); "
                f"tried {fmt(min(xs))} … {fmt(max(xs))}"
            )
        rows.append([f"`{name}`", lever["hypothesis"], text])
    return rows


def _facts(run: Run, r: int) -> dict:
    """Everything summary.md says about round r, computed from the run's files."""
    cfg, state, hyps = run.config(), run.state(), run.hypotheses()
    direction = cfg["direction"]
    defaults = store.lever_defaults(hyps)
    trials = run.trials()
    by_id = {t["trial"]: t for t in trials}
    round_trials = [t for t in trials if t["round"] == r]
    before, _ = warmstart.select(trials, hyps, r)
    after, _ = warmstart.select(trials, hyps, r + 1)
    floor = stats.noise_floor(trials, defaults)
    # Judged against the floor from before this round, so the measurement can't widen its own threshold.
    earlier_trials = [t for t in trials if t["round"] < r]
    drift_floor = stats.noise_floor(earlier_trials, defaults)
    drift_base = stats.noise_base(earlier_trials, defaults)
    queued = next((t for t in round_trials if t.get("queued") in ("baseline", "incumbent")), None)
    inc = warmstart.incumbent(trials, hyps, r + 1, direction)
    inc_trial = None
    if inc is not None:
        inc_trial = min(inc["trials"], key=lambda n: (stats.better(direction) * by_id[n]["metric"], n))
    return {
        "r": r,
        "cfg": cfg,
        "rec": state["rounds"][str(r)],
        "defaults": defaults,
        "space": store.search_space(hyps),
        "round_trials": round_trials,
        "after": after,
        "noise_floor": floor,
        "noise_trials": stats.noise_trials(trials, defaults),
        "all_trials": trials,
        "queued": queued,
        "drift": _drift(queued, before, defaults, drift_floor, drift_base),
        "noise_base": stats.noise_base(trials, defaults),
        "incumbent": inc,
        "incumbent_trial": inc_trial,
        "hyps": hyps,
    }


def _diagnostics(f: dict, r: int) -> list[str]:
    """The summary's Diagnostics section: lever effects on each diagnostic, bound observables, investigations."""
    keys = control.diag_keys(f["after"])
    if keys:
        rows = [[f"`{key}`", *row] for key in keys for row in _lever_effects(f["after"], f["space"], f["defaults"], key)]
        out = [
            "Each lever's marginal effect on each numeric diagnostic (the medians from diag.json), over the same "
            "trials as the lever effects. Diagnostics never affect the metric, the warm start or the incumbent.",
            "",
            md_table(["diagnostic", "lever", "hypothesis", "effect"], rows),
        ]
    else:
        out = ["No trial in the warm-start set or this round wrote numbers to diag.json."]
    active = store.active_hypotheses(f["hyps"])
    bound = "; ".join(f"{h['id']}: `{h['observable_key']}`" if h.get("observable_key") else f"{h['id']}: none yet"
                      for h in active)
    out += ["", f"Observable keys: {bound or 'no active hypothesis'}."]
    probes = [(h, control.investigation(h, r - 1)) for h in active if control.investigation(h, r - 1)]
    if probes:
        out += ["", "Investigations measured this round:", ""]
        out += [f"- {h['id']} (round {r - 1}): cause: {inv['cause']}; measures {', '.join(inv['measure'])}" for h, inv in probes]
    return out


def write_summary(run: Run, r: int, facts: dict | None = None) -> Path:
    f = facts or _facts(run, r)
    cfg, rec, defaults = f["cfg"], f["rec"], f["defaults"]
    better = "lower" if cfg["direction"] == "min" else "higher"
    ids = [t["trial"] for t in f["round_trials"]]
    out = [
        f"# Round {r} summary",
        "",
        f"Commit `{rec.get('commit')}`. Trials run: {id_list(ids)} ({len(ids)} of {control.round_size(cfg, rec)}). "
        f"Metric direction: {cfg['direction']} ({better} is better).",
        "",
        "## Incumbent",
        "",
    ]
    q, d = f["queued"], f["drift"]
    if q is None:
        out.append("No config was re-measured this round.")
    else:
        earlier = (
            f"{fmt(d['earlier'])} pooled over {d['earlier_n']} earlier repeats" if d.get("earlier") is not None
            else "no earlier repeats"
        )
        out.append(
            f"Re-measured this round ({q['queued']}): trial {q['trial']}, {config_text(store.non_default(q['config'], defaults))}, "
            f"{q['state']}, metric {fmt(q.get('metric'))} vs {earlier}. **drift: {d['flag']}** ({d['why']})."
        )
    out.append("")
    inc = f["incumbent"]
    if inc is None:
        out.append("After the round: no feasible trial in the warm-start set plus this round's trials, so no incumbent.")
    else:
        out.append(
            f"After the round: {config_text(inc['config'])}, pooled metric {fmt(inc['metric'])} over "
            f"{len(inc['repeats'])} repeats (trials {id_list(inc['trials'])}; best single trial {f['incumbent_trial']})."
        )
    out += ["", "## Noise floor", ""]
    noise = f["noise_trials"]
    if f["noise_floor"] is None and noise:
        out.append(
            f"Not known: {sum(stats.spread(t['repeats']) == 0 for t in noise)} of the {len(noise)} complete baseline "
            f"trials with two or more dev repeats (trials {id_list([t['trial'] for t in noise])}) returned the same "
            "metric bit for bit, so their median spread is 0 and measures no noise."
        )
    elif f["noise_floor"] is None:
        out.append("Not known yet: no complete baseline trial has two or more dev repeats.")
    else:
        out.append(
            f"{fmt(f['noise_floor'])}: the median, over {len(noise)} complete baseline trials (trials "
            f"{id_list([t['trial'] for t in noise])}), of the spread (max − min) of each trial's own dev repeats. "
            "Effects smaller than this count as no effect at the baseline's scale."
        )
        base, inc = f["noise_base"], f["incumbent"]
        if base:
            out.append(f"That is {fmt(100 * f['noise_floor'] / abs(base), 3)}% of the baseline's median {fmt(base)}.")
        scaled = stats.floor_at(f["noise_floor"], base, inc["metric"]) if inc else None
        if inc and scaled != f["noise_floor"]:
            by_id = {t["trial"]: t for t in f["all_trials"]}
            own = stats.median(
                stats.spread(by_id[n]["repeats"]) for n in inc["trials"] if len(by_id[n].get("repeats") or []) >= 2
            )
            out.append(
                f"Repeat spread grows with the metric, so among configs far from the baseline compare effects with "
                f"the floor scaled to their metric: {fmt(scaled)} at the incumbent's {fmt(inc['metric'])}"
                + (f" (the incumbent's own trials: median repeat spread {fmt(own)})" if own is not None else "")
                + ". The drift check uses the scaled floor."
            )
    warm = rec.get("warm") or {"copied": 0, "excluded": {}}
    ex = warm.get("excluded") or {}
    out += [
        "",
        "## Warm start",
        "",
        f"Copied {warm['copied']} earlier trials into study `round-{r}`. Left out: "
        f"rule 1 (state failed) {ex.get(warmstart.RULE1, 0)}; "
        f"rule 2 (a lever outside the search space set away from default) {ex.get(warmstart.RULE2, 0)}; "
        f"rule 3 (a lever of a hypothesis marked fix after the trial ran) {ex.get(warmstart.RULE3, 0)}.",
        "",
        "## Lever effects",
        "",
        f"Over the warm-start set plus this round's new trials: {sum(t['state'] == 'complete' for t in f['after'])} "
        f"complete trials. A numeric lever shows Spearman's ρ between its value and the metric "
        f"({better} metric is better).",
        "",
        md_table(["lever", "hypothesis", "effect"], _lever_effects(f["after"], f["space"], defaults)),
        "",
        "## Diagnostics",
        "",
        *_diagnostics(f, r),
        "",
        "## Infeasible and failed trials",
        "",
    ]
    bad = [t for t in f["round_trials"] if t["state"] != "complete"]
    if bad:
        out.append(md_table(
            ["trial", "state", "error", "logs"],
            [[t["trial"], t["state"], t.get("error") or "a guard failed", f"[trials/{t['trial']}/](../../trials/{t['trial']}/)"]
             for t in bad],
        ))
    else:
        out.append("None.")
    out += ["", "## All trials this round", ""]
    out.append(md_table(
        ["trial", "queued", "state", "metric", "duration (s)", "config", "diagnostics"],
        [[t["trial"], t.get("queued") or "", t["state"], fmt(t.get("metric")), fmt(t.get("duration_s"), 4),
          config_text(store.non_default(t["config"], defaults)),
          f"[diag](../../trials/{t['trial']}/)" if any(run.trial_dir(t["trial"]).glob("diag-*")) else ""]
         for t in f["round_trials"]],
    ))
    path = run.summary_path(r)
    store.write_text(path, "\n".join(out) + "\n")
    return path
