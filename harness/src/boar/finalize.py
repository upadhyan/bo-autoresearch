"""Finalize: the holdout check of baseline against the incumbent, and the report skeleton.

The two configs alternate (B, I, B, I, …) so that drift in the machine hits both alike. The
report's tables are generated here; its narrative sections carry store.TODO_MARKER for the agent.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from boar import control, evalrun, gitops, rounds, stats, store, warmstart, worker
from boar.errors import Refused
from boar.rounds import config_text, fmt, id_list, md_table
from boar.store import Run

NOT_STARTED, RUNNING, DONE = "not_started", "running", "done"
LABELS = ("baseline", "incumbent")


def _fin(state: dict) -> dict:
    return state.setdefault("finalize", {"status": NOT_STARTED, "commit": None, "incumbent_config": None})


def _prepare(run: Run) -> str:
    fin = _fin(run.state())
    if fin.get("status") == DONE:
        raise Refused(f"already finalized; write the narrative sections marked {store.TODO_MARKER} in {run.report_path}")
    rounds.check_eval_frozen(run)
    last = _last_round_commit(run.state())
    if fin.get("commit") or last is None:
        commit = rounds.ensure_commit(run, fin.get("commit"), "boar: finalize", "finalize", "boar finalize")
    else:
        # Holdout must measure the code the dev trials ran, not edits made after the last round.
        r, commit = last
        rounds.require_code(
            run.root, commit, f"the last round (round {r}) ran", "finalize must measure the code the dev trials ran",
            "boar finalize",
        )

    def mark(state: dict) -> None:
        f = _fin(state)
        f["commit"] = commit
        f["status"] = RUNNING

    run.update_state(mark)
    return commit


def _last_round_commit(state: dict) -> tuple[int, str] | None:
    done = [(int(k), rec["commit"]) for k, rec in state.get("rounds", {}).items()
            if rec.get("run_status") == DONE and rec.get("commit")]
    return max(done) if done else None


def start_finalize(run: Run, foreground: bool) -> str:
    with worker.claim(run, worker.FINALIZE, None) as c:
        _prepare(run)
        if foreground:
            run_finalize_worker(run)
            return f"finalize finished; write the sections marked {store.TODO_MARKER} in {run.report_path}"
        rec = c.spawn()
    return (
        f"finalize is running in the background (pid {rec['pid']}, log {rec['log']}). "
        "Run `boar wait` until it finishes."
    )


def _final_round(run: Run, trials: list[dict]) -> int:
    """A round number past every trial, so the warm-start rules see the whole run."""
    return 1 + max([run.config()["rounds"], *(t["round"] for t in trials)])


def final_incumbent(run: Run) -> dict:
    """Non-default levers of the best feasible config in the warm-start set for the final search space."""
    cfg, hyps, trials = run.config(), run.hypotheses(), run.trials()
    inc = warmstart.incumbent(trials, hyps, _final_round(run, trials), cfg["direction"])
    return inc["config"] if inc else {}


def run_finalize_worker(run: Run) -> None:
    commit = _prepare(run)
    cfg, hyps = run.config(), run.hypotheses()
    if _fin(run.state()).get("incumbent_config") is None:
        chosen = final_incumbent(run)
        run.update_state(lambda s: _fin(s).__setitem__("incumbent_config", chosen))
    inc = _fin(run.state())["incumbent_config"]
    configs = {"baseline": store.full_config(hyps, {}), "incumbent": store.full_config(hyps, inc)}
    done = {rec["i"] for rec in store.read_jsonl(run.holdout_path)}
    total = 2 * cfg["holdout_repeats"]
    budget = 3 * cfg["trial_target_s"] / cfg["repeats"]
    rounds.log(f"finalize: holdout, baseline vs {config_text(inc)}, {total} runs ({len(done)} already done)")
    for i in range(total):
        if i in done:
            continue
        label = LABELS[i % 2]
        rounds.check_unchanged(run, commit, "finalize", "boar finalize")
        dirs = gitops.untracked_dirs(run.root)
        out_dir = run.path("holdout", str(i))
        res = evalrun.run_trial(run, configs[label], "holdout", 1, out_dir, budget)
        # The writes are put back before the next run and nothing is committed after holdout, so a run
        # that wrote keeps its measurement; the report flags it.
        wrote, stuck = rounds.undo_trial_writes(run, commit, dirs, out_dir, "finalize", "boar finalize")
        ok = res["state"] != evalrun.FAILED
        store.append_jsonl(run.holdout_path, {
            "i": i, "label": label, "config": configs[label], "ok": ok,
            "guards_ok": res["state"] == evalrun.COMPLETE if ok else None,
            "metric": res["metric"], "error": res["error"], "wrote": wrote, "duration_s": res["duration_s"],
        })
        rounds.log(f"holdout {i + 1}/{total} ({label}): {res['state']} metric {fmt(res['metric'])}"
                   + (f"; {wrote}" if wrote else ""))
        if stuck:
            raise Refused(f"holdout run {i} was recorded, but {stuck}")
    write_report(run)
    run.update_state(lambda s: _fin(s).__setitem__("status", DONE))
    rounds.log(f"finalize done; report: {run.report_path}")


# --- report ---------------------------------------------------------------------------

_GOAL_LINE = re.compile(r"^\s*(?:[-*]\s*)?[*_]*goal[*_]*\s*:\s*[*_]*\s*(.+?)\s*$", re.IGNORECASE)
_GOAL_HEADING = re.compile(r"^#+\s*goal\b", re.IGNORECASE)


def goal_line(spec_text: str) -> str | None:
    """The spec's Goal: a `Goal: …` line, or the first line under a `## Goal` heading."""
    lines = spec_text.splitlines()
    for i, line in enumerate(lines):
        m = _GOAL_LINE.match(line)
        if m and m.group(1).strip("*_ "):
            return m.group(1).strip("*_ ")
        if _GOAL_HEADING.match(line):
            for nxt in lines[i + 1:]:
                if nxt.lstrip().startswith("#"):
                    break
                if nxt.strip():
                    return nxt.strip().lstrip("-*> ").strip()
    return None


def _change(new: float | None, base: float | None, direction: str, floor: float | None) -> str:
    """Percent change with a verdict; a difference no bigger than the noise floor is no effect."""
    if new is None or base is None:
        return "—"
    if base == 0:
        return "n/a (baseline median is 0)"
    pct = (new - base) / abs(base) * 100
    gain = stats.better(direction) * (new - base)
    if floor is not None and abs(new - base) <= floor:
        return f"{pct:+.1f}% (within noise)"
    return f"{pct:+.1f}% ({'better' if gain < 0 else 'worse' if gain > 0 else 'same'})"


def _compact(ids: list[int]) -> str:
    """1, 2, 3, 5 -> "1–3, 5"."""
    parts = []
    for _, run_ in itertools.groupby(enumerate(sorted(ids)), key=lambda p: p[1] - p[0]):
        seq = [n for _, n in run_]
        parts.append(str(seq[0]) if len(seq) == 1 else f"{seq[0]}–{seq[-1]}")
    return ", ".join(parts) if parts else "none"


def _active_in(h: dict, r: int) -> bool:
    start, end = h.get("activated_round"), h.get("removed_round")
    return start is not None and start <= r and (end is None or end >= r)


def write_report(run: Run) -> Path:
    cfg, state, hyps = run.config(), run.state(), run.hypotheses()
    direction = cfg["direction"]
    defaults = store.lever_defaults(hyps)
    levers = store.all_levers(hyps)
    by_hid = {h["id"]: h for h in hyps}
    trials = run.trials()
    by_id = {t["trial"]: t for t in trials}
    fin = _fin(state)
    inc_cfg = fin.get("incumbent_config") or {}
    holdout = store.read_jsonl(run.holdout_path)
    valid, _ = warmstart.select(trials, hyps, _final_round(run, trials))
    groups = stats.pooled(valid, defaults)
    spec = run.spec_path.read_text() if run.spec_path.exists() else ""

    samples = {
        "holdout": {lb: [h["metric"] for h in holdout if h["label"] == lb and h["ok"] and h["guards_ok"]] for lb in LABELS},
        "dev": {lb: (g["repeats"] if g else []) for lb, g in (
            ("baseline", groups.get(stats.BASELINE_KEY)), ("incumbent", groups.get(store.config_key(inc_cfg, defaults))))},
    }
    med = {split: {lb: stats.median(xs) for lb, xs in by.items()} for split, by in samples.items()}
    floor = stats.noise_floor(trials, defaults)
    change = {split: _change(med[split]["incumbent"], med[split]["baseline"], direction, floor) for split in samples}
    inc_holdout = [h for h in holdout if h["label"] == "incumbent"]
    failed = sum(1 for h in inc_holdout if not h["ok"])
    broke = sum(1 for h in inc_holdout if h["ok"] and not h["guards_ok"])
    holdout_note = " and ".join(
        f"{what} in {k}/{len(inc_holdout)} holdout runs" for what, k in (("failed", failed), ("broke a guard", broke)) if k
    ) or None
    if not inc_cfg:
        change = {split: "— (incumbent is the baseline)" for split in change}
    if holdout_note:
        # The median over the passing runs alone would read as a plain gain, and none at all as no news.
        change["holdout"] = f"**{holdout_note}**"
    better = "lower" if direction == "min" else "higher"

    out = [
        f"# BOAR report: {goal_line(spec) or run.id}",
        "",
        f"Run `{run.id}`; metric direction {direction} ({better} is better); trials: {len(trials)}; rounds run: "
        f"{sum(1 for r in state.get('rounds', {}).values() if r.get('run_status') == 'done')}.",
        "",
        "## 1. Result",
        "",
        md_table(
            ["config", "split", "median", "spread", "n", "change vs baseline"],
            [[lb, split, fmt(med[split][lb]), fmt(stats.spread(samples[split][lb])), len(samples[split][lb]),
              change[split] if lb == "incumbent" else "—"]
             for split in ("holdout", "dev") for lb in LABELS],
        ),
        "",
        "Dev rows pool every complete dev repeat of the config across rounds; holdout rows are the "
        f"{cfg['holdout_repeats']} alternating finalize runs of each config.",
    ]
    bad = [h for h in holdout if not (h["ok"] and h["guards_ok"])]
    if bad:
        out += ["", "Holdout runs that failed or broke a guard:", ""]
        out += [f"- run {h['i']} ({h['label']}): {h.get('error') or 'a guard failed'} — [holdout/{h['i']}/](holdout/{h['i']}/)"
                for h in bad]
    wrote = [h for h in holdout if h.get("wrote")]
    if wrote:
        out += ["", "Holdout runs that wrote into the repo or eval/ (still counted; the writes were put back, new "
                "files moved to the run's `holdout/<i>/repo-writes/`):", ""]
        out += [f"- run {h['i']} ({h['label']}): {h['wrote']} — [holdout/{h['i']}/](holdout/{h['i']}/)" for h in wrote]
    out += [""]
    if holdout_note:
        why = "Its guards are correctness conditions; check" if broke else "Check"
        out += [f"**Warning: the recommended config {holdout_note}.** {why} the failed runs above before using it.", ""]
    if inc_cfg:
        out += ["Recommended config:", "", "```json", json.dumps(inc_cfg, indent=2, sort_keys=True), "```"]
    elif warmstart.incumbent(trials, hyps, _final_round(run, trials), direction) is None:
        out += ["Recommended config: the baseline. No config's latest dev trial completed feasibly, so nothing was compared."]
    else:
        out += [_baseline_line(trials, hyps, valid, levers, med["dev"]["baseline"], direction)]
    out += ["", f"The code is on branch `{state.get('branch')}` at commit `{fin.get('commit')}`.", ""]

    out += ["## 2. What worked", ""]
    if inc_cfg:
        out += ["The incumbent sets these levers away from default (generated):", ""]
        for name, value in sorted(inc_cfg.items()):
            hid = levers[name]["hypothesis"]
            out.append(f"- `{name}` = `{json.dumps(value)}` ({hid}: {by_hid[hid]['statement']})")
    else:
        out.append("The incumbent is the baseline: no lever is set away from default (generated).")
    out += ["", store.TODO_MARKER, ""]

    out += ["## 3. What didn't", ""]
    removed = [h for h in hyps if h["status"] == store.REMOVED]
    if removed:
        rows = []
        for h in removed:
            rm = next((x for x in state.get("removals", [])
                       if x["hypothesis"] == h["id"] and (x.get("verdict") or {}).get("decision") == "accept"), None)
            rows.append([h["id"], h["statement"], fmt(h.get("removed_round")), h.get("removed_reason") or "—",
                         id_list(rm["trials"]) if rm else "—",
                         f"accept: {rm['verdict']['reason']}" if rm else "—"])
        out.append(md_table(["hypothesis", "statement", "removed in round", "reason", "cited trials", "reviewer verdict"], rows))
    else:
        out.append("No hypothesis was removed.")
    out += ["", store.TODO_MARKER, ""]

    out += ["## 4. Rejected by review", ""]
    removals = {x["id"]: x for x in state.get("removals", [])}
    rejects = []
    for rv in run.reviews():
        if rv.get("decision") != "reject":
            continue
        item = rv["item"]
        if rv.get("kind") == "proposal" and item in by_hid:
            item = f"{item}: {by_hid[item]['statement']}"
        elif rv.get("kind") == "removal" and item in removals:
            hid = removals[item]["hypothesis"]
            item = f"{item} (remove {hid}: {by_hid[hid]['statement'] if hid in by_hid else '?'})"
        rejects.append([item, rv.get("kind", ""), fmt(rv.get("round")), rv.get("reason", "")])
    out.append(md_table(["item", "kind", "round", "reviewer reason"], rejects) if rejects else "Nothing was rejected.")
    out += [""]

    out += ["## 5. Caveats", "", "Generated hints:", ""]
    out.append(f"- Incumbent vs baseline: {change['dev']} on dev, {change['holdout']} on holdout.")
    out.append(
        f"- Noise floor: {fmt(floor)} (median within-trial repeat spread of "
        f"{len(stats.noise_trials(trials, defaults))} baseline dev trials); holdout spread "
        f"baseline {fmt(stats.spread(samples['holdout']['baseline']))}, incumbent {fmt(stats.spread(samples['holdout']['incumbent']))}."
    )
    drift = [r for r, rec in sorted(state.get("rounds", {}).items(), key=lambda kv: int(kv[0])) if rec.get("drift") == "yes"]
    out.append(f"- Drift flags: rounds {', '.join(drift)}." if drift else "- Drift flags: none.")
    if bad:
        out.append(f"- {len(bad)} of {len(holdout)} holdout runs failed or broke a guard (see section 1).")
    if wrote:
        out.append(f"- {len(wrote)} of {len(holdout)} holdout runs wrote into the repo or eval/ (see section 1); the "
                   "eval's check ran dev only, so its holdout path writes outside a temp dir.")
    untested = _untested_pairs(trials, store.search_space(hyps), defaults)
    if untested:
        shown = ", ".join(f"`{a}`+`{b}`" for a, b in untested[:10]) + (" …" if len(untested) > 10 else "")
        out.append(f"- Active lever pairs never set away from default together in a complete trial: {shown}.")
    out += ["", store.TODO_MARKER, ""]

    out += ["## 6. Experimental log", ""]
    log_rows = []
    for key, rec in sorted(state.get("rounds", {}).items(), key=lambda kv: int(kv[0])):
        r = int(key)
        ids = [t["trial"] for t in trials if t["round"] == r]
        if not ids and rec.get("run_status") != "done":
            continue
        warm = rec.get("warm") or {}
        excluded = warm.get("excluded") or {}
        it = by_id.get(rec.get("incumbent_trial"))
        if rec.get("incumbent_metric") is not None:
            metric = f"pooled {fmt(rec['incumbent_metric'])}, n={rec.get('incumbent_n')}"
        else:
            metric = f"best trial {fmt(it['metric'])}" if it else ""
        decisions = [f"{h['id']} {control.decision_text(state, h, key)}" for h in hyps if key in (h.get("decisions") or {})]
        log_rows.append([
            r,
            ", ".join(h["id"] for h in hyps if _active_in(h, r)) or "—",
            _compact(ids),
            f"{warm.get('copied', 0)} / {sum(excluded.values())}",
            f"#{it['trial']} {config_text(store.non_default(it['config'], defaults))} ({metric})" if it else "—",
            ", ".join(decisions) or "—",
        ])
    out.append(md_table(["round", "active hypotheses", "trials", "warm copied / left out", "incumbent (pooled metric)", "decisions"], log_rows)
               if log_rows else "No round ran.")
    out += ["", "Full log: [trials.jsonl](trials.jsonl). Round summaries: `rounds/<r>/summary.md`.", ""]

    path = run.report_path
    store.write_text(path, "\n".join(out))
    return path


def _baseline_line(
    trials: list[dict], hyps: list[dict], valid: list[dict], levers: dict, base: float | None, direction: str
) -> str:
    """Why the baseline is recommended: nothing in the final search space beat it, and maybe what did outside it."""
    head = "Recommended config: the baseline. No config in the final search space beat it on dev."
    if base is None:
        return head
    space, defaults = store.search_space(hyps), store.lever_defaults(hyps)
    kept = {t["trial"] for t in valid}
    sign = stats.better(direction)
    beat = [
        t for t in trials
        if t["trial"] not in kept and t["state"] == "complete" and isinstance(t.get("metric"), (int, float))
        and sign * (t["metric"] - base) < 0 and warmstart._space_rule(t, hyps, space, defaults)
    ]
    if not beat:
        return "Recommended config: the baseline. No tested config beat it on dev."
    fixed = {name for t in beat for name in warmstart._fixed_after(hyps, t["round"])}
    moved = sorted({
        name for t in beat for name in store.non_default(t["config"], defaults)
        if name not in space or name in fixed
    })
    named = ", ".join(f"`{n}` of {levers[n]['hypothesis']}" for n in moved if n in levers) or "levers outside it"
    return (
        f"{head} Trials {_compact([t['trial'] for t in beat])} did, but they set levers of hypotheses removed, "
        f"superseded or marked fix ({named}) away from default, so they are not comparable in the final search "
        "space and can't be recommended (see sections 3 and 6)."
    )


def _untested_pairs(trials: list[dict], space: dict, defaults: dict) -> list[tuple[str, str]]:
    """Lever pairs from different hypotheses that no complete trial moved together (an untested interaction)."""
    moved = [set(store.non_default(t["config"], defaults)) for t in trials if t["state"] == "complete"]
    return [(a, b) for a, b in itertools.combinations(space, 2)
            if space[a]["hypothesis"] != space[b]["hypothesis"] and not any(a in m and b in m for m in moved)]
