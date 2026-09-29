"""The human-readable files, rendered from the run's state (itself a fold of the events): pure
functions, no clock and no randomness, so deleting them and rebuilding reproduces them byte for byte."""
import csv
import io
import json
import math

from . import directives

META = ["trial", "round", "epoch", "commit", "fidelity", "seed", "replicate_of", "kind", "status",
        "objective", "wall_clock_s", "peak_mem_mb"]
CSV_NAMES = {"trial": "seq", "kind": "chosen_by", "peak_mem_mb": "peak_mem"}  # the spec's column names
GROUPS = {"c:": "constraints", "t:": "telemetry", "compat:": "compat", "L:": "levers", "sampled:": "sampled"}


def render(st: dict, view: dict) -> dict[str, str]:
    """Every generated file: {path relative to the run dir: contents}.

    `view` holds what the scheduler derives from the same state: `status` (the queue by priority, the
    conflicts), `revivals` (each removed hypothesis a revival is planned for -> its partner) and
    `testing` (each active hypothesis -> its current verdict record, or None, and its schedule)."""
    status = view["status"]
    return {"SUMMARY.md": summary(st, view), "exports/trials.csv": trials_csv(st["trials"]),
            "exports/hypotheses.csv": hypotheses_csv(st),
            **{f"rounds/{r:03d}.md": round_summary(st, status, r) for r, x in st["rounds"].items() if x["ended"]}}


CAP = 10  # lines per SUMMARY.md section
MORE = "see exports/hypotheses.csv and rounds/"
IN_WORDS = {"irrelevant": "doesn't matter", "no-improvement": "matters, but doesn't help"}
RUN_END = {"budget_spent": "the budget is spent", "user_stop": "the user stopped it",
           "target_reached": "the target is reached and confirmed", "exhausted": "the hypothesis list is exhausted",
           "max_trials": "the trial ceiling is reached"}


def _num(x) -> str:
    return f"{x:.3g}"


def _interval(s: dict) -> str:
    return f"{_num(s['estimate'])} [{_num(s['lower'])}, {_num(s['upper'])}]"


def _capped(lines: list[str]) -> list[str]:
    if not lines:
        return ["- none"]
    return lines if len(lines) <= CAP else lines[:CAP - 1] + [f"- … and {len(lines) - CAP + 1} more: {MORE}"]


def _progress(t: dict) -> str:
    """How far an active hypothesis is from a verdict: its current record (None: in burn-in) and
    its schedule (needed, spacing, cap)."""
    v, spacing, cap = t["record"], t["spacing"], t["cap"]
    if v is None:
        return f"burn-in: its first verdict check comes after {t['needed']} fresh sampler trials"
    fresh = v["burn_in"]["fresh"]
    if v["outcome"] == "pending-reject":
        return f"a reject is pending, confirmed or dropped at {v['confirmation']['due_at']} fresh sampler trials"
    held = f"{v['held']}; " if v["held"] else ""
    return f"{held}undecided after {fresh} fresh sampler trials (next check at {fresh + spacing}, inconclusive at {cap})"


ASIDE = {"parked": "parked: {}", "pruned": "pruned under {}", "inconclusive": "inconclusive: {}", "merged": "{}"}


def _set_aside(h: dict) -> str | None:
    return ASIDE[h["status"]].format(_reason(h)) if h["status"] in ASIDE else None


def _baseline_mean(st: dict, fidelity: dict, epoch: int) -> float | None:
    """The mean of the baseline replicates at this fidelity in this epoch."""
    return next((x["mean"] for x in reversed(st["baselines"]) if x["fidelity"] == fidelity and x["epoch"] == epoch),
                None)


def _where(st: dict, status: dict) -> list[str]:
    run, rounds = st["run"], st["rounds"]
    ended = [r for r in sorted(rounds) if r >= 1 and rounds[r]["ended"] and rounds[r].get("incumbent")]
    if ended:
        r = rounds[ended[-1]]
        inc, base = r["incumbent"], _baseline_mean(st, r["fidelity"], r.get("epoch", 0))
        line = (f"- Incumbent: {run['objective']} {inc['mean']:.4g} (trial {inc['trial']}, "
                f"{'confirmed' if inc['confirmed'] else 'unconfirmed'})")
        if base is not None:
            gain = (base - inc["mean"]) * _sign(st)
            line += f" against the baseline's {base:.4g}: {abs(gain):.4g} {'better' if gain >= 0 else 'worse'}"
            if st["delta"]:
                line += f", {abs(gain) / st['delta']:.3g}δ"
    else:
        line = "- Incumbent: none yet (no BO round has ended)"
    b = status["budget"]
    lines = [line, f"- Budget: {b['spent_s']:.1f}s used, {b['remaining_s']:.1f}s of {b['total_s']:.1f}s remaining"]
    if rounds:
        n = max(rounds)
        lines.append(f"- Round: R{n} " + (f"ended ({rounds[n]['ended']})" if rounds[n]["ended"] else "running"))
    else:
        lines.append("- Round: none yet (the calibration round is next)")
    if st["run_ended"]:
        lines.append(f"- Run ended: {RUN_END.get(st['run_ended'], st['run_ended'])}")
    return lines


def summary(st: dict, view: dict) -> str:
    """SUMMARY.md: one line per hypothesis (its title), each section capped, no statistics beyond one
    effect interval per line."""
    hyps, status, revivals = st["hypotheses"], view["status"], view["revivals"]
    revived = {h["revived_from"]: h["id"] for h in hyps.values() if "revived_from" in h}

    def line(h: dict, what: str) -> str:
        return f"- {h['id']} {h['spec']['title']}: {what}"

    working = [line(h, f"Δ {_interval(h['verdicts'][-1]['delta_stat'])}")
               for h in hyps.values() if h["status"] == "retained"]
    not_working = []
    for h in hyps.values():
        if h["status"] == "rejected":
            what = IN_WORDS[h["condition"]]
            if h["id"] in revived:
                what += f"; revived as {revived[h['id']]}"
            elif h["id"] in revivals:
                what += f"; a revival is planned with {revivals[h['id']]}"
            not_working.append(line(h, what))
    testing = [line(hyps[i], _progress(t)) for i, t in view["testing"].items()]
    aside = [line(h, why) for h in hyps.values() if (why := _set_aside(h))]
    queue = status["schedule"]["queue"]
    prio = {h["id"]: h["priority"] for h in status["hypotheses"]}
    up = [f"- {len(queue)} queued" + ("; the first 3 by priority:" if queue else "")]
    up += [f"- {i} {hyps[i]['spec']['title']} (priority {prio[i]})" for i in queue[:3]]
    body = {"Where we are": _where(st, status), "Working": _capped(working), "Not working": _capped(not_working),
            "Still testing": _capped(testing), "Set aside": _capped(aside), "Up next": up}
    run = st["run"]
    out = [f"# Research summary: run {run['run_id']}", "",
           f"{run['objective']} ({run['direction']}), δ {_num(st['delta']) if st['delta'] else 'not set yet'}."]
    for name, lines in body.items():
        out += ["", f"## {name}", "", *lines]
    return "\n".join(out) + "\n"


def _cell(v) -> str:
    """A value as a CSV cell; never empty (empty means the lever didn't exist)."""
    if isinstance(v, (dict, list)) or v is None:
        return json.dumps(v, sort_keys=True)
    return str(v)


def _csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def trials_csv(trials: list[dict]) -> str:
    """One row per trial of every kind. A lever cell is empty when the lever didn't exist in the
    trial's code and holds what it ran at otherwise (its baseline when parked or pinned): the levers
    the trial logged, never backfilled."""
    cols = [(p + k, g, k) for p, g in GROUPS.items()
            for k in sorted({k for t in trials for k in t.get(g) or {}})]
    out = []
    for t in trials:
        meta = [_cell(t["fidelity"]) if k == "fidelity" else t.get(k, "") for k in META]
        out.append(meta + [_cell(t[g][k]) if k in (t.get(g) or {}) else "" for _, g, k in cols])
    return _csv([CSV_NAMES.get(k, k) for k in META] + [c for c, _, _ in cols], out)


HYP_COLS = ["id", "title", "lens", "provenance", "mechanism", "state", "reason", "sqrt_vt_lower",
            "sqrt_vt_upper", "delta_lower", "delta_upper", "trials_used", "expected_right", "revived_from",
            "prediction_flag"]


def _actual(st: dict, hid: str, r: int) -> str:
    """H's verdict at the end of round r, in an expected record's words: retain, reject or undecided
    (a reject a proxy fidelity's drift check turned inconclusive is undecided)."""
    past = [v for v in st["hypotheses"][hid]["verdicts"] if v["round"] == r]
    if not past:
        return "undecided"
    v = past[-1]
    if v["outcome"] == "reject" and not any(x["removal"] == "inconclusive" and x["verdict"] == v["id"]
                                            for x in st["removals"]):
        return "reject"
    return "retain" if v["outcome"] == "retained" else "undecided"


def _expected(st: dict) -> dict:
    """{(hypothesis, round): expected verdict} for every round that has ended (the latest record wins)."""
    return {(x["record"]["hypothesis"], x["round"]): x["record"]["verdict"] for x in st["records"]
            if x["kind"] == "expected" and st["rounds"].get(x["round"], {}).get("ended")}


def _reason(h: dict) -> str:
    """Why H is where it is: irrelevant or no-improvement for a reject, else the stated reason."""
    if h["status"] == "rejected":
        return h["condition"]
    if h["status"] == "pruned":
        return f"{h['pruned_by']}: {h['prune_reason']}"
    if h["status"] == "merged":
        return f"merged into {h['merged_into']}"
    return {"parked": h.get("park_reason"), "inconclusive": h.get("reason")}.get(h["status"]) or ""


def hypotheses_csv(st: dict) -> str:
    """One row per hypothesis version: its spec, final state and reason, its latest verdict record's
    bounds, and how often its expected verdict was right."""
    expected = _expected(st)
    out = []
    for h in st["hypotheses"].values():
        s, last = h["spec"], h["verdicts"][-1] if h["verdicts"] else None
        mine = [(r, v) for (i, r), v in expected.items() if i == h["id"]]
        right = sum(v == _actual(st, h["id"], r) for r, v in mine)
        bounds = ([last["sqrt_vt"]["lower"], last["sqrt_vt"]["upper"], last["delta_stat"]["lower"],
                   last["delta_stat"]["upper"], len(last["trials"])] if last else [""] * 5)
        out.append([h["id"], s["title"], s["lens"], s["provenance"], s["mechanism"], h["status"], _reason(h),
                    *bounds, f"{right}/{len(mine)}" if mine else "", h.get("revived_from", ""),
                    (last["prediction"]["flag"] or "") if last else ""])
    return _csv(HYP_COLS, out)


TOP = 5  # the best sampler trials of a round a narrowing suggestion reads
NARROW = 0.5  # suggested when those trials and the baseline span at most this share of a range


def _sign(st: dict) -> int:
    return 1 if st["run"]["direction"] == "minimize" else -1


def _narrowings(st: dict, r: int) -> list[str]:
    """Levers whose posterior mass looks concentrated: the round's best TOP sampler trials (the
    baseline with them) fill at most half the searched range, on a hypothesis whose last record of
    the round passed every gate (what `narrow` needs). -> lines, each with the command."""
    # ponytail: the top trials stand in for the posterior's mass (other hypotheses vary under them);
    # read the verdict GP's posterior over each lever if suggestions misfire
    rnd, hyps, sign = st["rounds"][r], st["hypotheses"], _sign(st)
    space = rnd.get("search_space", {})
    ran = [t for t in st["trials"] if t.get("round") == r and t["kind"] == "sampler" and t["status"] == "finished"]
    top = sorted(ran, key=lambda t: (sign * t["objective"], t["trial"]))[:TOP]
    out = []
    for hid in rnd.get("hypotheses", []):
        past = [v for v in hyps[hid]["verdicts"] if v["round"] == r]
        if not past or past[-1]["outcome"] not in ("active", "retained") \
                or not all(g["passed"] for g in past[-1]["gates"].values()):
            continue
        for n in past[-1]["group"]:
            lv = space.get(n)
            vals = [{**t["levers"], **t.get("sampled", {})}[n] for t in top if n in t["levers"]]
            if lv is None or len(vals) < TOP:
                continue
            if lv["kind"] in ("float", "int"):
                lo, hi = min(vals + [lv["baseline"]]), max(vals + [lv["baseline"]])
                if lo < hi and hi - lo <= NARROW * (lv["high"] - lv["low"]):
                    out.append(f"- {hid} {n}: the best {TOP} sampler trials and the baseline lie in [{_num(lo)}, "
                               f"{_num(hi)}] of [{lv['low']}, {lv['high']}]: "
                               f"`narrow {hid} --lever {n} --low {lo!r} --high {hi!r}`")
            elif lv["kind"] == "categorical":
                opts = [o for o in lv["options"] if o in vals or o == lv["baseline"]]
                if 2 <= len(opts) < len(lv["options"]):
                    out.append(f"- {hid} {n}: the best {TOP} sampler trials and the baseline use {opts} of "
                               f"{lv['options']}: `narrow {hid} --lever {n} --options '{json.dumps(opts)}'`")
    return out


def _sigma(st: dict, r: int) -> float | None:
    """σ in round r: its replicates' re-estimate, else the calibration round's at its fidelity."""
    rnd = st["rounds"][r]
    if (n := rnd.get("noise")) and n["sigma"] is not None:
        return n["sigma"]
    return next((x["sigma"] for x in (st["rounds"].get(0, {}).get("noise") or {}).get("rungs", [])
                 if x["fidelity"] == rnd["fidelity"]), None)


def _honouring(st: dict, r: int) -> list[str]:
    """For each discouraged directive the incumbent breaks: the best config logged up to this round (at
    its fidelity and epoch; mean over the config's trials) that honours it, and the gap to the
    incumbent with its 95% interval from σ (from the log only: no extra runs)."""
    rnd, sign, obj = st["rounds"][r], _sign(st), st["run"]["objective"]
    inc, out = rnd.get("incumbent"), []
    trials, sigma = {t["trial"]: t for t in st["trials"]}, _sigma(st, r)
    for d in st["registry"]["directives"]:
        if d["severity"] != "discouraged" or not inc:
            continue
        head = f"- {d['id']} ({d['statement']}): "
        if trials[inc["trial"]].get("compat", {}).get(d["id"]):
            out.append(head + "the incumbent honours it")
            continue
        ok = [t for t in st["trials"] if 1 <= t.get("round", 0) <= r and t["status"] == "finished"
              and t["fidelity"] == rnd["fidelity"] and t["epoch"] == rnd["epoch"] and t.get("compat", {}).get(d["id"])]
        if not ok:
            out.append(head + "no logged trial at the round's fidelity honours it")
            continue
        configs: dict[int, list[float]] = {}
        for t in ok:
            configs.setdefault(t.get("replicate_of", t["trial"]), []).append(t["objective"])
        root, ys = min(configs.items(), key=lambda c: (sign * sum(c[1]) / len(c[1]), c[0]))
        mean = sum(ys) / len(ys)
        gap = sign * (mean - inc["mean"])  # > 0: honouring it costs this much
        half = 1.96 * sigma * math.sqrt(1 / len(ys) + 1 / (inc["replicates"] + 1)) if sigma is not None else None
        out.append(head + f"trial {root} ({obj} {mean:.4g}, the mean over {len(ys)} trial{'s' * (len(ys) > 1)}) "
                   f"costs {_num(gap)}" + (f" [{_num(gap - half)}, {_num(gap + half)}]" if half is not None else "")
                   + " against the incumbent's mean")
    return out


def _proposed_directives(st: dict, r: int) -> list[str]:
    """Two or more hypotheses parked off-intent for one concern (similar reasons): a directive the
    user may state (the reviewer never writes one)."""
    parks = {x["id"]: x["reason"][len("off-intent: "):] for x in st["removals"]
             if x["removal"] == "parked" and (x["reason"] or "").startswith("off-intent: ")
             and x["seq"] < st["rounds"][r]["ended_seq"] and st["hypotheses"][x["id"]]["status"] == "parked"}
    groups: list[list[str]] = []
    for hid, why in parks.items():
        g = next((g for g in groups if directives.similar(parks[g[0]], why)), None)
        g.append(hid) if g else groups.append([hid])
    return [f"- {', '.join(g)} were parked off-intent for one concern (\"{parks[g[0]]}\"): the user may state "
            "it as a directive" for g in groups if len(g) >= 2]


def round_summary(st: dict, status: dict, r: int) -> str:
    """rounds/NNN.md: the mechanical sections, then the round analyst's narrative."""
    rnd, run, hyps = st["rounds"][r], st["run"], st["hypotheses"]
    this = [t for t in st["trials"] if t.get("round") == r and t["status"] != "running"]
    verdicts = [v for h in hyps.values() for v in h["verdicts"] if v["round"] == r]
    inc = rnd.get("incumbent")
    fid = json.dumps(rnd["fidelity"], sort_keys=True)
    head = [f"# Round R{r}" + (" (the calibration round)" if r == 0 else ""), "",
            f"Fidelity {fid}, epoch {rnd.get('epoch', 0)}, commit {rnd['commit'][:12]}"
            + (f", testing {', '.join(rnd['hypotheses'])}." if rnd.get("hypotheses") else ".")]

    def record(v: dict) -> str:
        failed = [g for g, x in v["gates"].items() if not x["passed"]]
        why = v["condition"] or v["reason"] or v["held"]
        return (f"- {v['id']}: {v['outcome']}" + (f" ({why})" if why else "")
                + f"; √V_T {_interval(v['sqrt_vt'])}, Δ {_interval(v['delta_stat'])}, δ {_num(v['delta'])}; "
                + (f"gates failed: {', '.join(failed)}" if failed else "gates passed")
                + f"; {v['burn_in']['fresh']} fresh sampler trials"
                + (f"; {v['prediction']['flag']}" if v["prediction"]["flag"] else ""))

    if r == 0:
        noise = [f"- σ at {json.dumps(x['fidelity'], sort_keys=True)}: {_num(x['sigma'])} from {x['n']} baseline "
                 f"replicates, {x['cost_s']:.3g}s per trial" for x in (rnd.get("noise") or {}).get("rungs", [])]
        if st["calibration"]:
            c = st["calibration"]
            noise.append(f"- ladder calibration chose {json.dumps(c['chosen'], sort_keys=True)}"
                         + (" (no rung passed: the reference fidelity)" if c["fallback"] else ""))
    else:
        n = rnd.get("noise")
        r0 = n["r0_sigma"] if n else None
        noise = [(f"- σ {_num(n['sigma'])} from the round's replicates (df {n['df']})" if n and n["sigma"] is not None
                  else "- σ not re-estimated (too few replicates)")
                 + (f"; the calibration round's σ {_num(r0)}" if r0 is not None else "")
                 + ("; shift flagged" if n and n["shift_flagged"] else "")]
    expected = _expected(st)
    exp = [f"- {h}: expected {v}, actual {a} ({'right' if v == a else 'wrong'})"
           for (h, rr), v in expected.items() if rr == r for a in [_actual(st, h, r)]]
    tested = set(rnd.get("hypotheses", []))
    rivals = [f"- {a}: {_actual(st, a, r) if a in tested else 'not in this round'}; "
              f"{b}: {_actual(st, b, r) if b in tested else 'not in this round'}"
              for a, b in status["conflicts"]["rivals"] if a in tested or b in tested]
    spent = rnd["spent_s"]
    body = {
        "Trigger": [f"- {rnd['ended']}"],
        "Verdict records": [record(v) for v in verdicts],
        "Incumbent": [f"- trial {inc['trial']}: {run['objective']} {inc['mean']:.4g}, the mean over "
                      f"{inc['replicates'] + 1} trials ({'confirmed' if inc['confirmed'] else 'unconfirmed'}), "
                      f"levers {json.dumps(inc['levers'], sort_keys=True)}"] if inc else [],
        "Noise": noise,
        "Budget": [f"- the round's trials: {sum(t['wall_clock_s'] for t in this):.1f}s; by its end {spent:.1f}s "
                   f"used, {run['budget_s'] - spent:.1f}s of {run['budget_s']:.1f}s remaining"],
        "Discouraged directives": _honouring(st, r),
        "Suggested narrowings": _narrowings(st, r) if r else [],
        "Expected versus actual verdicts": exp,
        "Rivals": rivals,
        "Proposed directives": _proposed_directives(st, r),
    }
    out = head
    for name, lines in body.items():
        out += ["", f"## {name}", "", *(lines or ["- none"])]
    out += ["", "## Narrative", ""]
    story = [x for x in st["records"] if x["kind"] == "narrative" and x["record"]["round"] == r]
    if story:
        x, n = story[-1], story[-1]["record"]
        out += [n["text"]]
        for name in ("diagnostics", "suggestions"):
            if n[name]:
                out += ["", f"{name.capitalize()}:", *(f"- {s}" for s in n[name])]
        out += ["", f"Cites: {', '.join(n['cites']) or 'none'}. By {x['actor']}"
                + (f" ({x['agent_id']})." if x["agent_id"] else ".")]
    else:
        out += ["_No narrative yet: the round analyst records one with `record narrative`._"]
    return "\n".join(out) + "\n"
