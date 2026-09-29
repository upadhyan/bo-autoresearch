"""Wrap-up and distillation (#37): after any run end, `wrapup` drives the hand-back through `next` —
the analyst's takeaways and DISTILL_SPEC.md, its review, the distilled branch (from the base commit, no
harness dependency, protected paths untouched), verification against the research incumbent at the
reference fidelity (outside the budget), then REPORT.md and `wrapup_finished`. `clean` reclaims disk.
"""
import json
import os
import shutil
from pathlib import Path

from conftest import bo, events, git, register, round_run, write_run_yaml
from test_verdicts import lever, spec

REPORT = ["Headline", "What to take", "Takeaways", "What worked and what didn't", "Interactions and revivals",
          "Terms of the run", "Process diagnostics", "Files"]


def of_type(run_dir, type):
    return [e["payload"] for e in events(run_dir) if e["type"] == type]


def record(repo, kind, body, actor="round-analyst", agent="an"):
    f = Path(repo).parent / f"{kind}-{len(list(Path(repo).parent.glob(kind + '-*')))}.json"
    f.write_text(json.dumps(body))
    return bo(repo, "record", kind, "--file", str(f), "--agent-id", agent, "--actor", actor,
              "--rationale", "wrap-up")


def sections(text):
    out, name = {}, None
    for ln in text.splitlines():
        if ln.startswith("## "):
            name = ln[3:]
            out[name] = []
        elif name and ln.strip():
            out[name].append(ln)
    return out


def test_no_confirm_skips_distillation_marks_the_result_unconfirmed_and_ends_enforcement(repo, run_dir):
    code, out = bo(repo, "wrapup", "--rationale", "too early")
    assert code == 1 and "stop" in out["reason"], out  # the run hasn't ended
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0
    assert bo(repo, "stop", "--rationale", "the user stops after R0")[0] == 0
    assert bo(repo, "status")[1]["next"][0].startswith("wrapup")
    code, out = bo(repo, "wrapup", "--no-confirm", "--rationale", "the user wants the report now")
    assert code == 0, out
    assert out["next"] == ["record takeaways (the round analyst: at most 5 bullets, each citing verdict records)"]
    assert not (run_dir / "REPORT.md").exists()
    assert bo(repo, "check", "stop")[1]["allow"] is False
    code, out = bo(repo, "wrapup", "--rationale", "a different mode")
    assert code == 1 and "--no-confirm" in out["reason"], out
    assert record(repo, "takeaways", {"takeaways": ["Nothing was tested beyond the calibration round."],
                                      "cites": []})[0] == 0
    code, out = bo(repo, "wrapup", "--no-confirm", "--rationale", "finish")
    assert code == 0 and out["next"] == [], out
    [done] = of_type(run_dir, "wrapup_finished")
    assert done["result"] == "unconfirmed" and done["confirmed"] is None
    assert not [t for t in of_type(run_dir, "trial_started") if t["kind"] == "wrapup"]
    text = (run_dir / "REPORT.md").read_text()
    s = sections(text)
    assert list(s) == REPORT
    assert s["Headline"][0].startswith("- Unconfirmed")
    assert s["Takeaways"][0] == "- Nothing was tested beyond the calibration round."
    assert bo(repo, "check", "stop")[1] == {"allow": True}
    code, out = bo(repo, "wrapup", "--no-confirm", "--rationale", "again")
    assert code == 1 and "finished" in out["reason"]


TOY = Path(__file__).parent / "toy_distill"
SIGMA = 0.05
# H1: a bowl whose bottom (x = 0.8) beats its no-op baseline (0.6) by 0.16; H2: a lever that does nothing
LEVERS = {"H1": ("The bowl has a better bottom", 0.6, '    term += 4 * ((lever("H1.x") - 0.8) ** 2 - 0.04)\n'),
          "H2": ("A lever that does nothing", 0.0, '    term += 0 * lever("H2.x")\n')}


def toy_run(d, python, extra="budget_s: 3600\n", **planted):
    """init on the wrap-up toy -> (repo, run_dir, env with TOY_<planted>)."""
    d.mkdir(exist_ok=True)
    repo = d / "project"
    shutil.copytree(TOY, repo)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "toy")
    y = write_run_yaml(d / "run.yaml", python, extra + "reference_fidelity: {epochs: 4}\nreplicates_k: 3\ndelta: 0.1\n")
    code, out = bo(repo, "init", str(y), "--rationale", "start")
    assert code == 0, out
    return repo, Path(out["run_dir"]), {**os.environ, **{f"TOY_{k.upper()}": str(v) for k, v in planted.items()}}


def code_levers(repo, run_dir, d, env, levers=LEVERS):
    """Propose, register, write, smoke and commit each planted hypothesis."""
    train = run_dir / "worktree" / "train.py"
    for n, (h, (title, base, line)) in enumerate(levers.items(), 1):
        f = d / f"{h}.json"
        f.write_text(json.dumps(spec({"x": lever(baseline=base)}, title=title)))
        assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
        assert register(repo, h)[0] == 0
        src = train.read_text()
        if n == 1:
            src = src.replace("import random\n", "import random\n\nfrom boautoresearch import lever\n")
        train.write_text(src.replace("    term = 0.0\n", "    term = 0.0\n" + line))
        code, out = bo(repo, "smoke", h, "--rationale", "try", env=env)
        assert code == 0 and out["passed"], out
        code, out = bo(repo, "commit-lever", h, "--rationale", "touches: train.py", env=env)
        assert code == 0, out


def researched(d, python, seed=1):
    """A toy research run, ended by the user once H1 is retained and H2 rejected -> (repo, run_dir, env)."""
    repo, run_dir, env = toy_run(d, python, f"budget_s: 3600\nseed: {seed}\n", sigma=SIGMA)
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    code_levers(repo, run_dir, d, env)
    for _ in range(6):
        code, out = round_run(repo, "--rationale", "search", env=env)
        assert code == 0, out
        states = {h["id"]: h["status"] for h in bo(repo, "status")[1]["hypotheses"]}
        if states["H1.v1"] == "retained" and states["H2.v1"] == "rejected":
            break
    assert states["H1.v1"] == "retained" and states["H2.v1"] == "rejected", states
    assert bo(repo, "stop", "--rationale", "the user has seen enough")[0] == 0
    return repo, run_dir, env


def last_verdict(repo, h):
    return bo(repo, "verdict", h)[1]["records"][-1]["id"]


def distill_records(repo, run_dir):
    """The round analyst's takeaways and distillation spec, then the reviewer's approval."""
    w = bo(repo, "status")[1]["wrapup"]
    v1, x = last_verdict(repo, "H1.v1"), w["config"]["H1.x"]
    wrapq = [{"record": "wrapup", "field": "config.H1.x", "value": x}]
    assert record(repo, "takeaways", {"takeaways": [f"The bowl's lever pays off ({v1})."], "cites": [v1]})[0] == 0
    code, out = record(repo, "distill_spec", {
        "content": f"Keep H1: set the bowl's x to {x} as a constant in train.py. Drop H2 and the lever() scaffolding.",
        "cites": [v1], "changes": [], "quotes": wrapq})
    assert code == 0, out
    assert (run_dir / "DISTILL_SPEC.md").read_text().startswith("# Distillation spec")
    assert record(repo, "distill_review", {"verdict": "approve", "rationale": "small, fits the brief"},
                  actor="registration-reviewer", agent="rr")[0] == 0
    return x


def write_distilled(run_dir, x):
    """The lever coder in distill mode: H1's mechanism at its tuned value, in the project's own code."""
    train = run_dir / "distilled" / "train.py"
    lines = [f"    term = 4 * (({x!r} - 0.8) ** 2 - 0.04)  # the bowl, tuned" if ln.startswith("    term =") else ln
             for ln in train.read_text().split("\n")]
    train.write_text("\n".join(lines))


def test_the_distilled_branch_holds_only_the_retained_mechanism_without_the_harness_and_is_verified(
        tmp_path, project_python):
    repo, run_dir, env = researched(tmp_path / "v", project_python)
    spent = bo(repo, "status")[1]["budget"]["spent_s"]
    code, out = bo(repo, "wrapup", "--rationale", "hand back", env=env)
    assert code == 0, out
    assert out["wrapup"]["retained"] == ["H1.v1"] and out["wrapup"]["distilled"]["branch"].endswith("-distilled")
    assert [d.split(" (")[0] for d in out["next"]] == ["record takeaways", "record distill_spec"]
    x = distill_records(repo, run_dir)
    assert abs(x - 0.8) < 0.15  # the tuned value is near the planted bottom
    assert bo(repo, "status")[1]["next"][0].startswith("write the distilled branch in ")

    # the harness refuses a distilled commit that touches a protected path or depends on the harness
    wt = run_dir / "distilled"
    (wt / "levers.json").write_text('{"H1.x": 0.8}\n')
    code, out = bo(repo, "wrapup", "--rationale", "commit it", env=env)
    assert code == 1 and "levers.json" in out["reason"], out
    git(wt, "checkout", "--", "levers.json")
    src = (wt / "train.py").read_text()
    (wt / "train.py").write_text("from boautoresearch import lever\n" + src)
    code, out = bo(repo, "wrapup", "--rationale", "commit it", env=env)
    assert code == 1 and "boautoresearch" in out["reason"], out
    (wt / "train.py").write_text(src)
    assert not of_type(run_dir, "distill_committed")

    write_distilled(run_dir, x)
    code, out = bo(repo, "wrapup", "--rationale", "commit and verify", env=env)
    assert code == 0 and out["result"] == "verified", out
    [v] = of_type(run_dir, "verification")
    assert v["passed"] and len(v["research"]["trials"]) == len(v["distilled"]["trials"]) == 3
    wrap = [t for t in of_type(run_dir, "trial_started") if t["kind"] == "wrapup"]
    assert [t["trial"] for t in wrap] == v["research"]["trials"] + v["distilled"]["trials"]
    assert all(t["fidelity"] == {"epochs": 4} for t in wrap)
    assert bo(repo, "status")[1]["budget"]["spent_s"] == spent  # wrap-up is outside the budget

    # the branch: taken from the base commit, only H1's mechanism, no harness import outside the runner
    branch, base = out["confirmed"]["branch"], of_type(run_dir, "run_started")[0]["base_commit"]
    assert git(repo, "merge-base", "--is-ancestor", base, branch) == ""
    research = of_type(run_dir, "run_started")[0]["branch"]
    assert git(repo, "rev-list", "--count", f"{branch}..{research}") != "0"
    assert all(c not in git(repo, "rev-list", branch).split() for c in (p["commit"] for p in of_type(run_dir, "lever_committed")))
    assert git(repo, "diff", "--name-only", base, branch).split() == ["train.py"]
    files = git(repo, "ls-tree", "-r", "--name-only", branch).split()
    assert [f for f in files if "boautoresearch" in git(repo, "show", f"{branch}:{f}")] == ["runner.py"]
    distilled = git(repo, "show", f"{branch}:train.py")
    assert "lever" not in distilled and "H2" not in distilled and repr(x) in distilled

    s = sections((run_dir / "REPORT.md").read_text())
    assert list(s) == REPORT
    trials = ", ".join(map(str, v["distilled"]["trials"]))
    assert s["Headline"][0].startswith(f"- Confirmed result: loss {v['distilled']['mean']:.4g} at the reference fidelity")
    assert f"(trials {trials}) of the distilled branch `{branch}` (verified)" in s["Headline"][0]
    assert s["What to take"][0] == f"- Distilled branch: `{branch}`, verified"
    assert s["Takeaways"][0].startswith("- The bowl's lever pays off (V-")
    assert "- H2.v1 A lever that does nothing" in s["What worked and what didn't"]
    assert bo(repo, "check", "stop")[1] == {"allow": True}

    # the report rebuilds byte for byte from the log
    before = {n: (run_dir / n).read_bytes() for n in ("REPORT.md", "DISTILL_SPEC.md")}
    for n in before:
        (run_dir / n).unlink()
    assert bo(repo, "rebuild")[0] == 0
    assert {n: (run_dir / n).read_bytes() for n in before} == before

    # clean: the worktrees and the venv go, the branches and the log stay
    code, out = bo(repo, "clean", "--rationale", "reclaim disk")
    assert code == 0, out
    assert not (run_dir / "worktree").exists() and not wt.exists() and not (run_dir / "venv").exists()
    assert (run_dir / "log.db").exists() and of_type(run_dir, "cleaned")
    assert set(git(repo, "branch", "--format=%(refname:short)").split()) >= {research, branch}
    assert str(wt) not in git(repo, "worktree", "list")


def test_a_planted_distillation_bug_gets_two_retries_then_the_research_branch_is_the_confirmed_result(
        tmp_path, project_python):
    repo, run_dir, env = researched(tmp_path / "u", project_python)
    assert bo(repo, "wrapup", "--rationale", "hand back", env=env)[0] == 0
    v1, x = last_verdict(repo, "H1.v1"), bo(repo, "status")[1]["wrapup"]["config"]["H1.x"]
    ok = {"content": "Keep H1 as a constant.", "cites": [v1], "changes": []}
    for kind, body, why in [
            ("takeaways", {"takeaways": [f"a point ({v1})"] * 6, "cites": [v1]}, "1 to 5"),
            ("takeaways", {"takeaways": ["no record named"], "cites": [v1]}, "takeaways[0] must name"),
            ("distill_review", {"verdict": "approve", "rationale": "fine"}, "no distillation spec"),
            ("distill_spec", {**ok, "cites": []}, "retained hypothesis's latest verdict record"),
            ("distill_spec", {**ok, "content": f"Set x to {x}."}, "not quoted"),
            ("distill_spec", {**ok, "content": f"Set x to {x}.",
                              "quotes": [{"record": "wrapup", "field": "config.H1.x", "value": x + 0.1}]}, "config.H1.x")]:
        code, out = record(repo, kind, body)
        assert code == 1 and why in out["reason"], (kind, out)
    x = distill_records(repo, run_dir)
    for attempt, bug in enumerate((0.3, 0.31, 0.32), 1):  # the coder keeps getting the bowl's bottom wrong
        write_distilled(run_dir, bug)
        code, out = bo(repo, "wrapup", "--rationale", f"attempt {attempt}", env=env)
        assert code == 0, out
        if attempt < 3:
            v = out["verification"]
            assert not v["passed"] and v["gap"] > v["tolerance"] and repr(bug) in out["diff"]
            assert out["next"] == [f"fix the distilled branch in {run_dir / 'distilled'}: verification attempt "
                                   f"{attempt} of 3 failed (gap {v['gap']:.4g}, tolerance {v['tolerance']:.4g}; the "
                                   "lever coder in distill mode gets the gap and the diff), then `wrapup`"]
            code, again = bo(repo, "wrapup", "--rationale", "nothing new", env=env)  # no new commit: no new attempt
            assert code == 0 and again["next"] == out["next"] and len(of_type(run_dir, "verification")) == attempt
    assert out["result"] == "unverified", out
    vs = of_type(run_dir, "verification")
    assert [v["attempt"] for v in vs] == [1, 2, 3] and not any(v["passed"] for v in vs)
    assert len({tuple(v["research"]["trials"]) for v in vs}) == 1  # the research replicates run once
    research = of_type(run_dir, "run_started")[0]["branch"]
    c = out["confirmed"]
    assert c["branch"] == research and c["trials"] == vs[0]["research"]["trials"] and c["config"]["H1.x"] == x
    head = sections((run_dir / "REPORT.md").read_text())["Headline"]
    assert head[0].startswith(f"- Confirmed result: loss {c['mean']:.4g} at the reference fidelity")
    assert f"(trials {', '.join(map(str, c['trials']))}) of the research branch `{research}` at the incumbent's values" in head[0]
    assert head[1].startswith(f"- The distilled branch `{research}-distilled` is unverified: after 3 attempts")
    assert bo(repo, "check", "stop")[1] == {"allow": True}


def takeaways(repo, cites=()):
    body = {"takeaways": [f"What the run showed ({', '.join(cites)})." if cites else "Nothing reached a verdict."],
            "cites": list(cites)}
    assert record(repo, "takeaways", body)[0] == 0


def test_wrapup_trials_run_after_the_budget_is_spent_and_are_not_charged(tmp_path, project_python):
    repo, run_dir, env = toy_run(tmp_path / "b", project_python, "budget_s: 0.1\n", sleep=0.2)
    code, out = bo(repo, "round-run", "--rationale", "calibrate", env=env)
    assert code == 0 and out["run_ended"] == "budget_spent", out  # the smoke trial spends it all
    status = bo(repo, "status")[1]
    assert status["budget"]["remaining_s"] < 0
    assert bo(repo, "wrapup", "--rationale", "hand back", env=env)[1]["next"][0].startswith("record takeaways")
    takeaways(repo)
    refused = len(of_type(run_dir, "trial_refused"))
    code, out = bo(repo, "wrapup", "--rationale", "the final confirmation", env=env)
    assert code == 0 and out["result"] == "confirmed", out  # nothing retained: nothing to distill
    wrap = [t for t in bo(repo, "trials")[1]["trials"] if t["kind"] == "wrapup"]
    assert len(wrap) == 3 and all(t["status"] == "finished" and t["wall_clock_s"] >= 0.2 for t in wrap)
    assert out["confirmed"]["trials"] == [t["trial"] for t in wrap]
    assert bo(repo, "status")[1]["budget"] == status["budget"] and len(of_type(run_dir, "trial_refused")) == refused
    assert not (run_dir / "distilled").exists()
    assert sections((run_dir / "REPORT.md").read_text())["What to take"][0] == \
        "- Distilled branch: none (no hypothesis was retained)"


def test_a_run_ended_early_makes_active_hypotheses_inconclusive_and_lists_the_untested(tmp_path, project_python):
    # the trial ceiling ends R1 before H1's burn-in; H3 was never registered
    repo, run_dir, env = toy_run(tmp_path / "e", project_python, "budget_s: 3600\nmax_trials: 20\n", sigma=SIGMA)
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    code_levers(repo, run_dir, tmp_path, env, {"H1": LEVERS["H1"]})
    (run_dir / "worktree" / "NOTES.md").write_text("How the toy's loss is built.\n")
    code, out = bo(repo, "commit-change", "--reason", "document the loss", "--rationale", "a non-lever change", env=env)
    assert code == 0, out
    sha = out["commit"]
    f = tmp_path / "H2.json"
    f.write_text(json.dumps(spec({"x": lever()}, title="An idea never tested")))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "later")[0] == 0
    code, out = round_run(repo, "--rationale", "search", env=env)
    assert code == 0 and out["run_ended"] == "max_trials", out
    assert bo(repo, "status")[1]["hypotheses"][0]["status"] == "active"
    assert bo(repo, "wrapup", "--rationale", "hand back", env=env)[0] == 0
    [inc] = of_type(run_dir, "hypothesis_inconclusive")
    assert inc["id"] == "H1.v1" and inc["reason"] == "the run ended before a verdict"
    takeaways(repo)
    spec_ = {"content": "Nothing was retained.", "cites": [], "changes": []}
    code, out = record(repo, "distill_spec", spec_)
    assert code == 1 and sha in out["reason"], out  # a keep-or-drop decision for each non-lever change
    changes = [{"commit": sha, "decision": "keep", "reason": "the notes help"}]
    assert record(repo, "distill_spec", {**spec_, "changes": changes})[0] == 0
    code, out = bo(repo, "wrapup", "--rationale", "finish", env=env)
    assert code == 0 and out["result"] == "confirmed" and len(out["confirmed"]["trials"]) == 3, out  # past the ceiling
    worked = sections((run_dir / "REPORT.md").read_text())["What worked and what didn't"]
    assert worked[worked.index("### Inconclusive otherwise") + 1] == \
        "- H1.v1 The bowl has a better bottom: the run ended before a verdict"
    assert worked[worked.index("### Untested") + 1] == "- H2.v1 An idea never tested: proposed"
    take = sections((run_dir / "REPORT.md").read_text())["What to take"]
    assert take[take.index("### Non-lever changes") + 1] == f"- {sha[:12]} (document the loss): keep, the notes help"


def test_a_final_confirmation_whose_replicates_fail_leaves_the_result_unconfirmed(tmp_path, project_python):
    repo, run_dir, env = toy_run(tmp_path / "f", project_python)
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    assert bo(repo, "stop", "--rationale", "enough")[0] == 0
    assert bo(repo, "wrapup", "--rationale", "hand back", env=env)[0] == 0
    takeaways(repo)
    broken = {**env, "TOY_SIGMA": "nan"}  # every objective degenerates: the trials fail
    code, out = bo(repo, "wrapup", "--rationale", "the final confirmation", env=broken)
    assert code == 0 and out["result"] == "unconfirmed" and out["confirmed"]["mean"] is None, out
    trials = ", ".join(map(str, out["confirmed"]["trials"]))
    head = sections((run_dir / "REPORT.md").read_text())["Headline"]
    assert head[0].startswith(f"- Unconfirmed (the confirmation replicates (trials {trials}) did not all finish)")
    assert bo(repo, "rebuild")[0] == 0
    assert bo(repo, "clean", "--rationale", "disk")[0] == 0
    code, out = bo(repo, "clean", "--rationale", "again")
    assert code == 1 and "already clean" in out["reason"], out
    code, out = bo(repo, "clean", "../x", "--rationale", "elsewhere")
    assert code == 1 and "not a run id" in out["reason"], out
