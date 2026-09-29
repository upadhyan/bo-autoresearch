"""Directives, the research brief and protected paths (#31): the review gate at registration,
prohibited limits enforced mechanically, discouraged ones flagged, revisions at a checkpoint, and
the protected-path hash manifest."""
import json

from conftest import bo, events, register, review
from test_rounds import BASE, init, make_repo, of_type
from test_verdicts import lever, repeat, spec


def propose(repo, s):
    f = repo.parent / f"spec-{len(list(repo.parent.glob('spec-*')))}.json"
    f.write_text(json.dumps(s))
    code, out = bo(repo, "propose", "--file", str(f), "--rationale", "an idea")
    assert code == 0, out
    return out["hypothesis"]["id"]


def status_of(repo, hid):
    return next(h for h in bo(repo, "status")[1]["hypotheses"] if h["id"] == hid)


def test_registration_without_a_review_is_refused(tmp_path, project_python):
    repo = make_repo(tmp_path)
    init(repo, project_python)
    hid = propose(repo, spec({"a": lever()}))
    code, out = bo(repo, "register", hid, "--rationale", "unreviewed")
    assert code != 0 and "review" in out["reason"] and hid in out["reason"]
    assert status_of(repo, hid)["status"] == "proposed"
    code, out = register(repo, hid)
    assert code == 0, out
    assert status_of(repo, hid)["status"] == "registered"


REGISTRY = """brief:
  purpose: find what limits the toy's loss
  contribution: a lever that moves the loss by at least delta
  complexity: small changes only
  provenance: standard techniques are fine
directives:
- id: no-pretrained
  severity: prohibited
  statement: no pretrained encoder weights downloaded from external model hubs
  reason: the paper's contribution is training from scratch
  scope: mechanism
- id: small-x
  severity: prohibited
  statement: keep x at or below 0.8
  reason: larger values crash the user's cluster
  scope: lever values
  predicate: "toy.x <= 0.8"
- id: no-cuda
  severity: prohibited
  statement: no custom CUDA kernels
  reason: the code must run on CPU
  scope: resource
  forbidden_patterns: ["torch\\\\.cuda", "import\\\\s+triton"]
- id: prefer-y-low
  severity: discouraged
  statement: y above 0.5 costs memory
  reason: the user's deployment is memory-bound
  scope: lever values
  predicate: "toy.y <= 0.5"
- id: no-extra-loss
  severity: discouraged
  statement: auxiliary losses make the model harder to explain
  reason: interpretability
  scope: mechanism
protected_paths: ["eval.py"]
"""


def setup(tmp_path, python, extra=""):
    repo = make_repo(tmp_path)
    (repo / "eval.py").write_text("def score(loss):\n    return loss\n")
    from conftest import git
    git(repo, "add", "eval.py")
    git(repo, "commit", "-q", "-m", "an evaluation module")
    return repo, init(repo, python, BASE + REGISTRY + extra)


def test_review_verdicts_prune_park_or_lower_priority_and_taste_never_prunes(tmp_path, project_python):
    repo, run_dir = setup(tmp_path, project_python)
    [reg] = of_type(run_dir, "registry_revised")
    assert reg["version"] == 1 and reg["brief"]["purpose"] == "find what limits the toy's loss"
    assert [d["id"] for d in reg["directives"]] == ["no-pretrained", "small-x", "no-cuda", "prefer-y-low",
                                                    "no-extra-loss"]
    assert set(reg["manifest"]) == {"runner.py", "levers.json", "eval.py"}

    pruned = propose(repo, spec({"a": lever()}, mechanism="Initialise the encoder from ImageNet pretrained weights."))
    # its mechanism is close to a prohibited directive's statement: only a strict review will do
    code, out = review(repo, pruned, directive_verdict="prune", directive="no-pretrained")
    assert code != 0 and "strict" in out["reason"] and "no-pretrained" in out["reason"], out
    code, out = review(repo, pruned, directive_verdict="prune", directive="no-pretrained", strict=True)
    assert code == 0, out
    assert status_of(repo, pruned)["status"] == "pruned"
    # the same idea under a new name is caught by its mechanism
    laundered = propose(repo, spec({"b": lever()}, title="Borrow a head start",
                                   mechanism="Start the encoder from weights pretrained on ImageNet."))
    code, out = review(repo, laundered)
    assert code != 0 and "strict" in out["reason"] and pruned in out["reason"], out
    assert review(repo, laundered, strict=True)[0] == 0
    code, out = bo(repo, "register", pruned, "--rationale", "try anyway")
    assert code != 0 and "pruned" in out["reason"]
    code, out = bo(repo, "smoke", pruned, "--rationale", "try anyway")
    assert code != 0 and "pruned" in out["reason"]
    [p] = of_type(run_dir, "hypothesis_pruned")
    assert p == {"id": pruned, "directive": "no-pretrained", "by": "review",
                 "reason": "nothing in the registry or the brief speaks against it"}

    # taste never prunes: a prune names a prohibited directive
    taste = propose(repo, spec({"b": lever()}, mechanism="Add an auxiliary reconstruction loss."))
    code, out = review(repo, taste, directive_verdict="prune", directive="no-extra-loss")
    assert code != 0 and "prohibited" in out["reason"] and "no-extra-loss" in out["reason"]
    code, out = review(repo, taste, directive_verdict="prune", directive="no-such-directive")
    assert code != 0 and "no-such-directive" in out["reason"]
    code, out = review(repo, taste, intent="off-intent", rationale="a new contribution, not a fix")
    assert code == 0, out
    assert status_of(repo, taste)["status"] == "parked"
    [park] = of_type(run_dir, "hypothesis_parked")
    assert park["id"] == taste and park["reason"] == "off-intent: a new contribution, not a fix"
    assert of_type(run_dir, "removal")[-1]["id"] == taste  # owed an interplay review like any park

    fits = propose(repo, spec({"c": lever()}, mechanism="Tune the step size."))
    stretch = propose(repo, spec({"d": lever()}, mechanism="Change the batch order."))
    disc = propose(repo, spec({"e": lever()}, mechanism="Weight the samples.", directives=["no-extra-loss"]))
    assert register(repo, fits)[0] == 0
    assert register(repo, stretch, "worth one round: cheap to test", intent="stretch")[0] == 0
    assert register(repo, disc, "the user may accept the loss", directive_verdict="deprioritize",
                    directive="no-extra-loss")[0] == 0
    assert [status_of(repo, h)["priority"] for h in (fits, stretch, disc)] == [0, -1, -1]
    lowered = {p["id"]: p.get("lowered") for p in of_type(run_dir, "hypothesis_registered")}
    assert lowered == {fits: None, stretch: {"by": 1, "reason": "worth one round: cheap to test"},
                       disc: {"by": 1, "reason": "the user may accept the loss"}}
    assert bo(repo, "status")[1]["schedule"]["queue"] == [fits, stretch, disc]


def test_lever_ranges_must_lie_inside_the_allowed_region(tmp_path, project_python):
    repo, run_dir = setup(tmp_path, project_python)
    wide = propose(repo, spec({"x": {**lever(), "path": "toy.x"}}))
    code, out = register(repo, wide)
    assert code != 0 and "small-x" in out["reason"] and '"H1.x": 1.0' in out["reason"], out
    assert status_of(repo, wide)["status"] == "proposed"
    inside = propose(repo, spec({"x": {**lever(high=0.8), "path": "toy.x"}}))
    assert register(repo, inside)[0] == 0
    # the predicate names the lever itself just as well; categorical options are all checked
    opts = propose(repo, spec({"x": {"kind": "categorical", "options": [0.1, 0.9], "baseline": 0.1,
                                     "path": "toy.x"}}))
    code, out = register(repo, opts)
    assert code != 0 and "small-x" in out["reason"] and '"H3.x": 0.9' in out["reason"], out


HOLE = """directives:
- id: no-mid-x
  severity: prohibited
  statement: x strictly between 0.26 and 0.49 overheats the rig
  reason: hardware
  scope: lever values
  predicate: "not (0.26 < toy.x < 0.49)"
seeds: [{H1.x: 0.4}, {H1.x: 0.9}]
"""


def test_trials_outside_the_allowed_region_never_run_and_such_seeds_are_skipped(tmp_path, project_python):
    # the hole lies between the grid points registration checks: only the per-trial check stops it
    flat = (spec({"x": {**lever(), "path": "toy.x"}}), '    term += 0 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [flat], extra=HOLE)
    refused_in = 0
    for run_dir, _ in runs:
        xs = [t["levers"]["H1.x"] for t in of_type(run_dir, "trial_started") if "H1.x" in t["levers"]]
        assert xs and not [x for x in xs if 0.26 < x < 0.49]
        assert 0.9 in xs
        [skip] = of_type(run_dir, "agent_trial_skipped")
        assert skip["kind"] == "seed" and skip["config"] == {"H1.x": 0.4} and "no-mid-x" in skip["reason"]
        refused = of_type(run_dir, "prohibited_check_refused")
        assert all(r["kind"] == "sampler" and r["directive"] == "no-mid-x"
                   and 0.26 < r["levers"]["H1.x"] < 0.49 for r in refused)
        refused_in += bool(refused)
    assert refused_in >= 2  # the sampler does ask inside the hole


def test_a_commit_adding_a_forbidden_pattern_is_refused(tmp_path, project_python):
    repo, run_dir = setup(tmp_path, project_python)
    worktree = run_dir / "worktree"
    (worktree / "kernels.py").write_text("import torch\nDEVICE = torch.cuda.current_device()\n")
    code, out = bo(repo, "commit-change", "--reason", "a faster kernel", "--rationale", "speed")
    assert code != 0 and "no-cuda" in out["reason"] and "kernels.py" in out["reason"], out
    (worktree / "kernels.py").write_text("DEVICE = 'cpu'\n")
    code, out = bo(repo, "commit-change", "--reason", "pin the device", "--rationale", "clarity")
    assert code == 0, out


def test_tampering_with_a_protected_file_by_any_means_is_caught_at_the_next_round_or_commit(
        tmp_path, project_python):
    import subprocess
    import sys
    from conftest import git
    repo, run_dir = setup(tmp_path, project_python)
    worktree = run_dir / "worktree"
    # a write the Bash hook's parser can't see
    subprocess.run([sys.executable, "-c", "open('eval.py', 'a').write('    # easier\\n')"], cwd=worktree,
                   check=True)
    (worktree / "notes.py").write_text("NOTE = 1\n")
    code, out = bo(repo, "commit-change", "--reason", "a note", "--rationale", "notes")
    assert code != 0 and "eval.py" in out["reason"] and "protected" in out["reason"], out
    code, out = bo(repo, "round-run", "--rationale", "calibrate")
    assert code != 0 and "eval.py" in out["reason"], out
    # committed behind the harness's back: the worktree is clean, the manifest still catches it
    git(worktree, "add", "-A")
    git(worktree, "commit", "-q", "-m", "sneaky")
    code, out = bo(repo, "round-run", "--rationale", "calibrate")
    assert code != 0 and "eval.py" in out["reason"], out
    git(worktree, "reset", "-q", "--hard", "HEAD~1")
    # nor can a commit behind the harness's back slip code past the forbidden-pattern scan
    (worktree / "kernels.py").write_text("import triton\n")
    git(worktree, "add", "-A")
    git(worktree, "commit", "-q", "-m", "sneaky kernel")
    code, out = bo(repo, "round-run", "--rationale", "calibrate")
    assert code != 0 and "not committed by the harness" in out["reason"], out
    git(worktree, "reset", "-q", "--hard", "HEAD~1")
    assert bo(repo, "round-run", "--rationale", "calibrate")[0] == 0

    # a change to a protected path is refused outright, a new file under a protected glob too
    (worktree / "runner.py").write_text("print('no')\n")
    code, out = bo(repo, "commit-change", "--reason", "rewrite the runner", "--rationale", "?")
    assert code != 0 and "runner.py" in out["reason"], out
    assert not of_type(run_dir, "commit_change")


def test_compat_flags_on_every_trial_and_a_revision_prunes_parks_restores_and_asks_for_re_reviews(
        tmp_path, project_python):
    import csv
    import yaml
    from conftest import round_run
    from test_rounds import toy_env
    from test_scheduler import in_background, round_trials
    from test_verdicts import code_hypothesis
    env = toy_env(sigma=0.01, sleep=0.1)
    repo, run_dir = setup(tmp_path, project_python, "delta: 0.1\n")
    pretrained = propose(repo, spec({"a": lever()}, mechanism="Load pretrained encoder weights from a hub."))
    assert review(repo, pretrained, directive_verdict="prune", directive="no-pretrained", strict=True)[0] == 0
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    two = spec({"x": {**lever(high=0.8), "path": "toy.x"}, "y": {**lever(), "path": "toy.y"}},
               directives=["no-extra-loss"])
    code_hypothesis(repo, run_dir, tmp_path, 2, two, '    term += 0 * (lever("H2.x") + lever("H2.y"))', env)
    p = in_background(repo, run_dir, env, lambda d: len(round_trials(d, 1)) >= 6)
    assert bo(repo, "checkpoint", "--rationale", "the user wants to revise the directives")[0] == 0
    assert json.loads(p.communicate(timeout=60)[0])["trigger"] == "checkpoint"
    assert status_of(repo, "H2.v1")["status"] == "active"
    # H3 fits the current limit on x (0.8), H4 is untouched by any directive
    h3 = propose(repo, spec({"x": {**lever(high=0.6), "path": "toy.x"}}))
    h4 = propose(repo, spec({"z": {**lever(), "path": "toy.z"}}, mechanism="Reorder the data shards."))
    assert register(repo, h3)[0] == 0 and register(repo, h4)[0] == 0

    def flags():
        return {t["trial"]: (t["levers"], t["compat"]) for t in bo(repo, "trials")[1]["trials"]}

    before = flags()
    assert len(before) == len(of_type(run_dir, "trial_started"))
    with_y = [(lv, c) for lv, c in before.values() if "H2.y" in lv]
    assert with_y and all(c == {"prefer-y-low": lv["H2.y"] <= 0.5,
                                "no-extra-loss": lv["H2.x"] == 0 and lv["H2.y"] == 0} for lv, c in with_y)
    assert any(not c["prefer-y-low"] for _, c in with_y) and any(c["prefer-y-low"] for _, c in with_y)
    assert all(c == {"prefer-y-low": True, "no-extra-loss": True} for lv, c in before.values() if "H2.y" not in lv)
    with open(run_dir / "exports" / "trials.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["compat:prefer-y-low"] for r in rows] == [str(c["prefer-y-low"]) for _, c in before.values()]

    reg = yaml.safe_load(REGISTRY)
    revised = [d for d in reg["directives"] if d["id"] != "no-pretrained"]  # relaxed: removed
    for d in revised:
        if d["id"] == "small-x":
            d["predicate"] = "toy.x <= 0.4"  # tightened
        if d["id"] == "prefer-y-low":
            d["predicate"] = "toy.y <= 0.9"
    f = tmp_path / "revision.yaml"
    f.write_text(yaml.safe_dump({"directives": revised}))
    code, out = bo(repo, "checkpoint", "--revise", str(f), "--rationale", "the user's new limits")
    assert code == 0, out
    [_, rev] = of_type(run_dir, "registry_revised")
    assert rev["version"] == 2 and rev["brief"] == reg["brief"] and rev["protected_paths"] == ["eval.py"]
    # tightened: the queued H3 is pruned, the active H2 parked; relaxed: the pruned H1 is proposed again
    assert status_of(repo, h3)["status"] == "pruned"
    assert [p["by"] for p in of_type(run_dir, "hypothesis_pruned")] == ["review", "revision"]
    assert status_of(repo, "H2.v1")["status"] == "parked"
    assert of_type(run_dir, "hypothesis_parked")[-1]["reason"].startswith("prohibited: small-x")
    code, out = bo(repo, "unpark", "H2.v1", "--reason", "the user wants it back", "--rationale", "?")
    assert code != 0 and "small-x" in out["reason"], out  # never back into a prohibited region
    assert status_of(repo, pretrained)["status"] == "proposed"
    assert of_type(run_dir, "hypothesis_unpruned") == [{"id": pretrained, "directive": "no-pretrained",
                                                        "version": 2}]
    # the flags are recomputed from the resolved configs under the revised registry
    after = flags()
    assert all(c["prefer-y-low"] == (lv.get("H2.y", 0) <= 0.9) for lv, c in after.values())
    assert any(before[n][1]["prefer-y-low"] != after[n][1]["prefer-y-low"] for n in before)

    # every hypothesis is re-reviewed: the queued H4, and the restored H1 before it registers
    status = bo(repo, "status")[1]
    assert status["review_missing"] == [h4]
    assert any(d.startswith(f"record review {h4}") for d in status["next"])
    assert bo(repo, "checkpoint", "--resume", "--rationale", "go on")[0] == 0
    code, out = bo(repo, "round-run", "--rationale", "go", env=env)
    assert code != 0 and "record review" in out["reason"] and h4 in out["reason"], out
    code, out = bo(repo, "register", pretrained, "--rationale", "allowed now")
    assert code != 0 and "review" in out["reason"], out
    assert review(repo, h4)[0] == 0
    assert bo(repo, "status")[1]["review_missing"] == []
    code, out = bo(repo, "checkpoint", "--revise", str(f), "--rationale", "again")
    assert code != 0 and "paused" in out["reason"], out
