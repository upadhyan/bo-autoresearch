from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types

import pytest

import boar as boar_pkg
from boar import cli, control, store
from boar import evalrun as real_evalrun

SPEC = "# Spec\n\n## Goal\nFaster.\n\n## Metric\nwall time in s\nDirection: min\n"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in {
        "HOME": str(tmp_path),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "XDG_STATE_HOME": str(tmp_path / "state"),  # the session registry
    }.items():
        monkeypatch.setenv(k, v)
    spec = tmp_path / "spec.md"
    spec.write_text(SPEC)
    return tmp_path


@pytest.fixture
def repo(env, monkeypatch):
    root = env / "target"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("print('hi')\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def boar(capsys):
    def call(*args):
        code = cli.main([str(a) for a in args])
        out, err = capsys.readouterr()
        return code, out, err

    return call


def fake_module(monkeypatch, name, **funcs):
    mod = types.ModuleType(f"boar.{name}")
    for k, v in funcs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, f"boar.{name}", mod)
    monkeypatch.setattr(boar_pkg, name, mod, raising=False)
    return mod


@pytest.fixture
def evalfake(monkeypatch):
    """A stand-in for the engine's evalrun: `ok` and `hash` steer the next check; `current` is eval/'s hash now."""
    ctl = {"ok": True, "hash": "h1", "current": "h1"}

    def eval_check(run):
        res = {
            "ok": ctl["ok"],
            "hash": ctl["hash"],
            "problems": [] if ctl["ok"] else ["guards failed at baseline"],
            "trial": {"state": "complete", "metric": 10.0, "repeats": [9.0, 10.0, 11.0], "metrics": {}, "duration_s": 3.0, "error": None},
            "limit_s": 1200,
            "at": store.now_iso(),
        }
        store.write_json(run.eval_check_path, res)
        return res

    fake_module(
        monkeypatch, "evalrun", eval_check=eval_check, eval_hash=lambda run: ctl["current"],
        eval_manifest=lambda run: {"run": ctl["current"]}, eval_diff=real_evalrun.eval_diff,
        eval_unchanged=lambda run, manifest: real_evalrun.eval_diff(manifest, {"run": ctl["current"]}),
        diff_text=real_evalrun.diff_text,
    )
    return ctl


def lever(name, **kw):
    return {"name": name, "type": "int", "low": 1, "high": 8, "default": 1, **kw}


def prop(*names, supersedes=None):
    return {
        "statement": f"levers {names} help",
        "mechanism": "less work per item",
        "source": "research",
        "citations": [],
        "levers": [lever(n) for n in names],
        "supersedes": supersedes,
    }


def write_props(tmp_path, *props, name="props.json"):
    path = tmp_path / name
    path.write_text(json.dumps(list(props)))
    return path


def the_run():
    return store.active_run()


def init(boar, env, *extra):
    code, out, err = boar("init", "--spec", env / "spec.md", *extra)
    assert code == 0, err
    return the_run()


def to_round1(boar, env, *extra, names=(("a",), ("b",))):
    """Init, propose, pass and accept everything: the run sits at R1 of round 1."""
    run = init(boar, env, *extra)
    assert boar("propose", write_props(env, *(prop(*n) for n in names)))[0] == 0
    assert boar("eval", "check", "--foreground")[0] == 0
    for i in range(len(names)):
        assert boar("review", "record", f"H{i + 1}", "accept", "--reason", "in scope")[0] == 0
    code, out, err = boar("review", "record", "eval", "accept", "--reason", "representative")
    assert code == 0, err
    return run


def finish_round(run, r, n_trials=3, config=None):
    """Pretend the engine ran round r: append trial records and mark it done."""
    start = len(run.trials()) + 1
    ids = list(range(start, start + n_trials))
    for n in ids:
        run.append_trial(
            {"trial": n, "round": r, "commit": "abc", "config": config or {}, "state": "complete", "metric": 1.0,
             "repeats": [1.0], "metrics": {}, "duration_s": 1.0, "error": None, "queued": None, "diagnostics": {"obs": 1.0}}
        )

    def done(s):
        s["rounds"][str(r)].update(run_status="done", new_trials=ids)

    run.update_state(done)
    return ids


def edit_hyps(run, fn):
    hyps = run.hypotheses()
    fn(hyps)
    run.save_hypotheses(hyps)


# --- init -----------------------------------------------------------------------------


def test_init_creates_run(boar, env, repo):
    code, out, err = boar("init", "--spec", env / "spec.md", "--session", "sess-1", "--rounds", "3", "--trials-per-round=4")
    assert code == 0, err
    run = the_run()
    state, config = run.state(), run.config()
    assert state["phase"] == "setup" and state["round"] == 0 and state["rounds"] == {}
    assert state["session_id"] == "sess-1"
    assert state["eval"] == {"status": "none", "checked_hash": None, "accepted_hash": None, "rejections": 0}
    assert state["base_branch"] == "main" and state["base_commit"] == git(repo, "rev-parse", "HEAD")
    assert config["rounds"] == 3 and config["trials_per_round"] == 4 and config["direction"] == "min"
    assert config["repeats"] == 3
    assert run.hypotheses() == []
    assert run.spec_path.read_text() == SPEC
    assert git(repo, "branch", "--show-current") == f"boar/{run.id}" == state["branch"]
    assert "/.boar/" in (repo / ".git" / "info" / "exclude").read_text().splitlines()
    assert git(repo, "status", "--porcelain") == ""
    assert run.id in out and f"boar/{run.id}" in out and str(run.dir) in out
    assert "next: [S2]" in out


def test_init_treats_unsubstituted_session_as_absent(boar, env, repo):
    run = init(boar, env, "--session", "${CLAUDE_SESSION_ID}")
    assert run.state()["session_id"] is None


def test_init_refuses_outside_git(boar, env, monkeypatch):
    plain = env / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    code, out, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and err.startswith("refused: not inside a git repository")


def test_init_refuses_unborn_head(boar, env, monkeypatch):
    fresh = env / "fresh"
    fresh.mkdir()
    git(fresh, "init", "-q", "-b", "main")
    monkeypatch.chdir(fresh)
    code, _, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and "no commits" in err


def test_init_refuses_detached_head(boar, env, repo):
    git(repo, "checkout", "-q", "--detach")
    code, _, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and "detached" in err


def test_init_refuses_tracked_changes(boar, env, repo):
    (repo / "app.py").write_text("print('changed')\n")
    code, _, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and "uncommitted changes" in err
    git(repo, "checkout", "app.py")
    assert boar("init", "--spec", env / "spec.md")[0] == 0


def test_init_refuses_untracked_files_the_first_round_would_commit(boar, env, repo):
    (repo / ".env").write_text("API_KEY=secret\n")
    (repo / "notes").mkdir()
    (repo / "notes" / "todo.md").write_text("x\n")
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"\0")
    code, _, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and ".env, notes/todo.md" in err and "pyc" not in err
    assert "commit them, add them to .gitignore, or move them" in err
    assert not (repo / ".boar").exists()
    (repo / ".git" / "info" / "exclude").write_text(".env\nnotes/\n")
    (repo / ".boar").mkdir()
    (repo / ".boar" / "spec-draft.md").write_text(SPEC)  # where the skill drafts the spec, before .boar/ is excluded
    assert boar("init", "--spec", repo / ".boar" / "spec-draft.md")[0] == 0


def test_init_refuses_without_a_git_identity(boar, env, repo, monkeypatch):
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(k)
    git(repo, "config", "user.useConfigOnly", "true")
    code, _, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and "git can't commit here" in err and "git config user.email" in err
    assert not (repo / ".boar").exists()
    git(repo, "config", "user.name", "u")
    git(repo, "config", "user.email", "u@example.com")
    assert boar("init", "--spec", env / "spec.md")[0] == 0


def test_init_warns_when_starting_from_an_earlier_runs_branch(boar, env, repo):
    first = init(boar, env)
    code, out, _ = boar("init", "--spec", env / "spec.md")
    assert code == 1
    boar("abort", "--reason", "done with it")
    code, out, err = boar("init", "--spec", env / "spec.md")
    assert code == 0, err
    assert f"warning: boar/{first.id} is an earlier BOAR run's branch" in out
    boar("abort", "--reason", "x")
    git(repo, "checkout", "-q", "main")
    assert "warning" not in boar("init", "--spec", env / "spec.md")[1]


def test_init_refuses_spec_without_direction(boar, env, repo):
    (env / "nodir.md").write_text("# Spec\nmetric: time\n")
    code, _, err = boar("init", "--spec", env / "nodir.md")
    assert code == 1 and "Direction" in err
    assert not (repo / ".boar").exists() or not (repo / ".boar" / "active").exists()


def test_init_refuses_bad_override(boar, env, repo):
    code, _, err = boar("init", "--spec", env / "spec.md", "--rounds", "0")
    assert code == 1 and "at least 1" in err
    code, _, err = boar("init", "--spec", env / "spec.md", "--colour", "3")
    assert code == 1 and "unknown config key" in err


@pytest.mark.parametrize(
    "key, value, needle",
    [
        ("--trials-per-round", "1", "trials_per_round must be at least 2"),
        ("--seed", "-5", "seed must be between 0 and 4294967295"),
        ("--seed", str(2**32), "seed must be between 0 and 4294967295"),
    ],
)
def test_init_refuses_overrides_the_engine_cannot_use(boar, env, repo, key, value, needle):
    code, _, err = boar("init", "--spec", env / "spec.md", key, value)
    assert code == 1 and needle in err
    assert not (repo / ".boar" / "active").exists()
    assert boar("init", "--spec", env / "spec.md", "--seed", str(2**32 - 1), "--trials-per-round", "2")[0] == 0


def test_init_refuses_while_a_run_is_active_then_allows_after_abort(boar, env, repo):
    first = init(boar, env)
    code, _, err = boar("init", "--spec", env / "spec.md")
    assert code == 1 and f"run {first.id} is still active" in err
    assert boar("abort", "--reason", "restart")[0] == 0
    git(repo, "checkout", "-q", "main")
    second = init(boar, env)
    assert second.id != first.id and second.id.startswith(first.id[:8])


def test_unrecognized_arguments_are_rejected(boar, env, repo):
    init(boar, env)
    with pytest.raises(SystemExit):
        cli.main(["status", "--bogus"])


# --- propose --------------------------------------------------------------------------


def test_propose_assigns_ids_and_normalizes(boar, env, repo):
    run = init(boar, env)
    code, out, err = boar("propose", write_props(env, prop("a"), prop("b", "c")))
    assert code == 0, err
    assert "proposed H1, H2 (setup)" in out
    hyps = run.hypotheses()
    assert [h["id"] for h in hyps] == ["H1", "H2"]
    assert all(h["status"] == "pending" and h["round"] == 0 and h["verdict"] is None for h in hyps)
    assert hyps[1]["levers"][1] == {"name": "c", "type": "int", "low": 1, "high": 8, "log": False, "default": 1}
    assert run.state()["next_hypothesis"] == 3


def test_propose_invalid_file_adds_nothing_and_lists_every_error(boar, env, repo):
    run = init(boar, env)
    bad = prop("a")
    bad["source"] = "vibes"
    worse = prop("b")
    worse["levers"][0]["default"] = 42
    code, _, err = boar("propose", write_props(env, prop("ok"), bad, worse))
    assert code == 1 and "2 problem(s); nothing was added" in err
    assert "proposal[1]: 'source'" in err and "proposal[2].levers[0] 'b'" in err
    assert run.hypotheses() == [] and run.state()["next_hypothesis"] == 1


def test_propose_refuses_unreadable_or_non_json(boar, env, repo):
    init(boar, env)
    assert "can't read" in boar("propose", env / "missing.json")[2]
    (env / "bad.json").write_text("{not json")
    assert "not valid JSON" in boar("propose", env / "bad.json")[2]


def test_propose_refuses_lever_name_already_in_run(boar, env, repo):
    init(boar, env)
    assert boar("propose", write_props(env, prop("a")))[0] == 0
    code, _, err = boar("propose", write_props(env, prop("a"), name="again.json"))
    assert code == 1 and "already used in this run" in err


def test_propose_is_gated_to_setup_and_r4(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    p = write_props(env, prop("c"), name="r1.json")
    code, _, err = boar("propose", p)
    assert code == 1 and "propose is allowed in setup or R4; now in R1" in err
    ids = finish_round(run, 1)
    code, _, err = boar("propose", p)
    assert code == 1 and "now in R3 (decide on H1, H2 first)" in err
    boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "helps")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "unclear", "--trials", ids[0])
    code, out, err = boar("propose", p)
    assert code == 0, err
    assert run.hypotheses()[-1]["round"] == 1 and "(round 1)" in out
    run.update_state(lambda s: s.update(phase="finalize"))
    code, _, err = boar("propose", write_props(env, prop("d"), name="f.json"))
    assert code == 1 and "now in finalize" in err


def test_propose_none(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    code, _, err = boar("propose", "--none", "--reason", "nothing")
    assert code == 1 and "now in R1" in err
    finish_round(run, 1)
    boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    assert boar("propose", "--none", "--reason", "  ")[0] == 1
    assert boar("propose", "--none")[0] == 1
    code, _, err = boar("propose", "--none", "--reason", "results are clear")
    assert code == 0, err
    assert run.state()["rounds"]["1"]["propose_none"] == "results are clear"


def test_propose_none_refused_in_setup(boar, env, repo):
    init(boar, env)
    code, _, err = boar("propose", "--none", "--reason", "x")
    assert code == 1 and "only for R4" in err


def test_supersedes_in_r4_must_name_an_active_hypothesis(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    finish_round(run, 1)
    boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    code, _, err = boar("propose", write_props(env, prop("c", supersedes="H7")))
    assert code == 1 and "must name an active hypothesis" in err
    assert boar("propose", write_props(env, prop("c", supersedes="H2")))[0] == 0
    assert run.hypotheses()[-1]["supersedes"] == "H2"


def test_setup_refuses_supersedes(boar, env, repo):
    init(boar, env)
    code, _, err = boar("propose", write_props(env, prop("a", supersedes="H1")))
    assert code == 1 and "must be null in setup" in err


# --- eval check and review -----------------------------------------------------------------


def test_eval_check_moves_eval_status(boar, env, repo, evalfake):
    run = init(boar, env)
    evalfake["ok"] = False
    code, out, _ = boar("eval", "check", "--foreground")
    assert code == 1 and "eval check: FAIL" in out and "guards failed" in out
    assert run.state()["eval"]["status"] == "none"
    evalfake["ok"] = True
    code, out, _ = boar("eval", "check", "--foreground")
    assert code == 0 and "eval check: PASS" in out and "metric 10.0" in out
    assert run.state()["eval"]["status"] == "pending" and run.state()["eval"]["checked_hash"] == "h1"
    evalfake["ok"] = False
    boar("eval", "check", "--foreground")
    assert run.state()["eval"]["status"] == "none" and run.state()["eval"]["checked_hash"] is None


def test_eval_accept_requires_the_checked_hash(boar, env, repo, evalfake):
    run = init(boar, env)
    boar("eval", "check", "--foreground")
    evalfake["current"] = "h2"
    code, _, err = boar("review", "record", "eval", "accept", "--reason", "ok")
    assert code == 1 and "changed since its last passing" in err
    assert run.state()["eval"]["status"] == "pending"
    evalfake["current"] = "h1"
    assert boar("review", "record", "eval", "accept", "--reason", "ok")[0] == 0
    ev = run.state()["eval"]
    assert ev["status"] == "accepted" and ev["accepted_hash"] == "h1"
    code, _, err = boar("review", "record", "eval", "accept", "--reason", "again")
    assert code == 1 and "not pending review (status accepted)" in err


def test_eval_check_after_acceptance_warns_when_eval_changed(boar, env, repo, evalfake):
    run = init(boar, env)
    boar("eval", "check", "--foreground")
    boar("review", "record", "eval", "accept", "--reason", "ok")
    evalfake["hash"] = "h3"
    code, out, _ = boar("eval", "check", "--foreground")
    assert code == 0 and "WARNING: eval/ has changed" in out
    assert run.state()["eval"]["status"] == "accepted" and run.state()["eval"]["accepted_hash"] == "h1"


def test_eval_reject_counts_and_stays_rejected_until_a_passing_check(boar, env, repo, evalfake):
    run = init(boar, env)
    code, _, err = boar("review", "record", "eval", "reject", "--reason", "x")
    assert code == 1 and "not pending review (status none)" in err
    boar("eval", "check", "--foreground")
    assert boar("review", "record", "eval", "reject", "--reason", "dev overlaps holdout")[0] == 0
    assert run.state()["eval"]["status"] == "rejected" and run.state()["eval"]["rejections"] == 1
    evalfake["ok"] = False
    boar("eval", "check", "--foreground")
    assert run.state()["eval"]["status"] == "rejected"
    evalfake["ok"] = True
    boar("eval", "check", "--foreground")
    assert run.state()["eval"]["status"] == "pending"


def test_review_record_refuses_items_that_are_not_pending(boar, env, repo, evalfake):
    run = init(boar, env)
    boar("propose", write_props(env, prop("a"), prop("b")))
    assert boar("review", "record", "H1", "accept", "--reason", "fine")[0] == 0
    code, _, err = boar("review", "record", "H1", "reject", "--reason", "changed my mind")
    assert code == 1 and "H1 is not pending review (status pending, verdict accept)" in err
    assert boar("review", "record", "H2", "reject", "--reason", "a cheat")[0] == 0
    code, _, err = boar("review", "record", "H2", "accept", "--reason", "x")
    assert code == 1 and "status rejected" in err
    assert "no hypothesis 'H9'" in boar("review", "record", "H9", "accept", "--reason", "x")[2]
    assert "no removal" in boar("review", "record", "rm-H1-r1", "accept", "--reason", "x")[2]
    assert "must not be empty" in boar("review", "record", "eval", "accept", "--reason", " ")[2]
    hyps = run.hypotheses()
    assert hyps[0]["status"] == "pending" and hyps[0]["verdict"]["decision"] == "accept"
    assert hyps[1]["status"] == "rejected"
    reviews = run.reviews()
    assert [(r["item"], r["kind"], r["decision"], r["round"]) for r in reviews] == [
        ("H1", "proposal", "accept", 0), ("H2", "proposal", "reject", 0),
    ]


def test_review_pending_and_show(boar, env, repo, evalfake):
    run = init(boar, env)
    boar("propose", write_props(env, prop("a")))
    (run.eval_dir).mkdir()
    (run.eval_dir / "run").write_text("#!/bin/sh\n")
    os.chmod(run.eval_dir / "run", 0o755)
    (run.eval_dir / "__pycache__").mkdir()
    (run.eval_dir / "__pycache__" / "x.pyc").write_text("")
    boar("eval", "check", "--foreground")
    code, out, _ = boar("review", "pending")
    assert code == 0
    assert f"spec: {run.spec_path}" in out and "pending: H1, eval" in out
    assert '"statement"' in out and "run (10 bytes, executable)" in out and "pyc" not in out
    assert "last `boar eval check`: PASS" in out and "eval/ now: unchanged" in out
    assert "decisions" not in out
    boar("review", "record", "H1", "reject", "--reason", "out of scope")
    code, out, _ = boar("review", "show", "H1")
    assert code == 0 and "status: rejected; verdict reject: out of scope" in out


# --- setup bookkeeping ----------------------------------------------------------------------


def test_setup_transition_activates_accepted_proposals(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    state, hyps = run.state(), run.hypotheses()
    assert state["phase"] == "round" and state["round"] == 1
    assert state["rounds"]["1"] == {
        "commit": None, "run_status": "not_started", "warm": None, "incumbent_trial": None,
        "new_trials": [], "propose_none": None, "closed": False,
    }
    assert all(h["status"] == "active" and h["activated_round"] == 1 for h in hyps)


def test_setup_waits_for_the_eval_and_every_verdict(boar, env, repo, evalfake):
    run = init(boar, env)
    boar("propose", write_props(env, prop("a"), prop("b")))
    boar("eval", "check", "--foreground")
    boar("review", "record", "eval", "accept", "--reason", "ok")
    boar("review", "record", "H1", "accept", "--reason", "ok")
    assert run.state()["phase"] == "setup"
    code, out, _ = boar("review", "record", "H2", "reject", "--reason", "cheat")
    assert "setup complete: H1 active; round 1 begins" in out
    assert run.state()["phase"] == "round"
    assert [h["status"] for h in run.hypotheses()] == ["active", "rejected"]


def test_failed_proposal_batches_are_counted_once_per_batch(boar, env, repo):
    run = init(boar, env)
    for attempt in range(1, 4):
        boar("propose", write_props(env, prop(f"a{attempt}"), prop(f"b{attempt}"), name=f"p{attempt}.json"))
        hid = len(run.hypotheses())
        boar("review", "record", f"H{hid - 1}", "reject", "--reason", "cheat")
        assert run.state()["setup_failed_batches"] == attempt - 1
        code, out, _ = boar("review", "record", f"H{hid}", "reject", "--reason", "out of scope")
        assert f"failed attempt {attempt}/3" in out
        assert run.state()["setup_failed_batches"] == attempt
        if attempt < 3:
            assert boar("next")[1].startswith(f"[S3] No hypothesis accepted (failed attempt {attempt}/3)")
    assert boar("next")[1].startswith("[ASK_USER] No hypothesis was accepted in 3 attempts")


# --- decide -----------------------------------------------------------------------------------


def test_decide_preconditions(boar, env, repo, evalfake):
    init(boar, env)
    code, _, err = boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    assert code == 1 and "decide is for R3 of a round; now in setup" in err
    boar("abort", "--reason", "x")
    git(repo, "checkout", "-q", "main")
    run = to_round1(boar, env)
    code, _, err = boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    assert code == 1 and "has not finished running" in err
    ids = finish_round(run, 1)
    code, _, err = boar("decide", "H1", "fix", "--reason", "crashed")
    assert code == 1 and "fix needs --trials" in err
    code, _, err = boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "no effect", "--trials", "99")
    assert code == 1 and "trial(s) 99 are not in this run" in err
    code, _, err = boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x", "--trials", "1,x")
    assert code == 1 and "integers" in err
    boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    boar("propose", write_props(env, prop("c")))
    code, _, err = boar("decide", "H3", "keep", "--observable-key", "obs", "--reason", "x", "--trials", ids[0])
    assert code == 1 and "H3 is not active (status pending)" in err


def test_decide_remove_creates_a_pending_removal_and_keep_drops_it(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    ids = finish_round(run, 1, n_trials=7)
    code, out, err = boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "3,4", "7", "3")
    assert code == 0, err
    assert "pending removal rm-H1-r1" in out
    (rm,) = run.state()["removals"]
    assert rm == {"id": "rm-H1-r1", "hypothesis": "H1", "round": 1, "reason": "flat", "trials": [3, 4, 7], "verdict": None}
    assert run.hypotheses()[0]["decisions"]["1"]["trials"] == [3, 4, 7]
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flatter", "--trials", ids[0])
    assert [rm["reason"] for rm in run.state()["removals"]] == ["flatter"]
    code, out, _ = boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "on reflection, unclear")
    assert "dropped the pending removal" in out
    assert run.state()["removals"] == []
    assert run.hypotheses()[0]["decisions"]["1"]["decision"] == "keep"


def test_after_a_rejected_removal_the_hypothesis_can_be_marked_fix(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    finish_round(run, 1)
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "1", "2")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    boar("propose", "--none", "--reason", "nothing new")
    boar("review", "record", "rm-H1-r1", "reject", "--reason", "implementation bug: this calls for fix")
    out = boar("next")[1]
    assert out.startswith("[R6]") and "rm-H1-r1 was rejected (implementation bug" in out and "boar decide H1 fix" in out
    assert "r1 remove (rejected)" in boar("status")[1], "a rejected removal left H1 active"
    code, _, err = boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "still flat", "--trials", "1")
    assert code == 1 and "the reviewer rejected rm-H1-r1" in err
    code, out, err = boar("decide", "H1", "fix", "--reason", "lever never applied", "--trials", "1", "2")
    assert code == 0, err
    assert "dropped" not in out
    assert run.hypotheses()[0]["decisions"]["1"]["decision"] == "fix"
    (rm,) = run.state()["removals"]
    assert rm["verdict"]["decision"] == "reject"
    assert "rm-H1-r1" not in boar("next")[1]
    code, _, err = boar("review", "record", "rm-H1-r1", "accept", "--reason", "x")
    assert code == 1 and "already has a verdict" in err
    assert boar("round", "close")[0] == 0
    assert boar("next")[1].startswith("[R1] Implement levers for: none; repair: H1")


def test_decide_refuses_fix_in_the_last_round(boar, env, repo, evalfake):
    """No R1 follows to repair it, and a fix would drop the hypothesis's trials from the final incumbent."""
    run = to_round1(boar, env, "--rounds", "1")
    finish_round(run, 1)
    code, _, err = boar("decide", "H1", "fix", "--reason", "crashed", "--trials", "1")
    assert code == 1 and "round 1 is the last round (1/1): no R1 follows to repair H1" in err
    assert "1" not in run.hypotheses()[0]["decisions"]
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "1", "2")
    boar("review", "record", "rm-H1-r1", "reject", "--reason", "a bug")
    code, _, err = boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "still flat", "--trials", "1")
    assert code == 1 and err.rstrip().endswith("Decide keep")
    assert boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "helps", "--trials", "1")[0] == 0


def _measured_round(run, r, points):
    """Round r ran these (config, metric) points; trial 1 of the run is the baseline at 10."""
    start = len(run.trials()) + 1
    if start == 1:
        points = [({}, 10.0), *points]
    ids = []
    for i, (config, metric) in enumerate(points):
        n = start + i
        ids.append(n)
        run.append_trial(
            {"trial": n, "round": r, "commit": "abc", "config": config, "state": "complete", "metric": metric,
             "repeats": [metric, metric], "metrics": {}, "duration_s": 1.0, "error": None, "queued": None, "diagnostics": {"obs": 1.0}}
        )
    run.update_state(lambda s: s["rounds"][str(r)].update(run_status="done", new_trials=ids))


def test_decide_refuses_a_last_round_removal_that_would_change_the_final_incumbent(boar, env, repo, evalfake):
    """A numeric lever is almost never sampled at its default: removing it would leave the baseline as the result."""
    run = to_round1(boar, env, "--rounds", "1")
    _measured_round(run, 1, [({"a": 3, "b": 5}, 2.0), ({"a": 7}, 9.0)])
    code, _, err = boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "no effect", "--trials", "2", "3")
    assert code == 1 and "round 1 is the last round (1/1): an accepted removal of H1 pins its levers" in err, err
    assert 'would change from {"a": 3, "b": 5} metric 2 (trials 2) to the baseline metric 10 (trials 1)' in err
    assert err.rstrip().endswith("argue the lack of effect in the report's What didn't")
    assert run.state()["removals"] == [] and "1" not in run.hypotheses()[0]["decisions"]
    assert "a removal that would change the final incumbent is refused" in boar("next")[1]


def test_decide_allows_a_last_round_removal_that_keeps_the_final_incumbent(boar, env, repo, evalfake):
    run = to_round1(boar, env, "--rounds", "1")
    _measured_round(run, 1, [({"b": 5}, 2.0), ({"a": 7}, 9.0)])
    code, out, err = boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "no effect", "--trials", "3")
    assert code == 0 and "pending removal rm-H1-r1" in out, err
    # With H1's removal pending, removing H2 too would leave the baseline.
    code, _, err = boar("decide", "H2", "remove", "--observable-key", "obs", "--reason", "no effect", "--trials", "2")
    assert code == 1 and 'from {"b": 5} metric 2 (trials 2) to the baseline' in err, err


def test_r6_and_round_close_warn_when_a_removal_moves_the_incumbent(boar, env, repo, evalfake):
    run = to_round1(boar, env, "--rounds", "2")
    _measured_round(run, 1, [({"a": 3, "b": 5}, 2.0), ({"b": 5}, 4.0)])
    assert boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "2", "3")[0] == 0
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "helps")
    boar("propose", "--none", "--reason", "nothing new")
    boar("review", "record", "rm-H1-r1", "accept", "--reason", "flat across the range")
    out = boar("next")[1]
    assert out.startswith("[R6]") and ('Warning: closing changes the incumbent from {"a": 3, "b": 5} metric 2 '
                                       '(trials 2) to {"b": 5} metric 4 (trials 3)') in out, out
    code, out, err = boar("round", "close")
    assert code == 0, err
    assert 'warning: closing changes the incumbent from {"a": 3, "b": 5} metric 2' in out, out


def test_decide_refuses_to_change_an_accepted_removal(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    finish_round(run, 1)
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "1", "2")
    boar("review", "record", "rm-H1-r1", "accept", "--reason", "flat across the range")
    for decision in ("keep", "fix", "remove"):
        code, _, err = boar("decide", "H1", decision, "--reason", "x", "--trials", "1")
        assert code == 1 and "rm-H1-r1 was accepted by the reviewer" in err
    assert run.reviews()[-1]["kind"] == "removal" and run.reviews()[-1]["round"] == 1


# --- round run --------------------------------------------------------------------------------


@pytest.fixture
def roundsfake(monkeypatch):
    calls = []

    def start_round(run, foreground):
        calls.append(foreground)
        return "round started"

    fake_module(monkeypatch, "rounds", start_round=start_round)
    return calls


def test_round_run_refused_in_setup(boar, env, repo, evalfake, roundsfake):
    init(boar, env)
    code, _, err = boar("round", "run")
    assert code == 1 and "setup isn't done" in err
    assert roundsfake == []


def test_round_run_preconditions(boar, env, repo, evalfake, roundsfake):
    run = to_round1(boar, env)
    edit_hyps(run, lambda hs: hs.append({**hs[0], "id": "H9", "status": "pending", "verdict": None, "levers": []}))
    evalfake["current"] = "changed"
    git(repo, "checkout", "-q", "main")
    code, _, err = boar("round", "run")
    assert code == 1 and "round 1 can't run" in err
    assert "verdicts are pending for H9" in err
    assert "eval/ has changed since the reviewer accepted it" in err
    assert f"lives on boar/{run.id} but main is checked out" in err
    assert roundsfake == []
    # With H9 settled, the other problems fixed, the round runs.
    edit_hyps(run, lambda hs: hs[-1].update(status="active"))
    evalfake["current"] = "h1"
    git(repo, "checkout", "-q", f"boar/{run.id}")
    code, out, err = boar("round", "run", "--foreground")
    assert code == 0, err
    assert out.strip() == "round started" and roundsfake == [True]
    finish_round(run, 1)
    code, _, err = boar("round", "run")
    assert code == 1 and "round 1 has already run" in err


def test_round_run_resumes_an_interrupted_round(boar, env, repo, evalfake, roundsfake):
    run = to_round1(boar, env)
    run.update_state(lambda s: s["rounds"]["1"].update(run_status="running", commit="abc"))
    assert boar("round", "run")[0] == 0
    assert roundsfake == [False]


def test_commands_refuse_while_a_worker_is_alive(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    store.write_json(
        run.running_path,
        {"pid": os.getpid(), "identity": store.process_identity(os.getpid()), "command": "round run", "round": 1},
    )
    for args in (("round", "run"), ("propose", "--none", "--reason", "x"), ("round", "close"), ("eval", "check", "--foreground")):
        code, _, err = boar(*args)
        assert code == 1 and "still running in the background" in err


# --- round close --------------------------------------------------------------------------------


def test_round_close_lists_every_unmet_condition(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    code, _, err = boar("round", "close")
    assert code == 1 and "R2: round 1 has not run" in err
    finish_round(run, 1)
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "1", "2")
    code, _, err = boar("round", "close")
    assert code == 1
    assert "R3: no round 1 decision on H2" in err and "R4: no proposal this round" in err
    assert "R5: no verdict yet on rm-H1-r1" in err
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    boar("propose", write_props(env, prop("c")))
    code, _, err = boar("round", "close")
    assert code == 1 and "R3" not in err and "R4" not in err and "R5: no verdict yet on H3, rm-H1-r1" in err
    assert run.state()["rounds"]["1"]["closed"] is False


def test_round_close_applies_removals_supersedes_and_activations(boar, env, repo, evalfake):
    run = to_round1(boar, env, names=(("a",), ("b",), ("x",)))
    finish_round(run, 1)
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat across 1..8", "--trials", "1", "2")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "helps")
    boar("decide", "H3", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "3")
    boar("propose", write_props(env, prop("b_wide", supersedes="H2"), prop("cheat")))
    boar("review", "record", "rm-H1-r1", "accept", "--reason", "wide spread, no effect")
    boar("review", "record", "rm-H3-r1", "reject", "--reason", "one point only")
    boar("review", "record", "H4", "accept", "--reason", "fine")
    boar("review", "record", "H5", "reject", "--reason", "caches eval output")
    assert boar("next")[1].startswith("[R6] Run `boar round close`.")
    code, out, err = boar("round", "close")
    assert code == 0, err
    hyps = {h["id"]: h for h in run.hypotheses()}
    assert hyps["H1"]["status"] == "removed" and hyps["H1"]["removed_round"] == 1
    assert hyps["H1"]["removed_reason"] == "removal rm-H1-r1 accepted: flat across 1..8"
    assert hyps["H2"]["status"] == "removed" and hyps["H2"]["removed_reason"] == "superseded by H4"
    assert hyps["H3"]["status"] == "active"
    assert hyps["H4"]["status"] == "active" and hyps["H4"]["activated_round"] == 2
    assert hyps["H5"]["status"] == "rejected"
    state = run.state()
    assert state["rounds"]["1"]["closed"] is True and state["round"] == 2 and state["phase"] == "round"
    assert state["rounds"]["2"]["run_status"] == "not_started"
    assert "active now: H3, H4" in out and "round 2 begins" in out
    code, out, _ = boar("next")
    assert out.startswith("[R1] Implement levers for: H4; repair: none; then `boar round run`.")
    assert "new: H4 (b_wide)" in out


def test_round_close_activates_every_accepted_proposal(boar, env, repo, evalfake):
    run = to_round1(boar, env)
    finish_round(run, 1)
    boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    boar("propose", write_props(env, prop("c")))
    boar("review", "record", "H3", "accept", "--reason", "ok")
    code, out, err = boar("round", "close")
    assert code == 0, err
    assert run.state()["round"] == 2
    assert [h["status"] for h in run.hypotheses()] == ["active", "active", "active"]


def test_last_round_close_moves_to_finalize(boar, env, repo, evalfake):
    run = to_round1(boar, env, "--rounds", "1")
    finish_round(run, 1)
    boar("decide", "H1", "keep", "--observable-key", "obs", "--reason", "x")
    boar("decide", "H2", "keep", "--observable-key", "obs", "--reason", "x")
    assert "This is the last round (1/1)" in boar("next")[1]
    boar("propose", "--none", "--reason", "last round")
    code, out, _ = boar("round", "close")
    assert code == 0 and "no rounds remain; the run moves to finalize" in out
    assert run.state()["phase"] == "finalize"


def test_close_with_nothing_active_moves_to_finalize_early(boar, env, repo, evalfake):
    run = to_round1(boar, env, names=(("a",),))
    finish_round(run, 1)
    boar("decide", "H1", "remove", "--observable-key", "obs", "--reason", "flat", "--trials", "1")
    boar("propose", "--none", "--reason", "nothing left")
    boar("review", "record", "rm-H1-r1", "accept", "--reason", "ok")
    code, out, _ = boar("round", "close")
    assert code == 0 and "no hypothesis is active" in out
    assert run.state()["phase"] == "finalize"


# --- finalize and abort -----------------------------------------------------------------------------


def test_finalize_preconditions(boar, env, repo, evalfake, monkeypatch):
    calls = []
    fake_module(monkeypatch, "finalize", start_finalize=lambda run, fg: calls.append(fg) or "finalize started")
    run = to_round1(boar, env)
    code, _, err = boar("finalize")
    assert code == 1 and "rounds remain and a hypothesis is active" in err
    run.update_state(lambda s: s.update(phase="finalize"))
    code, out, _ = boar("finalize", "--foreground")
    assert code == 0 and "finalize started" in out and calls == [True]
    run.update_state(lambda s: s["finalize"].update(status="done"))
    run.report_path.write_text("".join(f"{h}\n\n" for h in control.REPORT_SECTIONS))
    code, _, err = boar("finalize")
    assert code == 1 and "already finalized; write the narrative sections" in err


SKELETON = "".join(f"{h}\n{store.TODO_MARKER}\n" for h in control.REPORT_SECTIONS)


def test_a_lost_report_is_rewritten_only_by_finalize_report(boar, env, repo, evalfake, monkeypatch):
    calls = []

    def write_report(run):
        calls.append(run.id)
        run.report_path.write_text(SKELETON)

    fake_module(monkeypatch, "finalize", start_finalize=lambda run, fg: "x", write_report=write_report)
    run = to_round1(boar, env)
    run.update_state(lambda s: s.update(phase="finalize"))
    code, _, err = boar("finalize", "--report")
    assert code == 1 and "finalize is not_started; run `boar finalize`" in err and not calls
    run.update_state(lambda s: s["finalize"].update(status="done"))
    out = boar("next")[1]
    assert out.startswith("[REPORT] report.md is missing; run `boar finalize --report` to rewrite it from the stored "
                          "results (the holdout is not re-run)")
    assert boar("next", "--hook")[0] == 2
    code, _, err = boar("finalize")
    assert code == 1 and "report.md is missing; `boar finalize --report` rewrites it" in err and not calls
    code, out, err = boar("finalize", "--report")
    assert code == 0 and "rewrote" in out and "holdout was not re-run" in out, err
    assert "next: [REPORT] Write the narrative sections" in out and len(calls) == 1
    assert run.state()["phase"] == "finalize"


def test_a_report_with_a_renamed_heading_names_it_and_keeps_the_narrative(boar, env, repo, evalfake, monkeypatch):
    fake_module(monkeypatch, "finalize", start_finalize=lambda run, fg: "x",
                write_report=lambda run: run.report_path.write_text(SKELETON))
    run = to_round1(boar, env)
    run.update_state(lambda s: (s.update(phase="finalize"), s["finalize"].update(status="done")))
    narrative = SKELETON.replace(store.TODO_MARKER, "My narrative.").replace("## 1. Result\n", "## 1. Results (holdout)\n")
    narrative = narrative.replace("## 6. Experimental log", "## Experimental log")
    run.report_path.write_text(narrative)
    out = boar("next")[1]
    assert out.startswith("[REPORT] report.md has lost generated headings: ## 1. Result; ## 6. Experimental log; "
                          "put those heading lines back exactly as named, keeping the narrative;")
    assert "`boar finalize --report` restores them too, but discards the narrative" in out
    code, _, err = boar("finalize")
    assert code == 1 and "lost generated headings" in err
    assert run.report_path.read_text() == narrative, "nothing regenerates over the narrative unasked"
    run.report_path.write_text(narrative.replace("## 1. Results (holdout)", "## 1. Result").replace(
        "## Experimental log", "## 6. Experimental log"))
    assert boar("next")[1].startswith(f"[DONE] Run {run.id} is finished")


def test_abort_ends_the_run(boar, env, repo):
    run = init(boar, env)
    assert "must not be empty" in boar("abort", "--reason", "")[2]
    code, out, _ = boar("abort", "--reason", "wrong target")
    assert code == 0 and "aborted: wrong target" in out
    assert run.state()["phase"] == "aborted" and run.state()["abort_reason"] == "wrong target"
    assert boar("next")[1].startswith("[ABORTED]")
    assert "already aborted" in boar("abort", "--reason", "again")[2]
    code, _, err = boar("propose", write_props(env, prop("a")))
    assert code == 1 and "is aborted" in err


def test_abort_terminates_a_live_worker(boar, env, repo):
    run = init(boar, env)
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        store.write_json(
            run.running_path,
            {"pid": child.pid, "identity": store.process_identity(child.pid), "command": "round run", "round": 1},
        )
        code, out, _ = boar("abort", "--reason", "stop")
        assert code == 0 and "sent SIGTERM" in out
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()


def test_abort_kills_an_eval_a_dead_worker_left_running(boar, env, repo):
    run = init(boar, env)
    orphan = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        store.write_json(run.eval_group_path, {"pgid": orphan.pid, "identity": store.process_identity(orphan.pid)})
        code, out, _ = boar("abort", "--reason", "stop")
        assert code == 0 and f"killed the eval process group {orphan.pid}" in out
        assert orphan.wait(timeout=10) != 0
    finally:
        if orphan.poll() is None:
            orphan.kill()


def test_abort_signals_only_a_worker_that_does_not_lead_its_group(boar, env, repo):
    """An inline run shares its shell's process group; abort must not take the shell down with it."""
    run = init(boar, env)
    shell = subprocess.Popen(["sh", "-c", "sleep 30 & echo $!; sleep 30"], start_new_session=True,
                             stdout=subprocess.PIPE, text=True)
    try:
        inline = int(shell.stdout.readline())
        store.write_json(
            run.running_path,
            {"pid": inline, "identity": store.process_identity(inline), "command": "round run", "round": 1},
        )
        code, out, _ = boar("abort", "--reason", "stop")
        assert code == 0 and f"pid {inline}" in out
        for _ in range(100):
            if not store.pid_alive(inline):
                break
            subprocess.run(["sleep", "0.05"])
        assert not store.pid_alive(inline)
        assert shell.poll() is None
    finally:
        os.killpg(shell.pid, 9)
        shell.wait()


def test_foreground_eval_check_holds_the_worker_slot(boar, env, repo, evalfake, monkeypatch):
    run = init(boar, env)
    import boar.evalrun as fake

    seen = []
    inner = fake.eval_check

    def eval_check(r):
        seen.append(r.running())
        seen.append(boar("eval", "check", "--foreground"))
        return inner(r)

    monkeypatch.setattr(fake, "eval_check", eval_check)
    code, out, err = boar("eval", "check", "--foreground")
    assert code == 0, err
    rec, (code2, _, err2) = seen[:2]
    assert rec["pid"] == os.getpid() and rec["command"] == "eval check"
    assert code2 == 1 and "still running in the background" in err2
    assert run.running() is None and not run.running_path.exists()


# --- the eval freeze: a manifest and a copy, so a change can be named and undone -------------------


@pytest.fixture
def realeval(monkeypatch):
    """The real eval hashing over files in eval/, with a check that always passes without running anything."""

    def eval_check(run):
        res = {"ok": True, "hash": real_evalrun.eval_hash(run), "manifest": real_evalrun.eval_manifest(run), "problems": [],
               "trial": {"state": "complete", "metric": 1.0, "repeats": [1.0], "duration_s": 1.0}, "limit_s": 1200,
               "at": store.now_iso()}
        store.write_json(run.eval_check_path, res)
        return res

    fake_module(
        monkeypatch, "evalrun", eval_check=eval_check, eval_hash=real_evalrun.eval_hash,
        eval_manifest=real_evalrun.eval_manifest, eval_diff=real_evalrun.eval_diff, diff_text=real_evalrun.diff_text,
        eval_unchanged=real_evalrun.eval_unchanged,
    )


def accepted_eval_run(boar, env):
    run = init(boar, env)
    (run.eval_dir / "data").mkdir(parents=True)
    (run.eval_dir / "run").write_text("#!/bin/sh\necho '{}'\n")
    (run.eval_dir / "run").chmod(0o755)
    (run.eval_dir / "data" / "dev.json").write_text("[1, 2]\n")
    (run.eval_dir / "shared").symlink_to("data")
    (run.eval_dir / "__pycache__").mkdir()
    (run.eval_dir / "__pycache__" / "x.pyc").write_bytes(b"\0")
    boar("propose", write_props(env, prop("a")))
    assert boar("eval", "check", "--foreground")[0] == 0
    boar("review", "record", "H1", "accept", "--reason", "ok")
    assert boar("review", "record", "eval", "accept", "--reason", "ok")[0] == 0
    return run


def test_eval_accept_keeps_a_manifest_and_a_copy(boar, env, repo, realeval):
    run = accepted_eval_run(boar, env)
    ev = run.state()["eval"]
    assert ev["accepted_manifest"] == real_evalrun.eval_manifest(run)
    assert set(ev["accepted_manifest"]) == {"data/dev.json", "run", "shared"}
    snap = run.eval_accepted_dir
    assert (snap / "shared").is_symlink() and os.readlink(snap / "shared") == "data"
    assert os.access(snap / "run", os.X_OK) and not (snap / "__pycache__").exists()


def test_a_changed_eval_is_named_and_restored(boar, env, repo, realeval):
    run = accepted_eval_run(boar, env)
    (run.eval_dir / "run").write_text("#!/bin/sh\necho tweaked\n")
    (run.eval_dir / "data" / "dev.json").unlink()
    (run.eval_dir / ".DS_Store").write_text("x")
    code, _, err = boar("round", "run")
    assert code == 1
    assert "eval/ has changed since the reviewer accepted it (added: .DS_Store; changed: run; removed: data/dev.json)" in err
    assert "boar eval restore" in err
    out = boar("next")[1]
    assert out.startswith("[R1] eval/ has changed since the reviewer accepted it (added: .DS_Store;")
    assert "`boar eval restore`" in out
    code, out, err = boar("eval", "restore")
    assert code == 0, err
    assert "restored eval/ to the accepted version (undid: added: .DS_Store; changed: run; removed: data/dev.json)" in out
    assert real_evalrun.eval_hash(run) == run.state()["eval"]["accepted_hash"]
    assert os.access(run.eval_dir / "run", os.X_OK) and (run.eval_dir / "shared").is_symlink()
    assert not run.path("eval.replaced").exists()
    assert boar("next")[1].startswith("[R1] Implement levers for: H1")
    assert "nothing to restore" in boar("eval", "restore")[1]


def test_eval_restore_refuses_a_copy_that_does_not_match_and_leaves_eval_alone(boar, env, repo, realeval):
    run = accepted_eval_run(boar, env)
    (run.eval_dir / "run").write_text("mine\n")
    (run.eval_accepted_dir / "run").write_text("corrupted\n")
    code, _, err = boar("eval", "restore")
    assert code == 1 and "does not match the accepted hash" in err
    assert (run.eval_dir / "run").read_text() == "mine\n"
    assert not run.path("eval.replaced").exists()


def test_eval_restore_preconditions(boar, env, repo, realeval):
    run = init(boar, env)
    code, _, err = boar("eval", "restore")
    assert code == 1 and "the eval is not accepted (status none)" in err
    boar("abort", "--reason", "x")
    git(repo, "checkout", "-q", "main")
    run = accepted_eval_run(boar, env)
    store.write_json(
        run.running_path,
        {"pid": os.getpid(), "identity": store.process_identity(os.getpid()), "command": "round run", "round": 1},
    )
    code, _, err = boar("eval", "restore")
    assert code == 1 and "still running" in err
    run.running_path.unlink()
    run.update_state(lambda s: s["eval"].pop("accepted_manifest"))
    (run.eval_dir / "run").write_text("mine\n")
    code, _, err = boar("eval", "restore")
    assert code == 1 and "no copy of the accepted eval was kept" in err


def test_review_pending_holds_back_an_eval_changed_since_its_check(boar, env, repo, evalfake):
    init(boar, env)
    boar("propose", write_props(env, prop("a")))
    boar("eval", "check", "--foreground")
    evalfake["current"] = "edited"
    out = boar("review", "pending")[1]
    assert "pending: H1" in out and "== eval" not in out
    assert "eval: not ready for review (eval/ changed since its passing check)" in out


def test_eval_restore_clears_read_only_dirs_and_never_leaves_its_aside_copy(boar, env, repo, realeval):
    run = accepted_eval_run(boar, env)
    aside = run.path("eval.replaced")
    try:
        for mode in (0o555, 0o644):  # read-only; then readable but not searchable (`chmod -R a-x`)
            (run.eval_dir / "data").chmod(mode)
            (run.eval_dir / "run").write_text("tweaked\n")
            code, out, err = boar("eval", "restore")
            assert code == 0 and "restored eval/" in out, err
            assert not aside.exists()
            assert real_evalrun.eval_hash(run) == run.state()["eval"]["accepted_hash"]
        # A restore killed midway leaves its aside copy; the next one clears it first.
        (aside / "data").mkdir(parents=True)
        (aside / "data" / "x").write_text("old")
        (aside / "data").chmod(0o500)
        (run.eval_dir / "run").write_text("tweaked\n")
        code, out, err = boar("eval", "restore")
        assert code == 0 and "restored eval/" in out, err
        assert not aside.exists()
    finally:
        for d in (run.eval_dir / "data", aside / "data"):
            if d.exists():
                d.chmod(0o755)


def test_eval_restore_refuses_a_symlinked_eval_and_leaves_its_target_alone(boar, env, repo, realeval):
    run = accepted_eval_run(boar, env)
    elsewhere = env / "my-eval"
    run.eval_dir.rename(elsewhere)
    run.eval_dir.symlink_to(elsewhere)
    (elsewhere / "run").write_text("tweaked\n")
    code, _, err = boar("eval", "restore")
    assert code == 1 and f"eval/ is a symlink (to {elsewhere})" in err and f"`rm {run.eval_dir}`" in err
    assert (elsewhere / "run").read_text() == "tweaked\n" and run.eval_dir.is_symlink()
    assert not run.path("eval.replaced").exists()
    run.eval_dir.unlink()
    code, out, err = boar("eval", "restore")
    assert code == 0 and "removed: run" in out, err


def test_next_checks_a_frozen_eval_without_rereading_unchanged_files(boar, env, repo, realeval, monkeypatch):
    run = accepted_eval_run(boar, env)
    (run.eval_dir / "data" / "big.bin").write_bytes(b"\0" * 4096)
    assert "added: data/big.bin" in boar("next")[1]
    (run.eval_dir / "data" / "big.bin").unlink()
    assert boar("next")[1].startswith("[R1] Implement levers for: H1")
    read = []
    digest = real_evalrun._digest
    monkeypatch.setattr(real_evalrun, "_digest", lambda path: read.append(path) or digest(path))
    for _ in range(3):
        assert boar("next")[1].startswith("[R1] Implement levers for: H1")
    assert read == []
    (run.eval_dir / "run").write_text("tweaked\n")
    assert "changed: run" in boar("next")[1]
    assert read == [run.eval_dir / "run"]


def test_an_unreadable_eval_keeps_the_step_and_says_so(boar, env, repo, realeval):
    if os.geteuid() == 0:
        pytest.skip("root reads anything")
    run = accepted_eval_run(boar, env)
    data = run.eval_dir / "data" / "dev.json"
    data.chmod(0)
    try:
        out = boar("next")[1]
        assert out.startswith("[R1] eval/ could not be read (unreadable: data/dev.json: Permission denied)"), out
        assert boar("next", "--hook")[0] == 2
        code, _, err = boar("round", "run")
        assert code == 1 and "eval/ could not be read" in err
    finally:
        data.chmod(0o644)
    assert boar("next")[1].startswith("[R1] Implement levers for: H1")


def test_eval_check_names_repo_writes_outside_links_and_self_edits_and_sends_the_eval_back_to_s4(boar, env, repo):
    run = init(boar, env, "--repeats", "1")
    outside = env / "workloads"
    outside.mkdir()
    run.eval_dir.mkdir()
    (run.eval_dir / "run").write_text(
        "#!/bin/sh\necho row >> results.txt\necho x > \"$(dirname \"$0\")/cache\"\n"
        "echo '{\"metric\": 1.0, \"guards_ok\": true}'\n"
    )
    (run.eval_dir / "run").chmod(0o755)
    (run.eval_dir / "dev").symlink_to(outside)
    code, out, err = boar("eval", "check", "--foreground")
    assert code == 1 and "eval check: FAIL" in out, err
    assert "the eval wrote into the repo: results.txt" in out
    assert f"eval/ links outside itself: dev -> {outside}" in out
    assert "eval/ changed during the check: added: cache" in out
    assert not (repo / "results.txt").exists() and git(repo, "status", "--porcelain") == ""
    boar("propose", write_props(env, prop("a")))
    out = boar("next")[1]
    assert out.startswith("[S4] Build the eval") and "last check failed:" in out
    for needle in ("the eval wrote into the repo: results.txt", "eval/ links outside itself", "eval/ changed during the check"):
        assert needle in out
    assert run.state()["eval"]["status"] == "none"


def test_init_says_where_a_spec_draft_belongs_when_it_is_the_untracked_file(boar, env, repo):
    (repo / "spec.md").write_text(SPEC)
    code, _, err = boar("init", "--spec", "spec.md")
    assert code == 1 and "untracked files would be committed" in err
    assert "a spec draft belongs under .boar/, e.g. `boar init --spec .boar/spec-draft.md`" in err
    (repo / ".boar").mkdir()
    (repo / "spec.md").rename(repo / ".boar" / "spec-draft.md")
    (repo / "notes.txt").write_text("x")
    code, _, err = boar("init", "--spec", ".boar/spec-draft.md")
    assert code == 1 and "notes.txt" in err and "spec draft belongs" not in err
    (repo / "notes.txt").unlink()
    assert boar("init", "--spec", ".boar/spec-draft.md")[0] == 0


def test_the_no_run_hint_names_a_spec_path_init_accepts(boar, env, repo):
    code, _, err = boar("status")
    assert code == 1 and "start one with `boar init --spec .boar/spec-draft.md`" in err


def test_an_inline_round_stopped_by_abort_says_so_cleanly(boar, env, repo):
    run = init(boar, env, "--repeats", "1", "--trials-per-round", "2")
    sleep = env / "sleep_s"
    sleep.write_text("0")
    run.eval_dir.mkdir()
    (run.eval_dir / "run").write_text(
        f"#!/bin/sh\nsleep \"$(cat {sleep})\"\necho '{{\"metric\": 1.0, \"guards_ok\": true}}'\n"
    )
    (run.eval_dir / "run").chmod(0o755)
    boar("propose", write_props(env, prop("a")))
    assert boar("eval", "check", "--foreground")[0] == 0
    boar("review", "record", "H1", "accept", "--reason", "ok")
    assert boar("review", "record", "eval", "accept", "--reason", "ok")[0] == 0
    sleep.write_text("30")
    proc = subprocess.Popen([sys.executable, "-m", "boar", "round", "run", "--foreground"], cwd=repo,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(400):
            rec = run.running()
            if rec and rec["pid"] == proc.pid and run.eval_group_path.exists():
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"the inline round never started its eval: {proc.poll()} {proc.stderr.read() if proc.poll() is not None else ''}")
        text = boar("next")[1]
        assert text.startswith("[WAIT] boar round run is running"), text
        assert f"it runs inline (--foreground, pid {proc.pid})" in text and "log:" not in text
        assert "inline in another shell" in boar("status")[1]
        out = boar("abort", "--reason", "stop")[1]
        assert f"sent SIGTERM to the running `boar round run` (pid {proc.pid})" in out
        _, err = proc.communicate(timeout=30)
        assert proc.returncode == 143, err
        assert "interrupted: `boar round run` was stopped by SIGTERM; `boar next` says what now" in err
        assert "error:" not in err
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_eval_check_prints_a_warning_for_untracked_files_round_1_would_commit(boar, env, repo):
    run = init(boar, env, "--repeats", "1")
    run.eval_dir.mkdir()
    (run.eval_dir / "run").write_text("#!/bin/sh\necho '{\"metric\": 1.0, \"guards_ok\": true}'\n")
    (run.eval_dir / "run").chmod(0o755)
    (repo / "pred.txt").write_text("from a manual run\n")
    code, out, err = boar("eval", "check", "--foreground")
    assert code == 0 and "eval check: PASS" in out, err
    assert "warning: untracked files that round 1's commit will sweep onto the run branch: pred.txt;" in out
    assert "  warning: untracked files" in boar("review", "show", "eval")[1]
