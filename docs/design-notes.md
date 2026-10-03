# Design notes

Where the implementation deliberately differs from [design.md](../design.md), and why. The command reference and run directory layout are in [plugin/harness/README.md](../plugin/harness/README.md).

1. **Optuna 5 constraints.** Optuna 5 deprecates `TPESampler(constraints_func=…)`. Instead, each live
   trial calls `trial.set_constraint("guards", 0.0|1.0)` before `tell`, and copied warm-start trials pass
   `constraints={"guards": …}` to `optuna.trial.create_trial`. Both put the values in `system_attrs`
   under `constraints:guards`, which is where TPE reads them. The behaviour is the same and no deprecated
   API is used. The sampler is `TPESampler(multivariate=True, n_startup_trials=trials_per_round, seed=(seed + r) mod 2³²)`;
   a resumed round reseeds it (see 14).
2. **Detached workers and `boar wait`.** Claude Code kills background shells after a while, and one
   foreground Bash call is capped at 10 minutes. So `round run`, `finalize` *and* `eval check` (one baseline
   trial is about `trial_target_s` long, and up to 2× is a passing eval) check their preconditions in the
   foreground, start a detached worker (its own session, logging to a file) and return at once. `boar wait`
   polls the worker. `--foreground` runs the work inline. Re-running the same command resumes an
   interrupted worker. The worker runs from the run directory, so a target repo with its own `boar`
   module can't shadow the harness.
3. **`boar withdraw`** (new). Without it, the `max_active` refusals at setup and at R6 would have no way
   out, because an accepted proposal can't otherwise be dropped. It can't touch an active hypothesis:
   removing one still needs a reviewed removal.
4. **`boar review pending` and `boar review show`** (new, read-only). The reviewer pulls its material from
   the harness: the proposal; the eval's directory, file list and last check; for a removal, the
   hypothesis, the stated reason, the cited trial rows and the summary path. That way the main agent can
   only pass on its stated reason.
5. **Session binding.** The Stop hook fires in every Claude session open in the repo. `init --session` and
   `next --session` record the session id, and the hook blocks only a session whose id matches (or any
   session when none is recorded or the hook input has none). Values starting with `$`, such as an
   unsubstituted `${CLAUDE_SESSION_ID}`, are treated as absent. `next --session` says when it moves the
   binding from another session; plain `next` binds nothing. The hook looks for a run from the `cwd` field of
   its input, then `$CLAUDE_PROJECT_DIR`, then its own cwd, skipping runs that are done, aborted or bound to
   another session. If none is left it uses a per-user session registry,
   `${XDG_STATE_HOME:-~/.local/state}/boar/sessions/<session id>` (holding the repo root, written by
   `init --session` and `next --session`), so a session that has `cd`'d out of the repo, or was started in a
   parent directory, is still held. The entry leads only to a run bound to that session id; a stale one
   (no run there any more, or one that is finished, unbound or bound elsewhere) releases the session.
6. **New files.** `state.json`, `eval_check.json`, `holdout.jsonl`, `running.json`, `eval_group.json`,
   `eval.accepted/`, `.launch.lock`, the worker logs and `work/`. `state.json` also stores, per round,
   `drift` (yes, no or n/a) and the pooled `incumbent_metric` and `incumbent_n` for the report, and, for the
   eval, the `accepted_manifest` (path → sha256). The JSONL files are appended with `O_APPEND` and fsync; a
   torn last line (from a crash mid-write) is skipped on read and cut on the next append.
7. **Direction** comes from a `Direction: min|max` line in the spec and is stored in `config.json`.
   `init` refuses a spec without one.
8. **Noise floor** is the median, over complete baseline-config trials with at least 2 repeats, of each
   trial's own max − min. It is measured within trials so that drift between rounds doesn't widen it. It
   stays "not known yet" until one such trial exists; until then the drift flag is `n/a`. Round r's drift is
   judged against the floor from trials of earlier rounds. It is `yes` when the incumbent's re-measurement
   didn't complete (it failed or broke a guard) and the config was measured before; with no earlier
   measurement (round 1's baseline, say) it is `n/a` whatever the trial's state. The floor is measured at
   the baseline's metric, and repeat spread grows with the metric: a speed task's incumbent can run 20x
   faster with a far smaller spread. So the drift check compares |Δ| with the floor scaled from the
   baseline's median to the config's earlier metric (`stats.floor_at`; unscaled when either is zero or they
   differ in sign), and the summary gives the floor as a percentage of the baseline's median, the floor
   scaled to the incumbent's metric and the incumbent's own repeat spread. The report's "within noise"
   verdict on incumbent vs baseline keeps the absolute floor, since one side of it is at baseline scale.
9. **The eval freeze is a hash, a manifest and a copy.** `eval check` records the hash it checked, and
   `review record eval accept` refuses unless the current `eval/` hash equals the last *passing* check's
   hash. The hash is sha256 over sorted (path, bytes) pairs, skipping `__pycache__/` and `*.pyc`; a symlink
   counts as an entry (its target path) and is not followed. Since the hash can't see behind a link, the eval
   must be self-contained: `eval check` fails when a symlink in `eval/` (or `eval/` itself) resolves outside
   it. At acceptance the harness also stores a per-file manifest and copies `eval/` to `eval.accepted/`.
   Later checks compare against the manifest through a stat-keyed digest cache (`eval.fingerprint.json`),
   so `next`, the Stop hook and the per-trial checks reread only files whose size or times changed; an
   unreadable file is reported as such and never releases the hook. Once `eval/` changes, `round run` and
   `finalize` refuse, and they and `next` (at R1 and FINALIZE) name the changed files; `eval restore` (new)
   puts the copy back. An eval edited after its passing check but before review goes back to S4, and
   `review pending` holds it back.
10. **Bytecode is not code.** A round's commit leaves out `__pycache__/` and `*.pyc` (by git pathspec, so
    the user's ignore files are untouched). The resume check ignores them too, so running the eval can't
    make a round impossible to resume.
11. **Branch check.** `round run` and `finalize` refuse unless `boar/<id>` is checked out, so the harness
    never commits onto the user's branch.
12. **Proposal schema, stricter.** Unknown keys in a proposal or a lever are refused rather than silently
    dropped. `citations` (stored as `[]`) and `supersedes` (stored as `null`) may be omitted, and
    `supersedes` must be null in setup and must name an active hypothesis in R4. An empty list is refused:
    use `propose --none`. Categorical choices must be distinct and the default must match a choice in
    type as well as value. Float bounds and defaults are stored as floats, and `log` is made explicit.
13. **Warm-start internals.** `warmstart.build_frozen_trials(valid, space, defaults)` takes three
    arguments; the plan's `direction` argument was unused. The defensive "value outside the lever's
    distribution" check runs in `warmstart.select` and is counted under rule 2. A warm trial with no finite
    metric counts under rule 1.
14. **Resuming a round.** If the warm start was never recorded, the round's study is rebuilt from
    scratch. Otherwise RUNNING Optuna trials are marked FAIL, the incumbent (or the baseline) is queued
    again if its trial never made it into `trials.jsonl`, and the sampler is reseeded with
    `seed + r + 1000003 × (trials in the study)` so a resume doesn't replay the draws the interrupted
    worker already made. An interrupted trial is never recorded.
15. **Holdout runs** are single invocations, each with a timeout of 3 × `trial_target_s` / `repeats`.
    They measure the commit of the last round that ran, not the worktree: a fresh `finalize` refuses if HEAD
    or the worktree has moved since, and it commits nothing (it commits the worktree only when no round ever
    ran). The final incumbent comes from the warm-start set for the final search space. If there is none, it
    is the baseline. The report flags a recommended config that failed or broke a guard on holdout, and calls
    a change within the noise floor "within noise" rather than better or worse.
16. **Setup failures.** `S4` and `S3` say "failed attempt k/3". After 3 rejected evals or 3 batches with
    no proposal accepted, `next` prints `ASK_USER`, which releases the hook. A batch counts as failed only if
    the reviewer rejected something in it and accepted nothing; a batch the agent withdrew whole doesn't
    count. After the third eval rejection, a new `eval check` returns the run to S4, and a fourth rejection
    asks again. `next` also prints `ASK_USER`
    when the state is inconsistent (setup complete but round 1 not started), so the agent can't loop
    forever.
17. **The last round.** As the design says, `round close` still activates accepted proposals in the last
    round. An accepted supersede there would pin the superseded levers at default for the final
    incumbent, so the last round's R4 text recommends `propose --none`. For the same reason `decide … fix`
    is refused in the last round (no R1 follows to repair it, and its trials would drop out of the final
    incumbent), and R3's text says to decide `keep` and leave the bug for Caveats. `decide … remove` is
    refused in the last round if the removal, together
    with the round's other removals not rejected, would change the final incumbent: an accepted removal
    pins its levers at default too, and a numeric lever is almost never sampled exactly at its default, so
    the run would otherwise recommend the baseline or a worse config. A removal that leaves the incumbent
    as it is stays allowed. In every round, R6's `next` and `round close` warn when the close moves the
    incumbent to another config.
18. **Full commit shas** in `trials.jsonl` and `state.finalize.commit`.
19. **Worker slot.** `round run`, `finalize` and `eval check` claim the run before any work, foreground or
    detached: under `.launch.lock` they refuse if `running.json` names a live worker, then write their own
    record. A detached worker takes over the record from the CLI that spawned it, and refuses to start if
    the record names some other live process. A record counts as live only if its pid is alive *and* the
    process identity matches (the `/proc` start time and boot id, or `ps` start time), so a reused pid is
    not mistaken for the worker. While an eval runs, its process group is in `eval_group.json`; claiming
    the slot, starting a worker, `wait` on a dead worker and `abort` SIGKILL such a group when its worker
    is gone.
20. **The code is frozen between R1s, and trials may not write into the repo or `eval/`.** Trials run the
    round's commit in the worktree. Before each trial or holdout run the worker requires HEAD at that
    commit, an empty `git status` (untracked files included; bytecode and `.boar/` ignored) and `eval/`
    equal to the accepted manifest, and stops with a refusal otherwise ("changed between trials"). After
    each one, whatever it wrote is undone and the trial is recorded as failed, with an error naming the
    writes, so a writing lever costs its trials but never wedges the round. A holdout run that wrote keeps
    its measurement and the report lists it: `eval check` runs dev only, so a holdout-only write shows up
    first at finalize, which commits nothing afterwards. In both cases changed tracked files are checked
    out from the round's commit (their written content kept first), new files are moved to
    `trials/<n>/repo-writes/` (`holdout/<i>/repo-writes/`), and `eval/` entries are restored from
    `eval.accepted/` (replaced ones kept in `eval-writes/`). New files are moved before tracked paths
    are restored, so a directory standing where a tracked file was is emptied (what is left in it copied)
    before the checkout replaces it; a tracked file under a directory the trial replaced with a file or a
    link is checked out again too (the replacement is kept, under `dup-<n>/` if its place in the keep
    directory is taken). Nothing is deleted except the directories the trial made that the moves
    leave empty; one that was there before (untracked, even empty) stays. If something can't be
    put back, the trial is recorded first and then the worker stops naming what is left; a moved HEAD is
    refused without recording. So no trial output reaches the next round's commit, and a fresh `finalize`
    needs the same clean status. `eval check` applies the rule at baseline, before the eval is frozen, and
    also fails an eval that rewrites a file that was already uncommitted (by content, mtime or mode, so
    the same bytes count; not by ctime alone, which a same-mode chmod or a hard link also changes). Its
    problem says "put back" only for what was, and names what is still changed. It warns about every other
    untracked file and every uncommitted tracked change (an edit or deletion) it finds, since round 1's
    commit sweeps them in (a manual run's output or a primed cache is written before the check). The round's
    commit refuses when something stays uncommitted after `git add -A` (a nested repo or submodule with
    edits), before the round is marked started.
21. **Stricter `init`.** It refuses untracked, non-ignored files outside `.boar/` (the first round's
    `git add -A` would sweep them onto the run branch) and a repo where git has no identity (round commits
    happen unattended). `trials_per_round` must be at least 2 and `seed` must fit Optuna's 32-bit range. It
    warns when the base branch is an earlier run's `boar/*` branch.
22. **Incumbent eligibility.** A config can be the incumbent only if its latest trial completed, among the
    trials that still mean the same thing in the search space (warm-start rules 2 and 3, failed trials
    included): one that broke a guard or failed when measured again can't win on its earlier passing
    repeats, since a guard is a correctness condition, and a later passing trial makes it eligible again. The baseline is always eligible when it has complete trials, since default levers can't break a
    guard; a flaky baseline guard failure doesn't hand the win to a worse config.
23. **Deciding after a rejected removal.** The judged removal stays on record for the report. After a
    rejection the hypothesis can still be decided `keep` or `fix` in the same round (not `remove` again),
    and R6's `next` text names each rejected removal whose hypothesis is still marked `remove`. After an
    accepted removal the round's decision is final.
24. **A damaged report.** `next` stays at REPORT (and the hook holds) while `report.md` is missing,
    unreadable, or lacks any of its six generated headings (matched as exact lines, and named). A missing
    report is rewritten with `finalize --report` (new) from the stored results, without re-running the
    holdout; for missing headings `next` says to put those lines back, since `--report` would discard the
    narrative. Plain `finalize` after done always refuses.
25. **Result line.** The result is the last non-empty line of stdout; within it, text before a carriage
    return is dropped, so a progress line overwritten by the result (`working 99%\r{…}`) still parses.
26. **A `--foreground` run stopped by `abort`** exits 143 with `interrupted: … was stopped by SIGTERM`.
    `next` and `status` say it runs inline in another shell rather than pointing at a worker log. Once it
    has exited, `next` (R1 "Round r was interrupted") and `wait` say its output went to the shell that
    started it whenever no worker log exists.

