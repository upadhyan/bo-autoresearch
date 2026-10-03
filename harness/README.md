# boar: the BOAR harness

`boar` is the command-line harness for BOAR (Bayesian Optimization AutoResearch). It owns a run's
state, runs the trials, warm-starts each round's Optuna study, writes the round summaries and the
report tables, and refuses any step taken out of order. The agent finds out what to do next from
`boar next`. The design is in [`../design.md`](../design.md). This README covers installing it, the
commands, the run directory, and where the implementation deliberately differs from the design.

The harness does not depend on any particular host. The Claude Code side (the `/boar` skill, the
`boar-reviewer` agent and the Stop hook) lives in [`../plugin`](../plugin).

## Install

Python 3.10 or newer. The only dependency is `optuna>=5.0`.

```sh
uv tool install ./harness          # from the repo root; puts `boar` on PATH
# or
pip install ./harness
```

For development:

```sh
cd harness
uv sync --group dev
uv run pytest -q
uv run boar --help
```

Under `uv run boar` the harness's virtualenv is active; the worker drops it (`VIRTUAL_ENV` and its `bin/` on
`PATH`) from the environment evals inherit, so a `#!/usr/bin/env python3` eval runs the same interpreter as
the agent's manual runs. A virtualenv the user activated is passed through.

## Install the Claude Code plugin

The repo root is a plugin marketplace (`.claude-plugin/marketplace.json`) that lists one plugin, `boar`:

```sh
claude plugin marketplace add /path/to/bo-autoresearch
claude plugin install boar@bo-autoresearch
```

You can also load it for a single session with `claude --plugin-dir /path/to/bo-autoresearch/plugin`.

The plugin's Stop hook runs `boar next --hook` only when `boar` is on PATH, so install the harness first.
To start a run, type `/boar <request>` in the target repo. A run continues unattended for hours, so it
needs permissions that won't prompt: auto mode, or allow rules for `boar`, for whatever the eval and
research need, and for edits in the target repo.

## Commands

Run every command from inside the target repo. The run is found from the working directory, through
`.boar/active`. A refusal prints `refused: <why>` to stderr and exits 1. Any other error prints one line
and exits 1; set `BOAR_DEBUG=1` to get the traceback. After every command that changes state, the CLI
prints a `next: [TAG] …` line.

| Command | Does | Refuses when |
|---|---|---|
| `init --spec <path> [--session ID] [--key value…]` | Creates `.boar/<id>/` and branch `boar/<id>`, copies the spec, freezes `config.json` (defaults plus overrides, plus `direction` from the spec), adds `/.boar/` to `.git/info/exclude`, writes `.boar/active`. Warns when it starts from an earlier run's `boar/*` branch | A run is active; not a git repo; no commits; HEAD detached; tracked files have uncommitted changes; untracked, non-ignored files outside `.boar/` (named); git has no committer or author identity; the spec has no `Direction: min\|max` line; bad override |
| `propose <file.json>` | Adds pending proposals `H<n>` | Not in setup or R4; any schema error (all of them are listed and nothing is added) |
| `propose --none --reason …` | R4: records that this round has no new proposals | Not in R4 |
| `eval check [--foreground]` | One baseline trial on dev, checking the contract, the guards, the timing (≤ 2 × `trial_target_s`), that the run left the repo and `eval/` as they were, and that no symlink in `eval/` leads out of it. What the run wrote into the repo is put back (new files moved to `eval_check-repo-writes/<time>/`), and the problem names anything still changed. It runs in a detached worker; `boar wait` reports PASS or FAIL with the problems. A pass makes the eval pending review | A worker is running |
| `eval restore` | Puts the copy of the accepted eval (`eval.accepted/`) back into `eval/`, checks it against the accepted hash, and prints what it undid | No live run; a worker is running; the eval isn't accepted; no copy was kept; `eval/` is a symlink (remove the link; its target is left alone); an `eval.replaced/` left by a crashed restore can't be removed; the copy can't be restored or doesn't match the accepted hash (then `eval/` is left as it was) |
| `review pending` | Read-only. Lists every item awaiting a verdict, with exactly the material the reviewer may see. An eval edited since its passing check is held back, with a note to record no verdict on it | — |
| `review show <id>` | Read-only. Shows one item's material and verdict (`H<n>`, `eval`, `rm-H<n>-r<r>`) | Unknown id |
| `review record <id> accept\|reject --reason …` | Stores a verdict and appends it to `reviews.jsonl`. In setup, it starts round 1 once the eval and at least one proposal are accepted | The item isn't pending. Eval accept also needs a passing check on the current `eval/` hash |
| `decide <id> keep\|fix\|remove --reason … [--trials 3,4 7]` | R3 decision. `remove` creates the pending removal `rm-<id>-r<r>` | Not after R2 of the current round; the hypothesis isn't active; a cited trial isn't in `trials.jsonl`; `fix`/`remove` without `--trials`; `rm-<id>-r<r>` was accepted (the decision is final); it was rejected and the decision is `remove` again (`keep` or `fix` are still allowed) |
| `withdraw <id> --reason …` | Drops a proposal that is pending, or accepted but not yet active | It is active, rejected, removed or already withdrawn |
| `round run [--foreground]` | R2: commits the worktree to `boar/<id>`, builds and warm-starts study `round-<r>`, queues the incumbent (or the baseline), runs N trials, writes `rounds/<r>/summary.md`. Detached unless `--foreground`. Re-running it resumes an interrupted round. Each trial must find HEAD at the round's commit, a clean worktree and the accepted `eval/`; a trial that writes into the repo or `eval/` is recorded as failed and its writes are undone (see deviation 20) | Setup not done; the round already ran; any verdict pending; eval not accepted, or changed since acceptance (the changed files are named); 0 or more than `max_active` hypotheses active; the run's branch isn't checked out; a nested repo or submodule with uncommitted edits is left out of the commit; on resume, HEAD or the worktree moved since the round's commit |
| `round close` | R6: applies accepted removals and supersedes, activates accepted proposals, then starts round r+1 or moves to finalize | Any R2–R5 done-when condition is unmet; the result would exceed `max_active` |
| `wait [--timeout S]` | Blocks until the detached worker finishes, or for S seconds (default 540). Prints `finished: …, next: …`, `still running: …`, or, for a worker that died, its log tail (for a `--foreground` run that died, a pointer to the shell that ran it), and kills any eval it left running | — |
| `finalize [--foreground]` | Holdout check on the last round's commit (baseline and incumbent alternating, `holdout_repeats` runs each, with the same per-run checks and write handling as a round's trials), then `report.md` with `<!-- boar:todo -->` markers for the narrative. Detached unless `--foreground`. Resumable | Rounds remain and a hypothesis is active; already finalized (the refusal names a damaged report and its fix); eval changed since acceptance; the run's branch isn't checked out; HEAD or the worktree moved since the last round's commit |
| `finalize --report` | Rewrites `report.md` from the stored results, without re-running the holdout. Any narrative in it is discarded | Finalize isn't done |
| `next [--session ID]` | Prints `[TAG] instruction` plus detail lines. `--session` binds the run to a Claude session, and adds a `note:` line when that moves the binding from another session. Marks the run done once the report has no markers left and its generated sections are intact | — |
| `next --hook` | The Stop hook. Reads the hook's JSON on stdin and finds the run (see deviation 5), then exits 2 with the next step on stderr while work remains; exits 0 for NONE, ASK_USER, DONE or ABORTED, for another session's hook, and on any internal error | — |
| `status` | Run summary: phase, round, eval, the hypotheses table, trial counts, the current incumbent, pending items, the worker | — |
| `abort --reason …` | Ends the run and releases the hook. It sends SIGTERM to the live worker named in `running.json` (to its process group only if it leads one), and SIGKILLs the eval process group recorded in `eval_group.json` | The run is already done or aborted |
| `_worker round\|finalize\|eval` | Hidden: the detached worker's entry point | — |

While a worker holds the run (a detached worker, or a CLI running `--foreground` work), every state-changing
command except `abort` is refused, and a second `round run`, `finalize` or `eval check` is refused.

`next` tags: `NONE, S2, S3, S4, S5, ASK_USER, R1, WAIT, R3, R4, R5, R6, FINALIZE, REPORT, DONE, ABORTED`.

### Config keys (`init --key value`)

`rounds` 12, `trials_per_round` 6, `repeats` 3, `trial_target_s` 600, `holdout_repeats` 6, `max_active` 6,
`seed` 0. `trials_per_round` must be at least 2 (one trial each round re-measures the incumbent) and
`seed` must be in 0 … 2³² − 1. You can write them as `--trials_per_round 4`, `--trials-per-round=4` or `trials_per_round=4`.

## Run directory

`.boar/` lives in the target repo and is excluded through `.git/info/exclude`.

```
.boar/
  active                  # id of the active run; read by the Stop hook
  <run-id>/
    config.json           # defaults plus overrides plus direction, frozen at init
    state.json            # phase, round records, eval status, removals, counters (harness only)
    spec.md               # copy of the approved spec
    research.md           # agent: research notes
    work/                 # agent: scratch space (proposal files, profiles, lever-check configs)
    hypotheses.json       # written only by the harness
    reviews.jsonl         # every verdict
    eval/                 # agent-built eval; frozen at acceptance
    eval.accepted/        # copy of eval/ taken at acceptance; `boar eval restore` copies it back
    eval.fingerprint.json # stat-keyed digest cache, so checking the frozen eval rereads only changed files
    eval_check.json       # last `boar eval check` result, with the manifest it checked
    eval_check/           # its repeat stdout/stderr (cleared by the next check)
    eval_check-repo-writes/<time>/  # what a check's run wrote into the repo, one directory per check, kept
    eval_check.log        # detached eval check worker log
    study.db              # Optuna SQLite storage; one study `round-<r>` per round
    trials.jsonl          # one line per finished trial (append-only)
    trials/<n>/           # config.json and repeat-<k>.stdout / .stderr; repo-writes/ and eval-writes/ hold what
                          #   a trial wrote into the repo or eval/ (moved out, never deleted)
    rounds/<r>/summary.md
    rounds/<r>/worker.log
    holdout.jsonl         # finalize's holdout runs
    holdout/<i>/          # their stdout/stderr (and repo-writes/, eval-writes/ as for trials)
    finalize.log
    report.md
    running.json          # the worker slot: pid, process identity, command, round and log (null for a --foreground run)
    eval_group.json       # process group of the eval a worker is running, so a dead worker's eval can be killed
    .launch.lock          # serialises claiming the worker slot (and `eval restore`)
```

A trial record looks like this:

```json
{"trial": 12, "round": 2, "commit": "<full sha>", "config": {"parse_workers": 4, "use_set": false},
 "state": "complete|infeasible|failed", "metric": 41.7, "repeats": [41.2, 41.7, 43.0],
 "metrics": {"peak_rss_mb": 812}, "duration_s": 130.4, "error": null, "queued": "incumbent|baseline|null"}
```

`config` always lists every lever ever defined, with pinned levers at their default. The eval receives the
same config through `BOAR_CONFIG`.

## Deviations from design.md

These are deliberate.

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

## Tests

`uv run pytest -q` runs the unit tests (control, engine) and the acceptance tests. The acceptance
tests drive the real CLI in-process through whole runs on a toy target (no LLM), detached workers
included. They cover each of the three warm-start rules. For a manual run against an agent, see
[`../examples/toy`](../examples/toy).
