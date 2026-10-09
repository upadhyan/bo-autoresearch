# boar: the BOAR harness

`boar` is the command-line harness for BOAR (Bayesian Optimization AutoResearch). It owns a run's
state, runs the trials, warm-starts each round's Optuna study, writes the round summaries and the
report tables, and refuses any step taken out of order. The agent finds out what to do next from
`boar next`. The design is in [`design.md`](../../design.md). This README covers how the plugin runs
it, the commands, the run directory, and development. Installing and using BOAR is in the
[root README](../../README.md). Deliberate differences from the design are in
[`docs/design-notes.md`](../../docs/design-notes.md).

The harness does not depend on any particular host. The Claude Code side (the `/boar` skill, the
`boar-reviewer` agent and the Stop hook) is the rest of [the plugin](..). Python 3.10 or newer; the only
dependency is `optuna>=5.0`.

## How the plugin runs it

The plugin ships this package and runs it through [`bin/boar`](../bin/boar), which Claude Code puts on the
Bash tool's PATH while the plugin is enabled. The launcher runs `harness/.venv/bin/boar`, first building
that virtualenv with `uv sync --frozen --no-dev` (from `uv.lock`) if it is missing or its interpreter is
gone; without uv it prints how to install it and exits 127. The venv lives in the installed plugin's
directory, so each plugin version gets its own.

The Stop hook calls `harness/.venv/bin/boar next --hook` directly, and only when it exists: it never
installs anything, and a session where BOAR was never used ends its turns as usual.

The launcher doesn't activate the venv, so evals inherit the caller's environment as it is. Under
`uv run boar` (development) the venv is active; the worker drops it (`VIRTUAL_ENV` and its `bin/` on
`PATH`) from the environment evals inherit, so a `#!/usr/bin/env python3` eval runs the same interpreter as
the agent's manual runs. A virtualenv the user activated is passed through.

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
| `decide <id> keep\|fix\|investigate\|blocked\|remove --reason … [--trials 3,4 7] [--observable-key k]` | R3 decision. `remove` creates the pending removal `rm-<id>-r<r>`. `--observable-key` binds the hypothesis's observable to a diagnostic key. `investigate --cause "…" --measure a,b [--queue configs.json]` keeps it active one more round to measure the cause (`--reason` defaults to the cause); its queued partial configs, plus the hypothesis's off-state, run next round as extra trials. `blocked` resolves an investigation whose observable moved while another quantity got worse | Not after R2 of the current round; the hypothesis isn't active; a cited trial isn't in `trials.jsonl`; `fix`/`remove` without `--trials`; `rm-<id>-r<r>` was accepted (the decision is final); it was rejected and the decision is `remove` again (`keep`, `fix` or `investigate` are still allowed); a first decision other than `fix` without `--observable-key`, or a key no trial measured; `--cause`/`--measure`/`--queue` without `investigate`; `investigate` without `--cause` or `--measure`, in the last two rounds, or a second time for the hypothesis; a queue of more than 2 configs, or naming a lever or value outside the search space; `blocked` without an investigation the round before; resolving an investigation with `keep`, `blocked` or `remove` without cited trials that hold every measured key; removing half of an active enabler pair without a cited trial that moved both |
| `diag prune` | R6: deletes all but `diag.json` from the diagnostics directories of trials nothing protects (this round's, the incumbent's, every trial a decision or removal cites) | Not in a round; any R2–R5 done-when condition is unmet |
| `round run [--foreground]` | R2: commits the worktree to `boar/<id>`, builds and warm-starts study `round-<r>`, queues the incumbent (or the baseline), then the configs last round's investigations queued, each after its hypothesis's off-state, and the joint trial of each newly active enabler, leaving out any config already measured on the round's commit or queued before it (summary.md says `same as trial N`); runs N trials plus one per extra queued config, writes `rounds/<r>/summary.md`. Detached unless `--foreground`. Re-running it resumes an interrupted round. Each trial must find HEAD at the round's commit, a clean worktree and the accepted `eval/`; a trial that writes into the repo or `eval/` is recorded as failed and its writes are undone (see deviation 20) | Setup not done; the round already ran; any verdict pending; eval not accepted, or changed since acceptance (the changed files are named); no hypothesis active; the run's branch isn't checked out; a nested repo or submodule with uncommitted edits is left out of the commit; on resume, HEAD or the worktree moved since the round's commit |
| `round close` | R6: applies accepted removals and supersedes, activates every accepted proposal, then starts round r+1 or moves to finalize | Any R2–R5 done-when condition is unmet (R4 also owes a superseding proposal for each investigated hypothesis resolved `keep` and an enabling one for each `blocked`, unless `propose --none` ran) |
| `wait [--timeout S]` | Blocks until the detached worker finishes, or for S seconds (default 540). Prints `finished: …, next: …`, `still running: …`, or, for a worker that died, its log tail (for a `--foreground` run that died, a pointer to the shell that ran it), and kills any eval it left running | — |
| `finalize [--foreground]` | Holdout check on the last round's commit (baseline and incumbent alternating, `holdout_repeats` runs each, with the same per-run checks and write handling as a round's trials), then `report.md` with `<!-- boar:todo -->` markers for the narrative. Detached unless `--foreground`. Resumable | Rounds remain and a hypothesis is active; already finalized (the refusal names a damaged report and its fix); eval changed since acceptance; the run's branch isn't checked out; HEAD or the worktree moved since the last round's commit |
| `finalize --report` | Rewrites `report.md` from the stored results, without re-running the holdout. Any narrative in it is discarded | Finalize isn't done |
| `next [--session ID]` | Prints `[TAG] instruction` plus detail lines. `--session` binds the run to a Claude session, and adds a `note:` line when that moves the binding from another session. Marks the run done once the report has no markers left and its generated sections are intact | — |
| `next --hook` | The Stop hook. Reads the hook's JSON on stdin and finds the run (see deviation 5), then exits 2 with the next step on stderr while work remains; exits 0 for NONE, ASK_USER, DONE or ABORTED, for another session's hook, and on any internal error | — |
| `status` | Run summary: phase, round, eval, the hypotheses table, trial counts, the current incumbent, pending items, the worker | — |
| `abort --reason …` | Ends the run and releases the hook. It sends SIGTERM to the live worker named in `running.json` (to its process group only if it leads one), and SIGKILLs the eval process group recorded in `eval_group.json` | The run is already done or aborted |
| `_worker round\|finalize\|eval` | Hidden: the detached worker's entry point | — |

While a worker holds the run (a detached worker, or a CLI running `--foreground` work), every state-changing
command except `abort` is refused, and a second `round run`, `finalize` or `eval check` is refused. While a
trial's eval runs, the worker also looks every 2 s for another run of this repo's eval: a process with
`BOAR_CONFIG` in its environment and its working directory in the repo, outside the trial's own processes
(Linux, via `/proc`). Finding one stops the trial without recording it, and the worker stops with the reason
(see deviation 27).

`next` tags: `NONE, S2, S3, S4, S5, ASK_USER, R1, WAIT, R3, R4, R5, R6, FINALIZE, REPORT, DONE, ABORTED`.

### Config keys (`init --key value`)

`rounds` 12, `trials_per_round` 6, `repeats` 3, `trial_target_s` 600, `holdout_repeats` 6,
`seed` 0. `trials_per_round` must be at least 2 (one trial re-measures the incumbent on each new commit) and
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
    trials/<n>/           # config.json and repeat-<k>.stdout / .stderr; diag-<k>/ is repeat k's BOAR_DIAG_DIR (only
                          #   diag.json past 20 MB; `diag prune` thins it); repo-writes/ and eval-writes/ hold what
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

See [docs/design-notes.md](../../docs/design-notes.md).

## Development

From a clone of the repo:

```sh
cd plugin/harness
uv sync --group dev
uv run pytest -q
uv run --python 3.10 --isolated --group dev pytest -q   # the oldest supported Python
uv run boar --help
```

`uv run pytest -q` runs the unit tests (control, engine), the acceptance tests and the launcher and
hook tests. The acceptance tests drive the real CLI in-process through whole runs on a toy target (no
LLM), detached workers included. They cover each of the four warm-start rules. For a manual run
against an agent, see [`examples/toy`](../../examples/toy).

To try the plugin from the working tree, start Claude Code from the repo root with
`claude --plugin-dir plugin`. It loads in place, so the launcher uses the `plugin/harness/.venv` above,
and `/reload-plugins` picks up edits to the skill, agent or hook. `claude plugin validate .` checks the
marketplace and plugin manifests.

To release, bump `version` in both [`plugin.json`](../.claude-plugin/plugin.json) and `pyproject.toml`
and push. Installed copies only update when that version changes.
