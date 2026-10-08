---
name: boar
description: Start or resume an unattended BOAR run (Bayesian Optimization AutoResearch) that researches a measurable improvement, tunes hypothesis levers with Bayesian optimization against a frozen eval, and reports what worked.
argument-hint: "<request or path to ideas file>"
disable-model-invocation: true
---

# BOAR

Request: $ARGUMENTS

BOAR turns a measurable improvement task into **hypotheses**, each with **levers** (tunable parameters whose default reproduces the baseline), and runs Bayesian optimization over the levers of the active hypotheses against a fast, representative **eval**, round after round, for about 12 hours without supervision. You make every judgment call. The `boar` CLI owns the run state, runs the trials, and refuses any step taken out of order. A Stop hook keeps you working while the run has work left.

`<run dir>` below is `<repo root>/.boar/<run-id>/`. Run every `boar` command from inside the target repo: it finds the run from the working directory.

## Start

1. Run `boar next`. The first call on a machine sets up the harness with uv, which can take a minute. If the command fails, show the user what it printed and stop: the plugin's launcher names what is missing (usually uv), and `command not found` means this plugin's `bin/` isn't on PATH (the plugin is disabled, or Claude Code is too old to run plugin executables).
2. Any tag other than `[NONE]`, `[DONE]` or `[ABORTED]` means a run is in progress:
   - a request that is empty or matches the Goal in that run's `spec.md` (`boar status` prints the run dir): resume the run;
   - any other request: ask the user whether to resume the run or abort it, with your question tool (AskUserQuestion), and wait for the answer. Ending your turn to ask can fail: the run's Stop hook holds the session the run is bound to, and every session while it is unbound.

   To resume, run `boar next --session ${CLAUDE_SESSION_ID}`, which binds the Stop hook to this session and moves it from any other (one session drives a run), and continue in [the loop](#the-loop).
3. Otherwise start a new run at S1. With an empty request, ask the user what to improve.

## S1 Spec: the only human sign-off

Read the request and every file it names (often an ideas file). Draft the spec at `<repo root>/.boar/spec-draft.md` (repo root from `git rev-parse --show-toplevel`) with exactly these sections:

- **Goal**: one sentence, in the user's words.
- **Metric**: name, unit, how it is measured, and a line that reads literally `Direction: min` or `Direction: max` (the harness parses it).
- **Target population**: the real workloads or conditions an improvement has to carry over to.
- **Guards**: the conditions every trial must satisfy, each one something the eval can check on a single run (outputs correct, accuracy at or above a floor, RAM under a ceiling, …). A way to game the metric belongs under Known cheats, where the reviewer rules on it at proposal time.
- **Scope**: which changes are allowed (files, components, dependencies) and which are not.
- **Known cheats**: the ways this particular metric could be gamed, beyond the general list under [Hard rules](#hard-rules).
- **User ideas**: the user's list, copied word for word.
- **Budget**: changes to the defaults `rounds` 12, `trials_per_round` 6, `repeats` 3, `trial_target_s` 600, `holdout_repeats` 6, `seed` 0. `trials_per_round` is at least 2: one trial re-measures the incumbent on each new commit.

Show the user the spec and ask for explicit approval. Tell them that once approved the run continues unattended for about rounds × trials_per_round × trial_target_s plus finalize (12 h with the defaults), so it needs permissions that won't prompt (auto mode, or allow rules covering `Bash(boar *)`, the commands research and the eval need, and edits in this repo). Revise until they approve. If `git branch --show-current` shows a `boar/*` branch (a previous run's), ask whether to build on that run's code or check out their own branch first. Then:

    boar init --spec <repo root>/.boar/spec-draft.md --session ${CLAUDE_SESSION_ID} [--<key> <int> …]

Pass each Budget change as `--<key> <int>`: a request saying `rounds=3 trials_per_round=4` becomes `--rounds 3 --trials_per_round 4`. Hand every init refusal to the user, since it concerns their work or their git setup:

- no commits yet, uncommitted changes or a detached HEAD: they commit, stash or check out a branch;
- untracked files (init names them): they commit, gitignore or move them, since round commits would sweep them into the boar branch;
- no git identity: they set `user.name` and `user.email` for this repo, since round commits happen unattended.

Init freezes the spec and config, binds the Stop hook to this session, and checks out branch `boar/<run-id>`. From here on the run is unattended.

## The loop

1. Run `boar next`. Its first line is `[TAG] instruction`; the indented lines under it give ids, paths and exact commands.
2. Do what that tag's section below says, for exactly the items `boar next` names.
3. Repeat.

Take every next step from `boar next`: it is the run's single source of truth, and it survives context compaction where your memory doesn't. Keep looping until it prints `[ASK_USER]`, `[DONE]` or `[ABORTED]`. For every other tag the Stop hook refuses to let you end your turn and prints the same instruction back.

A **refusal** (`refused: …`, exit 1) names the precondition that doesn't hold: fix that cause, then run `boar next`. `boar status` summarises the run; `boar review show <id>` prints any item with its verdict. If you lose the thread, re-read `${CLAUDE_SKILL_DIR}/SKILL.md` and run `boar next`.

Long commands: `boar eval check`, `boar round run` and `boar finalize` each start a detached worker and return at once, because one trial can outlast the shell tool. Then go to [WAIT](#wait). Give `boar wait` and manual eval runs the Bash tool's maximum timeout (600000 ms).

## Tags

### [S2] Research

Collect candidate mechanisms from two places: the target system itself (profile it, read its code and logs) and outside sources (docs, papers, issue trackers, the web). Write them to `<run dir>/research.md`; keep profiles and other artifacts in `<run dir>/work/` so you can cite them. Carry every user idea that names a mechanism forward as a candidate, even one you expect to be useless or a cheat: BO and the reviewer rule on those, and their verdicts belong in the report. Reject in research.md, with a reason, only ideas that can't become a lever. Give each candidate its own section with an **Observable:** line: what should measurably change in the target if the mechanism works, and in which direction (a quantity the target code can count, such as "cache hits per request rise" or "passes over the input drop from 3 to 2"). The reviewer reads that line to judge the proposal, and R3 checks it against the measurements. Done when every user idea appears as a candidate or as a rejection with a reason, and every external candidate has a citation.

### [S3] Propose

Turn the candidates into hypotheses and submit them with `boar propose <run dir>/work/proposals-0.json`. Read [reference/proposals.md](${CLAUDE_SKILL_DIR}/reference/proposals.md) before the first proposal: it has the file format and the rules (one mechanism per hypothesis, a default that reproduces the baseline, realistic ranges, unique lever names, `supersedes` for retuning). Done when at least one proposal is pending.

When `boar next` says no hypothesis was accepted, read the reviewer's reasons (it prints them; `boar review show <id>` has the rest) and propose new hypotheses that answer them.

### [S4] Eval

Build the eval in `<run dir>/eval/` to the contract in [reference/eval.md](${CLAUDE_SKILL_DIR}/reference/eval.md); read it in full first. The point most easily missed: the eval is frozen when the reviewer accepts it, before most levers exist, so it forwards the whole `BOAR_CONFIG` to the target code generically, and the target code reads each lever, falling back to its default. It passes its whole environment through to the target the same way, so `BOAR_DIAG_DIR` reaches the target's instrumentation, and its metric and guard code never read `BOAR_DIAG_DIR`. Draw dev and holdout from the target population, disjoint, using typical cases. Size a trial (all repeats) at about `trial_target_s`. Run `boar eval check`, then `boar wait` until it prints `finished: eval check …`. Done when it reports PASS for eval/ as it now stands: any later edit needs a fresh passing check before review.

When `boar next` says the eval was rejected, rebuild it to answer the reviewer's reason, then pass `boar eval check` again.

### [S5] and [R5] Review

Dispatch the reviewer with the Agent tool in the foreground (`run_in_background: false`), so its verdicts are recorded when the call returns, subagent type `boar-reviewer` (installed as a plugin it may be listed as `boar:boar-reviewer`; use the name your agent list shows). The prompt is exactly:

    Review BOAR run <run-id>: <item ids from boar next>.

The reviewer starts with fresh context and fetches each item's material itself with `boar review pending`. Its judgment stays independent only while the prompt carries nothing more, so keep summaries, evidence and arguments out of it: the `--reason` you gave `boar decide` is the only part of your reasoning it ever sees. When it returns, run `boar next`, and dispatch again for any id still pending.

After setup verdicts: a rejected eval sends you back to S4, no accepted hypothesis back to S3, and three failed attempts at either to ASK_USER (`boar next` counts them).

### [ASK_USER]

Stop and ask the user: quote the `boar next` text, summarise what failed (the reviewer's reasons via `boar review show <id>`), and say what you would change. The Stop hook lets you stop here. When the user answers, act on it and return to the loop.

### [R1] Implement

On branch `boar/<run-id>` (check `git branch --show-current`), implement in the target code the mechanism of every hypothesis `boar next` lists as newly active, and repair every hypothesis it lists as marked `fix` (it prints the reason). Each lever reads its value from the `BOAR_CONFIG` JSON the eval passes through, via one small reader in the target code (pattern in [reference/eval.md](${CLAUDE_SKILL_DIR}/reference/eval.md)). At its default, each lever leaves the code behaving exactly as the baseline. For a hypothesis that supersedes another, its new levers drive the mechanism and the superseded levers stay pinned at their defaults.

Instrument the observable of every newly active hypothesis (its **Observable:** line in research.md), and add the measurements every `measure:` line in `boar next` names. Instrumentation writes a flat JSON object of numbers to `diag.json` in `$BOAR_DIAG_DIR`, and may write any other file there (a trace, a table); it runs only when `BOAR_DIAG_DIR` is set, and it costs almost nothing, since it runs in scored trials: counters, not profilers. The harness gives each eval invocation a fresh, empty directory and saves it as `trials/<n>/diag-<k>/`; the numbers become the trial's `diagnostics` (per-key medians over its repeats). Diagnostics never affect the metric, the guards, the warm start or the incumbent, so a broken `diag.json` never fails a trial (the trial's `diag_error` says what went wrong).

Done when every active lever changes behaviour when set away from its default. Check each one with the cheapest convincing run, either a direct call of the changed code or one eval invocation from the repo root with a config that sets only that lever:

    BOAR_CONFIG=<run dir>/work/check-<lever>.json BOAR_SPLIT=dev BOAR_DIAG_DIR=$(mktemp -d) <run dir>/eval/run

and confirm the instrumentation wrote the keys you expect into that directory's `diag.json`.

Run the eval once more with a config of `{}`, as a command of its own, and read its result line: start `boar round run` only once it shows `"guards_ok": true`, and fix whatever it reports first. `git status` must then show only your own edits: `boar round run` commits anything these runs wrote into the repo, and a lever whose code writes into the repo or eval/ fails every trial that sets it. A lever that crashes will show up in R2 as a failed trial.

Then run `boar round run`. It commits the whole working tree (`git add -A`) to the boar branch, so keep scratch files in `<run dir>/work/`, outside the repo's tracked tree. That commit is the round's code until the next R1 (see **Code** under [Hard rules](#hard-rules)). It returns at once while a detached worker runs the round (about an hour with the defaults). Go to WAIT.

"Round r was interrupted": run `boar round run` again to resume, leaving the worktree exactly as it was (resume refuses once HEAD or the worktree has moved since the round's commit).

### [WAIT]

A detached worker is running `boar eval check`, `boar round run` or `boar finalize`. Run `boar wait` (Bash timeout 600000 ms; it returns within about 9 minutes) and repeat until it prints `finished: …`, then run `boar next`. Keep calling `boar wait` rather than ending your turn: every other `boar` command that changes state is refused while the worker is alive. Keep HEAD, the worktree and eval/ untouched and the machine quiet meanwhile: run only `boar wait`, `boar status` and file reads, and leave the eval, the target code and prototypes until `finished`. Trials run the code in the worktree, and runtime metrics need an idle machine. A round or finalize worker compares all three before and after every trial: a change it finds before a trial stops it, and a change a trial leaves behind fails that trial and is undone. If `boar wait` reports that the worker died or stopped, read what it prints, fix the cause (restore what moved, or an environmental cause outside the worktree), and resume as it says.

### [R3] Analyse

Read `rounds/<r>/summary.md` (path in `boar next`): the incumbent and its drift flag, the noise floor, the warm start, each lever's marginal effect, each lever's effect on each diagnostic, the investigations measured this round, and the infeasible and failed trials with links to their logs and diagnostics. Then, for every active hypothesis:

    boar decide <id> keep|fix|investigate|blocked|remove --reason "…" --trials <ids> [--observable-key <key>]

A hypothesis's first decision other than `fix` binds its observable to the diagnostic key that measures it (`--observable-key`); the key must appear in some trial's diagnostics. Every reason says whether the prediction in research.md held: did the observable move the way it said?

- `keep`: it helps, or there isn't enough evidence yet (its levers barely varied, or few trials touched them).
- `fix`: a lever crashed, misbehaved, or wrote into the repo or eval/ because of a bug in its implementation (read the failed trials' errors and logs under `trials/<n>/`). It stays active, R1 repairs it, and earlier trials that set its levers away from default drop out of the warm-start set. In the last round `boar decide` refuses `fix` (no R1 follows, and those trials would drop out of the final incumbent too): decide `keep` and leave the bug for Caveats.
- `investigate`: the metric alone can't tell a bad idea from a bug, a bad range, or a bottleneck the change exposed elsewhere. `boar decide <id> investigate --cause "…" --measure k1,k2 [--queue <run dir>/work/queue-<id>.json]` keeps it active one more round: the next R1 adds the named measurements, and up to 2 queued partial configs (a JSON list; each is filled from the incumbent, and one must be the off-state, every lever of the hypothesis at its default) run as extra trials. Once per hypothesis, not in the last two rounds, allowed after a rejected removal, never reviewed. `--reason` defaults to the cause.
- In the round after an investigation, resolve it by what the measurements show, citing trials whose diagnostics hold every measured key (except for `fix`): the observable did not move → `fix`; moved weakly or only at some values → `keep`, then supersede it in R4; moved, but another quantity got worse → `blocked`, and R4 proposes a hypothesis that relieves that bottleneck; moved, nothing else changed, metric flat → `remove`.
- `blocked` stays active and lends the run one extra active slot until the next round closes. It exists only as the resolution of an investigation.
- `remove`: the trials that varied its levers show no effect beyond the noise floor, or show consistent harm. The floor is measured at the baseline's metric: when the configs you compare sit far from it (say more than 2x faster), compare their effect with the floor scaled to their metric (the summary gives it at the incumbent's metric) or with their own repeat spreads, not with the baseline's absolute floor. It takes effect only if the reviewer accepts it in R5, and the reviewer accepts only when the cited trials set the levers to a meaningful spread of values (not one or two points), the effect across them is within the noise floor or consistently harmful, nothing in the summary suggests it helps only in combination with another lever, and any crash comes from the mechanism itself. An accepted removal pins its levers at default, so every trial that set them elsewhere leaves the warm-start set and the incumbent can move (R6's `boar next` warns when it does). Decide removals before the last round where you can: in the last round `boar decide` refuses a removal that would change the final incumbent, since no round follows to search again; decide `keep` and argue the lack of effect in the report's What didn't.

Cite trial ids from the summary in every decision (`--trials 3,4,7`; required for fix and remove), and put the numbers in `--reason`: values tried, medians against the noise floor, rank correlation. It is the only reasoning the reviewer sees. A drift flag (`yes`) means the incumbent's re-measurement moved beyond the noise floor (scaled to its metric) or didn't complete: weigh small effects with care and keep it for Caveats. Done when every active hypothesis has a decision for this round. You may re-decide until the round closes, except that a judged removal stands: an accepted one is final, and after a rejected one only `keep`, `fix` or `investigate` remain.

### [R4] Synthesize

Propose new hypotheses from the results and the spec (`"source": "synthesis"`, citing trial ids and the summary), with targeted research if it helps (append it to research.md); submit with `boar propose <run dir>/work/proposals-<r>.json`. Typical moves: retune a lever whose best values sit at an edge of its range, as a new hypothesis with `"supersedes": "<id>"`, a widened or shifted range and fresh lever names; or chase the hotspot that a profile of the incumbent now shows. For each hypothesis decided `blocked` this round, propose an enabler: a hypothesis with `"enables": "<id>"` that addresses the bottleneck its reason names, and a `"joint_config"` (a partial config, filled from the incumbent, setting at least one of the enabler's own levers away from default) that the harness runs as an extra trial when the enabler activates. For each investigated hypothesis resolved `keep`, propose one that supersedes it. R4 isn't done until those exist, unless you run `propose --none`. Once an enabler is active, removing it or the hypothesis it enables needs a cited trial that moved both (the joint trial). If nothing is worth testing, or in the last round (new hypotheses would never run), run `boar propose --none --reason "…"`. Done when this round has at least one proposal, or `--none` was recorded.

### [R6] Close

When `boar next` suggests marking a hypothesis `fix` after its removal was rejected, do so with `boar decide` if the reviewer's reason points to a bug, so R1 repairs it. If the diagnostics directories under `trials/` grow large, `boar diag prune` (R6 only) deletes all but `diag.json` from the trials nothing needs any more: it spares this round's trials, the incumbent's and every cited one. Then run `boar round close`. It applies the accepted removals (a hypothesis whose removal was rejected stays active), activates the accepted proposals, and removes the hypotheses they supersede. The next round starts at R1, or the run moves to finalize.

### [FINALIZE]

Run `boar finalize`. It returns at once while a worker evaluates the baseline and the incumbent on holdout, alternating the two so machine drift cancels out, and then writes `report.md`. Go to WAIT. Holdout runs the last round's commit, the code the dev trials measured, so finalize refuses when HEAD or the worktree has moved since: restore that commit as the refusal says, then run it again.

### [REPORT]

Open the report (path in `boar next`) and replace every `<!-- boar:todo -->` line with narrative, leaving the generated sections as they are. The whole narrative fits on one page:

- **What worked**: the hypotheses the incumbent uses, and the evidence for each (trial ids, effect against the noise floor, the holdout result).
- **What didn't**: under the generated table, why the removed hypotheses failed and what that says about the problem.
- **Caveats**: disagreement between dev and holdout, noise, drift flags, and interactions between levers that were never tested.

Done when no marker is left; `boar next` then prints DONE. If it says report.md is missing, run `boar finalize --report` to rewrite the skeleton from the stored results (the holdout is not re-run), then write the narrative. If it names generated headings that are missing, put those heading lines back exactly as named; `boar finalize --report` restores them too, but discards the narrative.

### [DONE] and [ABORTED]

Report to the user: the holdout result (baseline and incumbent medians, relative change), the recommended config, the branch `boar/<run-id>` that holds the code, and the report path. For ABORTED, give the reason and how far the run got (`boar status`). Tell them the repo is still checked out on `boar/<run-id>`, and that `git checkout <base branch>` (the branch `boar status` shows after "from") returns them to their own.

### [NONE]

No active run here. Before init that is expected: do S1. After init it means you are outside the target repo: `cd` back into it.

## Hard rules

- **Files.** You write the spec draft (before init), `research.md`, `work/`, `eval/` (until the reviewer accepts it) and the narrative sections of `report.md`. Every other file under `.boar/` belongs to the harness: change run state only through `boar` commands.
- **Frozen eval.** Once accepted, `eval/` stays byte-for-byte unchanged for the rest of the run, and changing the eval means a new run. The harness keeps a copy of the accepted eval: when eval/ changes outside a trial, `boar round run` and `boar finalize` refuse and `boar next` lists the changed files; run `boar eval restore` to put the accepted copy back.
- **Branch.** Every change to the target code goes on `boar/<run-id>`; the user's branch stays untouched.
- **Code.** Edit the target code only in R1, before `boar round run`. Trials, the holdout check and the report all measure the round's commit, so HEAD and the worktree stay as committed until the next R1, and for good after the last round's `boar round run`. Prototype in `<run dir>/work/` in R3 and R4, once the round's worker has finished.
- **No cheats.** An improvement must carry over to the target population. These move the eval without moving the population, so they are never an improvement: treating the eval's inputs as a special case; caching or memoizing results across eval runs; doing less work than the task requires; changing how things are measured (timers, sampling, logging that hides work); weakening a guard; giving up a quality the spec implies (precision, determinism); using more resources than the deployment the spec describes; anything under the spec's Known cheats. A user idea of this kind still goes to the reviewer as a proposal, and every change in the target code's behaviour sits behind a reviewed lever.
- **Abort** with `boar abort --reason "…"` only when the run truly cannot continue (the target can no longer build or run and can't be repaired, the machine is unusable). A refusal or a hard phase calls for fixing the cause.
