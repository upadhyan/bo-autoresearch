# Handoff: BO Autoresearch after the v1 follow-ups (2026-09-29)

Repo: `upadhyan/bo-autoresearch`. All finished work is on `main`; nothing is in flight. The next session's job:
**first fix the blocking bug below** (the scripted benchmark crashes on a fresh clone of `main`), then **run the
dogfood benchmark once with the real `claude` caller** to confirm the harness and skills work, then tie off the
loose ends.

## BLOCKING BUG: the scripted dogfood benchmark crashes on `main`

### Symptom
A fresh clone of `main` (verified at fd77e81; nothing in the harness has changed since) running

```bash
uv venv -q --python 3.12 /tmp/dfenv && uv pip install -q --python /tmp/dfenv/bin/python pyyaml
/tmp/dfenv/bin/python benchmarks/dogfood/run_benchmark.py --caller scripted --seeds 1 --sigmas 0.025 --no-free --jobs 1 --work /tmp/df
```

gets through setup, R0, the ladder calibration, R1 and R2 (about 115 trials, ~5 min), then `scripted.py` dies.
From `/tmp/df/seed1-sigma0.025/caller.log`:

```
scripted.py:126 duties     -> self.interplay(st["interplay_missing"][0], st)
scripted.py:94  interplay  -> self.record("interplay", {**m, "flags": flags}, "interplay-reviewer", ...)
RuntimeError: record refused: interplay: flags[0].partner: H4.v1 is active, not rejected or inconclusive or parked
```

The matrix gate then FAILS, because the run never finished: not ended by exhaustion, distilled branch unverified,
incumbent not within δ, and the interaction, fidelity-sensitive and discouraged-but-wins verdicts are missing.
Everything else passed: every invariant, every adversarial action blocked, and the useful, useless,
matters-but-hurts, prohibited and off-intent verdicts. (The optuna "greenlet unavailable" warning in the log is
harmless.) This benchmark passed in v1 (PR #39), so this is a regression from the follow-ups in PR #40.

### What the event log shows (seed 1, σ = 0.025)
The hypotheses are H1 warm-up (retained), H2 label smoothing, H3 weight decay, **H4 "Clip the gradients"**,
H5 mixed precision, H6 pretrained (pruned), H7 attention (parked), **H8 "Keep an EMA"**, **H9 "Distil"**
(registered, untested), H10 cosine tail, and **H11 "Raise the learning rate"**. H11 is H4's planted partner; it
arrives in generation pass 3 (`fixtures/pass3.yaml`, "arriving after it was tested alone").
1. **R1:** H3, H4 and H8 each get pending-reject and then confirmed `reject`, all with condition **`no-improvement`**.
   The drift check passed (`undecidable: false`).
2. **R1's round end** (`cli._finalise_rejects`): only H3 is rejected. H4 and H8 get **`reject_deferred`**
   (events 357 and 358), with no reason field. That is the "substitutes" rule: *one `no-improvement` reject per
   round end, the smallest Δ upper bound first; the rest are re-judged next round*. A deferred reject then needs a
   fresh pending check **and** a fresh confirmation (see the `ponytail:` "k substitutes take k rounds" in
   `_finalise_rejects`).
3. **R2:** H4 gets a fresh pending-reject (`V-R2-H4.v1-1`) but no confirmation before the round ends. H2 is rejected.
4. **After R2:** pass 3 registers H11. The harness asks for a **newcomer interplay review** of H11
   (`interplay_missing: [{"newcomer": "H11.v1"}]`).
5. `scripted.py` `interplay()` (the newcomer branch, lines ~89–93) flags H4 as the partner whenever
   `verdict H4` has **any** records. It never checks H4's status. H4 is still **active**, because its reject was
   deferred, so the harness correctly refuses the flag (a flagged partner must be rejected, inconclusive or
   parked) and the scripted caller raises.

H8 (EMA) is in the same state. Its planted "interaction found by interplay review" story (EMA removed while its
partner Distil is untested, then revived) can't happen either, because H8 was deferred rather than removed.

### Root cause (strongly supported; confirm it)
Two things interact:
- **The calibrated-verdict amendment** (PR #40, "accept and amend": a δ-based signal floor on the verdict GP).
  A flat lever next to a curved partner is now labelled `no-improvement` where it used to be `irrelevant`.
- **The substitutes rule** in `_finalise_rejects` (older: it came with the M_u amendment, 244fd4e) exempts only
  `irrelevant` rejects. Every `no-improvement` reject beyond the first is deferred and must be re-earned from scratch.

So after the amendment, R1's three flat levers (H3, H4, H8) were all `no-improvement`, and two were deferred.
Presumably in v1, H4 and H8 came out `irrelevant` (exempt) and were rejected in R1, which is what the benchmark's
planted timeline assumes. The second bug is in `scripted.py`: it should never propose a flag the harness can't
accept, so a timing change should fail a verdict criterion rather than crash the run.

### What the fixing agent should do
1. **Reproduce** with the command above (a scratch `--work` dir outside the repo; about 5 minutes). Confirm the
   same crash and the `reject_deferred` events: `sqlite3 <work>/seed1-sigma0.025/project/.bo-research/*/log.db
   "select seq,type,payload from events where type in ('reject_deferred','hypothesis_rejected','verdict')"`.
2. **Confirm the cause** by checking that R1's labels changed with the signal floor. The commit that added it
   is `cdaab6d` ("Floor the verdict GP's signal so low-SNR intervals hold the truth"). Run the same command in
   `git worktree`s of `cdaab6d^` (just before it) and `cdaab6d`, plus `10cf86b` (the v1 merge) as a baseline.
   Compare R1's H4/H8 conditions and whether `reject_deferred` appears: expected `irrelevant` with no deferral
   before, `no-improvement` with deferrals after. If `cdaab6d^` also crashes, bisect the other follow-up merges
   instead (`git log --first-parent --oneline 06d6027^2` lists them). The evidence-cap change (`49efbbf`) and
   the two-stage checks (`bb8e8a2`) are the next suspects. Don't change code before the cause is confirmed; use
   the `diagnosing-bugs` skill.
3. **Fix `scripted.py` first, whatever else is decided.** Both interplay branches must flag only partners the
   harness allows. Build flags from `status` (the partner's `status` in rejected / inconclusive / parked), not
   from "has verdict records". With that alone the run should finish, but the two interaction cases may fail.
4. **Then fix the timing. This is a spec-level choice, so ask the user before implementing.** Options, with a
   recommendation:
   - **(a) Recommended: carry the confirmation over.** A deferred `no-improvement` reject is re-judged at the next
     verdict check, and if that check still says reject, it applies at that round end without a new
     pending + confirmation pair. This is the upgrade path already written in the `ponytail:` note. It keeps the
     point of the substitutes rule (once one substitute is frozen, the other is re-judged and can now show its
     effect) while cutting the extra round per deferred lever.
   - (b) Narrow the rule to levers that could actually be substitutes, e.g. defer only when the co-rejected
     hypotheses' levers are correlated in the GP posterior or share a config path. It's more precise but more code,
     and needs a definition the spec doesn't give.
   - (c) Change the benchmark (move `Raise the learning rate` to a later pass, or accept a late revival). Not
     recommended: it hides a real slowdown that users would hit too, since k flat levers now take about 2k rounds
     to leave.
5. **Tests:** a Seam 1 test (drive the CLI; assert only on JSON, refusals, events and generated files) that
   reproduces "two flat, non-substitute levers confirmed in the same round next to a curved one" and asserts the
   chosen rule. Mark any multi-seed rate test `@pytest.mark.slow`. Keep `test_two_substitute_levers_are_never_both_rejected_at_one_round_end`
   passing: true substitutes must still never both go at one round end.
6. **Acceptance:** the one-run command above passes every criterion, then the full scripted matrix
   `run_benchmark.py --caller scripted --jobs 2` (6 runs) passes the matrix gate. A verdict wrong in 1 of 6 is
   allowed; report the counts. Then update the Decision log and, if the rule changed, post an amendment to #21
   (ask the user first).
7. Only after that, go on to "What's left" below (the real `claude` benchmark).

## Read first (don't duplicate these)
- **Spec:** [issue #21](https://github.com/upadhyan/bo-autoresearch/issues/21), the source of truth, plus its
  amendment comments. The three that matter now:
  - [M_u replaces √V_T](https://github.com/upadhyan/bo-autoresearch/issues/21#issuecomment-5890844933)
  - [two-stage t-test on a pooled σ, α = 5%](https://github.com/upadhyan/bo-autoresearch/issues/21#issuecomment-5896122041)
    for the equivalence check, branch verification and drift check (it also clarifies budget = summed compute time)
  - [calibrated verdict intervals](https://github.com/upadhyan/bo-autoresearch/issues/21#issuecomment-5897420132):
    a δ-based signal floor; `no-improvement` is the expected label for a flat lever next to a curved partner
- **What was built:** [PR #39](https://github.com/upadhyan/bo-autoresearch/pull/39) (v1),
  [PR #40](https://github.com/upadhyan/bo-autoresearch/pull/40) (follow-ups: live verification, known limits,
  amendments, cheaper tests; its description is the summary of this session),
  [PR #41](https://github.com/upadhyan/bo-autoresearch/pull/41) (dogfood on demand only; README "Check that it works").
- **Design notes + Decision log:** `docs/agents/harness-design-notes.md`. Every non-obvious choice from this
  session is logged there (parallel trials, #26/#28/#29, enforcement, seed-rate convention, verdict calibration,
  two-stage checks, #36/#38 benchmark decisions, workers default, slow tests). Read it before changing harness
  behaviour, and append new decisions.
- **Glossary:** `CONTEXT.md` (now also defines narrowing of retained hypotheses and "context" levers).
- **Research:** `docs/research/equivalence-check-threshold.md` (+ `equivalence_check_mc.py`, `check_rules_mc.py`).
- **How to run the benchmark:** `README.md`, and the docstring of `benchmarks/dogfood/run_benchmark.py`.
  Planted truths: `benchmarks/dogfood/expected.yaml`.

## Set up on a new device
- **Claude login for headless runs:** `claude auth login` in a terminal. The desktop app's sign-in is separate
  from the CLI's. Check with `claude auth status`, then `claude -p "reply with just OK"`. A subscription is enough;
  no API key is needed.
- **Dev venv:** `cd harness && uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -e . pytest pytest-xdist optuna torch scipy numpy`.
- **Tests (the user's laptop overheats):** `cd harness && .venv/bin/python -m pytest -q -n 4 <files>`, never `-n auto`.
  Multi-seed rate tests are `@pytest.mark.slow` and skipped by default. Run only the fast tests while iterating,
  and the slow ones (`--slow`, or `-m slow` for just them) once before merging. The fast suite takes ~13 min at
  `-n 4`; the full suite with `--slow` takes about 10–11 min per agent run as measured (machine-dependent).
- **Typecheck:** `cd harness && uvx --quiet mypy --python-executable .venv/bin/python --ignore-missing-imports boautoresearch`.
- **CI:** `.github/workflows/dogfood.yml` runs on demand only. The `CLAUDE_CODE_OAUTH_TOKEN` repo secret is set.
  If that secret holds a token that was ever pasted into a chat, rotate it (`claude setup-token`, revoke the old one).

## What's left, in order (after the blocking bug above is fixed)
1. **Run the benchmark once, real caller (#38).**
   `python benchmarks/dogfood/run_benchmark.py --caller claude --jobs 2` (3 seeds × σ ∈ {δ/4, δ/2} scripted, plus
   1 free run). Measured: ~36 min per scripted run, 1.5–2 h for the free run; ~3 h total at 2 at a time.
   - Gate: hard invariants 6/6 (now including "no agent saw the planted truth"); each planted verdict right in at
     least 5/6; scripted runs end by exhaustion; distilled branch verified.
   - Watch the subscription **session limit**. It hit once this session; agents die with HTTP 429 and resume fine.
   - Known risk: in the one smoke run, the interplay reviewer (an LLM judgement) missed the "Clip the gradients" /
     "Raise the learning rate" interaction at registration, so that case failed and the incumbent ended +1.20 from
     the optimum. An earlier smoke passed it. If it misses in 2+ of 6, look at `agents/interplay-reviewer.md` and
     the newcomer review in `skills/start/SKILL.md`.
   - "useless → `irrelevant` or `no-improvement`": `expected.yaml` already accepts either (verdict amendment).
2. **Measure the amendment's open questions on those runs:** the false-reject rate across repeated verdict checks
   with the confirming check; whether M_u's 16×16 grid is big enough (`ponytail:` in `verdict.py`); the power cost
   of M_u. Whether `irrelevant` needs a corner-coverage pass stays open (the user accepted the stricter label instead).
3. **#35 manual check:** an interactive session asks the interview sections in order, gives a recommendation for
   each, and writes an editable `run.yaml`. The interview now recommends `workers: 1` and warns about parallel
   trials skewing runtime objectives.
4. **Two small decisions to ask the user about:**
   - The orchestrator records ~20 expected verdicts per round one at a time, because the orchestrator Bash rules
     refuse its batching (`$B` variable, heredoc in a loop, `> $F`). The shell-variable refusal wrongly says it
     "reads raw logs". Proposed fix: a batch `record expected` body plus a clearer refusal message.
   - Read-only agents can still use Read. Nothing raw was read in the #36 run, but it is allowed. Proposed: leave it.
5. **Close issues** #35, #36 and #38 once 1–3 pass, then #21. Nothing has been closed yet.
   #36's live acceptance already passed (52 min, R1→R3 plus R4 started, all checks except lens distinctness,
   which is partial: 9 mechanisms from 15 proposals in pass 1, 22/22 distinct in passes 2–3).

## Residual gaps (the user has not asked for these; build only if asked)
Each is logged in the Decision log and most carry a `ponytail:` comment.
- Parallel trials: only BO rounds run in parallel (R0, calibration, smoke, equivalence, drift and wrap-up are serial).
- A narrowed retained hypothesis's levers are still GP dimensions, so status `dimensions` can exceed `dimension_cap`.
- R1's `run.yaml` seeds aren't carried past an interrupted R1.
- An undecidable drift check defers rejects, and can undo a freeze every round at a proxy fidelity that orders
  nothing (bounded by the evidence cap). The user chose to keep this default.
- A stale ladder calibration is never re-run. An equivalence check on a rung not yet measured this epoch passes vacuously.
- Enforcement: names built inside code, scripts written outside the repo and then run, recursive reads that name
  nothing, and loading the harness in the distilled branch via a subprocess or `.pth`.
- `add-dependency`: the run venv can resolve versions other than the lock's; poetry locks against the project's
  Python constraint.

## User decisions this session (all reflected in the code and the Decision log)
- Two-stage checks at α = 5%, with defaults kept for pooling, α split, one-sided drift filter, and deferring on an
  undecidable drift check.
- Verdict intervals: "accept and amend": `irrelevant` needs evidence across the other levers' box.
- The budget is summed per-trial compute time, not elapsed time.
- `workers` defaults to 1 (sequential); parallel trials only when the user asks.
- The dogfood benchmark runs on demand, once, not on a schedule. Others should be able to run it from a clone (README).

## How this user likes to work
- One Opus subagent per unit of work (`isolation: "worktree"`), **at most 2 at a time**. Tell each agent to
  merge the integration branch (or `main`) into its worktree first, and to report its **actual** branch
  (`git branch --show-current`): agents twice named the wrong branch in their reports. Verify with
  `git log main..<branch>` before merging.
- Subagents can't wait on background agents they spawn; tell them to run review sub-agents with `run_in_background: false`.
- Commit per unit of work. Push, open PRs or merge only when asked. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Ask before posting to GitHub issues. The user approved posting the two amendments this session; that approval
  doesn't carry over.
- If the user says they need to move or go offline, stop agents (TaskStop) and resume them later with SendMessage.
  Their worktrees keep uncommitted work.
- Never use `bypassPermissions` / `--dangerously-skip-permissions`; the benchmark uses `--allowedTools`.

## Gotchas
- A hook blocks `python3 - <<EOF` heredocs; write a script file instead.
- The user's lean-ctx shell allowlist can reject odd tokens, and in agent worktrees it sometimes refuses plain
  `git`. Agents fell back to the lean-ctx `ctx_shell` tool with cwd set to the worktree.
- `ps`/`pgrep` and `claude` may need the Bash sandbox disabled (keychain, process table).
- Decision-log appends from parallel branches always conflict in `docs/agents/harness-design-notes.md`; resolve by
  keeping both sides.
- Scratch drivers from this session (`drive.py`, `drive36.py`, `measure.py`) lived in a local scratchpad and are
  not in the repo. `run_benchmark.py` covers the benchmark; write a new driver if you need a single headless
  `/boautoresearch:start` to R1.

## Suggested skills
- `diagnosing-bugs`: when a benchmark run fails a gate.
- `writing-for-agents`: when editing `skills/start/SKILL.md` or `agents/*.md` after the runs.
- `tdd`: for any harness change (Seam 1 only; assert on JSON, refusals, events, generated files).
- `code-review`: after each change, with the fixed point set to the commit before it; run its sub-agents in the foreground.
- `research`: before changing the spec over the amendment's open questions.
- `domain-modeling`: if new terms come up, to keep `CONTEXT.md` consistent.
