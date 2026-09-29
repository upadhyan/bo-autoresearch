# Handoff: BO Autoresearch after the v1 follow-ups (2026-09-29)

Repo: `upadhyan/bo-autoresearch`. `main` is at the merge of PR #41. All finished work is on `main`; nothing is
in flight. The next session's job is to **run the dogfood benchmark once with the real `claude` caller** to
confirm the harness and skills work, then tie off the loose ends below.

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

## What's left, in order
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
