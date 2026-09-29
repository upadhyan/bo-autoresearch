# BO Autoresearch — cross-ticket design notes

Shared decisions for every ticket agent. The spec (issue #21) and the ticket win over this file;
this file fills gaps the spec leaves open so tickets stay consistent. If you make a new cross-ticket
decision, APPEND it to the "Decision log" at the bottom (one line, ticket number first).

## Repo layout (as of #22)
- `harness/` — the `boautoresearch` package (pyproject, setuptools, `requires-python >=3.10`).
  - `boautoresearch/__init__.py` — the six runner-side names ONLY. Stdlib only (trials import it).
  - `boautoresearch/cli.py` — argparse CLI, JSON out, `Refused` → `{"refused": true, "reason": ...}` exit 1.
  - `boautoresearch/experiment_log.py` — events table (append-only triggers), `append`, `read`, `state` fold.
    (Named `experiment_log`, NOT `log`: a submodule named `log` would shadow the runner-side `log()`.)
  - Split more internal modules as needed (e.g. `trials.py`, `study.py`, `verdict.py`, `records.py`,
    `checks.py`, `reports.py`, `gitops.py`), but the public surface stays: six names + CLI.
- `harness/tests/` — Seam 1 tests. `conftest.py` has `bo(cwd, *args, env=None) -> (code, json)`,
  `events(run_dir)`, `git(...)`, fixtures `project_python`, `repo`, `run_yaml`, `run_dir`.
  `tests/toy_project/` is the toy (train.py, runner.py, levers.json). Add more toys as needed
  (toys with planted truth, instant trials). Tests assert only on JSON, refusals, events, generated files.
- Plugin shell (from #34–#36): repo root `.claude-plugin/plugin.json`, `skills/`, `agents/`, `hooks/`.
- Dogfood (#38): `benchmarks/dogfood/`.

## Testing policy (from #30 on — user decision)
- Do NOT run the whole suite per ticket. Run only the tests the change can affect: the ticket's new
  test file(s) plus the existing files whose behaviour you touched (e.g. `tests/test_scheduler.py
  tests/test_rounds.py`). Pick them by what your diff changes; say which you ran in your report.
- Run tests in parallel: `cd harness && .venv/bin/python -m pytest -q -n auto <files>` (pytest-xdist is
  installed). Tests must stay independent (own tmp repo each) so `-n auto` is safe.

## Dev commands
- Dev venv: `harness/.venv` (Python 3.10, editable harness + pytest; optuna 5.0, torch, scipy, numpy
  already installed). Run: `cd harness && .venv/bin/python -m pytest -q tests/<file>`.
- Typecheck: `cd harness && uvx --quiet mypy --python-executable .venv/bin/python --ignore-missing-imports boautoresearch`.
- Shell note: `python3 - <<EOF` heredocs are blocked by a hook; write a script file and run it instead,
  or use the Edit tool.
- Commit each ticket on `main` (don't push). End commit messages with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Reference the ticket, e.g. "(#23)".

## Principles (decisions, not suggestions)
- Everything trustworthy (statistics, verdicts, seeds, budget, stopping, enforcement) is in the harness.
  Invalid actions are refused with a JSON reason. Hooks are thin fail-closed adapters calling
  `boautoresearch check …`. The LLM only makes judgement calls; `next` + refusals force them.
- The experiment log is the single source of truth. State and EVERY generated file (SUMMARY.md, rounds/,
  CSVs, REPORT.md) must be rebuildable byte for byte from events (no wall-clock or randomness at render time;
  timestamps come from event `ts`). The Optuna study is disposable (`studies.db`, rebuilt each round).
- Deep module, small interface. No config/seams the ticket doesn't need. Stdlib + chosen deps.
- Glossary terms from CONTEXT.md exactly (lever, trial, round, verdict record, epoch, baseline, replicate,
  fidelity, reference fidelity, calibration round, verdict check, stall, budget, warm start, ...).
- Mark deliberate shortcuts with `# ponytail: <ceiling>, <upgrade path>`.
- Tests: TDD vertical slices at Seam 1 (the CLI). Expected values from independent truth (planted toys),
  never recomputed the way the code does. Statistical tests fix seeds and assert rates over repeats.

## Dependencies
- Runtime deps in pyproject: `pyyaml`, and from #25: `optuna`, `numpy`, `scipy`, `torch` (GPSampler needs
  torch + scipy). The verdict GP (#26) is our own numpy/scipy GP outside Optuna.
- Run venv creation (init): prefer `uv` when on PATH (`uv venv --python <py>`, `uv pip freeze --python <py>
  --exclude-editable`, `uv pip install --python <venv py> -r freeze <harness>`), fall back to stdlib venv +
  pip. This keeps test inits fast even with torch. (Decide/implement when adding deps in #25.)

## run.yaml (grows per ticket; unknown keys refused)
Required: `objective` (name), `direction` (minimize|maximize), `budget_s`.
Optional: `runner` (runner.py), `python`, `reference_fidelity` ({}), `ladder` (list of cheaper fidelity
dicts), `deterministic` (false), `replicates_k` (5), `workers` (1), `target`, `max_trials`, `checkpoint`
(false = autonomous; true = pause at each round boundary), `delta` (headless), `seeds` (≤5 configs over
lever names; round-1 agent-chosen trials), `brief` {purpose, contribution, complexity, provenance},
`directives` [...], `protected_paths` [globs, added to defaults], `lenses` [...], `generation`
(llm|scripted), `fixtures` (dir of scripted passes: pass1.yaml, pass2.yaml, ...), `go` (headless).

## Ids
- trial: global integer `trial` 1,2,3… (artifacts/trial-<n>/). Hypothesis `H<n>.v<k>`. Round `R<n>`
  (R0 = calibration round). Verdict record `V-R<n>-H<n>.v<k>-<check>` (check = that hypothesis's check
  counter within the round, from 1).
- Lever names are global and prefixed by hypothesis NUMBER (not version): spec says `warmup_frac`, the
  harness names it `H3.warmup_frac`. Code calls `lever("H3.warmup_frac")`. Revived versions reuse the same
  lever names (code stays in place).

## Hypothesis spec (propose payload / proposal record items)
```
title (≤80), rationale, mechanism, provenance (novel|adapted|standard; adapted needs `source`), lens,
directives: [ids], fidelity_sensitive: bool, fidelity_reason (required if true),
levers: {name: {kind: float|int (low, high, log?) | categorical (options) | bool (why_not_graded),
                baseline (inside range), predicted: higher|lower (graded levers only),
                path?: config path this lever controls (for conflict + predicate checks)}},
masked_by?: {H<m>: reason}, exclusive_with?: [H<m>], merges?: {from: [ids], mapping: {...}} (#32)
```
`predicted` = on which side of the baseline the improving lever values lie. Contradiction (flag
`retained-against-prediction`) = the best posterior point's lever value is on the other side.

## Actors
`--actor` default `orchestrator`; `harness` is reserved (refused). Subagents pass their role, e.g.
`registration-reviewer`, and `record` also takes `--agent-id`. Harness-computed events use actor `harness`.
Every action needs `--rationale` (stored in its first event payload). Probes are never logged.

## Calibration round (R0) ordering (#23)
- σ needs only baseline replicates; ladder calibration needs diverse configs, which need levers.
- `round-run` when R0 is missing runs R0: smoke (baseline trial) + k baseline replicates at the
  reference fidelity and each rung → per-rung σ + cost. Log `noise_estimate`.
- Ladder calibration (≈5 diverse configs per rung + reference, Spearman ρ ≥ 0.8, spread ≥ 3σ, cheapest
  expected cost to a verdict) runs as part of R0 as soon as registered hypotheses with committed code exist
  (i.e. at the start of the first `round-run` that would begin R1), before R1. Configs = quasi-random
  points over those levers. Log `fidelity_calibration` with the chosen rung. No ladder → nothing to calibrate.
- If none passes: fall back to reference, flag it. `accept-proxy --fidelity '<json>'` (action) logs
  `proxy_accepted_unvalidated` only when requested.
- δ is set by `set-delta <value>` (action, once; refused afterwards) or `delta:` in run.yaml (headless).
  Suggest 2σ with trials-needed estimate (cost ∝ σ²/δ²) and an infeasibility warning.
- Deterministic declaration, or all k replicates agreeing within float tolerance → σ=0, replicates off.

## Budget
Ledger = summed wall-clock of research-loop trials (calibration onward; smoke counts), computed from the
log; abandoned trials charged up to last heartbeat. A trial is refused (logged, e.g. `trial_refused`) when
its estimated cost (mean wall-clock at that fidelity so far; unknown → 0) exceeds the remaining budget.
Wrap-up trials (#37) are NOT charged. No reserve.

## Rounds (#25–#30)
- "In search" hypotheses = active ∪ retained (retained stay in the search; re-judged every round).
- The harness selects slots deterministically from state (priority desc, then registration order), within a
  dimension cap from the budget, co-placing interaction-flagged pairs, never co-selecting `exclusive_with`.
  Selection is computable before `round-run` so `next` can demand lever code for selected hypotheses.
- `round_started` payload: round, commit, fidelity, epoch, search space, hypotheses, pid.
  A round with `round_started` but no `round_ended` whose pid is dead is a crashed round: on the next
  command, abandoned trials get `trial_abandoned`, the round ends with trigger `interrupted`.
- Trial kinds (`chosen_by`/`kind`): smoke, calibration, baseline, seed, sampler, agent, replicate,
  confirmation, drift, equivalence, wrapup. Replicates carry `replicate_of`. Masked levers log
  `sampled` and effective `levers`.
- Fresh sampler trials = sampler trials in the current epoch and fidelity since the hypothesis entered the
  search (or was revived). Burn-in / confirmation / check spacing / evidence cap count only these.
- Agent-chosen trial cap per round = max(1, 6 − R) (round-1 seeds count against it).
- Replicate share ≥ ~10% (raised by escalation), picked from top-ranked evaluated configs; each new
  incumbent gets 2 confirmation replicates. Confirmed incumbent = best config (mean over it and its
  replicates) with ≥ 2 replicates.
- Proxy drift check at round end (only when the round's fidelity is a proxy): incumbent + one random
  config at the reference fidelity; if their order flips vs the proxy, the proxy is broken: that round's
  confirmed rejects are logged as inconclusive instead of rejected (rejects at a proxy are finalised only
  after the drift check).

## Directives (#31)
- Predicate = a boolean expression (restricted AST: comparisons, and/or/not, arithmetic, `in`, names,
  constants) over lever names and lever `path`s, stating the ALLOWED region. A predicate referencing names
  absent from a config doesn't apply to it. Registration checks the hypothesis's lever box on a grid
  (endpoints + interior points; all categorical options) with other names at their baselines.
- `compat:<id>` for a discouraged directive = its predicate if given, else "every lever of hypotheses that
  declare this directive id is at baseline".
- Protected-path manifest: sha256 of worktree files matching the protected globs (defaults: runner,
  levers.json, plus run.yaml `protected_paths`), recorded at init, re-checked before every round and commit.

## Records (#27)
`record <kind> --file f.json --agent-id <id> --actor <role>`. Kinds: proposal, review, interplay,
narrative, expected, distill_spec. Hand-written validators naming the failing field. Quote check: records
carry structured `quotes: [{record, field, value}]`, each must equal the verdict record's field exactly,
and every decimal number in the record's text fields must equal one of its quote values.

## Hooks (#34)
Claude Code hook formats: see `docs/research/claude-code-plugin-anatomy.md` on branch
`research/claude-code-plugin-anatomy` (`git show research/claude-code-plugin-anatomy:docs/research/claude-code-plugin-anatomy.md`).
Adapters call the run venv's `boautoresearch check …`. `check bash` may return an `updated_command` (e.g.
to append `--agent-id` to a subagent's `record` call) so no logic lives in hooks.

## Decision log (append here)
- #22: run dirs `.bo-research/<YYYYMMDD-HHMMSS>/` (latest wins); `.bo-research/` added to
  `.git/info/exclude`; `smoke` with no hypothesis = one baseline trial at the reference fidelity; baseline
  trial levers = committed worktree `levers.json`; state is an in-memory fold (no state tables yet).
- #23: R0 is `round-run` while R0 is incomplete (round 0 events `round_started`/`round_ended`, trigger
  `calibrated|smoke_failed|failed|budget_spent`); once R0 is complete `round-run` refuses ("no hypothesis
  is registered") until #25 wires R1. R0 runs the WHOLE calibration (smoke if none finished, k baseline
  replicates per fidelity → `noise_estimate`, then ladder calibration → `fidelity_calibration`) in one go,
  because #15 shows calibration results before δ. Ladder configs: 5-point centred Latin hypercube (stratum midpoints, shuffled per lever) over a box
  baseline ± max(|b|,1)/2 derived from worktree levers.json (ponytail in calibration.configs) — #24/#25 should
  swap in registered hypotheses' lever boxes. Rung choice: expected cost = mean cost_s × trials needed,
  n = max(1, 15.68·σ²/effect²) (1 if σ=0), effect = 2σ_ref (suggested δ; δ unset in R0) scaled by
  spread_rung/spread_ref; reference is always a candidate and wins ties; `fallback` = ladder given and no rung
  passed. `accept-proxy --fidelity` only after R0, only for a ladder rung that FAILED calibration.
  State fold gains `rounds`, `noise`, `calibration`, `r0_complete`, `fidelity` {fidelity, proxy:
  reference|validated|unvalidated, fallback?}; status exposes them + `sigma`, `replication`.
  Trials carry `kind` (smoke|baseline|calibration), R0 trials carry `round: 0` (calibration also `config`);
  harness-scheduled trials are logged with actor `harness`. Budget gate lives in cli._trial (every trial goes
  through it): refused + `trial_refused` {kind, fidelity, estimated_cost_s, remaining_s} when remaining ≤ 0 or
  estimate (mean wall-clock at that fidelity, 0 unknown) > remaining. run.yaml: `ladder`, `deterministic`,
  `replicates_k` (≥2). `set-delta` NOT built (not in #23's ticket) — left for whoever needs the R1 gate.
- #23 (review): a rerun of R0 after a stop reuses the logged `noise_estimate` (no replicates paid twice) and
  skips smoke if one finished. Budget refusals raise `BudgetRefused(Refused)` → R0 trigger `budget_spent`;
  any other error → `failed`. Toys may read `kind` from the BOAUTORESEARCH_TRIAL file to plant
  per-kind truth (tests/toy_project/train.py TOY_CHEAP_REPLICATE_SIGMA).
- #24: `propose --file spec.json` (JSON only) validates (hypotheses.validate; unknown fields refused, so #29/#32
  add masked_by/exclusive_with/merges to SPEC_KEYS) and logs `hypothesis_proposed` {id, number, version, spec with
  prefixed lever names}. `register <H>` (H = `H<n>.v<k>` or `H<n>` = latest version) logs `hypothesis_registered`;
  the reviewer-record / allowed-region refusals go where cmd_register has the "#27/#31 add here" comment. Graded =
  float|int; categorical/bool take no `predicted`. `smoke <H>` (registered, uncommitted only): baseline then random
  point (stops at the first failure) at the cheapest rung = argmin measured mean cost over reference+ladder (refused
  with a ladder until R0 is complete); trials are kind smoke with `hypothesis`, `point`, `tree`; then
  `lever_smoke` {id, tree, fidelity, trials, passed}. `tree` = git write-tree of the whole worktree via a temp
  index (untracked incl., ignored excl.). `commit-lever <H>`: latest smoke passed AND same tree, then static check
  on changed .py files (config reads counted by AST before vs after: environ/getenv/argv, argparse-like imports,
  'levers.json'/'trial.json'/BOAUTORESEARCH_TRIAL strings; every lever of H read as lever("literal") in the changed
  files; reads of undeclared names refused), writes levers.json = baselines of all committed hypotheses + H, commits
  as author `boautoresearch` with --no-verify, logs `lever_committed` {id, commit, levers}. Revivals (same lever
  names, code already in place) will need the "every lever read" check widened beyond changed files.
  R0's smoke ignores smoke <H> trials. Trials run with PYTHONDONTWRITEBYTECODE=1 so the worktree stays clean.
  `round-run` refuses a dirty worktree (R0 too); round_started.commit is the pin. status gains `hypotheses`.
  NOT wired: ladder-calibration boxes. JIT code exists only after R0 (smoke <H> needs R0's rung costs), so the
  lever boxes can't feed calibration until #25 moves ladder calibration to the start of the first R1 round-run.
- #24 (review): propose numbers past H<n> prefixes already in levers.json; commit-lever writes levers.json = worktree levers.json (project keys kept) + H baselines; smoke <H> fails (lever_smoke.reason) if its trials touch any non-ignored worktree file (mtime stamp); random smoke point never draws a categorical/bool baseline; smoke <H> trials keep commit=HEAD plus `tree` (what actually ran).
- #25: deps optuna/numpy/scipy/torch; init uses uv when on PATH (`uv venv`, `uv pip freeze --exclude-editable`
  minus pip/setuptools/wheel like pip's freeze, `uv pip install`), else stdlib venv + pip. `study.py` holds THE
  eligibility function `eligible(t, space, baseline, fidelity, epoch=0)` (finished, same fidelity, epoch (missing = 0),
  lever keys exactly baseline ∪ space, searched levers in range, others == baseline; #28 extends it), the `Study`
  wrapper (studies.db deleted and rebuilt from eligible trials every round via add_trials; GPSampler
  deterministic_objective=False; user_attr `trial` = log trial id) and `ranked`/`incumbent`. Selection (#29 replaces):
  every registered + active hypothesis; round-run refuses while any lacks a commit; registered ones get
  `hypothesis_activated` {id, round} at round start (status active, `activated_round`). Ladder calibration moved to the
  first post-R0 round-run (trials round 0 kind calibration, LHS over the selected levers' boxes, others at baseline);
  if δ is unset it then returns {round: 0, fidelity_calibration, next} (exit 0) so calibration is shown before δ.
  `set-delta <v>` (after R0, once; `delta_set` {delta, suggested}) or run.yaml `delta:`; round-run refuses R1 without δ
  (suggested 2σ in reason, status `suggested_delta`). status gains `delta`, `suggested_delta`, `run_ended`, `next`,
  trials.abandoned. round_started (R1+) payload: round, commit, fidelity, epoch 0, search_space, hypotheses, pid, cap_s,
  seeded (eligible trial ids). Trial kinds sampler | confirmation | replicate (both with `replicate_of`) | drift
  (`drift_of`); all trial_started carry `pid`. Loop: cap (round spend + est > 0.25·remaining-at-start, ≥1 trial) → stall
  (≥ max(5·d_total,20) fresh sampler trials since the confirmed incumbent last gained ≥ δ, no pending confirmation, all
  selected past burn-in max(10·d,20)) → owed confirmations (2 per sampler trial beating the confirmed incumbent's mean,
  or while none is confirmed) → replicate floor (10·replicates < round trials: least-replicated of the top 3 configs) →
  sampler. Incumbent = best mean config with ≥2 replicates (0 when replication off), reported with confirmed flag.
  Drift (round fidelity ≠ reference): incumbent + random other-levers config of the round at the reference →
  `drift_check` {round, configs, trials, proxy, reference, broken}; no downgrade (#26). round_ended {round, trigger,
  incumbent}; budget refusal → trigger budget_spent + `run_ended` {reason: budget_spent} (during post-R0 calibration only
  run_ended); round-run refused after run_ended. Heartbeats: thread appends `trial_heartbeat` {trial, elapsed_s} at
  1,2,4…60 s. Actions (not probes) first recover: running trial with dead pid → `trial_abandoned` {wall_clock_s = last
  heartbeat}, open round with dead pid → round_ended `interrupted`. commit-lever's "declared" now includes the
  project's own levers.json names. `workers` not built (ponytail in _run_round).
- #25 (review): owed confirmations and stall progress are replayed from the log each step (cli._progress): a finished
  sampler trial beating the confirmed incumbent's mean (or none confirmed) owes 2 confirmations; a confirmation counts
  once started (abandoned = paid, never retried). The stall window opens at the round, extended back over immediately
  preceding `interrupted` rounds (it still restarts per completed round; ponytail). run.yaml `seed` (int, optional; logged
  in run_started) roots every harness RNG stream (`_rng(run, *path)`), else the run id: statistical tests fix it. Runners
  start in their own process group; a `trial_heartbeat` at spawn (elapsed 0) and every beat carry `runner_pid`; recovery
  killpg's it; beats at 0,1,2,4,8 s then every 10 s (a kill is charged ≥ half its time). Missing pids read as dead.
  σ = 0 → `suggested_delta` None and the set-delta duty asks for the smallest effect worth having. Cap check reserves
  2 × mean reference cost for a proxy round's drift check; the first trial of a round always runs (only the budget
  refuses it). Drift `broken` = the reference reverses the proxy order by more than margin 2√2·σ_ref (logged `margin`).
  σ re-estimated each round (replication on): `noise_estimate` {round r, fidelity, sigma pooled within the round's
  replicated configs (None if df < 2), df, r0_sigma, shift, shift_flagged (outside [0.5, 2])}; the fold keeps R0's as
  `noise`, later ones on `rounds[r]["noise"]`; round-run returns it. Not logged on a budget_spent round end. Acting on
  the flag (escalation) is #26's. `elog.load(con)`, `hypotheses.options(lv)` helpers.
- #26: `verdict.py` = own numpy/scipy GP (Matern-5/2 ARD on [0,1] inputs, log levers in log space, categoricals by
  option index; y oriented larger=better, standardised; noise floor = R0 σ² at the round fidelity; LML by L-BFGS-B,
  3 restarts, seeded from `_rng(run, "verdict", <record id>)`). √V_T = Jansen on 256 joint posterior samples at 128
  Sobol A and A_B^G points over the in-search box; Δ = max over (128 Sobol + eligible trial points + baseline) with
  the group free − the same set with the group at baseline; bounds = 5%/95% sample quantiles; per-lever √V_T for
  freezing when d ≥ 2. Gates: trials (fresh sampler ≥ burn-in, agent count reported), coverage (fresh sampler trials
  reach the bottom and the top third of each graded lever's range, every option of a categorical — NOT every third:
  BO exploits, a lever whose best is at a range end never fills the middle third), fit (LOO mean z² ≤ 2),
  homogeneity (RMS of the STANDARDISED LOO residuals in the two halves of each lever's range, both ≥ 8 trials; ratio ≥ 3
  fails; raw residuals misread a steep, sparse region's misfit as noise on a quiet bowl).
  Verdict checks run at the top of every round-loop step: due when fresh sampler trials (eligible, kind sampler,
  round ≥ activated_round) reach burn-in max(10d,20), then last check's fresh + max(5d,10); "last" = the latest record
  with the same fidelity and group (a freeze or rung change restarts the schedule). Outcomes: retained | active |
  pending-reject | reject | inconclusive. Homogeneity failure → inconclusive (terminal, `hypothesis_inconclusive`
  reason "noise differs by region"); fidelity-sensitive at a proxy with a condition → active + `held`; active at
  fresh ≥ max(40d,80) → inconclusive (evidence cap). A reject = a pending-reject confirmed by the next check; it is
  finalised at round end after the drift check (`hypothesis_rejected` {id, verdict, condition, frozen: group for
  no-improvement}); at a proxy with drift broken or missing → `hypothesis_inconclusive` reason "broken proxy" |
  "proxy unchecked". `lever_frozen` {id, lever, verdict} when a d ≥ 2 group has some (not all) levers with own upper
  bound < δ; frozen levers leave `_space` and run at baseline (trials with them off baseline become ineligible until
  #28 drops their keys). A confirmed reject / freeze / inconclusive ends the round, trigger `search_space`; a pending
  reject blocks the stall. Retained stays selected (`_selected` includes retained); a later non-retain check sets it
  back to active. Escalation at a stall end for each hypothesis whose latest record is `active`: first
  `hypothesis_escalated` {step replicates, replicate_share 0.3} (the round's replicate floor becomes 30% if any
  selected hypothesis has it), then {step rung, fidelity, proxy} = cheapest passed rung dearer than the current one,
  else the reference (fold: overrides st["fidelity"]); no rung step at the reference. Record `V-R<r>-H<n>.v<k>-<check>`
  fields: id, hypothesis, round, check, group, outcome, condition, held, reason, sqrt_vt/delta_stat {estimate, lower,
  upper}, levers (per-lever √V_T), sobol_index (normalised, telemetry), best_point, gp, delta, gates, burn_in {fresh,
  needed, spacing, cap, passed}, confirmation {pending, confirms, due_at}, fidelity, proxy, prediction {flag,
  contradicted}, frozen, trials, context {retained, co_active}. Probe `verdict <H>` → {hypothesis, status, records,
  verdict (latest outcome | burn-in), burn_in}. round-run returns `verdicts` (the round's records). `sensitivity`
  not built. Note: √V_T of a linear effect over the box is range/√12, so a lever needs a spread ≳ 3.5δ to be retained
  and one with Δ up to ~3.5δ can be rejected `irrelevant` — the spec's form, flagged for #38's planted effects.
- #26: #25's killed-round test now ends R2 by search_space (δ=10 makes the bowl irrelevant); it asserts burn-in spans the interrupted round instead of the stall window.
- #26 (review): finalised-at-proxy reasons are "broken proxy fidelity" | "proxy fidelity unchecked"; held text "fidelity-
  sensitive: never rejected at a proxy fidelity" (CONTEXT: no "proxy" as a noun). A stall escalation moves the run up
  at most ONE rung (computed once; every stuck hypothesis's rung event names it). The fold lets a later
  fidelity_calibration / proxy_accepted_unvalidated override an escalated fidelity. Δ's free max is over C ∪ C_base
  (never below the baseline-group max). The round loop reloads state after checks, so a just-pending reject holds
  the stall in the same step (#25's no-stall-before-burn-in test now runs 30 + 15 samplers to the confirmed reject).
  Kept (known ceilings): homogeneity failure is terminal inconclusive at one check over every in-search lever; the
  evidence cap counts fresh trials at the current fidelity/group (a rung change or freeze restarts it); freezes at a
  broken proxy fidelity are not undone.
- #27: `records.py` holds the schemas (`KINDS`, `validate(kind, body, st)`); `record <kind> --file --agent-id`
  (agent id optional: the orchestrator's own `expected` records log `agent_id: null`) logs ONE `record` event
  {rationale, kind, agent_id, record: body[, hypotheses: ids]} with the caller's actor; the fold keeps them in
  `st["records"]` (+ actor, seq). Unknown fields refused per kind. Schemas: proposal {hypotheses: [specs]} (#24
  validation, ≥2 distinct mechanism texts; each spec also logs `hypothesis_proposed` with the record's actor);
  review {hypothesis, directive_verdict allow|prune|deprioritize, directive (id, required unless allow),
  intent fits|stretch|off-intent, conflict shared-lever|exclusive|rival|none, conflict_with [H ids] (empty iff
  none), rationale, strict bool}; interplay {removed XOR newcomer, flags [{partner, reason, cites (≥1 verdict
  id)}], quotes?}; narrative {round (logged), text, cites, quotes?, diagnostics [], suggestions [], generation
  bool}; expected {hypothesis, verdict retain|reject|undecided, reason (one line), quotes?}; distill_spec
  {content, cites, quotes?}. Directive ids aren't checked against a registry (#31 may). Quote check: quote
  `field` is a dotted path into the verdict record (longest key wins, so `best_point.H1.x` works); value must
  == the record's (bool never equals a number). Decimals (`\d+\.\d+`, optional exponent/sign) in the text fields
  of interplay/narrative/expected/distill_spec must equal a quote value ROUNDED TO THE DIGITS SHOWN
  (|value − text| ≤ half a unit in the last shown place); integers are not checked; proposal/review prose is
  not checked. `check recorded <agent_id>` (probe) → {"recorded": bool}. No gates on these records yet.
- #27 (review): a quote's record must be among the output's cites (interplay: any flag's cites; expected: any
  record); a narrative must cite ≥1 verdict record when its round has any; interplay `removed` must be
  rejected|inconclusive|parked with partners proposed|registered (untested), `newcomer` the reverse; a review
  can't conflict with itself; wrong-typed ids are refused, not crashes. Decimal regex also catches `.5`;
  version-like text ("3.11") reads as a decimal (ponytail in records.DECIMAL).
- #28: `study.eligible(t, space, baseline, fidelity, epoch, dropped, merges)` now RETURNS the trial rewritten into the
  round's levers (or None); every caller uses the mapped trials (cli._eligibility builds dropped/merges from state;
  running a config from a mapped trial fills missing keys from levers.json). Rules: merge (a trial that ran the merged
  lever, not the new one, while the merged lever is out of the search: `values` [[old,new]] pairs, none = identity,
  unmapped value → ineligible; the old key is consumed) → drop keys of `dropped` levers out of the search (rejected
  `irrelevant` hypotheses' levers + every `lever_frozen`) → backfill missing levers at their baseline (levers.json, else
  the spec's) → searched levers in range (never clipped), others == baseline (no-improvement/parked/masked/inconclusive
  = filter). R0 baseline replicates therefore now seed R1. Probe `trials --eligible` = the next round's warm start.
  Merge declaration (minimal, #32 registers the flow): spec `merges: {from: [H<n>.v<k>], mapping: {own lever: {lever:
  "H<m>.<name>", values?}}}`, mapping keys prefixed at propose. `in_range` moved to hypotheses (study re-exports).
- #28: epochs. Trials carry `epoch`; fold `st["epoch"] = {epoch, round}` (round = first round of the epoch);
  round_started.epoch real; `_latest` ignores verdict records from before the epoch's first round (schedule/burn-in
  restart). `commit-change --reason [--breaking]` (dirty worktree, all changes, refused mid-round / touching runner or
  levers.json — #31 adds the globs + manifest in cli._change_allowed) logs `commit_change` {reason, commit, paths,
  breaking}. `add-dependency <req>` (clean worktree; no leading "-"; `uv pip install --python <run venv>` else venv pip)
  writes the run venv's freeze minus the harness to worktree `requirements-freeze.txt` (ponytail: not the project's own
  lock file), commits, logs `dependency_added` {requirement, commit, freeze}. DECISION: commit-lever ALSO runs the check
  (spec: "after any code change"; #9: a baseline that isn't a no-op is caught) — test toys' lever code must be true
  no-ops at baseline (test_rounds BOWL now `- 0.08`). All three then `_after_change`: only once R0 is complete
  (before, nothing is measured: no check, no epoch). Check = `equivalence_check` {commit, change, fidelity, sigma,
  configs [{config baseline|incumbent, levers, logged_mean (mean of eligible trials at that config, current epoch,
  in-search space), logged_n, trials, mean (of the 2), tolerance 2σ + 1e-9·max(1,|mean|), passed}], passed, refused};
  trial kind `equivalence` (`equivalence_of`); incumbent skipped when it is the baseline; nothing logged at a config →
  it passes; a budget refusal → passed None (no epoch). Fail or --breaking → `epoch_started` {epoch, round, commit,
  change, breaking}, then k baseline replicates (1 if deterministic) in the new epoch → `noise_estimate` {epoch,
  fidelity, sigma (None if <2 finished), n, trials, mean, previous_sigma}; the fold replaces that rung's σ/mean in
  st["noise"]. NOTE: 2σ on 2 replicates fails a true no-op ~15% of the time with R0's df = 4 σ̂ (~28% at k = 3; ~4.5%
  only with a known σ; docs/research/equivalence-check-threshold.md; replaced by the two-stage rule, "#21 equivalence" below);
  tests now default to `seed: 0` (conftest.write_run_yaml) so runs replay identically.
- #28: `narrow <H> --lever <name> --low/--high` (graded; ints must be integral) or `--options '<json list>'`
  (categorical, ≥2, proper subset); refused mid-round (it takes effect at the next round boundary — rounds only run
  synchronously, so "ends the round" is automatic), unless H is active|retained, the lever is in its unfrozen group,
  and its latest verdict record (`_latest`) has every gate passed; the new range must be strictly smaller, inside,
  and keep the baseline. Logs `narrowed` {id, lever, before, after, verdict}; the fold rewrites h.spec.levers[lever].
- #28 (review): `narrowed` also carries `round` (the next round); the fold keeps `narrowed_round`, and `_latest`
  ignores records before it (the verdict schedule restarts on the narrowed ranges, like a freeze); narrow's gate reads
  the latest record ignoring that cut, but refuses once a round has run since the last narrowing without a new record
  (several levers may be narrowed at one boundary). The incumbent's equivalence trials carry `replicate_of` (they
  count as its replicates). Only each hypothesis number's latest version decides credibly-irrelevant levers (a
  revived version rejected no-improvement filters again). `study.eligible` args are all required; param renamed
  `irrelevant` (CONTEXT avoids "dropped"). Kept: retained status carries into a new epoch until its next check; σ
  re-measured at the current fidelity only (ponytail in cli._new_epoch); merge value match is type-strict like
  categoricals.
- #29: selection `cli._selected(st)` = in-search (active ∪ retained, always kept) + the registered queue sorted by
  (−priority, registration seq) while dims fit the dimension cap = largest d with max(10d, 20) ≤ affordable,
  affordable = 0.25·remaining / mean cost at the current fidelity (None = uncapped while cost unknown; 0 when
  affordable < 20). Strict priority: the first queued hypothesis that doesn't fit stops the fill (a lower priority
  never jumps it), except an empty round always takes the first. `exclusive_with: [H<n>|H<n>.v<k>]` (spec key, by
  number, symmetric) is skipped, not a stop. Interaction co-placement: hook comment for #30; retained-narrowing on a
  tight cap: ponytail. `prioritize <H> <int>` → `hypothesis_prioritized` {id, priority} (default 0). status gains
  `schedule` {selected, queue, dimension_cap, dimensions, expected_missing, agent_trials {round, cap, used}, starved
  [{id, rounds}]} and `paused`; hypotheses carry `priority`. `next` demands code only for selected hypotheses.
- #29: `park <H> --reason` (between rounds; proposed|registered|active|retained) → `hypothesis_parked` {id, reason,
  from}; `unpark <H> --reason` → `hypothesis_unparked`, back to registered (proposed if parked from proposed), same
  id/version/commit. Parked levers are filtered to baseline (#28 default). `_latest` now also ignores verdict records
  from before `activated_round`, so a returning hypothesis's schedule/burn-in restarts (no carried pending reject).
  Starvation: fold counts BO rounds (R≥1) a registered hypothesis was not in round_started.hypotheses
  (`unselected`, reset on unpark); flagged in status.schedule.starved at STARVED_AFTER = 3 (a flag, not a duty).
- #29: expected verdicts: fold gives every `record` its `round` = the next round to start (max(rounds)+1);
  round-run (R1+) is refused, and `next` lists "record expected <H> …", until every SELECTED hypothesis (those the
  upcoming round tests, newcomers included) has an `expected` record for that round. Tests use conftest
  `round_run`/`expect_all` (records `undecided`) and `ready(out)`.
- #29: `enqueue --config '<json>' --expected <float>` (after R0; config keys ⊆ the selection's space, in range) →
  `trial_enqueued` {queued n, round, config, expected}; cap per round max(1, 6−R), run.yaml `seeds` (≤5 configs,
  lever names as the harness names them) count against R1. At round start seeds (R1 only) then that round's queue
  run first, kinds `seed` / `agent` (agent carries `expected`, `queued`), re-bounds-checked: a failing one logs
  `agent_trial_skipped` {round, kind, config, reason}. Agent/seed trials never count as fresh; a new incumbent from
  one is owed confirmations like a sampler trial's.
- #29: run control: `stop` → `run_ended` {reason user_stop} by the caller; a running round (background round-run)
  sees it at its next step → round_ended `user_stop` (no drift check). `checkpoint` → `user_pause`; running round
  ends `checkpoint`; `checkpoint --resume` → `user_resume`; round-run refused while paused. run.yaml
  `checkpoint: true` → harness `checkpoint` {round} after each R1+ round end (not after R0; not when the run ended).
  #31's directive revisions hook in cmd_checkpoint. run.yaml `target` (number): ends the round (trigger target) and
  the run (`run_ended` {reason target_reached, incumbent}) once the CONFIRMED incumbent's mean (≥2 replicates) is at
  or past it, at the reference fidelity only (ponytail). `max_trials`: `_trial` refuses trial n > max_trials
  (`trial_refused` {max_trials}) via TrialRefused(reason="max_trials") (was BudgetRefused) → round trigger and run_ended reason max_trials.
- #29 (review): `BudgetRefused` renamed `TrialRefused` (reason budget_spent | max_trials: CONTEXT keeps "budget" for
  compute time); `agent_trial_dropped` renamed `agent_trial_skipped` (CONTEXT avoids "dropped"). A stop/pause is
  also seen before each agent-chosen trial; a stopped run skips the drift check; `_end_run` and the target end never
  log a second `run_ended` over a user stop. round-run's expected-verdict refusal now comes before ladder
  calibration (no trials spent on a refused call). park/unpark/prioritize are refused once the run has ended
  (`cli._not_ended`). Kept: in-search dims can exceed the cap (retained narrowing is ponytail); the starvation flag
  lives in status.schedule only; the selection can shift as measured costs move the cap (refusals catch it).
- #30: removals. Every hypothesis_rejected / hypothesis_inconclusive / hypothesis_parked is followed by a harness
  `removal` event {id, removal rejected|inconclusive|parked, reason, verdict (last record id | None), verdicts [ids],
  context (last record's), untested [proposed|registered ids]}; fold `st["removals"]` (+seq). Interplay reviews owed
  (`cli._interplay_missing`, status `interplay_missing` [{removed}|{newcomer}]): each removal needs an interplay record
  `removed: id` logged after it; each queued hypothesis (not a revival) needs a `newcomer: id` record after the latest
  removal that predates its queueing (registered/unparked seq, fold `queued`) and whose `untested` list lacked it (so
  a queued hypothesis the removal review already weighed owes nothing). Both are `next` duties and round-run (R1+)
  refusals. Parks and flags without verdict records: a flag must cite ≥1 verdict record (#27), so an untested park
  can only be reviewed with flags [].
- #30: revivals. Links = interplay flags (removed, partner), by hypothesis NUMBER pair; revived at most once per pair.
  At round-run (after the review and generation gates, before code/expected checks), each link whose removed
  number's latest version is removed and whose partner is registered and in `_selected` logs `hypothesis_revived`
  {id H<n>.v<k+1>, number, version, from, partner, spec = the removed version's `origin` (proposed/revived spec,
  narrowing undone), commit = its commit (code in place; None → the coder writes it), round}; round-run then RETURNS
  {revived, status} exit 0 without running (like the calibration-before-δ return), so the expected verdict (and
  #31's review) for the revived version land in `next` first. DECISION: no review of its own in #30; #31's register
  review gate should add revived versions to round-run's ready checks (they are logged before those checks).
  Revived versions register at once (status registered, `revived_from`, `partner`), priority 0; `unpark` of an older
  version is refused. Co-placement: `_selected` takes a queued hypothesis together with every queued one linked to
  it (a unit fits the cap or stops the fill; an empty round takes the first unit whole; ponytail).
- #30: generation. `generate` (action) logs `generation_pass` {pass n, mode llm|scripted, triggers}; every later
  `hypothesis_proposed` belongs to it (fold h["pass"]). run.yaml `generation` (llm|scripted, default llm) and
  `fixtures` (required iff scripted; dir relative to run.yaml, stored absolute): scripted serves
  `fixtures/pass<n>.yaml` (a YAML list of hypothesis specs, validated, proposed with actor harness); past the last
  fixture a pass is empty. Triggers (only once R0 is complete; status generation {mode, passes, due}): "run start" (no
  pass and no hypothesis), the queue < 2× max(1, len(selected)) with a BO round started since the last pass, a
  narrative record with generation: true since the last pass. A due pass is a `next` duty and a round-run refusal.
- #30: exhaustion (status `exhaustion`): queue_empty, none_undecided (no `active`), removals_reviewed, revivals_run
  (no pending link, no revived version still registered), final_pass_empty (a pass exists, no trigger since it, no
  hypothesis still `proposed`, none of its proposals ever registered). Judged at the round boundary: round-run (R1+,
  after the review/generation gates) logs `run_ended` {reason exhausted, exhaustion} and returns when all hold, and
  `next` then reads "round-run (the hypothesis list is exhausted: it ends the run)" — NOT at `generate`, because an
  llm pass opens before its generators record. A proposal that fails review must be parked (#31 adds pruned).
  Tests: conftest `expect_all` also records flag-less interplay reviews and runs `generate` when due (and, when that
  pass would exhaust the list, proposes an unregistered SPARE so older tests keep running rounds); `ready()` accepts
  those duties.
- #30 (review): a revived version rides with its partner uncounted by the dimension cap (same round, may exceed
  the cap; ponytail in `_selected`); a linked member exclusive with the unit or the chosen set leaves the unit
  instead of blocking it. Slots per round (generation trigger) = the queued hypotheses the next round takes (min 1),
  not retained ones. An llm pass counts toward `final_pass_empty` only once proposals are recorded against it
  (generators always propose ≥ 2); a scripted pass past its fixtures counts empty. `_propose(run_dir, con, spec,
  actor, rationale)`. CONTEXT.md gains **Generation pass** and **Exhaustion**. Kept: a revived version parked
  before it ran counts as "run" (parking is the orchestrator's logged, reviewed withdrawal); `generate` is never
  refused while the run is live (the orchestrator may request a pass); run start = no pass and no hypothesis.
- #31: `directives.py` holds the registry schema and pure checks. run.yaml `brief` (exactly purpose, contribution,
  complexity, provenance; optional), `directives` [{id, severity, statement, reason, scope, predicate?,
  forbidden_patterns? (prohibited only)}], `protected_paths` [globs; fnmatch, or a directory prefix]. init logs
  harness `registry_revised` {version 1, brief, directives, protected_paths, manifest {path: sha256}}; the fold keeps
  `st["registry"]` (+seq) and `st["manifest"]` (latest of registry_revised / lever_committed.manifest, since
  commit-lever rewrites levers.json). Manifest = worktree files (ls-files -co --exclude-standard: ignored files are
  NOT covered) matching runner, levers.json and the globs; checked (refusal names the files) at round-run (R0
  too), in `_change_allowed` (commit-lever, commit-change, add-dependency) and before a revision. round-run also
  refuses a worktree HEAD not in {base_commit, lever_committed/commit_change/dependency_added commits} (fold
  `st["commits"]`), so a raw `git commit` can't dodge the diff scan. Forbidden patterns: regex over the ADDED
  lines of `git diff-tree -p HEAD <tree>` in commit-lever / commit-change.
- #31: reviews. A `review` record is checked against the registry (directive must exist; prune names a prohibited
  one, deprioritize a discouraged one) and must be `strict: true` when the hypothesis's mechanism is similar
  (overlap coefficient of content words ≥ 0.5, ponytail) to a prohibited directive's statement or a pruned
  hypothesis's mechanism (other number). Its verdict applies AT RECORD TIME (harness events): prune → `hypothesis_pruned`
  {id, directive, by review|revision|revival, reason} when proposed|registered (status pruned: never tested), parked
  "prohibited: <id>: …" when active|retained; off-intent → parked "off-intent: <rationale>" (+ removal). A review of
  an active|retained hypothesis is refused mid-round. "Current review" = the latest for that id after the latest
  registry_revised. `register` needs a current review (then the grid check, `hypothesis_registered` carries
  `review` and, when lowered, `lowered` {by, reason = the register --rationale: the stated reason}). Effective
  priority = prioritize value − penalty (stretch +1; deprioritize or a declared discouraged directive +1),
  recomputed from the registry. status `review_missing` (registered|active|retained with no current review:
  revived versions, everyone after a revision) is a `next` duty and a round-run (R1+) refusal before interplay.
  unpark keeps the old review (the user overrides taste) but is refused while its box breaks a prohibition or its
  current review prunes it. A revived version whose box breaks a prohibition is pruned at once (by revival).
- #31: prohibited predicates = restricted-AST expressions over lever names and lever `path`s (dotted names), stating
  the allowed region; absent names → doesn't apply; eval error → outside (fail closed). Registration: grid (graded
  endpoints, baseline, quartiles; every option) over the hypothesis's box, others at baseline (ponytail: holes and
  cross-hypothesis predicates slip to the trial check). Every `_trial` checks the resolved config first → harness
  `prohibited_check_refused` {kind, levers, fidelity, directive, ...} + ProhibitedRefused; a refused sampler ask is
  told FAIL; 10 in a round → round trigger `prohibited` (ponytail). Seeds / agent trials outside → agent_trial_skipped;
  smoke's random point and ladder-calibration configs are drawn/filtered inside the region; `_eligibility` excludes
  trials breaking a current prohibition. compat flags: computed in the fold from the CURRENT registry (so a revision
  recomputes them), `t["compat"]` {id: bool} on every trial when any discouraged directive exists; trials.csv
  `compat:<id>` columns.
- #31: revisions: `checkpoint --revise <yaml>` (keys ⊆ brief/directives/protected_paths, others kept), only while
  paused and between rounds, applied at once (no round can run while paused, so this is the next boundary):
  `registry_revised` {version n+1, …, manifest over the new globs} by the caller; pruned hypotheses whose pruning
  directive changed at all (removed, reworded, relaxed or tightened) → `hypothesis_unpruned` {id, directive, version}
  (back to proposed; the re-review decides); registered hypotheses whose box now breaks a prohibition → pruned
  (by revision); active|retained → parked "prohibited: <id> (directives version n): …". Everything else (ineligible
  trials, compat, priorities, re-reviews) follows from the registry version. No new CONTEXT terms (Pruned now also
  covers registered-but-untested hypotheses a revision prunes, per spec).
- #32: conflicts. Live = registered|active|retained. `register` (after the prohibition grid) refuses a lever `path`
  shared with another live hypothesis (other number) unless H's `merges.from` names it (`cli._clash`); unpark checks
  it too; a revived version that clashes is parked by the harness (reason = the clash; a merge is the way on). The
  review's `conflict` is enforced at register against LIVE partners: shared-lever needs a merge of it; exclusive
  needs a merge, `masked_by` (either way) or `exclusive_with` (either way); rival needs nothing. Merges: mapping
  values may be one rule or a LIST of rules (several old levers → one new lever: exclusive same-slot mechanisms into
  one categorical with "none"); a trial's mapped values at the new lever's baseline yield to the one that is on, two
  on → ineligible; a merged-away lever out of the search loses its key in EVERY trial (study.eligible). Register
  checks merges.from exist (other numbers) and rule levers belong to them; sources still in the loop
  (proposed|registered|active|retained; active ones only between rounds) must have EVERY lever mapped over its whole
  range (`hypotheses.uncovered`: graded maps as itself onto a graded range covering it; each option maps into
  range) and get harness `hypothesis_merged` {id, into, from: status} after `hypothesis_registered` (status
  `merged`, `merged_into`; not a removal: no interplay review). Removed sources are lineage only (partial mappings
  allowed, as #28). CONTEXT gains **Merge**.
- #32: masking. Spec `masked_by: {H<m>: reason}` (by number). Declarations of hypotheses past proposed/pruned apply in
  `cli._trial` (every trial kind): when any lever of number m is off its spec baseline in the (sampled) config, the
  masked hypothesis's levers run at their baselines; trial_started carries `levers` = effective (trial.json, so
  lever() reads it), `sampled` {masked lever: sampled value} whenever a masked lever is in the config, and `masked`
  [ids masked in it]; trials.csv `sampled:<lever>` columns. The prohibited check runs on effective values.
  Eligibility (`cli._sampled`) and `Study._frozen` put SEARCHED masked levers at their sampled values (GP, Sobol,
  sampler, warm start, incumbent — reruns mask again), unsearched ones at what ran. `_fresh` skips trials where H
  was masked, so burn-in, check spacing, evidence cap, the trials gate and coverage count only effective trials.
  Chains judged on sampled values (ponytail in `_mask`). Precedence (bool masks graded) is not enforced.
- #32: rivals = pairs from each hypothesis's LATEST review (any registry version) with conflict rival;
  status `conflicts` {rivals [[a, b]] sorted, masking [{masked, by, reason}], merged [{id, into}]} (for #33's round
  summary). `_selected` co-places rivals like interplay links (join together or not at all). Verdicts stay
  independent (nothing couples them). test_directives' compat test moved H3 to path toy.w (+ an added small-w
  directive in the revision), since toy.x now belongs to the live H2.
- #32 (review): unpark also runs the review-conflict check (`_unresolved`) against its kept review; a merge excuses a
  conflict only for the exact ids in `merges.from` (both checks); two live hypotheses that mask each other are
  refused (one precedence). Kept: no check that a same-slot merge has a "none" option or that the route (merge /
  mask / exclusive_with) fits the shapes (the orchestrator's call); the masked hypothesis's GP/Sobol include
  masker-on trials at their sampled values, per spec, so its √V_T is averaged over that region too; a merged-away
  hypothesis's masked_by is not inherited (the merged spec declares its own).
- #33: `reports.py` = pure `render(st, status, revivals) -> {path: text}` (SUMMARY.md, rounds/NNN.md for every ENDED
  round incl. R0, exports/trials.csv, exports/hypotheses.csv); `cli._regenerate` writes them all (atomic replace) after
  every action, every trial and every batch of verdict checks. Rendering reads only the fold (never the worktree),
  no clock: `rebuild` (not logged, no rationale) rewrites everything; `summary` probe → {summary: SUMMARY.md text}.
  `round_summary` {round, path} logged by the harness once per ended round (in _regenerate); fold keeps
  rounds[r].incumbent/ended_seq/spent_s (budget spent at the round's end) and h.prune_reason. trials.csv keeps the
  JSON names (`trial`, `kind`) and `peak_mem_mb`: META = trial, round, epoch, commit, fidelity, seed, replicate_of, kind,
  status, objective, wall_clock_s, peak_mem_mb, then c:, t:, compat:, L:, sampled: (sorted within each); cells str(v)
  (bool "True"), dict/list/None as JSON; empty only when the key is absent. hypotheses.csv: id, title, lens,
  provenance, mechanism, state, reason (condition for rejects), √V_T/Δ bounds + trials_used from the LAST record,
  expected_right "k/n" (rounds with an expected record; actual = last record of the round: retained→retain,
  reject→reject unless a removal made it inconclusive, else undecided), revived_from, prediction_flag. SUMMARY
  incumbent = the latest R1+ round_ended incumbent vs the baseline mean at its fidelity (noise rungs). Suggested
  narrowing (ponytail): the round's best 5 sampler trials + baseline span ≤ half the round's searched range, on a
  hypothesis whose last record of the round passed every gate (and is active/retained) — printed as a runnable
  `narrow` command. Honouring config: best finished trial (rounds 1..r, round fidelity+epoch) with compat[id] true,
  gap vs the incumbent's mean. Proposed directives: ≥2 off-intent parks (before the round's end) whose reasons are
  `directives.similar`.
- #33 (review): `render(st, view)`, view = {status, revivals, testing: {active id: {record: cli._latest, needed, spacing,
  cap}}} so "Still testing" follows the harness's schedule (narrowing/epoch restarts). `rebuild` goes through
  `cli._write` (never logs); `_regenerate` = log pending round_summary events + `_write`. Fold keeps R0's own
  noise_estimate on rounds[0]["noise"] and `baselines` [{epoch, fidelity, mean}] so old summaries don't take a later
  epoch's σ/baseline. Honouring config = best honouring CONFIG (mean over it and its replicates), gap with a 95%
  interval ±1.96σ√(1/n_h + 1/n_inc), σ the round's (else R0's at its fidelity). Proposed directives count only
  hypotheses still parked. Kept: trials.csv names trial/kind/peak_mem_mb (spec says seq/chosen_by/peak_mem: same
  data, JSON names); between-round trials (equivalence, epoch baselines, smoke <H>) belong to no round's budget line
  (the running total counts them).
- #34: enforcement. `checks.py` = pure decisions (paths, a best-effort shlex Bash parser); cli `check write|read <path>
  --agent`, `check bash <cmd> --agent [--agent-id]`, `check stop`, `check recorded <id> [--agent]` (args after `--`)
  → {"allow": bool, "reason"?, "updated_command"?, "recorded"? (recorded only)}, exit 0; every block logs harness
  `hook_blocked` {check, agent, subject (resolved path | command | "stop" | agent id), reason}. No run, or a run whose
  log has `wrapup_finished` (new event, fold `st["wrapup_finished"]`; #37 logs it when distillation+verification+report
  are done, or at once under --no-confirm) → allow everything. ORCHESTRATOR = no agent_type (main session); the harness
  takes the role as the text after the last ":" of agent_type (`boautoresearch:round-analyst`). Read-only =
  registration-reviewer, interplay-reviewer, round-analyst; recording (held by `check recorded`) = those +
  hypothesis-generator (the lever coder records nothing: never held). Write rules: repo-root `run.yaml` (the draft) is
  allowed; any other path in the user's checkout blocked; under `.bo-research/`, only files inside a run worktree (a dir
  `.bo-research/<run>/<name>/` with a `.git` FILE — the research worktree, and #37's distilled one if it lives there)
  that aren't protected (runner, levers.json, registry globs, `.git`), and not while a harness process is live (open
  round or running trial with a live pid) unless the run has ended (wrap-up); everything else there (log, venv,
  run.yaml, artifacts, generated files) is the harness's. Outside the repo: allowed. Paths resolve directory
  symlinks, not the file's own. Bash: read-only agents may run only `cd` and `boautoresearch <probe>|record` (PROBES =
  status, summary, trials, verdict, next — add new probes to cli.PROBES), no output redirects or `$(`/backticks;
  installs (pip/`python -m pip`/uv pip|add|sync/conda/mamba/poetry/pipx) blocked for everyone; write targets
  (redirects, rm, mv, tee, touch, truncate, cp/ln last arg, sed -i files) go through the write rules; git
  commit/reset/rebase/checkout/switch/stash/merge/pull/push/cherry-pick/revert/am blocked when the git dir (cwd, `cd`,
  `-C`) is under `.bo-research/`; the orchestrator's non-harness commands may not name a path into a run's log.db*,
  artifacts/, or the run/.bo-research dir itself. A subagent's `boautoresearch record` gets `--agent-id <id> --actor
  <role>` inserted after `record` (`updated_command`); a subagent passing its own --agent-id/--actor is blocked.
  `record --file -` reads the body from stdin (read-only agents record through a heredoc). Read (orchestrator only):
  log.db*, artifacts/, or the run/.bo-research dir (Grep/Glob over it). Stop allowed: no run; live harness process;
  paused; pre-loop = no round ≥ 1 started AND run.yaml `go` false (new key `go`, default false, logged in run_started:
  headless runs never wait on the user); blocked with `next` mid-run and "wrap-up hasn't finished" after run_ended.
  New probe `next` → {next}. `verdict` is imported lazily in cli (checks start in ~20 ms, not 0.25 s).
- #34: plugin shell: `.claude-plugin/plugin.json` (name boautoresearch), `hooks/hooks.json` (every event → one adapter
  `python3 "${CLAUDE_PLUGIN_ROOT}/hooks/hook.py"`, timeout 30 s; PreToolUse matcher
  Edit|Write|NotebookEdit|Bash|Read|Grep|Glob|SubagentHandback, PostToolUse Bash, SessionStart, SubagentStop, Stop
  unmatched). hook.py (stdlib, py3.9-safe) maps hook_event_name/tool_name to the run venv's
  `.bo-research/<latest>/venv/bin/boautoresearch` call (cwd = hook cwd, timeout 20 s): inert when git finds no repo,
  no run, or `wrapup_finished` is in the log (one SELECT); allow → 0 (updated_command → hookSpecificOutput.updatedInput
  WITHOUT permissionDecision, so the permission flow still applies); block → 2 + reason on stderr; any nonzero exit,
  non-JSON, timeout or missing venv → 2. SessionStart prints `status` as context; PostToolUse on a command mentioning
  `boautoresearch` emits additionalContext with `next`.
- #34 (review): heredoc bodies are skipped only for a trailing heredoc opened on the first line whose delimiter is the
  last line and appears once (else the whole text is parsed: a false block, never a hidden command); the record
  rewrite touches only that command text, never the body. `>& file` is a write target (`2>&1`, `>&-` aren't); prefix
  flags (`env -i`) are skipped; `bash|sh|zsh -c '<cmd>'` and `eval` are checked recursively; NOBODY may pass
  --agent-id/--actor to `record` through Bash (the orchestrator can't forge a reviewer's record). The adapter: SubagentStop
  /SubagentHandback without agent_type are held (no `--agent`), Glob's pattern / Grep's glob are joined onto its path for
  `check read`, and any JSON without an answer blocks (all output handling inside the fail-closed try). Kept (ponytail):
  python -c / xargs / find -delete / perl -i / globbed names slip the parser (manifest backstop for writes, nothing for
  reads); a Glob like `**/log.db` from the root isn't caught; checkpoint-mode pauses share the fold's `paused` with the
  tested user pause; PROBES must grow with show/untested/sensitivity.
- #35: R1 gate = `cli._gate(run_dir, st) -> {condition: message}` checked at every BO round-run (R1's is the #15 gate):
  `_workspace` (venv, worktree, protected_paths, head, clean — also R0's precondition) + registry (brief present),
  review, interplay, generation, hypothesis, code, calibration (ladder not yet calibrated), expected, delta. ONE
  refusal: reason = messages joined by " | ", plus `failing: [conditions]` (Refused(msg, **extra) merges extra into
  the refusal JSON). Order: paused/ended → R0 (workspace only) → exhaustion (implies review/interplay/generation
  clear) → gate → revival only when review/interplay/generation hold → ladder calibration when the only failures are
  calibration (+ delta/expected while δ is unset) → refuse or run. R0's own conditions never reach the gate: round-run
  reruns R0 until it ends `calibrated`. Expected verdicts are now required (listed) even before δ is set.
  conftest.write_run_yaml adds a default `brief` unless the test states one (R1 needs a brief).
- #35: run.yaml gains `lenses` (distinct non-empty names; wildcard implicit), `proposals_per_lens` (int ≥2, default 3),
  `workers` (int ≥1; >1 refused: not built), all logged in run_started; status.generation gains lenses and
  proposals_per_lens. `go: true` is refused at init unless `delta` and (`lenses` or `generation: scripted`) are given
  (headless completeness is the harness's call). init's dirty-tree check skips the run.yaml being initialised (the
  uncommitted draft). δ advice: `_delta_guide(st, δ)` = {delta, trials = max(20, ceil(calibration.trials_needed(σ, δ)))
  at the run fidelity, cost_s = trials × mean cost, feasible = cost ≤ remaining}; status `delta_guide` (at the
  suggested 2σ); `set-delta` refuses an infeasible δ (JSON `guide`) unless `--infeasible-ok`; delta_set logs `guide`.
- #35: probes `show <H>` (the fold's hypothesis + effective priority + current review; verdicts = full records),
  `untested` (proposed|registered: id, status, title, mechanism, lens, priority), `sensitivity <H>` (per verdict record:
  id, round, check, group, sqrt_vt, levers, sobol_index, delta_stat, best_point; `telemetry: true`; no GP refit). All
  in cli.PROBES. Skill `skills/start/SKILL.md` (user-invoked): bootstrap `BOOT` = `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}/
  harness" uv run --no-project --python ">=3.10" --with pyyaml python -m boautoresearch` (init only; cli imports need
  only pyyaml; HARNESS_SRC resolves to the plugin source), then `BO` = init's `venv`/bin/boautoresearch. run.yaml must
  set `python:` (under uv, sys.executable is uv's throwaway env). Its "## Loop" section is a stub for #36 to replace.
- #35 (review): `cli._burn_in(d)` = max(10d, 20) is the one burn-in formula (_schedule, _delta_guide). Every gate
  condition has its own Seam-1 test (test_start.py) except `calibration` alone, which can't fail alone (round-run
  then calibrates). Kept: an infeasible δ is a refusal (`--infeasible-ok`), not a bare warning, so the user's
  judgement is forced; a headless run's run.yaml δ is not feasibility-checked; part 2 generates and codes the first
  hypotheses before δ (ladder calibration needs their lever boxes, #25) — #36's loop duties run there already.
  Headless `claude -p` NOT verified in #35 (no Claude auth in the ticket agent's environment); the harness path from a
  complete headless run.yaml to R1 was driven by hand through the CLI.
- #36: agents: `agents/<role>.md` for hypothesis-generator, lever-coder, registration-reviewer, interplay-reviewer,
  round-analyst (frontmatter `name` = the role #34's checks read after the last ":"; `model: opus`; `effort: high` for
  the generator and both reviewers per #14; read-only agents `tools: Read, Bash`). Each prompt takes `BO` (the run
  venv's boautoresearch, full path) from the orchestrator's delegation prompt, and ends with `BO record <kind> --file -
  --rationale … <<'EOF'` (JSON with no backticks: check bash scans the whole command for backticks/`$(`). The lever
  coder declares touched files in commit-lever's --rationale ("touches: a.py, b.py; …"): no harness flag. Distill-mode
  sections in lever-coder.md and round-analyst.md, and `## Wrap-up` in skills/start/SKILL.md, are marked stubs for #37.
- #36: the loop lives in the start skill's `## Loop` (spec: ONE orchestrator skill): a table from each `next` duty to
  the subagent/action. round-run runs with run_in_background and the orchestrator keeps its turn alive (completion
  notification, or Monitor on a `while pgrep -f "boautoresearch round-run"` loop that echoes on exit, re-armed): headless `claude -p` kills background
  Bash ~5 s after the final message (claude-code-guide). After a generation pass the orchestrator reviews every
  `proposed` hypothesis and registers those left standing (review_missing covers registered+ only).
- #36: the analyst after every round is harness-forced: `cli._narrative_missing(st)` = [latest ended BO round] while no
  narrative record names it; status `narrative_missing`, a `next` duty ("record narrative R<r> …", before reviews)
  and gate condition `narrative`. conftest.expect_all records a narrative (citing the round's verdict records) and
  ready() accepts the duty. New probe `registry` → {registry: {version, brief, directives, protected_paths}} (in
  PROBES): reviewers/generators had no way to read the current brief and directives.
- #36 (review): duplicates within one pass (both `proposed`) slip `register`'s live-partner conflict check, so the
  reviewer marks them in prose (rationale starts "duplicate of H<m>", class by shape) and the skill parks the weaker
  before registering either (prose, not harness-enforced: ponytail). Tier = `model: opus` in each agent file (no
  separate `tier:` key: Claude Code has no such field; the host adapter mapping is the frontmatter itself). Read-only
  prompts say one BO command per Bash call (pipes/redirects are blocked). Touched files are declared only in
  commit-lever's rationale, never checked (the diff is in the commit anyway).
- #37: wrap-up. `wrapup [--no-confirm]` (action; refused before `run_ended`, mid-round, after `wrapup_finished`, and
  when the flag differs from the first call's) logs `wrapup_started` {confirm, incumbent (latest BO round_ended's),
  config (levers.json baselines + the incumbent's values for RETAINED levers only: what the distilled branch must
  reproduce), retained, distilled {branch `<run branch>-distilled`, worktree `.bo-research/<run>/distilled` (git
  worktree from base_commit, created at start), spec path} | None (no-confirm or nothing retained), changes [{commit,
  reason}] (non-lever commits), fidelity = reference}; still-active hypotheses → `hypothesis_inconclusive` reason "the
  run ended before a verdict" (no removal: nothing is reviewed or revived after the run). Each call advances; after
  run end `_next` = wrap-up duties only: record takeaways → (confirm, retained) record distill_spec → record
  distill_review (approve|revise; revise asks for a new spec) → lever coder writes the distilled worktree → `wrapup`
  commits it (harness author; `distill_committed` {attempt, commit, paths}) after `_distill_allowed` (diff base..tree:
  no protected path, no boautoresearch import ADDED to a changed .py file vs base — static, `hypotheses.harness_imports`,
  no forbidden pattern) and verifies: `verification` {attempt, commit, research {trials, commit, mean} (run ONCE, reused
  by retries), distilled {…}, sigma (reference σ), replicates r (3; 1 when R0 measured σ = 0), tolerance 2σ√(2/r) +
  1e-9·max(1,|mean|), gap, passed}. Pass → `wrapup_finished` {result verified, confirmed {branch, commit, trials,
  mean, fidelity}}; 3 failed attempts → result unverified, confirmed = research branch at config; nothing retained →
  final confirmation (research replicates) → confirmed (unconfirmed if a replicate failed); --no-confirm → unconfirmed,
  confirmed None. A clean distilled tree already verified/unchanged = no new attempt (returns the duty). Wrap-up trials:
  kind `wrapup` (`wrapup_of` research|distilled, `attempt`), in the given worktree, skip the prohibited/budget/ceiling
  gates, excluded from `_spent`. The distilled branch keeps the base commit's runner (protected, untouched): the
  harness runs trials through it; "no harness dependency" = no import added (ponytail: dynamic imports slip).
- #37: records. New kinds `takeaways` {takeaways: 1–5 one-line bullets, each naming a cited verdict id (regex, when the
  run has verdicts), cites, quotes?} and `distill_review` {verdict approve|revise, rationale} (refused before a
  distill_spec of this wrap-up exists). `distill_spec` gains required `changes` [{commit, decision keep|drop, reason}]
  = exactly the non-lever commits, and cites ⊇ every retained hypothesis's latest verdict. Wrap-up records
  (distill_spec, takeaways) may quote pseudo-record `wrapup` (the wrapup_started payload: `config.H1.x`,
  `incumbent.mean`), exempt from citing. Only records logged after wrapup_started count. Status gains `wrapup` (the fold's
  st["wrapup"]: start payload + verifications + finished). Fold also keeps `equivalence` and `epochs` lists (REPORT
  diagnostics). Generated: DISTILL_SPEC.md (latest distill_spec) and REPORT.md (once finished; 8 `## ` sections, `### `
  subsections in What to take / What worked). `clean [run-id]` (after wrapup_finished; refused when nothing is left)
  logs `cleaned` {removed, branches} then removes both worktrees and the venv; `_open_run(run_id=)`.
- #38: R0 running out of budget ends the run (`_run_r0` → `_end_run`: round_ended + run_ended budget_spent, exit 0), like R1+.
  Dogfood lives in `benchmarks/dogfood/`: `toy/` (train.py CONFIG knobs set by lever code under "# research changes to
  cfg go here"; objective.py = planted truth, protected; runner passes epochs/seed so train.py never imports the harness),
  run.yaml (δ 0.1, ladder [{epochs: 2}], reference {epochs: 8}), fixtures pass1 (H1–H9), pass2 (the fidelity-sensitive
  tail, AFTER ladder calibration: with its lever in the calibration configs the proxy fails ρ ≥ 0.8), pass3 (lr, the
  newcomer partner of clipping). The interplay-review case is made structural: the partner (distillation) declares
  `exclusive_with` the label-smoothing hypothesis (3 levers → decided after the 1-lever EMA), so it is queued and
  untested at the EMA's removal. σ via env DOGFOOD_SIGMA, sleep DOGFOOD_SLEEP_PER_EPOCH. scripted.py plays every role;
  run_benchmark.py (driver + adversary via hooks/hook.py stdin and direct CLI calls with --actor benchmark-adversary);
  check.py per run; matrix gate = invariants in all runs, each case ≥ n−1 of n scripted runs.
- #38 (dogfood findings, harness): a fidelity-sensitive hypothesis at a proxy fidelity is exempt from the evidence cap
  (it waits for stall escalation; ponytail: unbounded until a stall or the budget); the drift check runs the other config
  with the incumbent's fidelity-sensitive levers (and skips candidates identical after that), else a tail that pays off
  only at the reference "breaks" the proxy and downgrades every co-active reject to inconclusive (seen 3/6 before).
- amendment M_u: (#21 amendment, supersedes #26's √V_T reject) `verdict.judge` returns `delta_stat`, `m_u`, `sqrt_vt`
  (telemetry) for the group and, d ≥ 2, per lever in `levers[n]` = {sqrt_vt, delta_stat, m_u}; all from ONE set of latent
  joint samples (GP.samples excludes noise). Candidates C = 128 Sobol + trials + baseline (drawn before the Jansen points).
  M_s per path = max(largest range on a grid: rows = 16 Sobol others-settings + baseline + best posterior-mean point, each
  with s moved over 16 other candidates' s-values + baseline; |f(c) − f(c with s at baseline)| over every c ∈ C) — the second
  term makes Δ_s ≤ M_s hold on every path by construction. `cli._condition(stats, δ)`: UB(Δ) < δ → no-improvement,
  and UB(M_u) < δ too → irrelevant; retain = LB(Δ) > δ only. Freeze: d ≥ 2, at most ONE lever per check (smallest UB(Δ_i)
  among those with UB(Δ_i) < δ, even if every lever qualifies); `lever_frozen` gains `condition` (irrelevant iff UB(M_i) < δ);
  fold `h["frozen"]` is now {lever: condition}; `_eligibility` drops keys only for irrelevant-frozen levers and, for a group
  rejected irrelevant, only its levers not frozen no-improvement (those stay filtered). Substitutes: `_finalise_rejects`
  applies every confirmed `irrelevant` reject but only the confirmed `no-improvement` reject with the smallest UB(Δ) (ties by
  record id) per round end; the others log harness `reject_deferred` {id, verdict, round} (fold h["deferred"]), stay active,
  and must reach a fresh pending-reject + confirmation again (a deferred reject confirms nothing). reports: hypotheses.csv
  `m_u_lower/m_u_upper` replace the √V_T columns; round lines show Δ and M_u (+ a deferral note); `_actual` counts a deferred
  reject as undecided. `sensitivity` adds m_u. `_kernel` accumulates per lever (no (n, m, D) array). Measured: 1 flat lever,
  σ = δ/2: reject at 30 fresh 6/6 before and after; 3 flat levers, σ = δ: before 6/6 at 45, after 4/6 at 45, 1 at 75, 1 still
  active at 70 (the sup statistics' power cost).
- Tests (seed rates): a statistical test runs a contiguous seed set (`range(N)`, or `repeat`'s (1, 2, 3)) and asserts a rate ≥ a
  threshold; its comment gives the rate measured over a wider sweep (seeds 0..19 or 0..9) and the chance the threshold fails at
  that rate (aim ≤ ~10%). A single fixed seed is only a replay seed for a mechanics test whose premise holds on (nearly) every
  seed, never a seed kept because it passes. Eligibility oracles in tests are epoch-aware.
- verdict calibration: the verdict GP's hyperparameters had no floor on the signal (var ≥ 1e-3 standardised, lengthscale
  ≤ 100), so at low signal-to-noise ML fit a flat function and the posterior was certain the lever was flat (a gain of
  exactly δ, σ = 1.5δ: rejected in 10 of 20 runs, Δ's 5/95% bounds held δ in 5 of 32 checks). Now signal sd ≥
  `SIGNAL_DELTAS`·δ = 1.5δ (`judge` takes δ) and each lengthscale ∈ [0.1, √(signal var / signal floor)] (a clamp in
  `_parts`: the prior's slope along every lever stays at least the floor's, however large the other levers make the
  signal); the record's `gp` gains `signal_floor`, `lengthscales` are the clamped ones. Measured over seeds 0..19: gain of
  δ (σ = 1.5δ) rejected 0/20 (bounds held δ in 22/24 checks); gain of 2δ 0/20 (3/20 before); 1 flat lever σ = δ/2 19/20 at
  30 fresh (20/20 before); 3 flat levers σ = δ 20/20 within 6 rounds, at 49–110 sampler trials after two single-lever
  `no-improvement` freezes (12/20 at 45 before). Cost: `irrelevant` needs M_u's bound below δ at every setting of the other
  levers, and BO leaves settings unexplored, so next to a curved co-active lever (a bowl) a flat lever is labelled
  `no-improvement`, not `irrelevant` (0/10, even at σ = δ/20; before, the flat fit supplied the label); a smooth partner
  (linear) still allows it at σ = δ/10 (20/20). Rejects are unaffected; the label and the warm start's key drop are.
  DECIDED (user, option 1; #21 amendment): the signal floor and lengthscale clamp are part of the verdict model, and
  `irrelevant` needs evidence across the other active levers' box, so `no-improvement` is the expected label for a flat
  lever next to a curved partner whose box is unexplored. Tests follow: the flat-next-to-a-bowl freeze asserts
  `no-improvement` + filter-to-baseline; test_reports' planted H2 is rejected `no-improvement`, deferred one round behind
  H3's; the broken-proxy test is a rate over range(10) (its drift premise is being reworked separately); Seam 2's
  useless case (benchmarks/dogfood/expected.yaml) accepts `irrelevant` or `no-improvement` (`condition` may be a list).
- #35: the skill's round-run wait loop is `pgrep -f "[/]<BO minus its leading slash> round-run"`: procps pgrep
  (Linux, the dogfood CI) doesn't exclude its ancestors, so a plain `"boautoresearch round-run"` matched the Monitor's
  own shell and never exited (BSD pgrep hid it on macOS); the full `BO` path also keeps one run's wait off another
  run's round (the 7-run matrix). Headless start verified live: `claude -p "/boautoresearch:start"` on the dogfood toy
  reaches R1 unprompted (scripted ≈5.5 min, free ≈17 min).
- #34 follow-up (orchestrator raw reads, no parser): the orchestrator's Bash may compute no words (`$(`, backticks,
  `$'`, `<(`/`>(` outside a trailing heredoc → `checks.COMPUTED`); a command not only `boautoresearch`/`cd` may not
  name `log.db`/`artifacts/trial-` in its raw text (inline code, heredoc bodies) or its shlex words; and every word,
  in Bash or as a Read/Grep/Glob path, is refused if as a glob (braces, `$VAR` glued to text → `*`; `*` crosses `/`)
  resolved from its cwd it fnmatches a `checks.raw_paths` entry (bo dir, run dirs, log.db*, artifacts, a stand-in
  trial-0, every file under artifacts). Names over parsing: every route must carry one. Known false positives
  (orchestrator only): a lone `*` word (`ls *`, `-m '* fix'`), a user file named log.db. Residual: names built in code
  (`'lo'+'g.db'`), a script file outside the repo, recursive reads naming nothing (`grep -r x .`).
- #37 follow-up (distilled branch, dynamic loads): `harness_imports` returns a Counter of sites (static harness imports,
  `import builtins`, loaders `__import__`/`import_module`/`run_module`/`run_path`/`exec_module`/`load_module`, bare
  `exec`/`eval`, `builtins|sys.<exec|eval|modules|__import__>`, any str matching `\bboautoresearch\b`); refused when
  added against the base (Counter difference). Not "verify with the harness unimportable": the harness runner imports
  it in-process. Residual: a subprocess, a .pth file.
- workers (parallel trials, #21 story 16): a BO round (R1+) keeps up to `workers` trials in flight; everything else (R0, ladder
  calibration, smoke, equivalence, drift, wrap-up) stays one at a time (ponytail: their trials are few). `cli._start` (gates +
  trial_started) and `cli._outcome` (trial_finished/failed) run on the main thread; only the subprocess wait (`_run_trial`,
  heartbeats on its own connection) runs in a `ThreadPoolExecutor` worker, so the log and the Optuna study have one writer.
  Pending points: GPSampler (Optuna 5) already conditions asks on RUNNING trials (qLogEI, Kriging believer), and an ask
  is RUNNING until told, so no constant liar is added; an ask equal to an in-flight ask is told FAIL and the round waits
  for a landing. Budget: the gate charges running trials their estimated cost (`_in_flight`, 0 serially), and so does
  the round cap (which now also exempts only the first trial: `spent > 0 or flight`). Budget stays summed per-trial
  wall-clock (spec), so W workers spend it ~W× faster in real time. Triggers with trials in flight: a stall is judged
  only with nothing in flight (wait for a landing, re-evaluate: a trial still running may be a new incumbent owed
  confirmations); every other trigger (search_space, cap, target, user_stop/checkpoint, prohibited, and a budget/ceiling
  refusal) stops launching, lets the in-flight trials finish, and records them in the round before round_ended. Landed
  trials are logged in trial order within one wait; "new incumbent"/owed confirmations replay in trial (start) order as
  before. A verdict check may see up to `workers` trials past its due count. Replicate-floor picks count in-flight
  replicates. A crash leaves every in-flight trial running with a dead pid: recovery abandons and killpgs each (no
  change); a main-thread exception killpgs the in-flight runners. Timing makes parallel runs non-replayable (the log
  still rebuilds every file). workers: 1 takes the same path with at most one trial in flight (same events).
- workers (review): the in-flight estimate never ends the run: `launch` (in `_run_round`) checks `_headroom` first, and while
  trials are in flight and the next would not fit, it lands one and asks again; only the ledger (nothing in flight) refuses,
  so budget_spent means spent. A worker error in a landing (also during the budget/ceiling end) goes through
  `_abort_round` (killpg the in-flight runners, round_ended failed). Kept: confirmations owed by a trial that landed after
  a round-ending trigger are not carried to the next round (the stall window and owed list restart per round; serial
  rounds already dropped them on a cap); an unknown cost (0) charges nothing, so a round at a never-measured fidelity could
  start `workers` trials uncharged (R1+ fidelities are always measured by R0 or ladder calibration).
- #29 limit (queued trials): `cli._queued(st, r)` = trials enqueued for r plus those of an earlier round that never started
  (no trial carries its `queued` seq) and were never skipped (the fold marks `agent_trial_skipped` {queued} entries
  `skipped`). They run first in r, oldest first, and count against r's agent-trial cap (so `used` may exceed it once; they
  were admitted under their own round's cap). A started one (abandoned in flight, or finished) is never run again. Kept:
  R1's run.yaml seeds are not carried past an interrupted R1 (ponytail in `_run_round`).
- #29 limit (tight cap; #10 resolution 14: "the oldest retained hypotheses whose posterior is concentrated are narrowed,
  never frozen"): `cli._selection(st)` -> (selection, narrowings). When the next queued unit doesn't fit, the harness
  takes retained hypotheses oldest first (registration seq) that have a window in the last round's suggested narrowings
  (`reports.narrowings`, the summary's rule: best 5 sampler trials + baseline span ≤ half the range, last record of the
  round passed every gate), as few as let the unit fit; none if even all of them wouldn't. The round start logs each
  window as `narrowed` {id, lever, before, after, verdict, round, cap: true} (actor harness; the usual narrowing fold, so
  its checks restart), and the fold keeps `context` [levers]: while retained, those levers are searched but uncounted by
  the cap (what "stay in the search as context" buys; like a revived version riding uncounted). A lever narrowed at
  this boundary already (by `narrow`) is not narrowed again. status.schedule gains `narrowing` (pending) and `context`
  (ids). Kept: in-search dims can still exceed the cap when nothing retained is concentrated (unconcentrated retained
  and active hypotheses are never narrowed or frozen for the cap).
- #29 limit (review): a window counts only when its verdict is the hypothesis's `_latest` record (current fidelity,
  epoch, group and ranges: what `narrow` reads; this also skips one narrowed at this boundary). `context` is cleared when
  the hypothesis stops being retained (a verdict makes it active again, or it is parked), so its levers count again.
  `enqueue` judges configs against the space with the pending windows applied. CONTEXT.md's Narrowing now covers
  retained hypotheses and context. Kept (ponytail in `_selection`): context levers are still GP dimensions, so status
  `dimensions` can pass `dimension_cap`; no window when the last round was interrupted before a record.
- #21 equivalence (user decision; docs/research/equivalence-check-threshold.md): the three noise checks use a t quantile at
  `CHECK_ALPHA` = 0.05 on σ̂ = `cli._pooled_noise(st, fidelity)`: pooled within configs (identical `levers` JSON) over the
  current epoch's finished trials at that fidelity, wrap-up trials aside, df = Σ(n_c − 1); no replicated config → the
  recorded σ (`_sigma`) on k − 1 df. `_sigma` itself stays R0's/the epoch's baseline σ everywhere else (GP noise floor, δ
  advice, previous_sigma): the spec keeps those on the baseline σ and the per-round re-estimate as a flag only. σ̂ is taken
  before a check's own trials. Equivalence: first look 2 replicates within 2σ̂ (+ float tol); a config outside it gets 4 more
  and fails iff |mean₆ − logged| > t(1 − α/(2·#configs), df)·σ̂·√(1/6 + 1/m); σ̂ = 0 or a failed trial → no second look; epoch
  iff a config fails (a budget refusal keeps any config's already-failed look: passed False, not None). The incumbent's logged mean excludes the trial that selected it when it has replicates (winner's curse).
  Row: final `trials`/`mean`/`tolerance`/`passed` + `stage1` (the first look, only when a second ran); payload + `df`,
  `alpha`. Verification: same shape, 3 + 3 more distilled replicates, t(1 − α/2, df)·σ̂·√(1/6 + 1/3); `distilled`, `gap`,
  `tolerance`, `passed` are final, + `stage1`, `df`, `alpha` (σ = 0.0, not None, before R0). Drift: margin t(1 − α, df)·√2·σ̂
  at the reference (one-sided: only a reversal breaks) + float tol; the other config is drawn only among the round's configs
  the proxy orders against the incumbent (|Δ_proxy| > t(1 − α, df_p)·σ̂_p·√(1/n_i + 1/n_o) + float tol); none → `drift_check`
  {undecidable: true, broken: None, trials: []} (also when the round has no other config at all), nothing run, and the
  round's rejects log `reject_deferred` {…, reason "undecidable proxy fidelity"} (stay active; ponytail: a proxy that never
  orders anything defers until budget/stall). Before → after (docs/research/check_rules_mc.py, seeds 0..19 × 20k, σ = 1,
  δ = 2σ; false alarm / power δ/2 / power δ / trials per no-op check): equivalence 14.8% / .36 / .74 / 4.00 → 3.6% / .17 /
  .53 / 4.76 at df 4, 2.5% / .19 / .69 / 4.34 at df 20; verification 11.6% / .30 / .68 / 6 → 4.2% / .18 / .55 / 6.35 (df 4),
  3.0% / .19 / .65 / 6.18 (df 20); drift 5.7% (df 4; 9% at df 2) / .17 / .35 / 2 → 5.0% / .15 / .32 / 2 (df 4), 5.0% / .17 /
  .39 / 2 (df 20). End to end: a no-op check (k = 2, df 2) failed 5 of 40 on the first look alone, 2 of 40 in full; a faithful
  proxy broke 2 of 20 before, 1 of 20 after; an exactly reversed proxy was caught 19 of 20 before, 20 of 20 after, and 7
  → 20 of 20 with a useless co-active lever (test_a_broken_proxy_downgrades…, noise off). Tests that declare
  `deterministic: true` must turn the toy's noise off (σ̂ = 0 lets noise pass for a proxy order).
