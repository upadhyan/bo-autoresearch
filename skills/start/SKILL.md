---
name: start
description: Start a BO Autoresearch run in this repository — pre-loop grilling (interview, init, calibration round, δ, the hypothesis list), or a headless start from a complete run.yaml — then run the loop.
disable-model-invocation: true
---

# /boautoresearch:start

You are the **orchestrator**. The `boautoresearch` harness decides everything trustworthy (statistics, verdicts, budget, stopping, enforcement); you make the judgement calls it asks for. Every harness command prints JSON; a refusal is `{"refused": true, "reason": ..., "failing"?: [...]}` — read the reason, fix what it names, retry. Refusals and `next` are your map.

## The harness command

The plugin root is `${CLAUDE_PLUGIN_ROOT}` (already expanded here; Bash does not have it as a variable, so always write the full path).

- **Before a run exists** (only `init` needs it), bootstrap the harness from the plugin's own source:
  `BOOT` = `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}/harness" uv run --no-project --python ">=3.10" --with pyyaml python -m boautoresearch`
  Without `uv`, use any Python ≥ 3.10 that has `pyyaml`: `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}/harness" <python> -m boautoresearch`.
- **After `init`**, use the run venv's copy: `BO` = `<venv>/bin/boautoresearch`, where `<venv>` is the `venv` field of init's output (an absolute path). Every later command in this skill is `BO …`, written out in full each time.

Every action takes `--rationale "<why>"`. Probes (`status`, `next`, `summary`, `show <H>`, `untested`, `verdict <H>`, `sensitivity <H>`, `trials`) are free.

## Headless or interview

Read `run.yaml` at the repository root.

- **Headless** — it has `go: true`: skip both interviews. Run `BOOT init run.yaml --rationale "headless start"`; if init refuses, stop and report its reason (the user fixes run.yaml; headless never asks). Then go straight to **Loop** — its first `round-run` runs R0, and δ comes from run.yaml. Ask the user nothing on this path.
- **Interview** — anything else (no run.yaml, or `go` absent/false): run **Interview part 1**.

## Interview part 1 (no data needed)

A grilling interview: one question at a time, in this order, each with **one recommended answer** you derive from the code and from earlier answers, which the user accepts or changes. Later topics depend on earlier ones.

1. **Objective.** The function that produces it and its `direction`; where evaluation code and data splits live (they become protected paths); the project's Python (`python:` — the interpreter the project runs with). Then generate the tiny runner at `runner.py` (it only wires the entry point):
   ```python
   import boautoresearch as bo
   from <module> import train_and_eval   # returns the objective, or (objective, {constraint: value})
   bo.run(train_and_eval)
   ```
   plus `levers.json` holding `{}`. Run it once at baseline, outside the harness: `PYTHONPATH="${CLAUDE_PLUGIN_ROOT}/harness" <python> runner.py` — it must print `{"objective": ...}`. Fix it until it does. Ask whether the objective is **deterministic** (same config, same number); `deterministic: true` turns replicates off.
2. **Research brief** — `brief: {purpose, contribution, complexity, provenance}`: why the research is done (speed up an implementation, a publishable contribution, a production model), what counts as in scope and as a contribution, the complexity appetite, the provenance rules.
3. **Directives** — `directives: [{id, severity, statement, reason, scope, predicate?, forbidden_patterns?}]`, severity `prohibited` or `discouraged`. Rewrite every "preferred" wish as a **discouraged** directive against its opposite. A predicate states the *allowed* region over lever names or config paths; forbidden patterns are regexes over added code (prohibited only).
4. **Protected paths** — confirm the defaults (the runner, `levers.json`, the objective and evaluation code, data loading and splits) and take additions: `protected_paths: [globs]`.
5. **Fidelity** — `reference_fidelity` (the fidelity the result must hold at, e.g. `{epochs: 20}`, passed to the code as `bo.fidelity()`), and a `ladder` you propose: 1–3 cheaper rungs, the cheapest fidelity that still ranks configs the same way. The user confirms both.
6. **Budget** — `budget_s` (compute seconds for the research loop; wrap-up is outside it), optional `target` (an objective value that ends the run once confirmed) and `max_trials` ceiling, `workers: 1` (tell the user: the harness runs one trial at a time), `replicates_k` (default 5), `checkpoint` (true: pause at each round boundary; false: autonomous).
7. **Seeds and ideas** — up to 5 named `seeds` (configs the user already believes in; they run in R1 as agent-chosen trials); the user's own hypotheses, with plausible lever ranges where the user has them (keep them for part 2); planned breaking changes (keep them: each is declared when it lands, with `commit-change --breaking`). Results from outside the harness inform rationales only.
8. **Lenses** — propose 4–6 `lenses` from the brief (for ML e.g. data, optimisation, architecture, regularisation, systems; for a speed-up e.g. algorithmic, memory, parallelism, compiler); the wildcard lens is always added. Ask `proposals_per_lens` (default 3).

**Done when** every topic has an answer. Write the `run.yaml` draft at the repository root — human-readable, a comment per section — with `go: false`, and tell the user they may edit it. Commit the runner and `levers.json` (ask first: `init` refuses a dirty tree, and the run worktree is taken from HEAD; the draft itself may stay uncommitted). Then `BOOT init run.yaml` (fix what it refuses, with the user), and `BO round-run --rationale "the calibration round"`: R0 runs the smoke test and the baseline replicates.

## Interview part 2 (after R0)

1. **Show R0's results** from `BO status`: σ (`sigma`, per rung in the round-run output's `noise_estimate`), R0's cost (`budget.spent_s` of `budget.total_s`), and — once a ladder is calibrated — `fidelity_calibration` (ρ, spread and cost per rung, the chosen rung, or a fallback).
2. **Hypotheses.** Run the first generation pass (`BO generate`, then the generators), turn the user's own hypotheses into specs through the same `propose` / review / `register` path, and get the selected hypotheses' lever code committed (`next` lists each duty). With a ladder, the next `BO round-run` then calibrates it and returns `fidelity_calibration` without starting R1; show it. If no rung passed and the user explicitly wants a rung anyway: `BO accept-proxy --fidelity '<rung json>'`.
3. **Set δ**, the minimum meaningful effect, in objective units. Recommend `status.suggested_delta` (2σ) and quote `status.delta_guide`: the trials a one-lever verdict needs at that δ and their cost against the budget left. `BO set-delta <δ>` is final for the run (it can never change: that would re-grade every verdict). When the harness refuses a δ as infeasible, show the user its numbers; pass `--infeasible-ok` only when the user accepts verdicts may never come. σ = 0: ask for the smallest effect worth having.
4. **Review the list once.** `BO status` and `BO untested`: title, lens, priority and the review's intent verdict per hypothesis, plus pruned and parked ones with their reasons. The user may `park`, `prioritize` or add hypotheses, or say go (recommend go). This is the last pause unless checkpoint mode is on.

**Done when** δ is set and the user said go. Then **Loop**.

## Loop

`BO round-run` for R1 is refused until the R1 gate holds; the refusal's `failing` list names every unmet condition. Do each duty in `next`, run `BO round-run` again, and repeat until `run_ended`.

<!-- #36 replaces this section with the full loop: which subagent does each duty, round boundaries, wrap-up. -->
