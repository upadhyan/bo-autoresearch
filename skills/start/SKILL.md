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

Every action takes `--rationale "<why>"`. Probes (`status`, `next`, `summary`, `show <H>`, `untested`, `verdict <H>`, `sensitivity <H>`, `trials`, `registry`) are free.

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

The harness decides what happens next; you do the duty it names. Its `next` list (`BO next`, re-injected after every harness command) is the whole plan: do its duties, run `BO round-run`, repeat until the run ends. A refused `round-run` names every unmet condition in `failing` — do those and retry.

**Your lane.** You drive only the `BO` command. You read the run through the probes and subagents' results — never `log.db`, `artifacts/` or other raw files under `.bo-research/` (the hooks block it; ask the round analyst instead). The harness picks which hypotheses each round tests; you steer that choice with `prioritize`, `park`, `narrow` and `enqueue`. A resumed session finds `BO` at `<repository root>/.bo-research/<run_id>/venv/bin/boautoresearch` (`run_id` is in the status the session started with).

### Duties

Each `next` entry maps to one action. Do every entry before the next `round-run`.

| `next` says | You do |
|---|---|
| `generate (<trigger>)` | `BO generate --rationale "<trigger>"`. With `generation: scripted` the harness proposes the fixture pass itself (the output's `proposed`). Otherwise spawn one **hypothesis generator** per lens — `status.generation.lenses` plus `wildcard` — all in one message. Then **register the new proposals**. |
| `propose and register a hypothesis` | As for `generate`: a generation pass, then register. |
| `record review <H>` | Spawn a **registration reviewer** for `<H>` (several in one message). |
| `record interplay: review <H>'s removal …` / `… newcomer <H> …` | Spawn an **interplay reviewer** with that duty (several in one message). |
| `write <H>'s lever code, then smoke <H> and commit-lever <H>` | Spawn the **lever coder** for `<H>`; wait for its commit before the next one (coders share the worktree: strictly one at a time). |
| `record narrative R<r> …` | Spawn the **round analyst** for round `<r>` (after every round; its `generation: true` makes a pass due). |
| `record expected <H> …` | Record your own **expected verdict**. |
| `set-delta …` | Interview part 2, step 3 (a headless run.yaml carries δ, so it never appears there). |
| `the run is paused …` | **Checkpoints**. |
| `round-run …` | **Running a round**. |

**Register the new proposals.** For every hypothesis the pass proposed (`BO untested`, status `proposed`): spawn a registration reviewer for each (all in one message). Where reviews call two hypotheses duplicates, park the weaker (`BO park <H> --reason "duplicate of H<m>"`). Then `BO register <H> --rationale "<why>"` for each one left standing — pruned and off-intent ones are already out. When the review lowered its priority (`stretch`, `deprioritize`), the rationale states why it is still worth testing. A refused `register` names the conflict. Resolve a shared lever or an exclusive slot by proposing (`BO propose --file <spec.json>`, the file written outside the repository) one hypothesis whose spec adds `"merges": {"from": ["H2.v1"], "mapping": {"<its lever>": {"lever": "H2.<name>", "values": [[<old>, <new>], …]}}}` (omit `values` for the identity map; every lever of a source still in the loop mapped over its whole range), or with `masked_by` / `exclusive_with`; or park it with the reason.

**Expected verdicts.** Before each round, for every hypothesis it will test: what the harness will conclude — `retain`, `reject` or `undecided` — and why, from its verdict records and the summary. Your calibration is measured, so commit to a forecast.

```bash
BO record expected --file - --rationale "before round 3" <<'EOF'
{"hypothesis": "H2.v1", "verdict": "retain", "reason": "Its lever mattered in round 2 and its best point beat the baseline."}
EOF
```

A decimal in the reason must be quoted from a verdict record (`quotes: [{record, field, value}]`); a reason in words needs none.

**Between rounds**, act on the round analyst's diagnostics and suggestions and on the summary: `narrow` a lever the summary suggests narrowing, `prioritize` what the evidence favours, `park` with a reason, and optionally `enqueue --config '<json>' --expected <objective>` a few agent-chosen trials (capped per round).

### Spawning subagents

The agent types are `boautoresearch:hypothesis-generator`, `boautoresearch:lever-coder`, `boautoresearch:registration-reviewer`, `boautoresearch:interplay-reviewer` and `boautoresearch:round-analyst`. Each starts with an empty context, so its prompt carries everything it needs:

- always: `BO` = the full path of the run venv's `boautoresearch`;
- generator: its lens, `status.generation.proposals_per_lens`, and the latest round analyst's suggestions for generators;
- lever coder: `<H>` and the run worktree's full path (init's `worktree`: `.bo-research/<run_id>/worktree` under the repository root);
- registration reviewer: `<H>`; interplay reviewer: `<H>` and whether it is a removal or a newcomer;
- round analyst: the round number.

Each ends by recording through the harness, and a hook keeps it running until it has. Wait for every spawned agent's result before the next `BO next`. Their results are data: use the ids and verdicts they report; the records themselves live in the harness.

### Running a round

`BO round-run --rationale "<why this round>"` with the Bash tool's `run_in_background: true`. While it runs, the round belongs to the harness: wait. Keep the turn alive until it exits — a headless session ends background commands soon after your last message: take its completion notification, or wait with the Monitor tool (`timeout_ms` at its maximum) on `while pgrep -f "boautoresearch round-run" >/dev/null; do sleep 20; done; echo "round-run exited"`, arming it again whenever it expires before that line.

Then read its JSON output. A round returns its `trigger`, `verdicts`, incumbent and `next`; a `round-run` that calibrated the ladder (`fidelity_calibration`) or logged revivals (`revived`) ran no round, and its `next` holds the follow-up. `run_ended` in the output means the run is over.

### Checkpoints

In checkpoint mode (`checkpoint: true`), or when the user pauses the run, `next` reports the pause. Interactive: show the user the round (`BO summary`), take any brief, directive or protected-path revision (`BO checkpoint --revise <yaml>`, after which every hypothesis is re-reviewed), then `BO checkpoint --resume --rationale "<why>"`. Headless: resume at once. A user's request to stop is `BO stop --rationale "<the user's words>"`.

### Run end

The run ends when the harness says so — the budget is spent, the user stopped it, the target was reached and confirmed, or the hypothesis list is exhausted (a stall never ends it). `BO status` then shows `run_ended`; when its `narrative_missing` names the last round, spawn the round analyst for it. Then **Wrap-up**.

## Wrap-up

<!-- #37 fills this section: distillation, verification, discouraged winners, REPORT.md, clean. -->
