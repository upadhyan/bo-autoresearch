# #47: does an Ax GP beat TPE at our trial budgets?

## Hypothesis

A GP through Ax (BoTorch underneath) finds better configs than the harness's multivariate TPE at 6-24 trials per
round, under the same warm-start rules. Evidence: in independent benchmarks Ax/BoTorch matched or beat other engines
at about 8-100 evaluations (arXiv 2311.15854, 2603.29730), but against default univariate TPE.

## Setup

**Scenario.** The benchmark's default `SCENARIO` (6 rounds): 7 levers of every type (bool, 3-way categorical, log
int, log float, linear float), a guard, a crash, a fix, a removal and metric shifts. It has every lever type the
hypothesis is about, so no scenario was added.

**Variants.** Both run the harness's round: warm-start rules 1-3, the incumbent queued, the queued-config dedupe.

| variant | optimizer |
|---|---|
| `harness` | `OptunaBackend`: `TPESampler(multivariate=True, n_startup_trials=trials per round)` |
| `ax` | `ax_gp.AxBackend`: Ax 1.3.1's `Client` with its default strategy (`method="fast"`): Sobol startup, then BoTorch's GP and acquisition |

`AxBackend`'s choices, fixed before any results:

- Startup: `initialization_budget` = trials per round, with warm and queued trials counted, as TPE counts them
  towards `n_startup_trials`. Ax's centre point is off: a new `Client` with trials already attached skips it but
  still spends a Sobol draw, so every round would start with a random trial that TPE doesn't make.
- Guards: a second metric, `guards` (0 held, 1 broke), with the outcome constraint `guards <= 0.5`. TPE gets the
  same 0/1 as a constraint, so both see the same information; #49 explores alternatives.
- Crashes: `mark_trial_failed`, as TPE gets `FAIL`. A failed trial's config carries no data either way.
- Warm trials: `attach_trial` and `complete_trial` with their metric and guard, levers new to the space at default.
- Seeds: `Client(random_seed=seed)` and `initialization_random_seed=seed`, seed = the harness's seed + round.

**Budgets.** 6 trials per round (the real run's) and 12 and 24 (the issue's 10-30), each with 2 repeats (the
real run's), 6 rounds.

**Seeds.** 0-29 for both variants. A GP suggestion takes seconds, so 30 rather than 100.

**Measures.**

- Regret and eval runs: `bench.report` per variant and `bench.compare` for `ax − harness` per seed.
- GP fits: suggestions Ax made with its GP (generation node `MBM`). One *fell back* when BoTorch logged a failed
  fit attempt (`Fit attempt #k of 5 triggered retry policy` or `failed with exception`) and refit from resampled
  hyperparameters, or when Ax logged `switching to fallback model` and drew a Sobol point instead. A fit that raises
  stops the run.
- Time per suggestion: wall clock of each ask the queue didn't answer, in processes of one thread each, 8 at a time
  on a shared 20-core WSL2 machine. Timings vary with load; every other number is reproducible exactly.

**Commands**, from the repo root:

```sh
uv sync --frozen --project benchmark  # Ax 1.3.1, BoTorch 0.18.1, torch 2.14.1+cpu, Python 3.14
uv run --frozen --project benchmark pytest benchmark/tests -q
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 30 --trials 6 --repeats 2 --jobs 8
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 30 --trials 12 --repeats 2 --jobs 8
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 30 --trials 24 --repeats 2 --jobs 8
```

`--jobs` changes nothing but the wall clock.

**Decision rule**, fixed before the runs: go only if, at each of 6, 12 and 24 trials per round, the 95% CI of
`ax − harness` regret lies below zero and Ax's extra eval runs, if any, are at most 5% of the harness's median total.
Otherwise no-go. A CI whose end nearest zero is within a tenth of its width of zero is rerun on seeds 0-59 and the
rule applied to that. Time per suggestion and fit failures don't decide; on a go they size #50 and #53.
