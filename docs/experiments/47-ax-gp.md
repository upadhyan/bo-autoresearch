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

**Post hoc: `ax-crash`.** Added after the `ax` runs at 6 and 12 trials, before running it. `ax` re-suggests a
crashed config until the round ends: Ax drops a failed trial's data and may suggest its config again, and with the
GP unchanged it does. `ax-crash` (`CrashAsGuard`) tells Ax a crash as a broken guard with no metric instead, so the
guard model learns where crashes are. It runs on the same budgets and seeds, and the rule above is applied to
`ax-crash − harness`. Chosen after seeing results, it can't make the decision go on its own: a pass would call for a
confirmatory run on fresh seeds.

```sh
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 60 --trials 6 --repeats 2 --jobs 8 --variant harness --variant ax-crash
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 30 --trials 12 --repeats 2 --jobs 8 --variant harness --variant ax-crash
uv run --frozen --project benchmark python benchmark/ax_gp.py --seeds 30 --trials 24 --repeats 2 --jobs 8 --variant harness --variant ax-crash
```

**Confirmation of `ax-crash`**, fixed after its 12-trial result and before its 6- and 24-trial results. If
`ax-crash` passes the rule at every budget above, it runs again on fresh seeds: 100-159 at 6 trials, 100-129 at 12
and 24. Go only if it passes the rule there too, at every budget.

```sh
uv run --frozen --project benchmark python benchmark/ax_gp.py --first-seed 100 --seeds 60 --trials 6 --repeats 2 --jobs 8 --variant harness --variant ax-crash
uv run --frozen --project benchmark python benchmark/ax_gp.py --first-seed 100 --seeds 30 --trials 12 --repeats 2 --jobs 8 --variant harness --variant ax-crash
uv run --frozen --project benchmark python benchmark/ax_gp.py --first-seed 100 --seeds 30 --trials 24 --repeats 2 --jobs 8 --variant harness --variant ax-crash
```
