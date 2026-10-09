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
- Sampled trials (suggestions) that broke the guard or crashed, per variant: added after the results, from the
  same runs.
- GP fits: suggestions Ax made with its GP (generation node `MBM`). A fit *failed* when BoTorch logged a failed
  attempt (`Fit attempt #k of 5 triggered retry policy` or `failed with exception`) and refit from resampled
  hyperparameters; a fit that raises stops the run. Ax 1.3.1 falls back only when 5 candidates in a row repeat
  configs it has (`exceeded MAX_GEN_ATTEMPTS ... switching to fallback model`): the fit is fine, and the suggestion
  is a Sobol draw (*Sobol instead*).
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

## Results

Every number but the seconds comes exactly from the commands above, which also print each variant's medians by
round and for the whole run (`bench.report`). At 6 trials, `ax − harness` on seeds 0-29 had the CI [−0.725, 0.0607],
within a tenth of its width of zero, so 6 trials ran again on seeds 0-59 (the same command with `--seeds 60`).

**Each variant − `harness`, per seed**: mean regret over the run's 6 rounds, total eval runs.

| trials | seeds | variant | regret (s): mean [95% CI] | lower / tied / higher | eval runs: mean [95% CI] | rule |
|---|---|---|---|---|---|---|
| 6 | 0-59 | `ax` | −0.553 [−0.842, −0.253] | 41 / 0 / 19 | −4.1 [−4.7, −3.5] | pass |
| 12 | 0-29 | `ax` | −0.0894 [−0.284, 0.0549] | 13 / 0 / 17 | −15.2 [−16.6, −13.8] | **fail** |
| 24 | 0-29 | `ax` | −0.105 [−0.16, −0.0474] | 23 / 0 / 7 | −35.3 [−37.3, −33.1] | pass |
| 6 | 0-59 | `ax-crash` | −0.628 [−0.917, −0.327] | 42 / 0 / 18 | −2.3 [−3.0, −1.7] | pass |
| 12 | 0-29 | `ax-crash` | −0.178 [−0.369, −0.0345] | 20 / 0 / 10 | −11.4 [−12.9, −9.8] | pass¹ |
| 24 | 0-29 | `ax-crash` | −0.193 [−0.247, −0.137] | 26 / 0 / 4 | −25.5 [−27.8, −23.1] | pass |
| 6 | 100-159 | `ax-crash` | −0.701 [−1.04, −0.373] | 42 / 0 / 18 | −1.9 [−2.5, −1.4] | pass |
| 12 | 100-129 | `ax-crash` | −0.239 [−0.428, −0.0863] | 21 / 0 / 9 | −12.2 [−13.1, −11.2] | pass |
| 24 | 100-129 | `ax-crash` | −0.162 [−0.22, −0.103] | 25 / 0 / 5 | −25.8 [−27.9, −23.8] | pass |

¹ Its end nearest zero, 0.0345, is just past a tenth of its width (0.0335), so no rerun.

Ax never costs extra eval runs, but its savings are wasted trials: a sampled trial that breaks the guard or crashes
stops its repeats, and Ax's do so more often.

**Sampled trials that broke the guard or crashed**, over all seeds (the suggestion table's `broke guard` and
`crashed`; – not run).

| trials | seeds | sampled | broke guard: `harness` | `ax` | `ax-crash` | crashed: `harness` | `ax` | `ax-crash` |
|---|---|---|---|---|---|---|---|---|
| 6 | 0-59 | 1860 | 170 | 348 | 348 | 154 | 220 | 116 |
| 12 | 0-29 | 2010 | 204 | 467 | 467 | 106 | 300 | 184 |
| 24 | 0-29 | 4170 | 372 | 1019 | 1019 | 249 | 660 | 368 |
| 6 | 100-159 | 1860 | 176 | – | 332 | 154 | – | 114 |
| 12 | 100-129 | 2010 | 179 | – | 464 | 97 | – | 177 |
| 24 | 100-129 | 4170 | 379 | – | 1022 | 256 | – | 386 |

Ax breaks the guard 1.9-2.7× as often as TPE. Guards break only from round 4, where `prefetch` arrives with the
optimum on the guard (0.8) and the metric still falling past it. Crashes happen only in round 2: `ax-crash` crashes
0.74-0.75× as often as TPE at 6 trials and 1.5-1.8× as often at 12 and 24.

**Round 2**, where `batch` arrives and batches over 16 crash: median regret (s), and `ax`'s re-measured trials.

| trials | `harness` | `ax` | `ax-crash` | `ax` re-measured |
|---|---|---|---|---|
| 6 | 0.121 | 0.583 | 0.0689 | 4 [3, 4] |
| 12 | 0.0574 | 0.583 | 0.00371 | 10 [10, 10] |
| 24 | 0.00371 | 0.583 | 0.00371 | 22 [22, 22] |

`ax` suggests its first crashed config again for the rest of the round (its re-measured trials;
`test_ax_suggests_a_crashed_config_again_unless_told_the_crash_as_a_broken_guard` pins this), so its round 2 ends
with whatever it found before that crash. TPE samples, so it doesn't keep repeating one. In rounds 3-6 the two Ax
variants have the same medians, as expected: crashes never join a warm-start set (rule 1), and H4's fix drops round
2's batch trials (rule 3). On seeds 0-59/0-29, Ax's median regret in rounds 3-6 is below TPE's except in round 4 at
12 trials, and furthest below in rounds 5-6.

**GP fits and time per suggestion.** Suggestions are asks the queue didn't answer; Ax's fast quartile is its Sobol
startup. Seconds are wall clock in one thread, with the machine's load average between 4 and 40. TPE's median was
3-5 ms in every run.

| trials | seeds | variant | suggestions | seconds each, median [IQR] | GP fits | fit failed | Sobol instead |
|---|---|---|---|---|---|---|---|
| 6 | 0-59 | `ax` | 1860 | 8.04 [0.105, 16.4] | 1329 | 0 | 0 |
| 12 | 0-29 | `ax` | 2010 | 7.31 [0.1, 17.4] | 1387 | 0 | 1 |
| 24 | 0-29 | `ax` | 4170 | 9.06 [0.1, 19.2] | 2834 | 0 | 0 |
| 6 | 0-59 | `ax-crash` | 1860 | 12.4 [0.128, 20.6] | 1329 | 0 | 0 |
| 12 | 0-29 | `ax-crash` | 2010 | 9.17 [0.113, 19.8] | 1387 | 0 | 1 |
| 24 | 0-29 | `ax-crash` | 4170 | 12.3 [0.0997, 23.8] | 2834 | 0 | 1 |
| 6 | 100-159 | `ax-crash` | 1860 | 11.6 [0.126, 19.9] | 1331 | 0 | 0 |
| 12 | 100-129 | `ax-crash` | 2010 | 10.9 [0.113, 20] | 1391 | 0 | 0 |
| 24 | 100-129 | `ax-crash` | 4170 | 11.2 [0.109, 21] | 2835 | 0 | 3 |

None of the 16,657 GP fits failed or raised; 6 suggestions were Sobol draws. A GP suggestion takes about 10-25 s
against TPE's few milliseconds, with 7 levers and runs of at most 144 trials. The NeuralSGT run had 23 levers and 246
trials over 30 rounds (median trial 922 s); GP fitting and the acquisition slow down with both, so the overhead there
is unmeasured. Each process holds about 0.5 GB with torch loaded.

**How Ax 1.3.1 models the levers** (`method="fast"`, read from the fitted model in round 4):

- A `SingleTaskGP` for the metric and one for the guard, each with an RBF kernel and dimension-scaled LogNormal
  lengthscale priors: not the Matern kernel or the categorical kernel the hypothesis names. Outcomes are winsorized
  and standardized; the guard is also bilog-transformed around its bound.
- The 3-way categorical (`route_table`) is one-hot encoded into three features on [0, 1], relaxed to continuous in the
  acquisition and mapped back to the largest. Bools are 0/1 features. Log ints (`batch`) become discrete choices on a
  log10 scale; log floats are continuous in log10.
- A categorical kernel (`MixedSingleTaskGP` with `CategoricalKernel`, Ax's `Generators.BO_MIXED`) isn't among
  `configure_generation_strategy`'s methods (`fast`, `quality`, `random_search`); it takes a hand-built
  `GenerationStrategy`.
- The acquisition is `qLogNoisyExpectedImprovement` weighted by the guard's probability of holding, optimised by
  BoTorch's `optimize_acqf_mixed_alternating`: continuous gradient steps alternating with a local search over the
  discrete features.

**API churn met on Ax 1.3.1, BoTorch 0.18.1, torch 2.14.1, Python 3.14:**

- The 1.x `Client` (`ax.api.client`) replaces `AxClient`: parameters as `RangeParameterConfig` and
  `ChoiceParameterConfig`, objective and constraints as strings (`"-metric"`, `"guards <= 0.5"`), `get_next_trials`
  returning `{index: params}`, `attach_trial`, `complete_trial(raw_data)`, `mark_trial_failed`, `summarize()`. It has
  no queue: a queued config is an attached trial.
- 1.x renamed models to generators throughout (`Models` → `Generators`, `ModelSpec` → `GeneratorSpec`,
  `GeneratorRun._model_key` → `_generator_key`), so code against 0.x internals breaks.
- Ax 1.3.1's own `configure_optimization` calls its deprecated `add_tracking_metric`, and every bool lever warns that
  `is_ordered` is unset.
- The centre-point quirk above, and a failed trial's config being suggested again.
- Fit diagnostics are only in logs: BoTorch reports a failed fit attempt at DEBUG, Ax a fallback at WARNING, and the
  node that made a trial at INFO. `AxBackend` reads them from the log text.
- torch warns that `torch.jit.script`, which GPyTorch uses, is unsupported on Python 3.14; it ran. BoTorch 0.18.1 turns
  off its batched L-BFGS-B for scipy 1.18 and later (the lock has 1.18.1), so the acquisition runs its slower path.

## Decision

**Go**, for Ax with a crash told as a broken guard.

- The pre-registered `ax` fails the rule at 12 trials and passes at 6 and 24: it loses round 2 by suggesting its
  first crashed config again until the round ends (Round 2).
- `ax-crash` changes only that, and passes the rule at every budget, on the original seeds and on the fresh ones.
  Being post hoc, it decides only through that confirmation, fixed before its 6- and 24-trial results. Its fewer
  eval runs are not a saving: they are repeats skipped by sampled trials that broke the guard or crashed (Sampled
  trials).
- Costs: seconds per GP suggestion against TPE's milliseconds (unmeasured at the NeuralSGT run's 23 levers), torch's
  memory, and those wasted trials. No GP fit failed or raised (GP fits).
- Limits: one synthetic scenario, and a GP's best case: the metric is a sum of one term per lever, smooth in each
  numeric one (quadratic in log `batch` and log `gc_scale`, linear in `prefetch`), with the optimum on the guard at
  `prefetch` 0.8. The issue's evidence found no framework better on mixed hierarchical spaces. 7 levers (the
  NeuralSGT run had up to 23), and 30-60 seeds per budget.

For the roadmap:

- #50 goes ahead: an Ax backend behind a config flag, built as `ax_gp.CrashAsGuard` is (Setup). Before it, re-check
  the adopted outcomes of #44-#46 on `ax-crash`, as the roadmap says.
- #48 and #49 go ahead on the benchmark's `ax-crash`; #49 starts from its guard and crash handling and the trials it
  wastes, and #52 from #49.
- #53 goes ahead: Ax and CPU-only torch as an opt-in install, as `benchmark/pyproject.toml` does here, and a fallback
  for a missing torch or a fit that raises (none did here).
- #51 follows #48 as planned. Nothing in `plugin/` changes here.
