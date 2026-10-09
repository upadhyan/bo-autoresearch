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

These numbers were measured before #46's warm-start rule 4, which changed `harness` and, through `Harness.warm`, both
Ax variants (copying only the last 3 rounds into each round's study). They reproduce at commit `0da9252`.

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
with whatever it found before that crash. TPE samples, so it doesn't keep repeating one. Outside round 2 the two Ax
variants have the same medians, as expected: round 1 has no crash, crashes never join a warm-start set (rule 1), and
H4's fix drops round 2's batch trials (rule 3). On seeds 0-59/0-29, Ax's median regret in rounds 3-6 is below
TPE's except in round 4 at 12 trials, and furthest below in rounds 5-6.

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

**Go**, for Ax with a crash told as a broken guard. Under the rule as first written, which applies to the
pre-registered `ax`, it is no-go: the go rests on the post hoc `ax-crash`, a deviation to accept knowingly.

- The pre-registered `ax` fails the rule at 12 trials and passes at 6 and 24: it loses round 2 by suggesting its
  first crashed config again until the round ends (Round 2). That is Ax's design, not the scenario's: it treats a
  failed trial as transient and leaves its config out of its dedupe set (`arms_by_signature_for_deduplication`),
  so `mark_trial_failed` was unsound for a crash that repeats.
- `ax-crash` changes only that, and passes the rule at every budget, on the original seeds and on the fresh ones.
  Being post hoc, it decides only through that confirmation, fixed before its 6- and 24-trial results. Outside
  round 2 it has `ax`'s medians, and in round 2 TPE's or lower (Round 2): it removes `ax`'s round-2 loss, and the
  rest of its lead over TPE is `ax`'s. Its fewer eval runs are not a saving: they are repeats skipped by sampled
  trials that broke the guard or crashed (Sampled trials).
- Costs: seconds per GP suggestion against TPE's milliseconds (unmeasured at the NeuralSGT run's 23 levers), torch's
  memory, and those wasted trials. No GP fit failed or raised (GP fits).
- Limits: one synthetic scenario, the same one whose round-2 crash prompted `ax-crash`, so the fresh seeds confirm
  its crash handling only on that crash; and a GP's best case: the metric is a sum of one term per lever, smooth in
  each numeric one (quadratic in log `batch` and log `gc_scale`, linear in `prefetch`), with the optimum on the guard
  at `prefetch` 0.8. The issue's evidence found no framework better on mixed hierarchical spaces. 7 levers (the
  NeuralSGT run had up to 23), and 30-60 seeds per budget.

For the roadmap:

- #50 goes ahead: an Ax backend behind a config flag, built as `ax_gp.CrashAsGuard` is (Setup). Before it, re-check
  the adopted outcomes of #44-#46 on `ax-crash`, as the roadmap says.
- #48 and #49 go ahead on the benchmark's `ax-crash`; #49 starts from its guard and crash handling and the trials it
  wastes, and #52 from #49.
- #53 goes ahead: Ax and CPU-only torch as an opt-in install, as `benchmark/pyproject.toml` does here, and a fallback
  for a missing torch or a fit that raises (none did here).
- #51 follows #48 as planned. Nothing in `plugin/` changes here.

## Re-validation (pre-registered 2026-10-09, not yet run)

The go above rests on `ax-crash`, chosen after seeing results, on the one scenario whose crash prompted it, and its
numbers predate #46's warm-start rule 4. Before #50 builds on it, `ax-crash` runs again under rule 4 on fresh seeds
(A) and on a second, harder scenario it wasn't fitted to (B). Everything in this section, the rule included, is
fixed before either runs. Nothing has been run on the new scenario but the scenario's own checks (its tests and the
properties below, from random configs).

### The `training` target

`benchmark/training.py`: a model trained for a fixed wall-clock time, tuned for dev error (%), shaped after the one
real run (the NeuralSGT retro: 23 levers, most numeric defaults on a range bound, inert and conditional levers, a
regression-guard floor, harmful hypotheses removed). `bench.simulate` runs it as a target; `ax_gp.py --target` picks
it. Three versions differ only in two switches; #48 and #49 use the other two.

| target | aux_weight changes how lr behaves | guard moves at round 7 | used by |
|---|---|---|---|
| `training` | yes | yes (+0.06 R²) | #47 B, #49 stage 2 |
| `training-fixed-guard` | yes | no (round 7 is a commit that changes nothing) | #48, #49 |
| `training-additive` | no (aux_weight only adds error) | no | #48 |

**Levers.** 20 levers in 15 hypotheses: 4 bool, 2 categorical, 4 log int, 1 int, 2 log float, 7 float; 10 of the
14 numeric defaults sit on a bound, so a sampled trial never has one exactly at default. Active: 10 in round 1, 13,
15, 16, then 19 from round 5.

| round | change (a non-empty step is a new commit) |
|---|---|
| 1 | lr_scale + warmup_frac, batch, pin_memory + num_workers, aug_strength, schedule + lr_floor, init_scale, weight_decay |
| 2 | aux_weight (harmful), width, label_smoothing: the guard can now break |
| 3 | checkpointing, precision: fp16 or checkpointing halves the memory; from round 4 the best width fits only with one |
| 4 | dropout; H4 investigated: its off-state (the incumbent with aux_weight 0) and aux_weight 0.1 are queued |
| 5 | H4 removed (rule 2 drops every round 2-4 trial with aux_weight off 0); ema + ema_halflife, cudnn_benchmark + log_every |
| 6 | same commit |
| 7 | `training`: the guard loosens by 0.06; the others: a new commit that changes nothing |
| 8, 9 | same commit |

**Metric** (dev error; regret is in the same units, and the tables' "regret (s)" header is the toy target's):

- lr is best at √(batch/64), times 1 + 1.5 × aux_weight where aux_weight interacts; warmup's best grows with lr; an
  lr past 3× its best adds 2.0 (the run diverges and recovers late), the scenario's numeric step.
- Wider is better but runs fewer steps (best width 548 alone); dropout's best grows with width.
- Smooth terms for aug_strength, label_smoothing, init_scale, weight_decay; schedule and precision are choices;
  lr_floor matters only under cosine (cosine without it loses to step), ema_halflife only with ema on (ema at the
  default halflife hurts).
- 4 dead levers (pin_memory, num_workers, cudnn_benchmark, log_every).
- Crash: out of memory when batch × width × bytes per value (× 0.35 with checkpointing) passes a limit, at every
  commit. It sits just past the best batch × width without binding there: the round-2 optimum (fp32) uses 93% of
  it, round 3's (fp16) 46%; from round 4, when dropout moves the best width to 548, that width fits only with fp16 or
  checkpointing (53% with fp16).
- Guard: the regression head's R² ≥ 0.73. Augmentation and smoothing together cost R², and so does aux_weight above
  0.7; neither augmentation nor smoothing alone can break it. The best feasible config sits on the floor. R² is
  measured with noise on every repeat (sd 0.01 shared per repeat plus 0.01 per config, paired like the metric), so a
  config on the floor fails a repeat about half the time, and the trial records its lowest measured margin.
- Noise: 0.05 relative per part (#47's scenario: 0.02), closer to the real run's ratio of gain to seed noise.

**Properties** (random configs, log-uniform for log levers; 20,000 per round):

| round | levers | optimum | baseline regret | crash | guard broken |
|---|---|---|---|---|---|
| 1 | 10 | 5.910 | 3.30 | 7% | 0% |
| 2 | 13 | 5.045 | 4.16 | 25% | 53% |
| 3-4 | 15-16 | 4.795, 4.725 | 4.41, 4.48 | 10% | 63% |
| 5-6 | 19 | 4.475 | 4.73 | 10% | 26% |
| 7-9 `training` | 19 | 4.271 | 4.93 | 10% | 4% |
| 7-9 others | 19 | 4.475 | 4.73 | 10% | 25% |

Leaving one lever at default from the round-5 optimum costs: aug_strength 1.37, batch 0.94, width 0.68, init_scale
0.60, lr_scale 0.46, label_smoothing 0.40, schedule 0.35, weight_decay 0.31, ema_halflife 0.29, lr_floor 0.25, ema
0.25, warmup_frac 0.10, dropout 0.08; precision's default crashes there. A repeat's noise sd at the optimum is about
0.30. The round-7 move is worth 0.20 a round.

`best_config` gives each round's optimum exactly (enumeration plus closed forms); `test_training.py` checks it against
an independent grid search over every lever and 2000 random configs per round and target.

### Setup

**Arms**, each the harness's round with rules 1-4 (current main):

| arm | optimizer | crash told as |
|---|---|---|
| `harness` | `OptunaBackend` (multivariate TPE) | `FAIL`: TPE never sees it |
| `harness-crash` | `OptunaBackend` | a broken guard (reported only; see below) |
| `ax-crash` | `ax_gp.CrashAsGuard` | a broken guard |

`harness-crash` is TPE told a crash as an infeasible trial, as `CrashAsGuard` tells Ax. It separates a GP beating TPE
from crash-awareness beating crash-blindness, which the new target makes count more than #47's did (its crash region
exists in every round; #47's only in round 2).

| part | target | trials per round | seeds | repeats |
|---|---|---|---|---|
| A | `default` (bench.SCENARIO, 6 rounds) | 6, 12 / 24 | 200-229 / 200-214 | 2 |
| B | `training` (9 rounds) | 6, 12 / 24 | 0-29 / 0-14 | 2 |

Amended before any run, to halve the compute: these were 60 seeds at 6 and 12 trials and 30 at 24. At 30 seeds the
pass probability at 12 trials was 0.66-0.88 even at #47's own effect sizes, so UNRESOLVED is more likely; the rerun
clause still doubles the seeds of a comparison that lands near its threshold.

**Deciding measure:** per seed, the mean regret over the run's rounds, `ax-crash − harness`, and its 95% bootstrap CI
[L, U] (`bench.compare`: 2000 resamples of the seeds, percentiles; at 30-60 seeds its one-sided miss rate is about
3% against a nominal 2.5%). Regret ignores the true guard: an incumbent that passed its noisy repeats but breaks the
true guard can score a little below 0 (about −0.03 to −0.1 a round). `bench.compare` also prints, per seed, the
difference in rounds that ended on such an incumbent ("infeasible incumbents"), with the same CI [L′, U′].

**Rerun clause**, for every deciding CI in #47-#49: if the CI end nearer the threshold (0 here) is within a tenth of
the CI's width of it, the comparison runs once more on twice the seeds (same first seed) and is decided on that CI.

### Decision rule

Each of the six comparisons (A and B at 6, 12 and 24 trials) is PASS if U < 0 and L′ ≤ 0, WORSE if L > 0, else
UNRESOLVED. L′ > 0 means `ax-crash` ships more incumbents that break the true guard, so a lower regret doesn't pass
on it (the default target's guard has no noise, so its L′ is always 0).

- **GO** if all six PASS.
- **NO-GO** if any is WORSE.
- **INCONCLUSIVE** otherwise: #48 and #49 don't start, and you decide between more seeds and stopping.

The earlier rule's cap on Ax's extra eval runs is dropped: with the trials per round fixed, a variant runs more eval
runs only by breaking the guard or crashing less, so the cap could only fail an improvement. Eval runs are reported.

**Reading `harness-crash`**, also fixed now: at a budget where `ax-crash − harness` passes but `ax-crash −
harness-crash` doesn't (U ≥ 0), the write-up says the gain there is crash handling, not the GP; on a GO with that
caveat, you decide between #50 and giving TPE crash-as-infeasible (the cheaper fix) before #50 starts.

**Reported, not deciding:** sampled trials that broke the guard or crashed per arm; in B, round-2-4 trials rule 2
keeps at round 5 per arm (Ax can land exactly on aux_weight's bound, TPE can't); seconds per suggestion and GP fits;
eval runs. Each command saves its runs with `--out`, and the counts not printed come from that file.

| outcome | roadmap |
|---|---|
| GO | #50 proceeds (re-check #46's rule 4 there); #48 and #49 run |
| GO with the crash caveat | you choose: #50, or crash-as-infeasible on TPE; #48 and #49 run only if you choose #50 |
| NO-GO | the roadmap closes #48-#53; you confirm first |
| INCONCLUSIVE | you choose: more seeds or stop |

### Commands

`ax_gp.py` compares each arm with every earlier one, so `ax-crash − harness-crash` prints alongside `ax-crash −
harness`.

```sh
uv run --frozen --project benchmark python benchmark/ax_gp.py --first-seed 200 --seeds 30 --trials 6 --repeats 2 --jobs 8 --variant harness --variant harness-crash --variant ax-crash --out a6.json
uv run --frozen --project benchmark python benchmark/ax_gp.py --first-seed 200 --seeds 30 --trials 12 --repeats 2 --jobs 8 --variant harness --variant harness-crash --variant ax-crash --out a12.json
uv run --frozen --project benchmark python benchmark/ax_gp.py --first-seed 200 --seeds 15 --trials 24 --repeats 2 --jobs 8 --variant harness --variant harness-crash --variant ax-crash --out a24.json
uv run --frozen --project benchmark python benchmark/ax_gp.py --target training --seeds 30 --trials 6 --repeats 2 --jobs 8 --variant harness --variant harness-crash --variant ax-crash --out b6.json
uv run --frozen --project benchmark python benchmark/ax_gp.py --target training --seeds 30 --trials 12 --repeats 2 --jobs 8 --variant harness --variant harness-crash --variant ax-crash --out b12.json
uv run --frozen --project benchmark python benchmark/ax_gp.py --target training --seeds 15 --trials 24 --repeats 2 --jobs 8 --variant harness --variant harness-crash --variant ax-crash --out b24.json
```

**Compute**, at #47's 10-25 s per GP suggestion (a planning probe on 20 random levers took 11-29 s): A about 10,000
Ax suggestions, B about 15,000 at the original seeds; with the seeds halved, about 6.5 h on 8 processes (one thread
each: a single process with 8 threads was 2.5 times slower per suggestion). TPE's arms take minutes.

### Limits fixed in advance

- Still synthetic, and still mostly smooth: most of the regret sits in terms smooth in a GP's log encoding; the
  divergence step, the choices and the conditional levers are the rest. The crash is permanent and spans batch,
  width, precision and checkpointing, but is a half-space in that encoding, the easiest shape to learn.
- One removal (a harmful numeric lever), no supersede; the benchmark queues only the incumbent and that removal's
  investigation, while probes made 64% of the real run's gain, so the optimizer's share of the search is overstated.
- Guard noise is gentler than the real run's (a 2% change to one lever moved its R² by more than its whole tolerance).
- 9 rounds against the real run's 30.
