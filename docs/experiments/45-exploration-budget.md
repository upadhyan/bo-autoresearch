# #45: an explicit per-round exploration budget

## Hypothesis

Each round's TPE has `n_startup_trials = trials_per_round` (N), and Optuna counts the copied warm-start trials and
the round's finished ones toward it. So round 1 samples every trial after the baseline at random, and a later round
samples at random only until copied plus finished trials reach N: none, unless the warm-start rules left out most
earlier trials (in `default`, round 4, after H3's removal). A fixed budget of k random trials in every round, whatever
was copied in, should find better configs.

## Setup

The offline benchmark ([benchmark/](../../benchmark/README.md)) on its two scenarios, `default` and `staggered` (one
new hypothesis per round; see [#44](44-seeding.md)).

Variants (`bench.VARIANTS`), each running N trials a round as the harness does:

| variant | the trials a round samples |
|---|---|
| `harness` | TPE with `n_startup_trials` N, counting copied and finished trials |
| `explore-0`, `explore-2`, `explore-5` | the first k (0, 2, 5) at random, then TPE with no random start-up |
| `seed-mixed` | as `harness`, after #44's mixed seeds, queued |
| `seed-mixed-explore-0`, `-2`, `-5` | as `explore-k`, after #44's mixed seeds, queued |

A random trial is drawn by Optuna's `RandomSampler`, the way TPE draws its start-up trials: round 1's random trials
of `explore-5` at 6 trials are the harness's. Queued configs (the incumbent, seeds) come first, as today, then the k
random trials, then TPE, so the round's TPE trials see what the random ones found, as after TPE's own start-up.
The random trials are part of N, so every variant runs the harness's trials; eval runs differ only where a crash or
a broken guard stops a trial's repeats (in the results, by at most 1.3 a run on average, against medians of 66-153).
#44 decided against seeding, so the harness doesn't seed; the `seed-mixed` rows show how the answer would change if
it did.

Budgets: `--trials 6 --repeats 2` (the NeuralSGT run's) and `--trials 12 --repeats 2`, 6 rounds, seeds 0-499. From
the repo root:

```sh
for s in default staggered; do for t in 6 12; do
  b="uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 500 --trials $t --repeats 2 --scenario $s"
  $b --variant harness --variant explore-0 --variant explore-2 --variant explore-5
  $b --variant seed-mixed --variant seed-mixed-explore-0 --variant seed-mixed-explore-2 --variant seed-mixed-explore-5
done; done
```

Regret is each run's mean over its 6 rounds; differences are paired per seed (`bench.compare`).

**Decision rule**, fixed before the runs:

- **Yes** for a k when `explore-k` − `harness` has a regret CI entirely below zero on both scenarios at both budgets,
  with eval runs up by no more than 5% of the harness's median. If several k pass, the one with the lowest mean
  difference in most of the four cells. Otherwise **no**.
- `seed-mixed-explore-k` − `seed-mixed` is the interaction with seeding; it doesn't enter the rule.
- A deciding CI that ends within a tenth of its width of zero is run again on 2000 seeds.

## Results

Each run's mean regret over its rounds, in seconds. The `harness` and `seed-mixed` rows are median [IQR] over the 500
seeds; the others are paired differences, mean [95% CI], where negative means the exploring variant does better.

| | default, 6 trials | default, 12 | staggered, 6 | staggered, 12 |
|---|---|---|---|---|
| harness | 0.977 [0.672, 2.01] | 0.53 [0.403, 0.685] | 0.27 [0.225, 0.592] | 0.242 [0.211, 0.321] |
| **explore-0 − harness** | 0.212 [0.0857, 0.346] | 0.125 [0.0603, 0.2] | -0.00179 [-0.0221, 0.0192] | 0.00133 [-0.017, 0.0206] |
| **explore-2 − harness** | 0.003 [-0.107, 0.108] | 0.043 [-0.0125, 0.0985] | 0.0654 [0.0438, 0.0876] | 0.0359 [0.0161, 0.0556] |
| **explore-5 − harness** | 0.137 [0.0673, 0.21] | 0.0728 [0.0305, 0.117] | 0.299 [0.26, 0.338] | 0.0568 [0.0372, 0.0756] |
| seed-mixed | 0.993 [0.657, 1.85] | 0.5 [0.39, 0.665] | 0.465 [0.214, 0.606] | 0.246 [0.209, 0.601] |
| seed-mixed-explore-0 − seed-mixed | 0.228 [0.098, 0.366] | 0.116 [0.044, 0.19] | 0.00605 [-0.0176, 0.0291] | 0.0121 [-0.0119, 0.0364] |
| seed-mixed-explore-2 − seed-mixed | 0.047 [-0.0581, 0.147] | 0.0127 [-0.0416, 0.069] | -0.00602 [-0.0285, 0.0171] | -0.0261 [-0.0477, -0.00352] |
| seed-mixed-explore-5 − seed-mixed | 0.172 [0.09, 0.249] | 0.0781 [0.0297, 0.129] | -0.00433 [-0.0264, 0.0175] | -0.0392 [-0.0611, -0.0176] |

Every k has a cell whose CI lies well above zero, so more seeds can't make a yes.

Where it comes from: median regret per round, `default`, 6 trials (the first command's per-variant tables).

| round | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| harness | 0 | 0.046 | 0.286 | 2.13 | 1.25 | 0.986 |
| explore-0 | 0 | 0.142 | 0.499 | 4.11 | 1.35 | 0.994 |
| explore-2 | 0 | 0.142 | 0.69 | 2.11 | 1.2 | 1.07 |
| explore-5 | 0 | 0.9 | 1.09 | 2.15 | 1.37 | 1.41 |

## Decision

**No.** No k beats the harness in a single cell, and the rule needs all four: `explore-0` loses on `default` at both
budgets (at 12 trials on the mean, through its tail), `explore-2` on `staggered` at both, `explore-5` everywhere.
Nothing changes in the harness.

- The harness's start-up already samples at random where the model has least data: round 1, and `default`'s round 4,
  where rule 2 leaves out nearly every earlier trial (H3 removed, `buffer_kb` almost never sampled at its default).
  Without it (`explore-0`) at 6 trials round 4's median regret doubles. A fixed k spends TPE's trials in rounds with
  plenty of data: `explore-5`'s round 2 median is 0.9 against 0.046.
- On `staggered` every earlier trial is copied, so the harness samples at random only in round 1, where the only
  lever, `buffer_kb`, has no effect: `explore-0` doesn't differ from it.
- With seeds, no k beats `seed-mixed` at the real run's budget or on `default` either. On `staggered` at 12 trials,
  k = 5 does: its random trials cut the tail seeding adds (upper quartile 0.601 → 0.38, the harness's 0.321) while
  the median rises (0.246 → 0.296). #44 left seeding out, so this changes nothing; a revisit of seeding
  should test it with `seed-mixed-explore-5`.
- Roadmap: #45 adopts nothing, so #50 has nothing of it to re-check on Ax.
