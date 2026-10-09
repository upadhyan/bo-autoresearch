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
a broken guard stops a trial's repeats. #44 decided against seeding, so the harness doesn't seed; the
`seed-mixed` rows show how the answer would change if it did.

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
