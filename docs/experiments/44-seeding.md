# #44: seeding a new hypothesis with a deliberate first trial

## Hypothesis

From round 2 on, the copied warm-start trials use up `n_startup_trials`, so a round samples nothing at random, and
every copied trial has the new hypothesis's levers at their default. TPE then rarely tries a new lever off default.
Queuing one trial per newly active hypothesis, its levers at a proposed setting (its seed) and every other lever at
the incumbent, should find the wins of new hypotheses sooner.

## Setup

The offline benchmark ([benchmark/](../../benchmark/README.md)) on two scenarios:

- `default` (`SCENARIO`): H1-H3 start in round 1; H4 (int) in round 2, crashing above 16; H5 (float) in round 3; H6
  (float and bool, the guard breaks above `prefetch` 0.8) in round 4; then metric shifts.
- `staggered`: H3 in round 1, then H1 (bool), H2 (categorical), H4, H5 and H6, one per round; the world never changes.
  Added because in `default` the only bool or categorical lever that starts after round 1 is `prefetch_async`, which
  pays only with `prefetch` moved too, while TPE almost never samples a numeric lever exactly at its default (design
  note 17). In the NeuralSGT run 22 of 35 accepted hypotheses started after round 1.

Variants (`bench.VARIANTS`):

| variant | queue |
|---|---|
| `harness` | the incumbent (round 1: the baseline), then `rounds.extra_queue`'s configs |
| `seed-good`, `seed-bad`, `seed-mixed` | also one trial per hypothesis activated this round, from round 2 on: its levers at the seed, the rest at the incumbent |
| `seed-mixed-r1` | the same, from round 1 on |
| `extra-sampled`, `extra-sampled-r1` | the control: as many trials as `seed-*(-r1)` (every seed set seeds every hypothesis), the seeds' slots sampled by TPE |

A seed is the proposal's setting of each lever it names; a bool lever it doesn't name is seeded at its non-default
value, a numeric or categorical one not at all (a categorical lever has several non-default choices, and taking the
first would hang on the order they were listed). A seed is an extra trial, deduped like every queued config.

Each seed against the baseline, in `default`'s world in the round its hypothesis starts (true metric, seconds):

| hypothesis | levers (default) | good seed | | bad seed | |
|---|---|---|---|---|---|
| H1 | `dedup_set` (false) | true (bool rule) | 5.9 better | true (no worse setting) | 5.9 better |
| H2 | `route_table` (per_request) | once | 4.2 better | per_file (the weaker choice) | 3.6 better |
| H3 | `buffer_kb` (8) | 64 | no effect | 4096 | no effect |
| H4 | `batch` (1) | 8 | 1.5 better | 64 | crash (`staggered`: 1.2 better) |
| H5 | `gc_scale` (1.0) | 2.0 | 0.24 better | 0.05 | 2.85 worse |
| H6 | `prefetch` (0.0), `prefetch_async` (false) | 0.5, true | 0.83 better | 1.0, true | guard broken |

`seed-mixed` takes the good seed for odd-numbered hypotheses and the bad one for even ones: from round 2 on that is
H4 bad, H5 good, H6 bad in `default`, and H1 good, H2 bad, H4 bad, H5 good, H6 bad in `staggered`.

Budgets: `--trials 6 --repeats 2` (the NeuralSGT run's) and `--trials 12 --repeats 2`, 6 rounds, seeds 0-499. From
the repo root:

```sh
for s in default staggered; do for t in 6 12; do
  b="uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 500 --trials $t --repeats 2 --scenario $s"
  $b --variant harness --variant seed-good --variant seed-bad --variant seed-mixed --variant extra-sampled
  $b --variant extra-sampled --variant seed-mixed
  $b --variant seed-mixed --variant seed-mixed-r1
  $b --variant extra-sampled-r1 --variant seed-mixed-r1
done; done
```

Added after the first results, to compare good seeds with the same extra trials sampled by TPE:

```sh
for s in default staggered; do for t in 6 12; do
  uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 500 --trials $t --repeats 2 --scenario $s \
    --variant extra-sampled --variant seed-good
done; done
```

Regret is each run's mean over its 6 rounds; differences are paired per seed (`bench.compare`).

**Decision rule**, fixed before the runs:

- **Yes** only when, on both scenarios at both budgets, `seed-mixed` − `harness` and `seed-mixed` − `extra-sampled`
  both have a regret CI entirely below zero. The second is the cost test: the eval runs the seeds add must buy more
  than the same runs given to TPE. Otherwise **no**.
- `seed-good` and `seed-bad` bound the best and worst case; they don't enter the rule.
- On yes, round 1 is seeded too when `seed-mixed-r1` − `seed-mixed` and `seed-mixed-r1` − `extra-sampled-r1` both have
  a regret CI entirely below zero at both budgets on `default` (`staggered` has only H3 in round 1).
- A deciding CI that ends within a tenth of its width of zero is run again on 2000 seeds.

## Results

These numbers were measured before #46's warm-start rule 4, which changed `harness` and every variant built on
`bench.Harness` (copying only the last 3 rounds into each round's study). They reproduce at commit `0da9252`.

Each run's mean regret over its rounds, in seconds. The first row is the harness's median [IQR] over the 500 seeds;
the others are paired differences, mean [95% CI], where negative means the first-named variant does better.

| | default, 6 trials | default, 12 | staggered, 6 | staggered, 12 |
|---|---|---|---|---|
| harness | 0.977 [0.672, 2.01] | 0.53 [0.403, 0.685] | 0.27 [0.225, 0.592] | 0.242 [0.211, 0.321] |
| seed-good − harness | -0.00218 [-0.0733, 0.0651] | -0.0155 [-0.0447, 0.013] | -0.272 [-0.288, -0.256] | -0.216 [-0.23, -0.201] |
| seed-bad − harness | -0.0213 [-0.099, 0.0548] | 0.0125 [-0.0119, 0.041] | 0.0394 [0.0158, 0.0628] | 0.0595 [0.0376, 0.0811] |
| **seed-mixed − harness** | -0.0364 [-0.112, 0.0386] | 0.00384 [-0.0225, 0.0296] | 0.0417 [0.0182, 0.0648] | 0.0586 [0.0371, 0.0798] |
| extra-sampled − harness | -0.148 [-0.2, -0.102] | -0.0282 [-0.0464, -0.00806] | -0.0604 [-0.0793, -0.0422] | -0.0139 [-0.0304, 0.00156] |
| **seed-mixed − extra-sampled** | 0.112 [0.0398, 0.181] | 0.032 [0.0056, 0.0596] | 0.102 [0.0802, 0.125] | 0.0724 [0.0496, 0.0955] |
| seed-mixed-r1 − seed-mixed | -0.291 [-0.396, -0.192] | 0.0155 [-0.0191, 0.051] | 0.00819 [-0.0149, 0.0312] | 0.0373 [0.0136, 0.0622] |
| seed-mixed-r1 − extra-sampled-r1 | 0.0465 [-0.0343, 0.127] | 0.0589 [0.0219, 0.0965] | 0.106 [0.0832, 0.128] | 0.0761 [0.0524, 0.0994] |

`seed-good` came out lower than `harness` on all 500 seeds in `staggered` at both budgets. The `default`, 12 trials
`seed-mixed − extra-sampled` CI ended close to zero (0.0056), so it was also run on 2000 seeds: 0.0276 [0.0153, 0.0404].

```sh
uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 2000 --trials 12 --repeats 2 \
  --scenario default --variant extra-sampled --variant seed-mixed
```

Added after the first results, the same way:

| | default, 6 trials | default, 12 | staggered, 6 | staggered, 12 |
|---|---|---|---|---|
| seed-good − extra-sampled | 0.146 [0.0745, 0.217] | 0.0127 [-0.0186, 0.0451] | -0.212 [-0.227, -0.197] | -0.202 [-0.215, -0.189] |

In `staggered`, `seed-good` came out lower than `extra-sampled` on all 500 seeds at both budgets.

Eval runs per run: the harness's median, then the mean difference.

| | default, 6 trials | default, 12 | staggered, 6 | staggered, 12 |
|---|---|---|---|---|
| harness | 66 | 134 | 72 | 144 |
| seed-good − harness | 5.8 | 4.4 | 10.0 | 10.0 |
| seed-bad, seed-mixed − harness | 4.2, 4.2 | 4.0, 3.9 | 9.0, 9.0 | 9.0, 9.0 |
| extra-sampled − harness | 5.2 | 5.0 | 10.0 | 10.0 |
| seed-mixed-r1 − seed-mixed | 8.0 | 7.2 | 2.0 | 2.0 |
| seed-good − extra-sampled | 0.6 | -0.6 | 0 | 0 |

## Decision

**No.** `seed-mixed` never beats the harness (its CI spans zero on `default` and lies above it on `staggered`), and
the same extra trials sampled by TPE beat it in all four cells. Round 1 is moot; it fails its own test too. Nothing
changes in the harness: proposals get no seed.

- `default` settles it alone: no seed set passes there, not even `seed-good`, whose CI against the harness spans zero
  at both budgets.
- On `staggered` good seeds beat the harness and `extra-sampled` on every seed, and bad and mixed ones lose to the
  harness: there the answer turns on how good the proposers' seeds are, which this benchmark can't measure.
- Extra trials help without seeds: `extra-sampled` beats the harness in three cells of four, and `seed-mixed-r1`'s
  gain over `seed-mixed` (`default`, 6 trials) is no more than `extra-sampled-r1` gets with the same trials.
- #45 can test its interaction with seeding on `seed-*` and `extra-sampled*`, which stay in `bench.VARIANTS`.
