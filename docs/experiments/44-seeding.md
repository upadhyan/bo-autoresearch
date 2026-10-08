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
  note 17). In the NeuralSGT run 22 of 35 accepted hypotheses started after round 1. `default` turned out to test
  the premise less still: after H3's removal, round 4 copies in about one of the 18 earlier trials, so H6's first
  round there starts from the baseline and samples mostly at random (first-round table below).

Variants (`bench.VARIANTS`):

| variant | queue |
|---|---|
| `harness` | the incumbent (round 1: the baseline), then `rounds.extra_queue`'s configs |
| `seed-good`, `seed-bad`, `seed-mixed`, `seed-flipped` | also one trial per hypothesis activated this round, from round 2 on: its levers at the seed, the rest at the incumbent |
| `seed-mixed-r1` | the same, from round 1 on |
| `extra-sampled`, `extra-sampled-r1` | the control: as many trials as `seed-*(-r1)` (every seed set seeds every hypothesis), the seeds' slots sampled by TPE |
| `seed-good-startup` | `seed-good`, with `n_startup_trials` raised by the seeds queued, so a seed takes none of the round's random draws (added after review) |

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
H4 bad, H5 good, H6 bad in `default`, and H1 good, H2 bad, H4 bad, H5 good, H6 bad in `staggered`. `seed-flipped`,
added after the first results, swaps them: H4 good, H5 bad, H6 good in `default`, and only H5 bad in `staggered`.

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

Added after the first results, to check the reasons given for the decision:

```sh
for s in default staggered; do for t in 6 12; do
  b="uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 500 --trials $t --repeats 2 --scenario $s"
  $b --variant extra-sampled --variant seed-good --variant seed-flipped
  $b --variant harness --variant seed-flipped
  f="uv run --frozen --project plugin/harness python benchmark/first_round.py --seeds 500 --trials $t --repeats 2 --scenario $s"
  $f --variant harness --variant seed-good --variant seed-mixed --variant seed-flipped --variant extra-sampled
done; done
```

Added after review, to split `default`'s comparisons by hypothesis (`staggered` copies in at least `trials_per_round`
trials from round 2 on, so it has no random draws for `seed-good-startup` to give back):

```sh
for t in 6 12; do
  f="uv run --frozen --project plugin/harness python benchmark/first_round.py --seeds 500 --trials $t --repeats 2"
  $f --own H1,H2 --own H4,H5,H6 --variant harness --variant seed-good --variant seed-mixed --variant seed-good-startup
  $f --own H1,H2 --own H4,H5,H6 --variant extra-sampled --variant seed-good --variant seed-mixed \
    --variant seed-good-startup
  b="uv run --frozen --project plugin/harness python benchmark/bench.py --seeds 500 --trials $t --repeats 2"
  $b --variant harness --variant seed-good-startup
  $b --variant extra-sampled --variant seed-good-startup
done
```

Regret is each run's mean over its 6 rounds; differences are paired per seed (`bench.compare`).

**Decision rule**, fixed before the runs:

- **Yes** only when, on both scenarios at both budgets, `seed-mixed` − `harness` and `seed-mixed` − `extra-sampled`
  both have a regret CI entirely below zero. The second is the cost test: the eval runs the seeds add must buy more
  than the same runs given to TPE. Otherwise **no**.
- `seed-good` and `seed-bad` bound the best and worst case; they don't enter the rule, nor does `seed-flipped`.
- On yes, round 1 is seeded too when `seed-mixed-r1` − `seed-mixed` and `seed-mixed-r1` − `extra-sampled-r1` both have
  a regret CI entirely below zero at both budgets on `default` (`staggered` has only H3 in round 1).
- A deciding CI that ends within a tenth of its width of zero is run again on 2000 seeds.

## Results

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
| seed-flipped − harness | 0.00604 [-0.0659, 0.0749] | -0.00609 [-0.0328, 0.0223] | -0.252 [-0.268, -0.235] | -0.208 [-0.223, -0.194] |
| seed-flipped − extra-sampled | 0.154 [0.0839, 0.227] | 0.0221 [-0.00957, 0.0549] | -0.191 [-0.207, -0.175] | -0.194 [-0.208, -0.181] |

In `staggered`, `seed-good` came out lower than `extra-sampled` on all 500 seeds at both budgets, `seed-flipped`
lower than both on 483-500.

Eval runs per run: the harness's median, then the mean difference.

| | default, 6 trials | default, 12 | staggered, 6 | staggered, 12 |
|---|---|---|---|---|
| harness | 66 | 134 | 72 | 144 |
| seed-good − harness | 5.8 | 4.4 | 10.0 | 10.0 |
| seed-bad, seed-mixed − harness | 4.2, 4.2 | 4.0, 3.9 | 9.0, 9.0 | 9.0, 9.0 |
| extra-sampled − harness | 5.2 | 5.0 | 10.0 | 10.0 |
| seed-mixed-r1 − seed-mixed | 8.0 | 7.2 | 2.0 | 2.0 |
| seed-flipped − harness | 5.8 | 4.4 | 10.0 | 10.0 |
| seed-good, seed-flipped − extra-sampled | 0.6, 0.6 | -0.6, -0.6 | 0, 0 | 0, 0 |
| seed-good-startup − harness, − extra-sampled | 5.9, 0.7 | 4.3, -0.7 | not run | not run |

A hypothesis's own regret is what its levers lose against their best setting; `true_metric` adds a term per
hypothesis, so it is that hypothesis's share of the regret. `default` split by hypothesis (`first_round.py --own`):
the paired difference over the run in the own regret of H1 and H2 (they start in round 1, so no seed touches them),
the same in round 4 alone, and over the run for H4-H6 (the seeded ones). H3's is always 0, so the parts add up to the
total (`bench.py`).

| 6 trials | total | H1, H2 | H1, H2 in round 4 | H4-H6 |
|---|---|---|---|---|
| seed-good − harness | -0.00218 [-0.0733, 0.0651] | 0.0229 [-0.0465, 0.0913] | 0.301 [0.108, 0.489] | -0.025 [-0.0461, -0.00345] |
| seed-mixed − harness | -0.0364 [-0.112, 0.0386] | -0.0169 [-0.0909, 0.0538] | 0.114 [-0.0896, 0.313] | -0.0195 [-0.0404, 0.0027] |
| seed-good-startup − harness | -0.032 [-0.0877, 0.0235] | -0.024 [-0.0781, 0.0327] | -0.0178 [-0.115, 0.0796] | -0.00808 [-0.025, 0.00813] |
| seed-good − extra-sampled | 0.146 [0.0745, 0.217] | 0.124 [0.0558, 0.193] | 0.482 [0.282, 0.681] | 0.0218 [0.00233, 0.0409] |
| seed-mixed − extra-sampled | 0.112 [0.0398, 0.181] | 0.0843 [0.0136, 0.151] | 0.295 [0.0948, 0.49] | 0.0274 [0.00777, 0.0469] |
| seed-good-startup − extra-sampled | 0.116 [0.0605, 0.179] | 0.0772 [0.021, 0.137] | 0.163 [0.0488, 0.283] | 0.0387 [0.021, 0.0565] |

| 12 trials | total | H1, H2 | H1, H2 in round 4 | H4-H6 |
|---|---|---|---|---|
| seed-good − harness | -0.0155 [-0.0447, 0.013] | 0.0126 [-0.0134, 0.0402] | 0.0783 [0.00372, 0.159] | -0.0281 [-0.0422, -0.0144] |
| seed-mixed − harness | 0.00384 [-0.0225, 0.0296] | 0.0136 [-0.0103, 0.0378] | 0.0653 [-0.00573, 0.138] | -0.0098 [-0.0222, 0.00316] |
| seed-good-startup − harness | -0.0257 [-0.0508, -0.00137] | -0.0024 [-0.0253, 0.0185] | -0.0277 [-0.0726, 0.0119] | -0.0233 [-0.0344, -0.0127] |
| seed-good − extra-sampled | 0.0127 [-0.0186, 0.0451] | 0.0139 [-0.0151, 0.0435] | 0.115 [0.0387, 0.201] | -0.00119 [-0.0142, 0.0117] |
| seed-mixed − extra-sampled | 0.032 [0.0056, 0.0596] | 0.0149 [-0.0112, 0.0406] | 0.102 [0.0339, 0.178] | 0.0171 [0.00629, 0.029] |
| seed-good-startup − extra-sampled | 0.00253 [-0.0249, 0.0277] | -0.00114 [-0.025, 0.0207] | 0.00924 [-0.0422, 0.0581] | 0.00366 [-0.00784, 0.0159] |

The round each hypothesis starts in (`first_round.py`). Columns: the earlier trials the round copies in (mean), the
regret of the incumbent it starts from (all levers), then the new levers' own regret at default (all they can save),
in the round's sampled trials and in the incumbent after the round, median [IQR], seconds. The harness, 6 trials:

| scenario, round: hypothesis | copied | incumbent before | at default | sampled trials | incumbent after |
|---|---|---|---|---|---|
| default, 2: H4, int, log | 6 | 1.6 [1.6, 2.2] | 1.6 | 0.0652 [0.0154, 0.165] | 0.0172 [0.00371, 0.0689] |
| default, 3: H5, float, log | 7.02 | 2.08 [2.08, 2.74] | 0.317 | 0.205 [0.0553, 0.609] | 0.0636 [0.013, 0.172] |
| default, 4: H6, float and bool | 1.15 | 13.2 [13.2, 13.2] | 1.32 | 0.644 [0.328, 0.991] | 0.7 [0.472, 0.996] |
| staggered, 2: H1, bool | 6 | 5.9 [5.9, 5.9] | 5.9 | 0 [0, 0] | 0 [0, 0] |
| staggered, 3: H2, categorical | 12 | 4.2 [4.2, 4.2] | 4.2 | 0.6 [0, 0.6] | 0 [0, 0.6] |
| staggered, 4: H4 | 18 | 1.6 [1.6, 2.2] | 1.6 | 0.4 [0.282, 0.583] | 0.2 [0.1, 0.4] |
| staggered, 5: H5 | 24 | 0.688 [0.489, 0.957] | 0.288 | 0.0424 [0.0087, 0.301] | 0.0191 [0.00433, 0.0545] |
| staggered, 6: H6 | 30 | 1.36 [1.25, 1.83] | 1.2 | 1.05 [0.998, 1.11] | 1.02 [0.962, 1.07] |

At 12 trials `default`'s round 4 copies in 1.38 of 36 trials and starts from 13.2 [9.23, 13.2] s, and in
`staggered`'s round 6 the sampled trials leave 1.11 [1.08, 1.14] s of H6's 1.2 and the incumbent 1.07 [1.04, 1.11].
13.2 s is the baseline's regret in `default`'s round 4; the harness's incumbent after that round has 2.13 [1.29, 5.67]
s, and 1.34 [0.977, 1.85] at 12 trials (`bench.py`'s round 4).

The incumbent's own regret after the round, `staggered`, 6 trials (H1's is 0 in every variant):

| round: hypothesis | harness | seed-good | seed-mixed | seed-flipped | extra-sampled |
|---|---|---|---|---|---|
| 3: H2 | 0 [0, 0.6] | 0 [0, 0] | 0.6 [0, 0.6] (bad seed) | 0 [0, 0] | 0 [0, 0] |
| 4: H4 | 0.2 [0.1, 0.4] | 0.0689 [0.0168, 0.1] | 0.057 [0.0154, 0.142] (bad seed) | 0.0689 [0.0168, 0.1] | 0.142 [0.046, 0.282] |
| 5: H5 | 0.0191 [0.00433, 0.0545] | 0.0126 [0.00295, 0.0383] | 0.0136 [0.00346, 0.0353] | 0.0318 [0.00723, 0.0893] (bad seed) | 0.0138 [0.00338, 0.0449] |
| 6: H6 | 1.02 [0.962, 1.07] | 0.45 [0.45, 0.45] | 1.04 [0.983, 1.1] (bad seed) | 0.45 [0.45, 0.45] | 1 [0.942, 1.06] |

## Decision

**No.** `seed-mixed` never beats the harness (its CI spans zero on `default` and lies above it on `staggered`), and
the same extra trials sampled by TPE beat it in all four cells. Round 1 is moot; it fails its own test too. Nothing
changes in the harness: proposals get no seed.

What the no rests on:

- Under the rule `default` settles it alone: no seed set passes there, not even `seed-good` (its CI against the
  harness spans zero at both budgets, and `extra-sampled` beats it at 6 trials). It tests seeds on H4 and H5 only:
  H3's removal leaves round 4 about one copied trial, so it starts from the baseline (13.2 s of regret) in most runs
  at 6 trials and at least half at 12. H6's seed puts H6's levers on it, at least 11.9 s off (13.2 less H6's 1.32)
  against 2.13 s (1.34 at 12 trials) for the harness's incumbent after the round: whatever its setting, it can't
  become the incumbent.
- Split by hypothesis, good seeds do lower the seeded hypotheses' regret on `default` (H4-H6: -0.025 and -0.0281 s,
  CIs below zero), but give much of it back on H1 and H2, which no seed touches (+0.0229 and +0.0126), so the totals
  span zero; most of that is in round 4 (+0.301 and +0.0783). That round has to find `dedup_set` and `route_table`
  again, and a queued seed counts toward `n_startup_trials` like any finished trial, so with about one trial copied
  in it takes one of the round's random draws. `seed-good-startup` gives the draw back: round 4's H1, H2 term drops
  to -0.0178 and -0.0277, and the total against the harness to -0.032 (CI spanning zero) and -0.0257. It still
  doesn't beat `extra-sampled` (+0.116, then +0.00253 with a CI spanning zero): a seed sits on the round's
  incumbent, in most runs the baseline for H1 and H2, which TPE's extra trial can find (round 4's H1, H2 term against
  `extra-sampled`: +0.163 at 6 trials). H1 and H2 carry most of what `extra-sampled` gains at 6 trials over
  `seed-good` (+0.124 of +0.146) and `seed-mixed` (+0.0843 of +0.112).
- On `staggered` the premise holds for H6: its round copies in 30 trials, all at `prefetch` 0, and TPE's samples stay
  near that default, the low end of a linear range. They leave 1.05 s of the 1.2 s H6 can save, though
  `prefetch` alone could save 0.8, and the incumbent after the round leaves 1.02 s. 19 of the NeuralSGT run's 23
  numeric levers had their default on a range bound (retro M6). There good seeds beat the same extra trials sampled
  by TPE (-0.212 and -0.202 s), and so does `seed-flipped`, with only H5's seed bad (-0.191 and -0.194). `seed-mixed`
  loses: its bad seeds fall on H2 and H6, and the odd/even split that put them there is arbitrary. On `staggered`
  alone the answer turns on how good the proposers' seeds are, which this benchmark can't measure.
- Bad seeds: H2 seeded at per_file, better than the default but not best, anchors the search (H2's own regret after
  its round: median 0.6 s, against 0 for the harness). A seed that breaks the guard buys nothing (H6's: 1.04 s
  against 1.02). H4's doesn't crash in `staggered` and helps (0.057 s against 0.2); H5's costs little (0.0318 against
  0.0191).
- Extra trials help without seeds too: `extra-sampled` beat the harness in three cells of four, and `seed-mixed-r1`'s
  -0.29 s over `seed-mixed` (`default`, 6 trials) is no better than `extra-sampled-r1` gets with the same trials.

For the roadmap:

- #45 (exploration budget) branches from here with `seed-*` and `extra-sampled*` registered. `extra-sampled` −
  `harness` shows that one more trial per new hypothesis lowers regret (by 0.148, 0.0282 and 0.0604 s, and by
  0.0139 s with a CI spanning zero at `staggered`, 12 trials), but it spends 5-10 more eval runs per run, and TPE
  samples those trials in every round: `n_startup_trials` is `trials_per_round`, so the extra trial comes after the
  round's random draws. A queued seed instead takes one of them where a round still has them (`default`'s round 4,
  and round 1 for `-r1`); `seed-good-startup` gives it back. It says nothing about #45's k random trials per round.
- NeuralSGT's single-lever probes (retro M5, 64% of the gain) were close to this variant: at round 1's close the
  agent queued a probe for each of its 13 hypotheses with identical boilerplate, as screening, and two of the four
  probe trials behind that gain (11 and 15) were among them. The retro recommends probing on activation. Its
  probes' settings were the agent's, in a space of 15-22 levers that TPE covers less than this one's 7, so that run
  is evidence that seeds can pay; how often a proposer's seeds are good, which `staggered` turns on, stays open.
