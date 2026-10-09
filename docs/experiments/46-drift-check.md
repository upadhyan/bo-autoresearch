# #46: does re-measuring the top copied configs detect drift, and does dropping what it contradicts help?

## Hypothesis

A commit between rounds can shift the metric. Each round already re-measures the incumbent on a new commit.
Re-measuring the next 2 best copied configs too (repeats pooled) and checking whether their order still agrees with
the earlier measurements should tell real reordering apart from noise. Leaving out the trials the check contradicts,
or every older trial once a check fails, or ageing old rounds out, should beat copying everything in, even after
paying for the re-measurements.

## Setup

Offline benchmark (`benchmark/bench.py`, #42), driven by `benchmark/drift_check.py`. Noise model unchanged: each
repeat's noise is shared by all configs at that repeat index plus each (commit, config)'s own, so configs compare
paired and a config measured again on the same commit gets the same values.

**Scenarios.** What changes before each round, and whether it reorders configs:

| round | `default` (`bench.SCENARIO`) | `drift` (`drift_check.DRIFT`) |
|---|---|---|
| 1 | H1-H3 | H1-H3 |
| 2 | H4 (crashes over batch 16): no shift | H4: no shift |
| 3 | fix H4, H5, everything 1.1x slower: uniform | H5, everything 1.1x slower: uniform |
| 4 | remove H3, H6, best batch 16 → 4: reorders | best batch 16 → 4: reorders |
| 5 | same commit | same commit |
| 6 | per_file now beats once: reorders | H6, best gc_scale 4 → 1: reorders |
| 7 | | per_file now beats once: reorders |
| 8 | | everything back to 1x: uniform |

In `default`, removing H3 before round 4 leaves out (rule 2) every copied trial but the baseline, since TPE never
samples buffer_kb at its default; so the round 4 shift has nothing to contradict, and its only check with copied
data is round 6. `drift` has no removal and a shift at most new commits. A test checks that `drift` reorders
configs exactly in rounds 4, 6 and 7.

**The check.** Each round queues, after the incumbent, the next n − 1 best configs of the warm-start set (pooled), as
`recheck i`, on top of its N trials (like an investigation's queue). A config already measured on the round's commit is
not run again, so on an unchanged commit only top configs not yet measured on it are. A pair of the re-measured configs
is *contradicted* when its order on the new commit goes against its pooled order over earlier trials by more than the
noise floor (from trials before the round, scaled to the pair's earlier metric) on both sides. With no floor (repeats
that never differ), any reversal counts.

**Detection.** A check is *misordered* when the same check on noise-free metrics (each earlier trial's true metric
at its own round, pooled the same way) finds a reversal: the copied data orders a pair of the re-measured configs
the other way from the truth on the new commit. Hit rate: flagged among misordered checks. False alarms: flagged
among the rest. Swept: noise (`bench.NOISE`) 0, 0.01, 0.02 (the benchmark's), 0.05, 0.1; n = 2, 3, 4.

**Variants**, fixed before any run (rule 4 is the new warm-start rule). `copy-all` is the harness this experiment
started from, which copies the whole warm-start set into each round's study (named `harness` until rule 4 went in):

| variant | re-measures (n) | rule 4 leaves out |
|---|---|---|
| `copy-all` | incumbent only (1) | nothing |
| `recheck` | 3 | nothing: the cost of the re-measurements alone |
| `drop-trials` | 3 | the earlier trials of both configs of each contradicted pair |
| `drop-older` | 3 | every earlier trial, once a check flags |
| `age-3` | incumbent only (1) | trials more than 3 rounds old |

A round's study is built before its re-measurements run, so a check's verdict reaches the warm start from the next
round on. The incumbent after the round, which regret scores, already applies it.

**Budgets and seeds.** 6 trials × 2 repeats per round (the real run's) and 12 × 2; seeds 0-199 (0-999 for reruns).
Regret at noise 0.02. Commands, from the repo root, with `drift_check` standing for
`uv run --frozen --project plugin/harness python benchmark/drift_check.py`:

```sh
drift_check --detect --scenario default --seeds 200 --trials 6
drift_check --detect --scenario default --seeds 200 --trials 12
drift_check --detect --scenario drift --seeds 200 --trials 6
drift_check --detect --scenario drift --seeds 200 --trials 12
V="--variant copy-all --variant recheck --variant drop-trials --variant drop-older --variant age-3"
drift_check --scenario default --seeds 1000 --trials 6 $V
drift_check --scenario default --seeds 1000 --trials 12 $V
drift_check --scenario drift --seeds 1000 --trials 6 $V
drift_check --scenario drift --seeds 200 --trials 12 $V
# the cost check: copy-all with 2 more trials per round, against the variants that re-measure
drift_check --scenario drift --seeds 1000 --trials 6 --base-trials 8 --variant copy-all --variant drop-trials --variant drop-older
drift_check --scenario drift --seeds 200 --trials 12 --base-trials 14 --variant copy-all --variant drop-trials --variant drop-older
# added after the first results (see "Without drift")
drift_check --scenario steady --seeds 200 --trials 6 --variant copy-all --variant age-3 --lever prefetch_async
drift_check --scenario steady --seeds 200 --trials 12 --variant copy-all --variant age-3
drift_check --scenario steady --seeds 200 --trials 6 --variant copy-all --variant recheck --variant drop-trials --variant drop-older
drift_check --scenario unchanged --seeds 200 --trials 6 --variant copy-all --variant age-3 --lever prefetch_async
drift_check --scenario unchanged --seeds 200 --trials 12 --variant copy-all --variant age-3 --lever prefetch_async
# added after review (see "Ageing the study's copy only")
S="--variant copy-all --variant age-3-study"
drift_check --scenario default --seeds 1000 --trials 6 $S
drift_check --scenario default --seeds 1000 --trials 12 $S
drift_check --scenario drift --seeds 1000 --trials 6 $S
drift_check --scenario drift --seeds 200 --trials 12 $S
drift_check --scenario steady --seeds 200 --trials 6 $S
drift_check --scenario steady --seeds 200 --trials 12 $S
drift_check --scenario unchanged --seeds 200 --trials 6 $S
drift_check --scenario unchanged --seeds 200 --trials 12 $S
```

**Decision rule** (written before any run). A rule is "yes" only if, against `copy-all` on the same seeds:

1. on `drift`, the mean regret difference's 95% CI lies below 0 at both budgets;
2. on `default`, that CI does not lie above 0 at either budget (it must not hurt where shifts are rarer);
3. a rule that re-measures must also have its regret CI below 0 against `copy-all` given 2 more trials per round
   (`--base-trials`), which spends at least the eval runs the re-measurements do, on `drift` at both budgets: else
   the same eval runs are better spent on more trials. `age-3` adds no eval runs.

A CI whose nearer end is within a fifth of its width of 0 is rerun with 1000 seeds. Otherwise "no", and the harness
stays as it is.

## Results

### Detection

At noise 0.02 with 3 configs re-measured:

| scenario | budget | checks | misordered | hits | false alarms |
|---|---|---|---|---|---|
| `default` | 6×2 | 622 | 18 | 11/18 | 23/604 (4%) |
| `default` | 12×2 | 651 | 7 | 6/7 | 56/644 (9%) |
| `drift` | 6×2 | 1309 | 519 | 135/519 (26%) | 75/790 (9%) |
| `drift` | 12×2 | 1353 | 620 | 100/620 (16%) | 90/733 (12%) |

`drift`, 6×2: hits / false alarms by noise and configs re-measured (n). At noise 0 the check is the truth: 100% / 0%.

| noise | n = 2 | n = 3 | n = 4 |
|---|---|---|---|
| 0.01 | 34% / 3% | 40% / 7% | 52% / 12% |
| 0.02 | 15% / 5% | 26% / 9% | 34% / 15% |
| 0.05 | 9% / 5% | 15% / 10% | 23% / 17% |
| 0.1 | 5% / 7% | 19% / 12% | 22% / 19% |

The check flags only reversals beyond the noise floor on both sides; the truth counts any reversal. Hits fall from
100% at noise 0 to 26% at 0.02. More configs catch more and raise more false alarms.

`drift`, 6×2, noise 0.02, n = 3, by round (the same command's second table):

| round | shift | checks | misordered | hits | false alarms |
|---|---|---|---|---|---|
| 2 | none (H4 added) | 200 | 0 | — | 7/200 |
| 3 | uniform (1.1x slower) | 200 | 0 | — | 24/200 |
| 4 | reorders (batch) | 200 | 154 | 55/154 | 5/46 |
| 5 | none (same commit) | 109 | 60 | 4/60 | 2/49 |
| 6 | reorders (gc_scale) | 200 | 158 | 49/158 | 10/42 |
| 7 | reorders (route_table) | 200 | 43 | 12/43 | 20/157 |
| 8 | uniform (1x again) | 200 | 104 | 15/104 | 7/96 |

- Shifts that reorder: it catches about a third of the misorders in rounds 4 and 6. Round 7's (per_file overtakes
  once) misorders the top 3 only when they differ in route_table.
- Uniform or no shifts misorder nothing in rounds 2-3 yet draw false alarms, most after the slowdown (24/200).
  Round 8 changes no order, yet half its checks are misordered: pooling mixes measurements from before and after
  the earlier shifts. Round 5 re-measures, on round 4's commit, top configs last measured before it.

### Regret

Each variant minus `copy-all` on the same seeds: mean over seeds of the run's mean regret (s), its 95% bootstrap CI,
seeds lower / tied / higher, and the mean difference in eval runs.

| scenario | budget | seeds | `copy-all` regret, median [IQR] | variant | Δ regret (s) | lower / tied / higher | Δ eval runs |
|---|---|---|---|---|---|---|---|
| `default` | 6×2 | 1000 | 0.95 [0.672, 1.97] | `recheck` | +0.0331 [0.00291, 0.0627] | 405 / 0 / 595 | +13.4 |
| | | | | `drop-trials` | +0.0285 [−0.00248, 0.0585] | 414 / 0 / 586 | +13.4 |
| | | | | `drop-older` | +0.0301 [−0.00121, 0.0619] | 418 / 0 / 582 | +13.4 |
| | | | | `age-3` | −0.0139 [−0.0346, 0.00741] | 426 / 121 / 453 | −0.1 |
| `default` | 12×2 | 1000 | 0.526 [0.402, 0.689] | `recheck` | −0.00949 [−0.0221, 0.00362] | 538 / 0 / 462 | +13.1 |
| | | | | `drop-trials` | −0.0133 [−0.0265, −0.000364] | 542 / 0 / 458 | +13.1 |
| | | | | `drop-older` | −0.0112 [−0.0255, 0.00339] | 548 / 0 / 452 | +12.9 |
| | | | | `age-3` | −0.00379 [−0.0139, 0.0061] | 457 / 67 / 476 | −0.2 |
| `drift` | 6×2 | 1000 | 0.861 [0.72, 1.09] | `recheck` | −0.039 [−0.0754, 0.00166] | 676 / 0 / 324 | +26.0 |
| | | | | `drop-trials` | −0.0395 [−0.074, −0.00129] | 686 / 0 / 314 | +26.0 |
| | | | | `drop-older` | −0.107 [−0.139, −0.0728] | 745 / 0 / 255 | +25.1 |
| | | | | `age-3` | −0.15 [−0.162, −0.139] | 850 / 0 / 150 | −0.5 |
| `drift` | 12×2 | 200 | 0.848 [0.757, 0.955] | `recheck` | −0.101 [−0.129, −0.0736] | 143 / 0 / 57 | +27.0 |
| | | | | `drop-trials` | −0.107 [−0.135, −0.0776] | 142 / 0 / 58 | +27.0 |
| | | | | `drop-older` | −0.183 [−0.214, −0.152] | 162 / 0 / 38 | +25.7 |
| | | | | `age-3` | −0.163 [−0.18, −0.147] | 187 / 0 / 13 | −0.1 |

The cost check, `drift`, against `copy-all` with 2 more trials per round:

| budget | seeds | `copy-all` regret, median [IQR] | variant | Δ regret (s) | lower / tied / higher | Δ eval runs |
|---|---|---|---|---|---|---|
| 6×2 (`copy-all` 8×2) | 1000 | 0.822 [0.718, 0.996] | `drop-trials` | +0.0448 [0.00259, 0.0914] | 648 / 0 / 352 | −6.0 |
| | | | `drop-older` | −0.0231 [−0.0615, 0.0184] | 699 / 0 / 301 | −6.9 |
| 12×2 (`copy-all` 14×2) | 200 | 0.889 [0.802, 1.01] | `drop-trials` | −0.143 [−0.168, −0.116] | 161 / 0 / 39 | −5.0 |
| | | | `drop-older` | −0.22 [−0.25, −0.186] | 168 / 0 / 32 | −6.3 |

### Without drift (added after the results above)

`age-3` passed the rule, but neither scenario has the one real run's shape: 30 rounds, no shift (all 29 incumbent
re-measures, rounds 2-30, matched), and a new commit in about a third of the rounds (19 of the 29 were on an unchanged
commit). Two scenarios were added after the results above, so they can only turn a "yes" into a "no": `steady` (H1-H6
added in rounds 1-4, then a commit that changes nothing every third round: 12 commits in 30 rounds) and `unchanged` (no
commit after round 4, so the incumbent is never measured again; the worst case for ageing by round).

| scenario | budget | `copy-all` regret, median [IQR] | variant | Δ regret (s) | lower / tied / higher | Δ eval runs |
|---|---|---|---|---|---|---|
| `steady` | 6×2 | 0.458 [0.427, 0.567] | `age-3` | −0.275 [−0.352, −0.209] | 163 / 0 / 37 | −7.4 |
| | | | `recheck` | +0.124 [0.00591, 0.25] | 76 / 0 / 124 | +53.4 |
| | | | `drop-trials` | +0.109 [−0.00501, 0.233] | 80 / 0 / 120 | +55.1 |
| | | | `drop-older` | −0.000234 [−0.11, 0.114] | 99 / 0 / 101 | +47.7 |
| `steady` | 12×2 | 0.404 [0.34, 0.469] | `age-3` | −0.0603 [−0.0775, −0.0432] | 133 / 0 / 67 | −45.6 |
| `unchanged` | 6×2 | 0.462 [0.427, 0.573] | `age-3` | −0.21 [−0.292, −0.141] | 152 / 0 / 48 | −4.4 |
| `unchanged` | 12×2 | 0.411 [0.344, 0.481] | `age-3` | −0.0804 [−0.101, −0.0612] | 134 / 0 / 66 | −42.3 |

Why ageing helps here: sampled trials that set prefetch_async (H6, from round 4) on:

| scenario | budget | `copy-all` | `age-3` |
|---|---|---|---|
| `steady` | 6×2 | 4977/30049 (17%) | 20353/30509 (67%) |
| `unchanged` | 6×2 | 5762/32198 (18%) | 21531/32198 (67%) |
| `unchanged` | 12×2 | 10995/64600 (17%) | 22853/64600 (35%) |

Every trial copied from rounds 1-3 carries prefetch_async at its default (off). Copying them all, `copy-all` turns
it on in 17% of sampled trials; copying 3 rounds, in 35-67%. On at prefetch 0.8 is worth 0.4 s, about where
`copy-all`'s regret stalls (median 0.43-0.445 s over rounds 15-30 at 6×2). This is the NeuralSGT run's M5 (TPE
rarely explores new levers) on the benchmark.

Round 30 alone, median [IQR] regret (s), tells a second story:

| scenario | budget | `copy-all` | `age-3` | `age-3-study` (below) |
|---|---|---|---|---|
| `steady` | 6×2 | 0.431 [0.412, 0.462] | 0.0948 [0.059, 0.145] | 0.103 [0.0593, 0.147] |
| `steady` | 12×2 | 0.0714 [0.0399, 0.18] | 0.0961 [0.0503, 0.432] | 0.0975 [0.04, 0.432] |
| `unchanged` | 6×2 | 0.43 [0.403, 0.461] | 0.214 [0.132, 0.31] | 0.103 [0.0601, 0.138] |
| `unchanged` | 12×2 | 0.0727 [0.0379, 0.419] | 0.112 [0.0733, 0.432] | 0.0754 [0.0448, 0.415] |

At 12×2 `copy-all` escapes the stall later but ends lower: by round 30 its full history beats 3 rounds of it.
`age-3` ends higher in `unchanged` than in `steady`; there the incumbent is never measured again, so its own trials
age out.

### Ageing the study's copy only (added after review)

`age-3-study` ages only the trials copied into the study; the incumbent is still picked over every round, as
`copy-all` does. Each against `copy-all` on the same seeds, `age-3`'s from the tables above:

| scenario | budget | seeds | `age-3` Δ regret (s) | `age-3-study` Δ regret (s) | lower / tied / higher | Δ eval runs |
|---|---|---|---|---|---|---|
| `default` | 6×2 | 1000 | −0.0139 [−0.0346, 0.00741] | −0.0139 [−0.0346, 0.00741] | 426 / 121 / 453 | −0.1 |
| `default` | 12×2 | 1000 | −0.00379 [−0.0139, 0.0061] | −0.00379 [−0.0139, 0.0061] | 457 / 67 / 476 | −0.2 |
| `drift` | 6×2 | 1000 | −0.15 [−0.162, −0.139] | −0.112 [−0.124, −0.101] | 802 / 0 / 198 | −0.5 |
| `drift` | 12×2 | 200 | −0.163 [−0.18, −0.147] | −0.106 [−0.122, −0.0901] | 164 / 0 / 36 | −0.1 |
| `steady` | 6×2 | 200 | −0.275 [−0.352, −0.209] | −0.272 [−0.345, −0.206] | 163 / 0 / 37 | −7.4 |
| `steady` | 12×2 | 200 | −0.0603 [−0.0775, −0.0432] | −0.0589 [−0.0754, −0.0425] | 134 / 0 / 66 | −46.5 |
| `unchanged` | 6×2 | 200 | −0.21 [−0.292, −0.141] | −0.277 [−0.36, −0.208] | 163 / 0 / 37 | −4.4 |
| `unchanged` | 12×2 | 200 | −0.0804 [−0.101, −0.0612] | −0.0954 [−0.116, −0.0759] | 146 / 0 / 54 | −42.3 |

On `default` it is `age-3` exactly. Without drift it keeps the whole gain, and more in `unchanged`, where the
incumbent's own trials no longer age out. On `drift` it keeps 65-75% of it.

## Decision

| rule | 1. lower on `drift`, both budgets | 2. not higher on `default` | 3. beats `copy-all` given 2 more trials | verdict |
|---|---|---|---|---|
| `drop-trials` | yes | yes | no: 6×2 +0.0448 [0.00259, 0.0914] | **no** |
| `drop-older` | yes | yes | no: 6×2 −0.0231 [−0.0615, 0.0184] | **no** |
| `age-3` | yes | yes | adds no eval runs | **yes** |
| `age-3-study` (added after review) | yes | yes | adds no eval runs | **yes** |

**The drift check: no.** On `drift` at the benchmark's noise it catches 16-26% of the misorders in the copied data,
with 9-12% false alarms per check. Dropping what it flags doesn't beat giving the harness those eval runs as trials
at the real run's budget. Without drift the re-measurements cost 15-17% more eval runs and lower no regret. The
harness gets no check; it keeps the incumbent's drift flag.

**Ageing old rounds out: yes by the rule, but not as a drift fix.** It lowers regret with no drift at all too (`steady`,
`unchanged`): copied trials from before a lever existed hold TPE at that lever's default. The gain comes from the trials
copied into the study (all of it without drift, 65-75% on `drift`), so rule 4 is `age-3-study`'s: it leaves trials more
than 3 rounds old out of the study's copy and nothing else. The warm-start set stays as rules 1-3 leave it, so the
incumbent, the summary's lever effects, diagnostics and drift flag, and finalize don't change. Ageing those too would
age the incumbent out on unchanged commits (`unchanged`), empty the report's dev baseline (usually measured only in
round 1), and age out every trial when a run stops early (`finalize._final_round`).

Rule 4 is now in the harness: `warmstart.copied` gives `rounds._open_study` the trials to copy, and the round summary
counts rule 4 next to rules 1-3 (design.md, "BO and warm start"). `bench.Harness` copies the same trials, so the
benchmark's `harness` now runs `age-3-study` (a test checks it), and `copy-all` keeps the baseline above. At 12×2
the full history ends lower by round 30 (table above): over longer runs at larger budgets 3 rounds may be too short
a window.

Roadmap: #48 (keep all lever data across rounds) bets the other way, but here more copied data held TPE back: its
experiment should include `steady`. Variants built on `bench.Harness`, #47's Ax variant among them, now copy only
the last 3 rounds too.
