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
configs exactly in rounds 4, 6 and 7, and that each round's regret reference is still the best config.

**The check.** On each new commit the round queues, after the incumbent, the next n − 1 best configs of the
warm-start set (pooled), as `recheck i`, on top of its N trials (like an investigation's queue; a config already
measured on the commit is not run again). A pair of the re-measured configs is *contradicted* when its order on the
new commit goes against its pooled order over earlier trials by more than the noise floor (from trials before the
round, scaled to the pair's earlier metric) on both sides. With no floor (repeats that never differ), any reversal
counts.

**Detection.** A check is *misordered* when the same check on noise-free metrics (each earlier trial's true metric
at its own round, pooled the same way) finds a reversal: the copied data orders a pair of the re-measured configs
the other way from the truth on the new commit. Hit rate: flagged among misordered checks. False alarms: flagged
among the rest. Swept: noise (`bench.NOISE`) 0, 0.01, 0.02 (the benchmark's), 0.05, 0.1; n = 2, 3, 4.

**Variants**, fixed before any run (rule 4 is the new warm-start rule; counted like rules 1-3):

| variant | re-measures (n) | rule 4 leaves out |
|---|---|---|
| `harness` | incumbent only (1) | nothing |
| `recheck` | 3 | nothing: the cost of the re-measurements alone |
| `drop-trials` | 3 | the earlier trials of both configs of each contradicted pair |
| `drop-older` | 3 | every earlier trial, once a check flags |
| `age-3` | incumbent only (1) | trials more than 3 rounds old |

A round's study is built before its re-measurements run, so a check's verdict reaches the warm start from the next
round on. The incumbent after the round, which regret scores, already applies it.

**Budgets and seeds.** 6 trials × 2 repeats per round (the real run's) and 12 × 2; seeds 0-199. Regret at noise
0.02. Commands (from the repo root, `uv run --frozen --project plugin/harness python benchmark/drift_check.py`
abbreviated `drift_check`):

```sh
drift_check --detect --scenario default --seeds 200 --trials 6
drift_check --detect --scenario default --seeds 200 --trials 12
drift_check --detect --scenario drift --seeds 200 --trials 6
drift_check --detect --scenario drift --seeds 200 --trials 12
V="--variant harness --variant recheck --variant drop-trials --variant drop-older --variant age-3"
drift_check --scenario default --seeds 200 --trials 6 $V
drift_check --scenario default --seeds 200 --trials 12 $V
drift_check --scenario drift --seeds 200 --trials 6 $V
drift_check --scenario drift --seeds 200 --trials 12 $V
# the cost check: the harness with 2 more trials per round, against the variants that re-measure
drift_check --scenario drift --seeds 200 --trials 6 --base-trials 8 --variant harness --variant drop-trials --variant drop-older
drift_check --scenario drift --seeds 200 --trials 12 --base-trials 14 --variant harness --variant drop-trials --variant drop-older
```

**Decision rule** (written before any run). A rule is "yes" only if, against `harness` on the same seeds:

1. on `drift`, the mean regret difference's 95% CI lies below 0 at both budgets;
2. on `default`, that CI does not lie above 0 at either budget (it must not hurt where shifts are rarer);
3. a rule that re-measures must also have its regret CI below 0 against the harness given 2 more trials per round
   (`--base-trials`), which spends at least the eval runs the re-measurements do, on `drift` at both budgets: else
   the same eval runs are better spent on more trials. `age-3` adds no eval runs.

A CI whose nearer end is within a fifth of its width of 0 is rerun with 1000 seeds. Otherwise "no", and the harness
stays as it is.
