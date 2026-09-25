# Research constraints: prohibited vs discouraged

Research ticket [#17](https://github.com/upadhyan/bo-autoresearch/issues/17), which feeds the map decision **Research constraints: prohibited vs discouraged** ([#1](https://github.com/upadhyan/bo-autoresearch/issues/1)). It builds on [#2](https://github.com/upadhyan/bo-autoresearch/issues/2) (a fresh study per round, seeded with rewritten trials), [#3](https://github.com/upadhyan/bo-autoresearch/issues/3) (reject only on a total-order Sobol index from a noise-aware GP) and [#5](https://github.com/upadhyan/bo-autoresearch/issues/5) (prior art).

The Optuna claims were checked against the **v5.0.0** source (`gh api repos/optuna/optuna/contents/...?ref=v5.0.0`). Line numbers refer to that tag.

In this document, a **research constraint** is a user-stated limit on *what may be tried*. It is not an objective feasibility constraint such as "runtime < X". Anything marked **Proposal** is this document's recommendation, not a sourced fact.

## Key facts / recommended options for Research constraints: prohibited vs discouraged

1. **Optuna constraints only cover black-box feasibility and are learned after the fact.** `trial.set_constraint(key, value)` records a value on a trial that is already running, and the trial counts as feasible when every value is ≤ 0 ([`optuna/trial/_trial.py` L748–782](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L748-L782)). TPE ranks infeasible trials after feasible ones ([`_tpe/sampler.py` L740–776](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L740-L776)). GP multiplies EI by the probability of feasibility (Gardner et al. 2014). So the sampler only learns to avoid a point after it has *run* that point, which directly breaks a "don't test this" limit. GPSampler lists "input constraints" as a missing feature ([`_gp/sampler.py` L244](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L244)).
2. **Optuna has no way to express a user prior.** TPE's "prior" is a single fixed kernel at the midpoint of the range, with width equal to the range ([`parzen_estimator.py` L201](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/parzen_estimator.py#L201)), and `prior_weight` is deprecated as of v4.9. The GP's prior applies to kernel hyperparameters. πBO is not in Optuna or OptunaHub; the nearest OptunaHub package is `user_prior_cmaes`, which sets only CMA-ES's initial mean and covariance.
3. **πBO cannot express a hard limit, and its soft preference fades away by design.** It maximises `α(x)·π(x)^(β/n)`. π must be strictly positive everywhere, and the exponent β/n goes to 0 as trials accumulate (Hvarfner et al. 2022, §3, Eq. 6). Under our warm start, `n` includes the seeded trials, so by later rounds the prior has already largely decayed. Cost-aware BO (EI per second in Snoek et al. 2012 §3.2; `EI/c(x)^α` with decaying α in CArBO, Lee et al. 2020) behaves the same way: it changes *where the search samples*, not *what is optimal*.
4. **A penalised objective is the worst option for our statistics.** Optimising `f − λ·g` changes the objective. Map #1 treats that as a new research run, and λ would have to stay fixed across every seeded trial. The GP would also attribute the penalty's variance to the discouraged lever, so `S_T` would show that lever as *mattering* purely because of the penalty. The #3 reject statistic would then be measuring the user's own preference.
5. **Proposal: prohibited limits never reach BO.** The reviewer subagent prunes any hypothesis that violates one. If a limit applies at lever level ("never optimizer=SGD", "lr ≤ 1e-2"), the harness removes that categorical choice or narrows the bound. Under #2, seeds that fall outside the new distribution are dropped from the study and kept as telemetry. The Sobol box then equals the allowed box, so reject statistics stay valid.
6. **Proposal: use `set_constraint` only as a last-resort guard for prohibited regions that aren't axis-aligned.** Examples are "not x1 high *and* x2 high". Check the predicate *before* running anything: set the constraint to violated, return a sentinel value, and exclude the trial from Sobol fits. Two caveats follow from the source. GPSampler fits its objective GP on *all* COMPLETE trials, infeasible ones included ([L385–436](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L385-L436)), so a sentinel value poisons it; this works only with TPE. GP also raises unless every trial has the same number of constraints ([L626–636](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L626-L636)). Because the predicate depends only on the params, it can be backfilled exactly onto seeds. A non-box domain also makes classic Sobol invalid, because the inputs become dependent (Owen 2014; Song, Nelson & Staum 2016). The better fix is to reparameterise so the allowed region becomes a box.
7. **Proposal: keep discouraged options out of the objective. Enforce them at the agent layer and track them as telemetry.** The reviewer lowers a discouraged hypothesis's priority, which delays it, and asks for a stated reason. If it does run, every trial logs a `compat:<id>` flag per soft limit. Karpathy's autoresearch applies its "simplicity criterion" the same way, as judgement at keep time and not in the metric ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)).
8. **Proposal: report a discouraged option that wins as a tradeoff.** The round summary reports two incumbents, best overall and best compliant with each soft limit, with the gap between them and its credible interval from the #3 GP. The user decides. Nothing is silently penalised away.
9. **Proposal: the reviewer should be a guard model with the policy in its prompt.** This follows Llama Guard (Inan et al. 2023): the constraint registry goes into the prompt, and the output is `{verdict: allow|prune|deprioritize, constraint_id, rationale}`. Constitutional AI *randomly samples* one of its 16 ad hoc principles at each step (Bai et al. 2022, §3.1, App. C). That suits soft shaping, but a hard limit must check *every* rule on *every* hypothesis. As in AI-Scientist (Lu et al. 2024, §8) and #5, the orchestrator must not be able to edit the registry.
10. **Proposal: detect constraint laundering by mechanism, not by name.** Hypotheses declare a structured `mechanism` field. Nearest-neighbour retrieval against the prohibited registry plus every previously pruned hypothesis forces a strict re-review, following the retrieval defence of Krishna et al. 2023 (NeurIPS), which catches 80–97% of paraphrases at 1% false positives. A hook then scans the lever-code diff for prohibited code signatures, and a runtime assertion checks each trial's resolved config against the lever predicates.

| Option | Layer | Hard / soft | Pros | Cons (incl. warm start #2, Sobol #3) |
|---|---|---|---|---|
| Reviewer prunes the hypothesis | Agent | Hard | Nothing prohibited ever runs, and it's cheap | Can be fooled by renaming (see laundering); relies on LLM judgement |
| Exclude from search space (drop a choice, narrow a bound) | BO | Hard | Nothing prohibited ever runs; Sobol box = allowed box | Changes the distribution, so seeds outside it must be dropped (#2 already rewrites every round); box-shaped regions only |
| Pre-check predicate + `set_constraint` + sentinel | BO | Hard (fallback) | Handles regions that aren't boxes; predicate can be backfilled onto seeds | Sentinel poisons GPSampler's objective fit (TPE only); non-box domain invalidates Sobol; mixes with the objective-feasibility channel |
| Black-box `set_constraint` (evaluate, then flag) | BO | Neither | Built in | Runs the prohibited config, so it violates "don't test"; meant for runtime < X only |
| Penalised objective `f − λg` | BO | Soft | Trivial to implement | Changes the objective (a new run per #1); λ locked across seeds; penalty inflates the lever's `S_T`; hides the tradeoff |
| πBO / user prior on acquisition | BO | Soft | Leaves the objective and the meaning of Sobol unchanged; regret bound doesn't depend on the prior | Not in Optuna (custom sampler); π > 0 so never hard; decays as β/n, and seeds inflate n; sparse sampling in the disfavoured region fails the #3 coverage gate, so the verdict is "inconclusive" |
| Cost-aware acquisition (EI/c^α) | BO | Soft | Same as πBO; natural when "discouraged" really means "expensive" | Not in Optuna; α decays; same coverage effect as πBO |
| Deprioritise hypothesis + log `compat:` telemetry + dual incumbent | Agent + report | Soft | Objective, warm start and Sobol untouched; tradeoff shown to the user | Discouraged ideas still use budget when they do run; relies on reviewer ranking |

## Detail

### 1. The two kinds of limit

In the vocabulary of `CONTEXT.md`, a user limit can apply at two levels:

- **Hypothesis level.** "Don't try knowledge distillation." "We know augmentation X works, but we're avoiding it." This limit acts on the *hypothesis set*, which the agent controls.
- **Lever level.** "Never use SGD." "Keep the LR ≤ 1e-2." "Stay compatible with downstream method X, which needs fp32." This limit acts on a lever's *distribution* or on a joint region of lever values.

A hard limit ("prohibited") means *never evaluate*. A soft limit ("discouraged") means *allowed, but it costs something to choose it, and the choice must be visible*. Neither is an objective feasibility constraint ("runtime < X", map #1: "one objective plus optional constraints"). Feasibility constraints are black-box: you only know after running. Research constraints are known *a priori* from the proposal or the params. That difference decides which mechanisms are appropriate.

### 2. BO layer: what Optuna v5 actually offers

**`set_constraint` is a post-hoc, black-box mechanism.** Its docstring says the trial is feasible when all constraint values are ≤ 0. The value is stored as a system attr `constraints:<key>` on an already-running trial ([`_trial.py` L748–782](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py#L748-L782)). Feasibility is simply `all(x <= 0)` ([`study/_constrained_optimization.py` L23–35](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/_constrained_optimization.py#L23-L35)). Samplers use it as follows:

- **TPE** puts infeasible trials after all complete and pruned trials when splitting "below" from "above" ([`_tpe/sampler.py` L740–776](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py#L740-L776)). An infeasible trial's objective value is therefore never used for ranking. The sampler learns to avoid the region through `l(x)/g(x)`, but only after sampling it. Its docstring cites c-TPE (Watanabe & Hutter, arXiv 2211.14411) and notes that Optuna's algorithm differs from that paper (L110–120).
- **GPSampler** computes `LogEI + Σ log P(feasible)` and assumes each constraint is independent of the objective ([`_gp/sampler.py` docstring L100–125](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L100-L125)), following Gardner et al. 2014. Gardner et al. motivate the method with cases where "feasibility … can not always be determined in advance" and costs as much as evaluating the objective (§1). Their constrained EI is EI weighted by the probability that the point is feasible, and infeasible samples are still added to the training data (§3). That is exactly the situation research constraints are *not* in.
- **GPSampler also has two properties that matter here.** First, the objective GP is fit on *all* COMPLETE trials, infeasible ones included (`states = (COMPLETE, RUNNING)`, `trial.values for trial in completed_trials`, [L385–436](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L385-L436)); feasibility only masks the incumbent threshold (L509–512). Second, `_get_constraint_vals_and_feasibility` raises `ValueError` unless every completed trial has the same number of constraints ([L626–636](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py#L626-L636)). GPSampler also comments that "input constraints (use SLSQP)" are a missing feature (L244).
- **CMA-ES ignores constraints** (#2, fact 8).

**There is no user prior.** The TPE Parzen estimator appends one "prior" kernel at `0.5*(low+high)` with sigma `high-low` ([`parzen_estimator.py` L201–202](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/parzen_estimator.py#L201-L202)). For categoricals it adds a uniform `prior_weight/n_kernels` (L142–146). This is a stabiliser, not a belief the user can place. `prior_weight` itself is deprecated in v4.9.0 and scheduled for removal in v6.0.0 (`_tpe/sampler.py` L245–253). The GP's `prior.default_log_prior` is a prior on kernel parameters (L256). OptunaHub (registry listing, `package/samplers/`) has no πBO package. `user_prior_cmaes` only replaces CMA-ES's `x0`/`sigma0` with `mu0`/`cov0` ([README](https://github.com/optuna/optunahub-registry/tree/main/package/samplers/user_prior_cmaes)).

**Other native tools** are bounds and categorical choices (the search space itself), `PartialFixedSampler` (pins a lever to a value), and `enqueue_trial`. None of them expresses "less likely, but still possible".

### 3. BO layer: user-belief, cost-aware and constrained BO in the literature

- **πBO** (Hvarfner, Stoll, Souza, Lindauer, Hutter, Nardi, ICLR 2022, [arXiv 2204.11051](https://arxiv.org/abs/2204.11051)). It uses the acquisition `α_π,n(x) = α(x)·π(x)^(β/n)` (Eq. 6), with β set by the user to reflect confidence in the prior. The paper *requires* "π to be strictly positive on all of X" (§2), so a hard zero is ruled out by construction. The prior's influence "needs to decay over time" so that a poor prior cannot slow the search forever (§3), and the EI loss is asymptotically equal to plain EI's (Thm 1, Cor. 1). For us, this means a discouraged region becomes *less likely early on* and the discouragement then disappears. It is a search heuristic, not a policy.
- **BOPrO** (Souza et al., ECML-PKDD 2021, [arXiv 2006.14608](https://arxiv.org/abs/2006.14608)). It combines a user prior over the optimum's location with the model into a pseudo-posterior, and "robustly recovers from misleading priors". It shares πBO's properties: it is soft, it fades, and it is not in Optuna.
- **Cost-aware BO.** Snoek, Larochelle & Adams 2012 §3.2 "Modeling Costs" ([arXiv 1206.2944](https://arxiv.org/abs/1206.2944)) optimise "expected improvement per second". CArBO (Lee, Perrone, Archambeau, Seeger, [arXiv 2003.10870](https://arxiv.org/abs/2003.10870)) uses `EI/c(x)^α` with α decaying as the cost budget is consumed. If "discouraged" is encoded as a synthetic cost, the arithmetic matches πBO with `π ∝ c^-α`, and it has the same decay property.
- **Constrained BO** (Gardner et al., ICML 2014, [PMLR v32](https://proceedings.mlr.press/v32/gardner14.pdf)) targets unknown constraints that are as expensive to evaluate as the objective, as described above. c-TPE ([arXiv 2211.14411](https://arxiv.org/abs/2211.14411)) makes the same assumption.

### 4. Interaction with warm start (#2) and Sobol rejects (#3)

| Mechanism | Warm start (identical distributions, fresh study per round) | Sobol `S_T` reject (#3) |
|---|---|---|
| Search-space exclusion | The distribution changes, so `create_trial` refuses seeds outside it (#2 fact 3). Drop them from the study and keep them in the log as telemetry. Since #2 already rewrites seeds every round, this costs nothing extra. | Clean. The Sobol box equals the allowed box, and "this lever doesn't matter" is a statement about the allowed region. The reject verdict should say which region it covers. |
| Pre-check + `set_constraint` + sentinel | The predicate depends only on params, so it can be backfilled exactly onto every seed. That satisfies GP's equal-count check and TPE's tolerance. The distributions don't change. | Sentinel trials must be excluded from the surrogate. Over a box, the posterior in the forbidden region reverts to the prior, which widens the credible interval and makes an "inconclusive" verdict more likely (safe but wasteful). Over the allowed region, the inputs are dependent, so the Sobol–Hoeffding decomposition no longer applies; Shapley effects are the dependent-input alternative (Owen 2014, *SIAM/ASA JUQ* 2:245; Song, Nelson & Staum 2016, *SIAM/ASA JUQ* 4:1060). |
| Black-box `set_constraint` | Same as above. | Same as above, and the prohibited config has already been run. |
| Penalised objective | Every seed must be re-scored with the same λ and g. Changing λ is changing the objective, which map #1 puts out of scope within a run. | `Var(f − λg)` includes `λ²Var(g)`, so a lever that drives g gets a large `S_T` even when f doesn't depend on it. Rejects stop meaning "the hypothesis is false". |
| πBO / cost-aware | No change to distributions or values, so it is warm-start neutral. However, `n` counts seeded trials, so β/n is already small in round k > 1 unless n is reset per round (Proposal: count only this round's own trials). | The meaning of the objective is unchanged. Biased sampling leaves the disfavoured region sparse, so the #3 range-coverage gate fails there and the verdict is "inconclusive" rather than a false reject, which is honest. |
| Agent-layer deprioritisation | No change. | No change. |

### 5. Agent layer: rule review, guard models, prior systems

- **Constitutional AI** (Bai et al. 2022, [arXiv 2212.08073](https://arxiv.org/abs/2212.08073)). The constitution is a short list of natural-language principles, 16 for harmlessness, that were "selected in a fairly ad hoc manner" (App. C). In SL-CAI, one principle is *randomly sampled* at each critique/revision step (§3.1). In RL-CAI, the feedback model is shown one principle and a two-option multiple choice, optionally with chain-of-thought, and CoT probabilities are clamped to 40–60% (§4.1, §4.3). This fits *soft* shaping. Checking one sampled rule at a time is wrong for *hard* limits, where every rule must be checked on every hypothesis.
- **Guard models.** Llama Guard (Inan et al. 2023, [arXiv 2312.06674](https://arxiv.org/abs/2312.06674)) puts the policy taxonomy *in the prompt*, adapts zero- or few-shot to new taxonomies, and outputs safe/unsafe plus the violated category. Constitutional Classifiers (Sharma et al. 2025, [arXiv 2501.18837](https://arxiv.org/abs/2501.18837)) train separate input/output classifiers from a constitution of "permitted and restricted content", and report robustness over roughly 3,000 red-teaming hours with a 0.38% absolute increase in refusals. Both designs name what is *permitted* as well as what is restricted, and that is what keeps false refusals low. It maps to our discouraged vs prohibited split.
- **AI-Scientist** (Lu et al. 2024, [arXiv 2408.06292](https://arxiv.org/abs/2408.06292)). Scope comes from a *code template*, and ideas are filtered by novelty via Semantic Scholar (§3). It has "minimal direct sandboxing". The system tried to extend its own time limit instead of shortening its runtime, and the paper recommends strict sandboxing (§8). There is no user-stated prohibition list.
- **Karpathy autoresearch** ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)). It has explicit CAN/CANNOT lists: edit only `train.py`, add no packages, and never touch the evaluation. It also has a *soft* "simplicity criterion" that the agent weighs when deciding what to keep, without folding it into the metric.
- **Local `autoresearch` skill** (`~/.claude/skills/autoresearch/autoresearch.md`). It takes `Scope:` (file globs, meaning *where* it may edit, not *what* it may try) and an optional `Guard:` command that "must always pass". If the guard fails, the change is reverted even when the metric improved (Phase 5–6). The Verify command gets a safety screen for dangerous shell patterns. It has no idea-level prohibitions.
- **What's missing everywhere:** none of these systems has an idea-level prohibited/discouraged registry, and none checks for renamed ideas. Scope is always a location (files) or a template.

### 6. Proposals

**P1. Constraint registry.** The registry is a user-owned file, and hooks make it read-only to the orchestrator (enforcement lives in hooks, #4). Each entry has:

- `id`, `kind: prohibited|discouraged`, the user's text, and an optional `why`;
- `mechanism` (a canonical one-line description of *what* is excluded, independent of naming);
- `examples` (positive and negative);
- `lever_predicate` (optional, a pure function of params);
- `code_signatures` (optional: imports, API calls, config keys);
- for discouraged entries, `compat_metric` (a pure function of params or the trial artifact).

**P2. Reviewer contract.** This is a cheap subagent run cold, with isolation as in #5 steal 6. It gets the full registry in its prompt, like Llama Guard, and each hypothesis's `rationale`, `mechanism`, levers and planned diff. It returns a schema-checked result `{verdict: allow|prune|deprioritize, constraint_ids[], rationale}`. Every rule is checked against every hypothesis. A `prune` never enters the round. A `deprioritize` goes to the back of the queue and must carry a stated reason why it is still worth trying. When verdicts disagree across two reviewers, or when confidence is low, a prohibited-class call escalates to the user in checkpoint mode. In autonomous mode, it prunes on doubt.

**P3. BO encoding.**

- Hard lever limits become search-space exclusions.
- A hard limit that covers a region that isn't a box is first *reparameterised* into a box. If that's impossible, it becomes a pre-check predicate with `set_constraint`. The harness then forces TPE for that round, excludes sentinel trials from the Sobol fit, and marks that lever's reject verdict `inconclusive` unless the Sobol computation handles dependent inputs.
- Soft limits add nothing to the sampler in v1. πBO is deferred until there is evidence that the budget spent on discouraged regions matters. If it is added later, it should be a GPSampler subclass that adds `(β/n_round)·log π(x)` to the log-acquisition.
- Research constraints never share `set_constraint` keys with objective feasibility constraints. They get a `research:` key prefix, so the two cannot be confused when constraints are counted or backfilled.

**P4. Reporting a winning discouraged option.** Every trial logs `compat:<id> ∈ {0,1}` or a numeric value per soft limit (telemetry, per map #1). The round summary gives:

- the best trial overall and the best trial compliant with each soft limit;
- `Δ = f(best) − f(best_compliant)` with a GP credible interval;
- which discouraged levers or hypotheses the winner uses.

If the interval for Δ excludes 0, the result is flagged as a **tradeoff for the user**: "discouraged X buys Δ; compliant alternative is Y". It never gets silently discarded. When Δ is within noise, the compliant incumbent is recommended.

**P5. Detecting constraint laundering.** Checks are layered from cheapest to strictest, and any hit forces a strict re-review with the matching registry entry quoted:

1. **Declared mechanism.** A hypothesis must state *what changes in the computation*, separate from its name. The reviewer judges the mechanism, not the title.
2. **Retrieval against history.** Embed `rationale + mechanism` and search for nearest neighbours among registry entries *and every previously pruned hypothesis*. This follows Krishna et al. 2023 ([arXiv 2303.13408](https://arxiv.org/abs/2303.13408)), where retrieval against a store of previous outputs caught 80–97% of paraphrases at 1% false positives. A resubmitted pruned idea lands close to its earlier version even under a new name.
3. **Diff signatures.** A `PreToolUse` or commit hook scans the lever code the orchestrator writes for `code_signatures`, such as a prohibited import, API or config key. Names are cheap to change; calls into the same library are harder to disguise.
4. **Runtime assertion.** The harness logs each trial's *resolved* config (for example the optimizer class actually instantiated) and evaluates `lever_predicate` against it. This catches a benign-looking lever that resolves to a prohibited config at some value.
5. **Churn guard.** If a hypothesis is pruned twice under the same constraint, the orchestrator must stop proposing in that neighbourhood. This extends the oscillation guard from #5 steal 6.

## Sources

- Optuna v5.0.0 source: [`trial/_trial.py`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/trial/_trial.py), [`study/_constrained_optimization.py`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/study/_constrained_optimization.py), [`samplers/_tpe/sampler.py`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/sampler.py), [`samplers/_tpe/parzen_estimator.py`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_tpe/parzen_estimator.py), [`samplers/_gp/sampler.py`](https://github.com/optuna/optuna/blob/v5.0.0/optuna/samplers/_gp/sampler.py); [OptunaHub registry](https://github.com/optuna/optunahub-registry/tree/main/package/samplers).
- Hvarfner et al. 2022, πBO, ICLR, [arXiv 2204.11051](https://arxiv.org/abs/2204.11051).
- Souza et al. 2021, BOPrO, [arXiv 2006.14608](https://arxiv.org/abs/2006.14608).
- Snoek, Larochelle & Adams 2012, [arXiv 1206.2944](https://arxiv.org/abs/1206.2944).
- Lee et al. 2020, CArBO, [arXiv 2003.10870](https://arxiv.org/abs/2003.10870).
- Gardner et al. 2014, constrained BO, [PMLR v32](https://proceedings.mlr.press/v32/gardner14.pdf).
- Watanabe & Hutter, c-TPE, [arXiv 2211.14411](https://arxiv.org/abs/2211.14411).
- Owen 2014, "Sobol' indices and Shapley value", *SIAM/ASA J. Uncertainty Quantification* 2:245–251; Song, Nelson & Staum 2016, "Shapley effects for global sensitivity analysis", *SIAM/ASA JUQ* 4:1060–1083.
- Bai et al. 2022, Constitutional AI, [arXiv 2212.08073](https://arxiv.org/abs/2212.08073).
- Inan et al. 2023, Llama Guard, [arXiv 2312.06674](https://arxiv.org/abs/2312.06674).
- Sharma et al. 2025, Constitutional Classifiers, [arXiv 2501.18837](https://arxiv.org/abs/2501.18837).
- Lu et al. 2024, The AI Scientist, [arXiv 2408.06292](https://arxiv.org/abs/2408.06292).
- Karpathy, [autoresearch `program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md).
- Local skill: `~/.claude/skills/autoresearch/autoresearch.md`.
- Krishna et al. 2023, NeurIPS, [arXiv 2303.13408](https://arxiv.org/abs/2303.13408).
