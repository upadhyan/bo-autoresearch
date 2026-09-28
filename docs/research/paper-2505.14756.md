# LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization (arXiv 2505.14756)

## Citation and summary

Chih-Yu Chang, Milad Azvar, Chinedum Okwudire, Raed Al Kontar (University of Michigan). *LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization.* arXiv:2505.14756, v1 20 May 2025, v2 9 Oct 2025. Code: github.com/UMDataScienceLab/LLM-in-the-Loop-BO. Read from the arXiv HTML (v2), including appendices B–F.

In each BO iteration an LLM proposes one candidate design and a GP evaluates it. The paper gives three ways to decide whether that design is used. **Transient** flips a coin whose GP-probability p_t rises to 1. **Justify** rejects the LLM point when its UCB value is more than ψ_t below the UCB maximum. **Constrained** conditions the GP on the belief that f(x_LLM) beats the current posterior-mean maximum, using rejection sampling, then maximizes a Monte-Carlo UCB. Each mechanism comes with a sublinear cumulative-regret bound. None of the bounds depends on how good the LLM is, which the authors call a "no-harm" guarantee (App. D). The pitch is that the LLM is useful early and the GP takes over later. Experiments use GPT-3.5-turbo on 2–6-D synthetic functions, small hyperparameter-tuning tasks and one 3D-printing case study.

## Method

- **What is generated.** Single design points, not hypotheses. The LLM gets a "Description Card" (a verbal description of the function or model, the dimensionality and ranges) and a "Data Card" (past (x, y) pairs), and returns a vector (§2.2, App. E, Tables 1–4). It also produces the D initial points ("warmstarting", §3).
- **How a proposal is tested.** The test is point-wise, against the GP's acquisition function (Alg. 1, §§2.3–2.5). In Justify, eq. (2) is an explicit accept/reject rule on a single proposal. In Constrained, an LLM point that contradicts the posterior keeps no samples, so the surrogate is left unchanged and the proposal is effectively discarded (Fig. 2c–d).
- **Lifecycle / states.** None. A proposal is used or dropped within one iteration and nothing persists. Nothing is ever falsified, and there is no concept of a claim.
- **Interactions.** Not discussed. The Matérn-5/2 ARD GP (App. E) models them implicitly, but no attribution or sensitivity analysis is done.
- **Noise.** Handled through the theory only (sub-Gaussian noise, Assumption 1). The experiments use 10 replications with 95% bands. The 3D-printing objective is an image-thresholded pixel ratio, with the threshold set by trial and error (App. F.1).
- **Multiple testing.** Not addressed. The Justify gate is applied every iteration with a decaying ψ_t, and there is no error-rate control across iterations.
- **Schedules.** All three trust schedules (p_t, ψ_t, S_t) are fixed before the run and set by hand (§3, App. D). The authors say that tying them to how well the LLM understands the problem is still open (§5).

## Results and evidence strength

- **Claims.** All three variants beat plain BO, LLAMBO and LLAMBO-light on six synthetic functions (Figs. 3–4) and five HPT tasks. The gain is largest early and narrows later. Pure-LLM optimizers often plateau.
- **Critical reading:**
  - **Confounded initialisation.** LLM methods start from D LLM-chosen points while BO starts from D random points (§3). The early lead may therefore come mostly from the warm start rather than the in-loop mechanisms. There is no ablation that gives BO the same LLM initial points.
  - **Tiny budgets and dimensions.** Budgets are T = 10D for synthetic tasks and T = 5D for HPT, and d ≤ 6 throughout. This is exactly the regime where any prior helps and where the GP has barely been fitted.
  - **Leaky prompts.** The Description Cards describe textbook functions (for example "a unique global maximum" plus landscape shape, App. E Table 1) that are well known from pretraining. Nothing prevents the LLM from recognising the benchmark (my inference; the paper does not test it).
  - **Weak LLM baselines.** The LLM baselines are GPT-3.5 at temperature 1. "LLAMBO-light" is the authors' own simplification of LLAMBO.
  - **No statistical tests** between methods beyond overlapping CIs. There is no ablation of the schedule hyperparameters.
  - **3D printing.** Only Transient is run. The case study appears to be one run per method on a single printer (inferred from "each run takes several hours" and one curve per method in Fig. 5c). It is a demonstration, not evidence.
  - **Theory.** The bounds are standard UCB-style arguments with a vanishing LLM contribution. "No harm" is asymptotic, and the bounds say nothing about early-stage gains.
- **Net.** The paper gives moderate evidence that a GP gate stops LLM proposals from hurting, and weak evidence that it helps beyond a better warm start.

## Relation to our work

**Same.**
- The LLM contributes priors, and a GP is the arbiter.
- Both reject the premise that an LLM can be the optimizer.
- Both put the reject/accept decision in a mechanical rule, not in the LLM's judgment.
- Their "warmstarting" is a cousin of our seeded studies.

**Different.**
- **Unit of analysis.** Their unit is a design point; ours is a hypothesis with graded levers. They optimize f. We use BO to search and then run an attribution test (total-order Sobol) on the fitted surrogate to judge claims.
- **Persistence.** They have no pre-registration, no persistent states, no rounds and no changing search space. Their search space is fixed for the whole run.
- **Where the LLM sits.** Their LLM is inside every iteration. Ours sits between rounds and never picks trials.
- **Interactions.** Their notion of "rejection" has nothing to do with interactions, which is our central concern.

**Borrow.**
1. **The no-harm framing as a design invariant.** No orchestrator input should be able to make the loop worse than plain Optuna BO asymptotically. For us that means agent-injected points are enqueued trials, and the sampler is otherwise untouched.
2. **A decaying LLM-trust schedule for seeding.** If the orchestrator may enqueue "hunch" trials at round start, cap them and make the cap shrink as rounds accumulate data. The Transient schedule is the template.
3. **Their stated open problem** (§5): trust should depend on how well the LLM understands the problem. For us, this could key the cap to a surrogate-fit diagnostic.

**Avoid.**
1. **Conditioning the surrogate on LLM beliefs (Constrained).** It is clever for optimizing, but it would contaminate the GP we use for Sobol verdicts with pseudo-observations. The verdict GP must be fit on real trials only.
2. **Their evaluation design.** Asymmetric initialisation and recognisable benchmarks make LLM gains look larger. Any evaluation of our plugin should give the BO-only baseline the same seeds and obfuscate the task.
3. **Hand-set schedules with no ablation.**

## Bearing on Q1–Q7

- **Q1 (state set): silent.** There is no lifecycle; proposals exist for a single iteration (Alg. 1, §2.2). Nothing supports or contradicts our state set.
- **Q2 ("retained" = gates + lower bound > τ): silent.** Their "retain" is point-level acceptance within a ψ_t-suboptimal UCB region (eq. 2, §2.4). There is no importance statistic and no credible-bound acceptance. The framing, where a failed check means "fall back", resembles our inconclusive-by-default (loose analogy only).
- **Q3 (narrowing as an event after gates pass): silent.** Bounds are fixed at [0,1]^D throughout (App. E.1).
- **Q4 (revival of rejected hypotheses): silent.** A rejected LLM point is gone. Nothing persists to revive, and interactions are not discussed.
- **Q5 (whole-hypothesis group index; freeze single levers): silent.** There is no attribution at any grouping.
- **Q6 (only mechanical evaluation rejects/retains): supports.** The whole paper argues that the statistical surrogate, not the LLM, must have final say. All three mechanisms let the GP override the LLM (§1 "Main considerations", §§2.3–2.5). The no-harm bounds hold precisely because the LLM cannot override the GP (App. D). The paper does not discuss an orchestrator-only "park" action.
- **Q7 (slot allocation, dimension caps, starvation): silent.** A single LLM proposes one point per iteration, and there is no competition among proposals. The Transient schedule, which gives the LLM a shrinking share of evaluations over time, is a loose analogue of a budget allocator.

## New question for the lifecycle

If the orchestrator (or a warm start) injects non-random, belief-driven trials into a round, do those trials count toward the trial-count and range-coverage gates? Clustered LLM-chosen points can satisfy "n ≥ 10·d" while leaving much of the lever range unexplored, and they shape where the GP is confident. LLINBO's gains come largely from such clustered early points. We should decide whether gates count only sampler-chosen trials, or whether coverage is measured on the actual trial spread regardless of where the trials came from.
