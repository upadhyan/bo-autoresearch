# Agentic BO through Surrogate-Augmented Autoresearch (arXiv 2608.00316)

Brunzema, Tiao, Le, De Angeli, Xuan, Gligorijevic (Meta / RWTH Aachen). *Agentic Bayesian Optimization through Surrogate-Augmented Autoresearch.* arXiv:2608.00316v2, 11 Aug 2026. Read from the HTML version, with the PDF used for Appendix F/G and Table 3, which the HTML does not render. Page numbers refer to PDF pages.

## Summary

The paper puts an LLM agent ("Sara") at the centre of a BO loop. The Bayesian machinery lives in "lenz", a BoTorch/GPyTorch backend exposed as a JSON CLI. Before committing each evaluation the agent can probe the GP, request proposals, score its own candidates, override the surrogate's pick, and persistently reconfigure bounds, acquisition function, objectives and constraints. With no prior, Sara roughly matches Ax. With a natural-language prior, it beats Ax early and matches or beats it at the end. It clearly beats LLAMBO and Centaur. It can also reformulate the problem mid-run (constraint to second objective) without restarting. There is no hypothesis layer, no code search, and no statistical verdict on anything.

## Method

- **Formalism (§4, p.5–7).** A metalevel decision process. The state holds trial data, the configuration (model, acquisition, active region, objective/constraint partition), an append-only context of instructions, and the deliberation history. The agent has three kinds of action: *evaluate* (the only one that spends budget), *computational* (probe, reconfigure, propose), and *stop*. Standard BO and LLM-as-optimizer both fall out as special cases.
- **Backend (§5.1, p.8–9; App. G–H).** The raw trial log is the single source of truth. GPs are rebuilt from it on each query, so any reconfiguration is lossless. Temporary steering (`suggest --bounds`, `suggest --around`) is kept apart from persistent steering (`set-bounds`, `set-acqf`, ...). The reference sheet says to use `set-bounds` only "after evidence and/or strong prior support", and to run `diagnostics` before narrowing. `diagnostics` returns LOO CV R², the noise estimate, lengthscales and a **first-order** per-parameter sensitivity (App. G, p.40).
- **Unit managed.** There is none beyond the single study and its configuration. Nothing like a hypothesis lifecycle exists, and nothing is ever pruned, retained or revived. The only "narrowing" is `set-bounds` on a whole study, and §6.4 reports that persistent `set-bounds` and `set-acqf` calls "remain rare".
- **Judging.** Nothing is judged formally. The system prompt (App. F, p.34–35) asks for a final report of "which priors held or broke". It also asks the agent to state, before every decision call, what it believes and what the next evaluation should learn. Stopping rules are listed: target reached, posterior converged, feasibility collapsed, or suggest keeps repeating points. A stalled or noisy incumbent is explicitly not a reason to stop.
- **Noise/replication.** The only noise handling is the GP's noise estimate and noisy log-EI as the default acquisition. The paper has **no replicates** and no harness-scheduled repeats. Apart from the scaling-law toy, the benchmarks look deterministic (inferred from their construction). Seeds (10 per method) measure run-to-run variance, not evaluation noise.
- **Prompt directives (Table 2, p.10).** These came out of observed failures. Without explicit ownership the agent "can become lazy" and defers to the surrogate. Coding-trained models batch trials in shell loops and skip reasoning; the footnote says fully enforcing one reasoning step per trial "would require harness-level constraints". Anti-patterns include deferring to an untrustworthy posterior, talking itself out of a context signal, and dropping a prior after one contradicting trial.

## Results and evidence strength

- **Setup.** 10 seeds; median with IQR bands. Table 3 (p.18) uses Mann–Whitney tests on final regret. Benchmarks are Branin, Hartmann-6 (plus a constrained variant), Ackley-10/20, a bi-objective GP-sample sweep, LCBench (3 tasks, 7-D, 80 evaluations) and four **author-constructed** synthetic reaction-yield functions (6-D, 40 evaluations).
- **Baselines are thin.** Sobol, default Ax (5 Sobol trials, then GP + LogEI), LLAMBO and Centaur. There is no TuRBO or other high-D method, and no comparison at matched cost (token usage is reported only in App. C.2).
- **No-prior results are only parity.** Hartmann: Ax 0.0116 vs Sara-Opus 0.0114. The text says Sara slightly beats Ax on Ackley-20, but in Table 3 Opus has a final median of 7.40 against Ax's 6.54. Only the GPT-5.5 variant (0.78) is clearly better. The text and the table disagree here; I read this myself and the paper does not comment on it.
- **Prior-driven gains are partly built in.** The reaction functions are products of per-factor Gaussians with one T×water interaction. The prompt names the reaction, and the true optima sit at textbook regimes (Table 5). App. D.1 admits these tasks are close to separable: a bash-only agent doing coordinate descent matches the full system on them.
- **The benchmarks are contaminated.** Despite renaming, shifting and sandboxing, the bash-only agent recognised Ackley-10 in 10/10 seeds and Hartmann in 4/10 (App. D.1, p.30). GP sample paths are the only uncontaminated test. There the surrogate's advantage grows with dimension (4–16-D).
- **Ablations run on few tasks.** Reasoning effort matters little, and turning it off gave a *better* warm start on Heck: extra reasoning led the agent to second-guess a strong prior (App. C.1). Haiku does worse on Ackley. None of the Table 2 directives is ablated; their value is anecdotal.
- **Tool-use inertia (§7, p.17).** Whatever call pattern a run adopts early tends to persist, which reduces strategy diversity. The paper lists this as an open problem.

## Relation to our work

**Same:**
- BO does the numerical search and the agent supplies priors and structure.
- Trials are fully preserved when the problem specification changes.
- Before-the-fact statements of belief are required.
- Surrogate fit (CV R²) gates how far the surrogate is trusted.

**Different:**
- They have no hypotheses, levers-as-claims or reject conditions. Their unit is one continuous study that is mutated in place. Ours is rounds under a fixed hypothesis set, with warm starts between rounds.
- They give the agent final authority over every evaluation and over reconfiguration. We deliberately take verdict authority away from the orchestrator.
- They report first-order sensitivity. We use total-order Sobol precisely because first-order hides interactions. D.1's finding that a surrogate helps on coupled landscapes and adds little on separable ones is indirect support for our core intuition about myopia.

**Borrow:**
1. An append-only raw trial log as the single source of truth, from which the GP and any importance statistic are re-derived. This makes warm-start mapping and verdict recomputation cheap and auditable.
2. Keep temporary probes (bounded suggestions) separate from persistent narrowing.
3. Gate narrowing on diagnostics.
4. Adapt the opening strategy to how strong the priors are, for round-1 seeding. Their rule of seeding one point per competing hypothesis when the hypotheses cannot be ranked maps directly onto us.
5. Their stop conditions, plus "a stalled incumbent is not a stop". Treat both as round-ending rules.
6. Enforce per-trial or per-round rules in the harness, not in prompts: the paper itself found prompting insufficient.
7. Name anti-patterns explicitly, including "discarding a prior on one contradicting trial", in our judge and orchestrator prompts.

**Avoid:**
- Letting the agent override the surrogate or mutate the space without a record. This is the unstructured behaviour we are designing against.
- Trusting first-order sensitivity for anything verdict-like.
- Benchmarks an LLM can recognise, when we evaluate the plugin. Use GP sample paths or private tasks.
- Relying on prompt wording for rigour: they report high sensitivity to it.

## Bearing on Q1–Q7

- **Q1 (state set): silent.** There is no managed unit with states (§4–5). The earlier skim was wrong: there is no "narrowed" state. Narrowing is only the `set-bounds` action, which fits our choice to make narrowing an event.
- **Q2 (retained = lower bound > τ): silent.** "Which priors held or broke" is an informal self-report (App. F). There is no acceptance criterion. Indirect support: the anti-pattern list warns against both blind trust in the posterior and dropping a prior on one trial (App. F, p.35).
- **Q3 (narrowing as a gated event): supports.** Reconfiguration is lossless and keeps the study's identity (§5.1, §6.4). `set-bounds` should come only after evidence, and diagnostics should be run first (App. G, p.39–40). Persistent narrowing is rarely used (§6.4), so under-use is a real risk.
- **Q4 (scoped revival): silent.** Nothing is ever rejected. "Priors can also form mid-run... test them" (App. F) is the closest point, and it is weak.
- **Q5 (whole-hypothesis group total-order verdict): silent on verdicts, mildly supportive on interactions.** D.1 (p.29–31) shows per-coordinate strategies fail on coupled functions. lenz's per-dimension first-order sensitivity is the kind of statistic Q5 exists to avoid.
- **Q6 (only mechanical evaluation rejects/retains): contradicts in design, supports in evidence.** The paper gives the agent final say ("You hold the controls", App. F). Yet its own failures support limiting the agent: lazy deference, skipped reasoning, more reasoning overriding a good prior, and inertia (Table 2, App. C.1, §7). They conclude that harness-level constraints are needed (Table 2 footnote).
- **Q7 (slot allocation / dimension cap): silent.** The search space is fixed per task. As an inference only: the surrogate's advantage in higher dimensions (App. D.1, GP 12/16-D) and Opus's weaker Ackley-20 result both suggest a dimension cap costs little. Neither was tested.

## Things the earlier skim got wrong

The paper has **no** harness-scheduled replicates, no judge agents (so nothing "anti-rubber-stamp"), and no forced variety. The nearest real points are these:
- the agent's tendency to defer to the surrogate is countered only by a prompt directive;
- tool-use inertia is listed as an open problem;
- "one seed per unrankable hypothesis" is an opening heuristic.

Recording an expectation happens per trial ("what the next evaluation should learn"), not per round.
