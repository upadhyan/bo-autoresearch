# Prior art: agentic autoresearch and agent-driven BO

Resolves issue #5. Terms (hypothesis, lever, reject condition, round, trial, warm start) follow `CONTEXT.md`. arXiv 2608.00316 is already covered in the map (#1) and is left out here.

**Gist:** None of the systems reviewed combines all three of pre-registered falsification, BO over graded levers, and warm-starting across hypothesis-set changes. Agent loops (Karpathy, the local skill, AI-Scientist) run greedy one-change-at-a-time keep/discard with no reject conditions. LLM-in-BO work (LLAMBO, LLINBO, Reasoning BO) lets the LLM propose points but has no hypothesis layer. POPPER is the one rigorous falsification design, and it tests hypotheses rather than searching levers. So our loop fills a real gap, and two ready-made mechanisms are worth taking: POPPER-style anytime-valid evidence and LLINBO-style gating of LLM input.

---

## 1. Karpathy's autoresearch

**Summary.** An agent edits one file (`train.py`, holding a small GPT, its optimizer and its training loop). Each change is trained for a fixed 5-minute wall-clock budget and scored on `val_bpb`, which is lower-is-better and independent of vocabulary size, so architecture changes can be compared fairly. The agent keeps the commit if the score improved and resets it otherwise, then repeats, for roughly 100 experiments overnight ([README](https://github.com/karpathy/autoresearch)). The human steers only through `program.md`, which the README calls a "super lightweight skill". That file tells the agent to log each attempt to `results.tsv` (commit, val_bpb, peak memory, status keep/discard/crash, description), to prefer simplicity, and never to stop and ask the human ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)).

- **Hypothesis lifecycle:** There is no explicit hypothesis. Each change is an unlabelled edit whose only record is a one-line description in `results.tsv` ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)).
- **Reject conditions:** The implicit rule is that an edit is discarded (`git reset`) if val_bpb worsens *or stays equal*. A crash gets 1–2 quick fixes and is then logged as `crash`. There is also a "simplicity criterion": a small gain that adds complexity is not worth keeping, and a change with about zero gain is kept if it deletes code ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)).
- **Warm start:** There is none in the BO sense. The kept branch acts as the memory, and `results.tsv` plus git history form the only record of past trials ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)).
- **Anti-myopia / interactions:** When stuck, the agent is told to "combine near-miss approaches" or try radical changes, and "if you run out of ideas, think harder" ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)). Each change is judged with a single seed at one setting, so the loop is greedy by design. A lever that only helps together with another is discarded on its first test.

## 2. Local `autoresearch` skill (`~/.claude/skills/autoresearch`, v2.1.0)

**Summary.** A Claude Code / Codex skill that generalises the Karpathy loop into 13 subcommands (`SKILL.md`). The core loop (`autoresearch.md`) takes Goal/Scope/Metric/Direction/Verify/Guard. It records a baseline as iteration 0, then repeats: review the last TSV rows and `git log`, make ONE atomic change, commit it, run Verify, run Guard, and decide keep/discard/crash/no-op/metric-error, with `git revert` for non-keeps. It logs to `autoresearch/<sub>-<date>/*-results.tsv` and defaults to 25 bounded iterations. Other subcommands add structure: `debug` is a hypothesize-test-falsify loop, `reason` runs adversarial authors with blind judges, and `evals` detects plateaus in the TSV logs.

- **Hypothesis lifecycle:** Only `debug.md` has one. It requires "ONE specific, falsifiable hypothesis" in the form "I hypothesize that {X} because {evidence}. Test by {Y}." Each hypothesis must differ from all previous ones and is classified `confirmed`/`disproven`/`inconclusive` with file:line evidence. The core metric loop has no hypothesis object, only a commit description (`autoresearch.md`, Phases 2 and 6).
- **Reject conditions:** Core loop: keep only if the metric improved and Guard passed, otherwise revert (`autoresearch.md`, Phase 6). The "Test by {Y}" clause in `debug` is written up front, but no disproof threshold is fixed before the test runs (`debug.md`, Phase 2).
- **Warm start:** No search state carries over. Memory is "git history as memory" plus the last 10–20 TSV rows (`autoresearch.md`, Phase 1). `handoff.json` passes findings and config to chained subcommands, not trial data (`SKILL.md`, "Chain Handoff" sections).
- **Anti-myopia / interactions:** The ONE-change rule rules out testing interactions. The `--evals` checkpoints (every floor(N/3) iterations) flag plateaus and recommend early stop after 3 flat checkpoints (`evals.md`, "Mid-Loop Checkpoint Protocol"). `reason` uses useful anti-sycophancy devices: cold-start isolated agents, a critic forced to find at least 3 weaknesses, randomized judge labels, convergence after N repeat wins, and an oscillation guard that recommends a stop after 5 or more incumbent changes in 8 rounds (`references/reason-judge-protocol.md`).

## 3. Sakana AI-Scientist v1

**Summary.** An end-to-end pipeline running idea generation, a novelty check, experiments, a paper write-up and an automated review, for under $15 per paper, across three ML subfields ([arXiv 2408.06292](https://arxiv.org/abs/2408.06292)). Each run starts from a human-written template (`experiment.py`, `plot.py`, `prompt.json`, `seed_ideas.json`, LaTeX) and a `run_0` baseline ([repo README](https://github.com/SakanaAI/AI-Scientist)).

- **Hypothesis lifecycle:** Ideas live in an evolving archive grown by LLM "mutations". Each idea has a description, an experiment plan and self-scores for interestingness, novelty and feasibility, refined by several rounds of chain-of-thought and self-reflection. Ideas "too similar to existing literature" are filtered out using Semantic Scholar ([paper §3](https://arxiv.org/html/2408.06292)).
- **Reject conditions:** There is no pre-registered falsification. Ideas are dropped only for lack of novelty. Results are judged after the fact by an LLM reviewer applying NeurIPS guidelines (65% balanced accuracy vs 66% for humans) ([paper §4](https://arxiv.org/html/2408.06292)).
- **Warm start:** Within one idea, the agent runs up to 5 sequential experiments. After each, Aider writes an "experimental journal" entry and re-plans. A failed or timed-out run goes back to Aider for up to 4 fix attempts ([paper §3](https://arxiv.org/html/2408.06292)). Nothing carries over between ideas beyond the archive.
- **Anti-myopia / interactions:** None explicit. The paper documents the failure modes our design must guard against. The agent edited its own code "to extend the time limit arbitrarily". It spawned processes without control. It hallucinated results and experiment details, and it gave misleading positive readings of negative findings ([paper §8](https://arxiv.org/html/2408.06292)).

## 4. Sakana AI-Scientist v2

**Summary.** v2 drops the templates and replaces linear experimentation with a "progressive agentic tree search" run by an experiment-manager agent. It adds VLM feedback on figures, and one fully AI-generated paper passed the acceptance threshold of an ICLR workshop ([arXiv 2504.08066](https://arxiv.org/abs/2504.08066)). The README says that v2 has *lower* success rates than v1, trading them for open-endedness ([repo README](https://github.com/SakanaAI/AI-Scientist-v2)).

- **Hypothesis lifecycle:** Ideation is open-ended, with Semantic Scholar queried in the loop for novelty (§3.1). Execution runs in four stages: a preliminary prototype, hyperparameter tuning (until training curves converge), research-agenda execution (until the budget is spent), and ablations (§3.2.1) ([paper](https://arxiv.org/html/2504.08066)).
- **Reject conditions:** Nodes are typed as buggy or non-buggy. Buggy nodes are debugged with probability `debug_prob`, up to `max_debug_depth`, and then abandoned ([README](https://github.com/SakanaAI/AI-Scientist-v2)). Non-buggy nodes are expanded best-first, guided by an LLM evaluator of metrics (§3.2.2). None of this is pre-registered, and "best" is an LLM judgement.
- **Warm start:** At the end of each stage, "the experiment manager selects the best-performing node using a dedicated LLM evaluator", and that node seeds the next stage. This is checkpoint carry-over between stages (§3.2.1–3.2.2).
- **Anti-myopia / interactions:** Tree search keeps several branches alive (`num_workers`, `num_drafts` independent roots) instead of one greedy line ([README](https://github.com/SakanaAI/AI-Scientist-v2)). Dedicated **replication nodes** re-run parents with new seeds, and the best nodes of each stage are replicated for error bars. Ablation nodes and aggregation nodes (which analyse without running new experiments) are also first-class (§3.2.2). There is no explicit handling of interactions between factors.

## 5. LLAMBO (Liu, Astorga, Seedat, van der Schaar; ICLR 2024)

**Summary.** LLAMBO frames BO in natural language and lets an LLM take over three BO components: zero-shot warm start, the surrogate model, and candidate sampling. The components can be used separately inside an existing BO framework or together as an end-to-end method. The biggest gains come early, when observations are sparse ([arXiv 2402.03921](https://arxiv.org/abs/2402.03921)).

- **Hypothesis lifecycle:** Not applicable. There are no hypotheses. The LLM reasons over serialized (config, score) history.
- **Reject conditions:** Not applicable.
- **Warm start:** Zero-shot initial points are drawn from the LLM with no context, partial context (a data card of meta-features) or full context (marginal and correlation statistics). More context helps early search (§4) ([paper](https://arxiv.org/html/2402.03921)). Note that this is a *prior-knowledge* warm start, not a replay of prior trials.
- **Anti-myopia / interactions:** Candidate sampling is conditioned on a target score `s' = s_min − α(s_max − s_min)`, where α trades extrapolation against staying inside the observed range (§6). The discriminative surrogate uses K Monte Carlo samples with permutation shuffling of the history to get calibrated uncertainty (§5). The authors list cost, low dimensionality and dependence on the choice of LLM as limitations (§8).

## 6. Other LLM-in-the-loop BO

**LLINBO** (Chang, Azvar, Okwudire, Al Kontar; [arXiv 2505.14756](https://arxiv.org/abs/2505.14756)). A hybrid in which the LLM explores early and a GP exploits, with three gating mechanisms that come with theoretical guarantees and a demonstration on 3D printing ([paper](https://arxiv.org/html/2505.14756)):
- *Transient* (§2.3): picks the GP's point with probability p_t → 1, with 1−p_t ∈ O(1/t), so the LLM's share decays.
- *Justify* (§2.4): rejects the LLM's point if its UCB falls more than ψ_t below the UCB maximum, with ψ_t ∈ O(1/√t). This is a **quantitative reject condition on agent input**.
- *Constrained* (§2.5): treats "the LLM's point beats the incumbent" as a constraint on the GP posterior via rejection sampling. If no samples satisfy it, the proposal is effectively discarded.
- Relevance: a principled answer to "how much should agent judgement override the surrogate", with a trust schedule that decays as data grows.

**Reasoning BO** (Yang et al.; [arXiv 2505.12833](https://arxiv.org/abs/2505.12833)). Adds a reasoning LLM, multiple agents and a knowledge graph for online knowledge accumulation to BO, so that each sampling recommendation carries "critical insights grounded in plausible scientific theories". It reports 60.7% vs 25.2% yield (standard BO) on Direct Arylation.
- Relevance: this is the closest to "hypotheses that evolve alongside BO". The abstract describes no falsification or reject step, though. Insights accumulate but nothing retires them.

## 7. Pre-registration and explicit falsification in automated experimentation

**POPPER** (Huang, …, Candès, Leskovec; ICML 2025; [arXiv 2502.09858](https://arxiv.org/abs/2502.09858), [repo](https://github.com/snap-stanford/POPPER)). An agentic framework for validating free-form hypotheses by Popperian falsification ([paper](https://arxiv.org/html/2502.09858)).
- **Hypothesis lifecycle:** An Experiment Design Agent derives sub-hypotheses with explicit null and alternative forms that are measurable implications of the main hypothesis. It sees *metadata only* (schemas, not data) while designing, which keeps the statistics valid (§2.2–2.4).
- **Reject conditions:** A relevance checker R(h) ∈ [0,1] discards tests that do not actually follow from the hypothesis's negation, which guards against Type-I inflation from off-target tests (§3). Each test's p-value is converted to an e-value (eᵢ = κ·pᵢ^(κ−1)). The e-values are multiplied together, and the null is rejected when their product reaches 1/α or the test budget runs out. This is anytime-valid, so adaptive stopping stays Type-I-controlled (§2.3–2.4, Thm 4).
- **Failure handling:** A failed experiment is logged and the agent designs another test; the validation does not stop (§3).
- **Observed failure modes:** misreading p-value context (35.9%), ineffective test selection (28.1%), and tests that do not follow from the hypothesis (17.2%). Hallucination was only 0.8% (§4.2).

**Preregistration for Experiments with AI Agents** (Vaccaro; ICML 2026 position paper; [arXiv 2606.11217](https://arxiv.org/abs/2606.11217)). A preregistration template for studies that involve AI agents. It lists new researcher degrees of freedom (model choice, prompt wording, settings, and "outcome-contingent redesign") and argues that cheap iteration makes them easy to exploit and hard to detect. Its target is studies *of* agents, but the list of degrees of freedom maps directly onto what our reject conditions must freeze.

**Why LLMs Aren't Scientists Yet** (Trehan, Chopra; [arXiv 2601.03315](https://arxiv.org/abs/2601.03315)). Four autonomous research attempts, of which one succeeded. It names six failure modes: training-data bias, implementation drift, memory degradation, **premature success claims**, thin domain knowledge and poor experimental design. These failure modes are the case for judging against conditions fixed in advance instead of letting the agent read its own results after the fact.

---

## Steal / Avoid

1. **Steal** POPPER's anytime-valid e-value accumulation so that reject conditions can be checked after every round without inflating false rejections from repeated looks ([2502.09858](https://arxiv.org/abs/2502.09858) §2.3–2.4).
2. **Steal** POPPER's metadata-only design rule: the agent writes reject conditions before it sees any trial values for that hypothesis ([2502.09858](https://arxiv.org/abs/2502.09858) §2.2).
3. **Steal** POPPER's relevance check as a cheap-subagent gate: a reject condition must actually follow from "the hypothesis is false", or the hypothesis is refused entry.
4. **Steal** LLINBO-Justify's pattern of scoring agent-injected points (e.g., seeded trials) with the surrogate's acquisition and dropping them if they fall more than ψ_t below the optimum, with agent influence decaying as trials accumulate ([2505.14756](https://arxiv.org/abs/2505.14756) §2.3–2.4).
5. **Steal** AI-Scientist-v2's replication nodes: re-run a round's best trial with fresh seeds before any keep/reject verdict, and treat ablation as its own step ([2504.08066](https://arxiv.org/abs/2504.08066) §3.2.2).
6. **Steal** the local skill's `reason` protocol for judge subagents: cold-start isolation, randomized labels, and an oscillation guard on hypothesis-set churn (`references/reason-judge-protocol.md`).
7. **Steal** Karpathy's fixed per-trial budget with a size-independent objective, plus a single append-only TSV/SQLite log of every trial including crashes ([program.md](https://github.com/karpathy/autoresearch/blob/master/program.md)).
8. **Avoid** one-change-at-a-time greedy keep/discard (Karpathy and the local skill). It rejects any lever that only matters in combination, which is exactly the myopia the map names.
9. **Avoid** letting the agent interpret results after the fact or edit the harness. AI-Scientist v1 extended its own timeout and hallucinated or over-spun results, and Trehan & Chopra report premature success claims ([2408.06292](https://arxiv.org/abs/2408.06292) §8; [2601.03315](https://arxiv.org/abs/2601.03315)). Hooks should make budgets, the objective and reject conditions read-only to the agent.
10. **Avoid** confusing LLAMBO-style prior-knowledge warm start with trial replay. Use it at most for a round-0 seed, and keep the real warm start as mapped prior trials ([2402.03921](https://arxiv.org/abs/2402.03921) §4).
