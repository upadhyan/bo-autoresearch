"""The verdict engine (#26): the fixed reject form against toys with planted truth.

Each toy is the bowl-free toy_bo trainer with lever code of our own: `term += ...` lines whose
effect on the loss is the planted truth. δ = 0.1 and the toy's noise σ = 0.05 (δ = 2σ).
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import bo, ready, round_run, register
from test_rounds import BASE, init, make_repo, of_type, started_in, toy_env

DELTA, SIGMA = 0.1, 0.05


def lever(baseline=0.0, predicted="higher", low=0.0, high=1.0):
    return {"kind": "float", "low": low, "high": high, "baseline": baseline, "predicted": predicted}


def spec(levers, **extra):
    return {"title": "A planted mechanism", "rationale": "Planted truth for the verdict engine.",
            "mechanism": "The toy's loss moves with this lever as planted.", "provenance": "novel",
            "lens": "optimisation", "directives": [], "fidelity_sensitive": False, "levers": levers,
            **extra}


def code_hypothesis(repo, run_dir, d, n, s, line, env):
    """Propose, register, write `line` (the planted term), smoke and commit hypothesis H<n>."""
    f = d / f"spec{n}.json"
    f.write_text(json.dumps(s))
    assert bo(repo, "propose", "--file", str(f), "--rationale", "idea")[0] == 0
    assert register(repo, f"H{n}")[0] == 0
    train = run_dir / "worktree" / "train.py"
    src = train.read_text().replace("import fidelity, seed", "import fidelity, lever, seed")
    train.write_text(src.replace("    term = 0.0\n", f"    term = 0.0\n{line}\n"))
    code, out = bo(repo, "smoke", f"H{n}", "--rationale", "try", env=env)
    assert code == 0 and out["passed"], out
    assert bo(repo, "commit-lever", f"H{n}", "--rationale", "smoke passed", env=env)[0] == 0


def verdict_run(d, python, seed, hyps, rounds=1, sigma=SIGMA, extra="", proxy=False, budget=3600,
                **planted):
    """R0, the hypotheses [(spec, planted line)] coded, then up to `rounds` BO rounds.

    proxy: a ladder rung {epochs: 1} (the toy's cheap rung), searched at even if calibration fails
    it. -> (run_dir, round-run outputs)
    """
    d.mkdir(exist_ok=True)
    repo = make_repo(d)
    extra += f"seed: {seed}\n" + ("ladder: [{epochs: 1}]\n" if proxy else f"delta: {DELTA}\n")
    run_dir = init(repo, python, BASE.replace("budget_s: 3600", f"budget_s: {budget}") + extra)
    env = toy_env(sigma=sigma, cheap_below=2, **planted)
    assert round_run(repo, "--rationale", "calibrate", env=env)[0] == 0
    for n, (s, line) in enumerate(hyps, 1):
        code_hypothesis(repo, run_dir, d, n, s, line, env)
    if proxy:
        code, out = round_run(repo, "--rationale", "calibrate the ladder", env=env)
        assert code == 0, out
        if out["fidelity_calibration"]["chosen"] != {"epochs": 1}:
            assert bo(repo, "accept-proxy", "--fidelity", '{"epochs": 1}', "--rationale", "cheap")[0] == 0
        assert bo(repo, "set-delta", str(DELTA), "--rationale", "the user's effect")[0] == 0
    outs = []
    for _ in range(rounds):
        code, out = round_run(repo, "--rationale", "search", env=env)
        assert code == 0, out
        outs.append(out)
        if not ready(out):
            break
    return run_dir, outs


def repeat(tmp_path, python, hyps, seeds=(1, 2, 3), **kw):
    """Independent runs, one per fixed seed, in parallel."""
    with ThreadPoolExecutor(len(seeds)) as pool:
        return list(pool.map(lambda s: verdict_run(tmp_path / f"s{s}", python, s, hyps, **kw), seeds))


def records(run_dir, h="H1.v1"):
    return [v for v in of_type(run_dir, "verdict") if v["hypothesis"] == h]


@pytest.mark.slow
def test_a_useless_lever_is_rejected_irrelevant_after_burn_in_and_a_confirming_check(tmp_path,
                                                                                  project_python):
    useless = (spec({"x": lever()}), '    term += 0 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [useless])
    rejected = [of_type(run_dir, "hypothesis_rejected") for run_dir, _ in runs]
    assert sum(r == [{"id": "H1.v1", "condition": "irrelevant", "verdict": r[0]["verdict"],
                      "frozen": []}] if r else False for r in rejected) >= 2, rejected
    for run_dir, [out] in runs:
        vs = records(run_dir)
        sampler = [t for t in started_in(run_dir, 1) if t["kind"] == "sampler"]
        # no verdict before burn-in: the first check comes after max(10·1, 20) = 20 sampler trials,
        # the next ones every max(5·1, 10) = 10
        assert [v["burn_in"]["fresh"] for v in vs] == [20 + 10 * i for i in range(len(vs))]
        for v in vs:
            if v["condition"] == "irrelevant":  # a sub-case of no-improvement, in every record
                assert v["m_u"]["upper"] < DELTA and v["delta_stat"]["upper"] < DELTA
            upto = [t["trial"] for t in sampler][:v["burn_in"]["fresh"]]
            assert set(upto) <= set(v["trials"])
            assert v["id"] == f"V-R1-H1.v1-{v['check']}" and v["delta"] == DELTA
        rej = of_type(run_dir, "hypothesis_rejected")
        if rej:
            *before, last = vs
            assert last["id"] == rej[0]["verdict"] and last["outcome"] == "reject"
            # the confirming check: the check before fired too, max(5·d, 10) sampler trials earlier
            assert before[-1]["outcome"] == "pending-reject" and before[-1]["condition"] is not None
            assert last["confirmation"]["confirms"] == before[-1]["id"]
            # `irrelevant` is a sub-case of `no-improvement`: Δ's upper bound is below δ too
            assert last["m_u"]["upper"] < DELTA and last["delta_stat"]["upper"] < DELTA
            assert out["trigger"] == "search_space"
            assert bo(run_dir.parents[1], "status")[1]["hypotheses"][0]["status"] == "rejected"


FIELDS = {"id", "hypothesis", "round", "check", "group", "outcome", "condition", "held", "reason",
          "sqrt_vt", "delta_stat", "m_u", "levers", "sobol_index", "best_point", "gp", "delta", "gates",
          "burn_in", "confirmation", "fidelity", "proxy", "prediction", "frozen", "trials", "context"}


@pytest.mark.slow
def test_a_harmful_lever_is_rejected_no_improvement_and_its_records_hold_every_field(tmp_path, project_python):
    harmful = (spec({"x": lever()}), '    term += 2 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [harmful])
    conditions = [[p["condition"] for p in of_type(run_dir, "hypothesis_rejected")] for run_dir, _ in runs]
    # rejected within R1 in 8 of 10 runs over seeds 0..9 (the rest still active); at 80%, fewer than 2 of 3
    # has probability ~10%
    assert conditions.count(["no-improvement"]) >= 2, conditions
    assert all(c in ([], ["no-improvement"]) for c in conditions), conditions
    for run_dir, [out] in runs:
        repo = run_dir.parents[1]
        vs = records(run_dir)
        assert vs and out["verdicts"] == vs
        for v in vs:
            assert set(v) == FIELDS
            assert set(v["gates"]) == {"trials", "coverage", "fit", "homogeneity"}
            assert all("passed" in g for g in v["gates"].values())
            assert set(v["gates"]["trials"]) == {"sampler", "agent", "passed"}
            for stat in ("sqrt_vt", "delta_stat", "m_u"):
                assert v[stat]["lower"] <= v[stat]["estimate"] <= v[stat]["upper"]
            assert v["fidelity"] == {"epochs": 4} and v["proxy"] == "reference"
            assert v["context"] == {"retained": [], "co_active": []}
            assert v["prediction"]["flag"] is None
        code, probe = bo(repo, "verdict", "H1")
        assert code == 0 and probe["records"] == vs
        assert bo(repo, "verdict", "H9")[1]["refused"]
        if not of_type(run_dir, "hypothesis_rejected"):
            continue
        # the lever matters (its range M_u, 2, is above δ) but only hurts, so Δ's upper bound is below δ
        [rej] = of_type(run_dir, "hypothesis_rejected")
        assert rej["frozen"] == ["H1.x"] and probe["status"] == "rejected"
        assert vs[-1]["m_u"]["lower"] > DELTA and vs[-1]["delta_stat"]["upper"] < DELTA
        assert vs[-1]["delta_stat"]["upper"] <= vs[-1]["m_u"]["upper"]


def test_no_verdict_before_burn_in(tmp_path, project_python):
    useless = (spec({"x": lever()}), '    term += 0 * lever("H1.x")')
    # slow trials and a small budget: the round cap ends R1 well short of 20 sampler trials
    run_dir, [out] = verdict_run(tmp_path / "r", project_python, 5, [useless], budget=16, sleep=0.3)
    assert out["trigger"] == "cap" and out["verdicts"] == [] and not of_type(run_dir, "verdict")
    code, probe = bo(run_dir.parents[1], "verdict", "H1.v1")
    fresh = sum(t["kind"] == "sampler" for t in started_in(run_dir, 1))
    assert code == 0 and probe["verdict"] == "burn-in" and probe["records"] == []
    assert probe["burn_in"] == {"fresh": fresh, "needed": 20, "spacing": 10, "cap": 80} and fresh < 20


@pytest.mark.slow
def test_a_useful_lever_is_retained(tmp_path, project_python):
    useful = (spec({"x": lever(0.2)}), '    term += 4 * (lever("H1.x") - 0.7) ** 2')
    runs = repeat(tmp_path, project_python, [useful])
    status = [bo(run_dir.parents[1], "status")[1]["hypotheses"][0]["status"] for run_dir, _ in runs]
    assert status.count("retained") >= 2, status
    for run_dir, [out] in runs:
        assert not of_type(run_dir, "hypothesis_rejected")
        assert out["trigger"] == "stall"  # a retain does not end the round
        for v in records(run_dir):
            if v["outcome"] == "retained":
                assert v["delta_stat"]["lower"] > DELTA and v["m_u"]["lower"] > DELTA
                assert v["prediction"] == {"flag": None, "contradicted": []}


@pytest.mark.slow
def test_a_contradicted_prediction_flags_a_retain_without_changing_it(tmp_path, project_python):
    # predicted lower, but the improving values lie above the baseline 0.2 (the optimum is 0.7)
    wrong = (spec({"x": lever(0.2, "lower")}), '    term += 4 * (lever("H1.x") - 0.7) ** 2')
    runs = repeat(tmp_path, project_python, [wrong])
    status = [bo(run_dir.parents[1], "status")[1]["hypotheses"][0]["status"] for run_dir, _ in runs]
    assert status.count("retained") >= 2, status
    for run_dir, _ in runs:
        for v in records(run_dir):
            if v["outcome"] == "retained":
                assert v["prediction"] == {"flag": "retained-against-prediction",
                                           "contradicted": ["H1.x"]}


@pytest.mark.slow
def test_a_lever_that_helps_only_with_a_co_active_one_is_retained_never_rejected(tmp_path, project_python):
    # H1.a does nothing alone; with H2.b both up the loss drops by 2 (a product: √V_T is 2/6 per lever)
    alone = (spec({"a": lever()}), '    term += 0 * lever("H1.a")')
    together = (spec({"b": lever()}), '    term += -2 * lever("H1.a") * lever("H2.b")')
    runs = repeat(tmp_path, project_python, [alone, together], rounds=2)
    status = [[h["status"] for h in bo(run_dir.parents[1], "status")[1]["hypotheses"]] for run_dir, _ in runs]
    assert status.count(["retained", "retained"]) >= 2, status
    for run_dir, outs in runs:
        assert len(outs) == 2
        assert not of_type(run_dir, "hypothesis_rejected")
        vs = records(run_dir, "H1.v1") + records(run_dir, "H2.v1")
        assert vs and all(v["outcome"] in ("active", "retained") for v in vs), vs
        for v in records(run_dir, "H1.v1"):
            assert v["context"]["co_active"] == ["H2.v1"]
            assert v["delta_stat"]["lower"] > DELTA  # B optimised in both terms: A's Δ is real


@pytest.mark.slow
def test_noise_that_differs_by_region_is_inconclusive_not_a_reject(tmp_path, project_python):
    # a useless lever whose upper half of the range is 6x noisier
    hetero = (spec({"x": lever()}),
              '    term += (0.3 if lever("H1.x") > 0.5 else 0.0) * random.Random(seed() + 1).gauss(0, 1)')
    runs = repeat(tmp_path, project_python, [hetero])
    outcomes = [of_type(run_dir, "hypothesis_inconclusive") for run_dir, _ in runs]
    assert sum(o == [{"id": "H1.v1", "verdict": o[0]["verdict"], "reason": "noise differs by region"}]
               if o else False for o in outcomes) >= 2, outcomes
    for run_dir, [out] in runs:
        assert not of_type(run_dir, "hypothesis_rejected")
        for v in records(run_dir):
            if v["outcome"] == "inconclusive":
                assert v["gates"]["homogeneity"]["passed"] is False
                assert v["gates"]["homogeneity"]["ratio"] >= 3
                assert out["trigger"] == "search_space"


@pytest.mark.slow
def test_a_stuck_hypothesis_escalates_then_is_inconclusive_at_the_evidence_cap(tmp_path, project_python):
    # a linear gain of exactly δ over the range: with σ = 0.15 Δ's interval keeps straddling δ
    stuck = (spec({"x": lever()}), '    term += -0.1 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [stuck], seeds=range(10), rounds=8, sigma=0.15)
    stuck_path = sum(stuck_run(run_dir, outs) for run_dir, outs in runs)
    # R1 stalled and escalated in 20 of 20 runs over seeds 0..19 (19 of 20 before the #21 checks
    # reshuffled trial seeds); at 95%, fewer than 8 of 10 has probability ~1% (before the GP's signal
    # floor, 4 of 10 seeds passed)
    assert stuck_path >= 8, stuck_path


def stuck_run(run_dir, outs) -> bool:
    """False unless R1 stalled with the hypothesis still active and escalated; then the rest of the
    stuck path must follow: a larger replicate share in R2, one escalation, inconclusive at the cap."""
    assert not of_type(run_dir, "hypothesis_rejected")
    esc = of_type(run_dir, "hypothesis_escalated")
    if not (outs[0]["trigger"] == "stall" and esc and esc[0]["round"] == 1):
        return False
    assert esc == [{"id": "H1.v1", "round": 1, "step": "replicates", "replicate_share": 0.3}]  # no rung at the reference
    r1, r2 = started_in(run_dir, 1), started_in(run_dir, 2)
    assert sum("replicate_of" in t for t in r2) >= 0.3 * len(r2) - 1
    assert sum(t["kind"] == "replicate" for t in r2) / len(r2) > sum(t["kind"] == "replicate" for t in r1) / len(r1)
    [inc] = of_type(run_dir, "hypothesis_inconclusive")
    last = records(run_dir)[-1]
    assert inc == {"id": "H1.v1", "verdict": last["id"], "reason": "no verdict after 80 sampler trials"}
    assert last["burn_in"]["fresh"] == last["burn_in"]["evidence"] == 80 and last["outcome"] == "inconclusive"
    assert outs[-1]["trigger"] == "search_space"
    assert bo(run_dir.parents[1], "status")[1]["next"] == [
        "propose and register a hypothesis",
        f"record narrative R{outs[-1]['round']} (the round analyst: a cited narrative, diagnostics, suggestions)",
        "record interplay: review H1.v1's removal against the untested list",
        "generate (the queue holds 0, below 2x the 1 slot(s) per round)"]
    return True


@pytest.mark.slow
def test_at_low_signal_to_noise_a_gain_of_delta_is_rarely_rejected_and_intervals_cover_it(tmp_path,
                                                                                          project_python):
    # a linear gain of exactly δ (Δ = M_u = δ) under noise σ = 1.5δ: the 95% one-sided bounds must
    # hold the truth, so the lever is almost never rejected
    gain = (spec({"x": lever()}), '    term += -0.1 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [gain], seeds=range(10), sigma=0.15)
    kept = sum(not of_type(run_dir, "hypothesis_rejected") for run_dir, _ in runs)
    vs = [v for run_dir, _ in runs for v in records(run_dir)]
    covered = sum(v["delta_stat"]["lower"] <= DELTA <= v["delta_stat"]["upper"]
                  and v["m_u"]["lower"] <= DELTA <= v["m_u"]["upper"] for v in vs)
    # over seeds 0..19: never rejected (20 of 20), and the bounds held δ in 23 of 26 checks; at 95%,
    # fewer than 9 of 10 kept has probability ~9%, at 88% coverage below 70% ~5% (before the GP's signal
    # floor: 10 of 20 rejected, and Δ's bounds held δ in 5 of 32 checks)
    assert kept >= 9, kept
    assert covered >= 0.7 * len(vs), (covered, len(vs))


@pytest.mark.slow
def test_a_fidelity_sensitive_lever_is_never_rejected_at_a_proxy(tmp_path, project_python):
    # the lever pays off only at the reference fidelity (4 epochs): at the 1-epoch rung it is flat
    late = (spec({"x": lever()}, fidelity_sensitive=True, fidelity_reason="pays off late"),
            '    term += 0 * lever("H1.x") if cheap else -lever("H1.x")')
    runs = repeat(tmp_path, project_python, [late], proxy=True, rounds=3, seeds=(1, 2))
    for run_dir, outs in runs:
        assert not of_type(run_dir, "hypothesis_rejected")
        at_proxy = [v for v in records(run_dir) if v["fidelity"] == {"epochs": 1}]
        assert at_proxy and all(v["outcome"] != "pending-reject" for v in at_proxy)
        held = [v for v in at_proxy if v["condition"]]
        assert held and all(v["held"] == "fidelity-sensitive: never rejected at a proxy fidelity"
                            for v in held)
        # stuck at the proxy: more replicates, then up a rung to the reference, where it can be retained
        steps = [e["step"] for e in of_type(run_dir, "hypothesis_escalated")]
        assert steps == ["replicates", "rung"], steps
        assert of_type(run_dir, "hypothesis_escalated")[1]["fidelity"] == {"epochs": 4}
        at_ref = [v for v in records(run_dir) if v["fidelity"] == {"epochs": 4}]
        assert at_ref and all(v["outcome"] in ("active", "retained") for v in at_ref)


@pytest.mark.slow
def test_a_broken_proxy_downgrades_its_rounds_rejects_to_inconclusive(tmp_path, project_python):
    # H1 is useless everywhere; H2's bowl ranks backwards at the proxy, so the drift check breaks it
    useless = (spec({"x": lever()}), '    term += 0 * lever("H1.x")')
    bowl = (spec({"y": lever(0.5)}), '    term += 4 * (lever("H2.y") - 0.7) ** 2')
    runs = repeat(tmp_path, project_python, [useless, bowl], seeds=range(10), proxy=True,
                  extra="deterministic: true\n", sigma=0, scramble=1)
    downgraded = 0
    for run_dir, [out] in runs:
        # the drift check compares the incumbent only with a config the proxy orders against it (a tie, or
        # one differing only in H1's useless lever, orders nothing), so the reversal always shows. Measured
        # 20 of 20 over seeds 0..19 (the toy's noise is off: under `deterministic: true` σ̂ = 0, so noise
        # would pass for an order, and was on in the old single-seed test: 17 of 20)
        assert out["drift"]["broken"] is True and not of_type(run_dir, "hypothesis_rejected")
        if records(run_dir)[-1]["outcome"] == "reject":
            assert out["trigger"] == "search_space"
            assert of_type(run_dir, "hypothesis_inconclusive") == [
                {"id": "H1.v1", "verdict": records(run_dir)[-1]["id"], "reason": "broken proxy fidelity"}]
            downgraded += 1
    # the rest ended inconclusive at a verdict check (with no noise at all, "noise differs by region"):
    # measured 17 of 20 reached the reject over seeds 0..19 (15 of 20 before the GP's signal floor); at 85%,
    # fewer than 6 of 10 has probability ~1%
    assert downgraded >= 6, downgraded


def test_a_proxy_that_orders_nothing_defers_the_rounds_rejects(tmp_path, project_python):
    # one useless lever and no noise: every config ties with the incumbent at the proxy, so the drift check
    # can't tell a faithful proxy from a broken one; it runs nothing, and the reject waits for a later round
    useless = (spec({"x": lever()}), '    term += 0 * lever("H1.x")')
    run_dir, [out] = verdict_run(tmp_path / "r", project_python, 1, [useless], proxy=True,
                                 extra="deterministic: true\n", sigma=0)
    assert records(run_dir)[-1]["outcome"] == "reject" and out["trigger"] == "search_space"
    [drift] = of_type(run_dir, "drift_check")
    assert out["drift"] == drift and drift["undecidable"] is True and drift["broken"] is None
    assert drift["trials"] == [] and not [t for t in started_in(run_dir, 1) if t["kind"] == "drift"]
    assert not of_type(run_dir, "hypothesis_rejected") and not of_type(run_dir, "hypothesis_inconclusive")
    assert of_type(run_dir, "reject_deferred") == [{"id": "H1.v1", "verdict": records(run_dir)[-1]["id"],
                                                    "round": 1, "reason": "undecidable proxy fidelity"}]
    assert bo(run_dir.parents[1], "status")[1]["hypotheses"][0]["status"] == "active"


@pytest.mark.slow
def test_a_flat_lever_next_to_a_curved_one_is_frozen_no_improvement_and_leaves_the_search(tmp_path,
                                                                                         project_python):
    # z does nothing, but next to the bowl in x BO leaves most of x's box unexplored with z moved, so
    # M_z's bound (taken over every setting of x) stays above δ: `irrelevant` lacks the evidence and z is
    # frozen `no-improvement` (#21 amendment). Frozen so in 20 of 20 runs over seeds 0..19;
    # at 95%, fewer than 2 of 3 has probability ~1%
    two = (spec({"x": lever(0.2), "z": lever()}),
           '    term += 4 * (lever("H1.x") - 0.7) ** 2 + 0 * lever("H1.z")')
    runs = repeat(tmp_path, project_python, [two], rounds=2)
    frozen = [of_type(run_dir, "lever_frozen") for run_dir, _ in runs]
    assert sum(f == [{"id": "H1.v1", "lever": "H1.z", "verdict": f[0]["verdict"], "condition": "no-improvement"}]
               if f else False for f in frozen) >= 2, frozen
    for (run_dir, outs), f in zip(runs, frozen):
        if not f or f[0]["condition"] != "no-improvement":
            continue
        v = next(v for v in records(run_dir) if v["id"] == f[0]["verdict"])
        z = v["levers"]["H1.z"]
        assert z["delta_stat"]["upper"] < DELTA <= z["m_u"]["upper"] and v["frozen"] == ["H1.z"]
        assert outs[0]["trigger"] == "search_space"  # a freeze changes the search space
        [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
        assert set(r2["search_space"]) == {"H1.x"}
        # no-improvement: z is filtered to its baseline, so only the R1 trials that ran z there seed R2
        r1 = [t for t in started_in(run_dir, 1) if t["status"] == "finished"]
        assert any(t["levers"]["H1.z"] != 0.0 for t in r1)
        assert {t["trial"] for t in r1 if t["levers"]["H1.z"] == 0.0} == {t["trial"] for t in r1} & set(r2["seeded"])
        assert all(t["levers"]["H1.z"] == 0.0 for t in started_in(run_dir, 2))
        assert not of_type(run_dir, "hypothesis_rejected")


def test_a_held_fidelity_sensitive_hypothesis_at_its_cap_moves_the_run_up_a_rung_and_never_breaks_the_drift_check(
        tmp_path, project_python):
    # an 8-lever flat hypothesis burns in over 80 fresh trials, so no stall (and no stall escalation) comes
    # first. The tail's proxy trials say nothing about it: its evidence cap can't make it inconclusive there,
    # and no check may freeze one of its levers there (a lever's reject); at the cap it escalates a rung
    late = (spec({"x": lever(), "u": lever()}, fidelity_sensitive=True, fidelity_reason="pays off late"),
            '    term += 0 * (lever("H1.x") + lever("H1.u")) if cheap else -lever("H1.x") - lever("H1.u")')
    names = "abcdefgh"
    wide = (spec({n: lever() for n in names}),
            "    term += 0 * (" + " + ".join(f'lever("H2.{n}")' for n in names) + ")")
    run_dir, outs = verdict_run(tmp_path / "r", project_python, 1, [late, wide], proxy=True, rounds=2)
    at_proxy = [v for v in records(run_dir) if v["fidelity"] == {"epochs": 1}]
    assert at_proxy and all(v["outcome"] == "active" and v["frozen"] == [] for v in at_proxy), at_proxy
    assert not [f for f in of_type(run_dir, "lever_frozen") if f["id"] == "H1.v1"]
    *before, cap = at_proxy
    assert cap["burn_in"]["evidence"] == 80 and all(v["burn_in"]["evidence"] < 80 for v in before)
    [esc] = [e for e in of_type(run_dir, "hypothesis_escalated") if e["id"] == "H1.v1"]
    assert esc == {"id": "H1.v1", "round": 1, "step": "rung", "fidelity": {"epochs": 4}, "proxy": "reference",
                   "stale_ladder": False, "verdict": cap["id"], "reason": "evidence cap"}
    assert outs[0]["trigger"] == "fidelity"  # a fidelity change ends the round
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert r2["fidelity"] == {"epochs": 4}
    # at the reference its count restarts (the proxy fidelity's trials were no evidence about it)
    at_ref = [v for v in records(run_dir) if v["fidelity"] == {"epochs": 4}]
    assert at_ref and all(v["outcome"] in ("active", "retained") for v in at_ref), at_ref
    assert all(v["burn_in"]["evidence"] == v["burn_in"]["fresh"] for v in at_ref)
    assert not [e for e in of_type(run_dir, "hypothesis_inconclusive") if e["id"] == "H1.v1"]
    # the drift check never blames the proxy fidelity for the tail it can't show: when it runs a pair, the
    # two configs differ in the other levers only and both run the incumbent's fidelity-sensitive one; here
    # every other lever is flat too, so the proxy mostly orders nothing and the check is undecidable
    checks = of_type(run_dir, "drift_check")
    assert checks and not any(c["broken"] for c in checks), checks
    drift = {t["trial"]: t["levers"] for t in of_type(run_dir, "trial_started") if t["kind"] == "drift"}
    pairs = [[drift[t] for t in c["trials"]] for c in checks if not c["undecidable"]]
    assert all(a["H1.x"] == b["H1.x"] and a != b for a, b in pairs)


def test_a_freeze_made_at_a_broken_proxy_fidelity_is_undone(tmp_path, project_python):
    # z does nothing and x's bowl ranks backwards at the proxy fidelity (scrambled): the check that freezes z
    # there rests on a fidelity the round's drift check finds broken, so, like a reject, the freeze is
    # downgraded: z returns to the search and H1 stays in it
    two = (spec({"x": lever(0.2), "z": lever()}),
           '    term += 4 * (lever("H1.x") - 0.7) ** 2 + 0 * lever("H1.z")')
    run_dir, outs = verdict_run(tmp_path / "r", project_python, 0, [two], proxy=True, rounds=2, scramble=1,
                                extra="deterministic: true\n", sigma=0)
    # noise off (σ̂ = 0) so the drift check sees the reversal: over seeds 0..9, z was frozen in R1 and R1's
    # check found the proxy fidelity broken in 9 of 10; seed 0 replays it
    frozen = of_type(run_dir, "lever_frozen")
    assert frozen and frozen[0]["lever"] == "H1.z" and frozen[0]["verdict"].startswith("V-R1-"), frozen
    assert outs[0]["trigger"] == "search_space" and outs[0]["drift"]["broken"] is True
    [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
    assert set(r2["search_space"]) == {"H1.x", "H1.z"}
    # every freeze stands only when its round's drift check passed (R2's may be undecidable: deferred likewise)
    reasons = {o["round"]: None if o["drift"]["broken"] is False else "undecidable proxy fidelity"
               if o["drift"]["undecidable"] else "broken proxy fidelity" for o in outs}
    assert of_type(run_dir, "lever_unfrozen") == [
        {"id": "H1.v1", "lever": f["lever"], "verdict": f["verdict"], "reason": reasons[int(f["verdict"][3])]}
        for f in frozen if reasons[int(f["verdict"][3])]]
    assert of_type(run_dir, "lever_unfrozen")[0]["reason"] == "broken proxy fidelity"
    assert not of_type(run_dir, "hypothesis_rejected") and not of_type(run_dir, "hypothesis_inconclusive")


@pytest.mark.slow
def test_a_stuck_hypothesis_carries_its_evidence_up_a_rung_to_the_cap(tmp_path, project_python):
    # the stuck lever (a gain of exactly δ, σ = 1.5δ) at a faithful proxy fidelity: a stall buys replicates,
    # the next one the reference; its proxy trials still count, so the cap is reached there before 80 fresh
    stuck = (spec({"x": lever()}), '    term += -0.1 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [stuck], seeds=range(10), rounds=6, sigma=0.15, proxy=True)
    capped = 0
    for run_dir, outs in runs:
        steps = [e["step"] for e in of_type(run_dir, "hypothesis_escalated")]
        inc = of_type(run_dir, "hypothesis_inconclusive")
        if steps != ["replicates", "rung"] or not inc:
            continue
        last = records(run_dir)[-1]
        assert last["fidelity"] == {"epochs": 4} and last["outcome"] == "inconclusive"
        assert last["burn_in"]["evidence"] >= 80 > last["burn_in"]["fresh"]
        assert inc == [{"id": "H1.v1", "verdict": last["id"],
                        "reason": f"no verdict after {last['burn_in']['evidence']} sampler trials"}]
        capped += 1
    # measured 20 of 20 over seeds 0..19 (capped at the reference at 20-40 fresh, 80-88 sampler trials in all);
    # at a true 90%, fewer than 8 of 10 has probability ~7%
    assert capped >= 8, capped


@pytest.mark.slow
def test_a_linear_lever_worth_twice_delta_is_retained_not_rejected_irrelevant(tmp_path, project_python):
    # the regression behind the #21 amendment: end to end the lever gains 0.2 = 2δ, but its √V_T is
    # 0.2/√12 ≈ 0.06 < δ, so the old form rejected it `irrelevant`; its range M_u is 0.2
    linear = (spec({"x": lever()}), '    term += -0.2 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [linear], seeds=range(10))
    status = [bo(run_dir.parents[1], "status")[1]["hypotheses"][0]["status"] for run_dir, _ in runs]
    # never rejected; retained within one round in 5 of 20 over seeds 0..19 (7 of 20 before the #21
    # equivalence change reshuffled trial seeds), the rest still active; at 25%, none of 10 has probability ~6%
    assert status.count("retained") >= 1, status
    for run_dir, _ in runs:
        assert not of_type(run_dir, "hypothesis_rejected")
        for v in records(run_dir):
            if v["outcome"] == "retained":
                assert v["delta_stat"]["lower"] > DELTA and v["m_u"]["lower"] > DELTA


@pytest.mark.slow
def test_a_single_lever_is_frozen_by_its_own_delta_one_per_check(tmp_path, project_python):
    # z hurts (Δ_z ≈ 0 but its range is 2): frozen `no-improvement`, its trials filtered to its baseline;
    # w gains 0.2 = 2δ linearly (√V_T_w ≈ 0.06 < δ, what the old form froze on): never frozen
    three = (spec({"x": lever(0.2), "z": lever(), "w": lever()}),
             '    term += 4 * (lever("H1.x") - 0.7) ** 2 + 2 * lever("H1.z") - 0.2 * lever("H1.w")')
    runs = repeat(tmp_path, project_python, [three], rounds=2)
    frozen = [of_type(run_dir, "lever_frozen") for run_dir, _ in runs]
    assert sum([(f["lever"], f["condition"]) for f in fs] == [("H1.z", "no-improvement")]
               for fs in frozen) >= 2, frozen
    for (run_dir, outs), fs in zip(runs, frozen):
        assert "H1.w" not in [f["lever"] for f in fs]
        assert len({f["verdict"] for f in fs}) == len(fs)  # at most one freeze per verdict check
        if [f["lever"] for f in fs] != ["H1.z"]:
            continue
        v = next(v for v in records(run_dir) if v["id"] == fs[0]["verdict"])
        assert v["levers"]["H1.z"]["delta_stat"]["upper"] < DELTA <= v["levers"]["H1.z"]["m_u"]["upper"]
        [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
        r1 = [t for t in started_in(run_dir, 1) if t["status"] == "finished"]
        # no-improvement: only the trials that ran z at its baseline seed R2
        assert {t["trial"] for t in r1 if t["levers"]["H1.z"] == 0.0} == {t["trial"] for t in r1} & set(r2["seeded"])
        assert any(t["levers"]["H1.z"] != 0.0 for t in r1)


@pytest.mark.slow
def test_two_substitute_levers_are_never_both_rejected_at_one_round_end(tmp_path, project_python):
    # H1.a alone gains 0.3, H2.b alone gains 0.3, both together still 0.3: each one's Δ is ≈ 0 because
    # the other is optimised in both terms, so both can reach a confirmed no-improvement reject at once
    a = (spec({"a": lever()}), '    term += 0 * lever("H1.a")')
    b = (spec({"b": lever()}), '    term += -0.3 * max(lever("H1.a"), lever("H2.b"))')
    runs = repeat(tmp_path, project_python, [a, b], rounds=3, seeds=(1, 2, 3, 4))
    for run_dir, _ in runs:
        rounds = [int(p["verdict"].split("-")[1][1:]) for p in of_type(run_dir, "hypothesis_rejected")]
        assert len(rounds) == len(set(rounds)), of_type(run_dir, "hypothesis_rejected")
        assert len(rounds) <= 1  # the survivor carries the gain, and its Δ shows it once the other is frozen
    deferred = [of_type(run_dir, "reject_deferred") for run_dir, _ in runs]
    assert any(deferred), deferred
    for (run_dir, _), ds in zip(runs, deferred):
        for d in ds:
            same = [v for h in ("H1.v1", "H2.v1") for v in records(run_dir, h)
                    if v["round"] == d["round"] and v["outcome"] == "reject"]
            rejected = [p["verdict"] for p in of_type(run_dir, "hypothesis_rejected")]
            [applied] = [v for v in same if v["id"] in rejected]
            deferred_v = next(v for v in same if v["id"] == d["verdict"])
            assert applied["delta_stat"]["upper"] <= deferred_v["delta_stat"]["upper"]
