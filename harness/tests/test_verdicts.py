"""The verdict engine (#26): the fixed reject form against toys with planted truth.

Each toy is the bowl-free toy_bo trainer with lever code of our own: `term += ...` lines whose
effect on the loss is the planted truth. δ = 0.1 and the toy's noise σ = 0.05 (δ = 2σ).
"""
import json
from concurrent.futures import ThreadPoolExecutor

from conftest import bo, ready, round_run
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
    assert bo(repo, "register", f"H{n}", "--rationale", "reviewed")[0] == 0
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
            assert last["sqrt_vt"]["upper"] < DELTA
            assert out["trigger"] == "search_space"
            assert bo(run_dir.parents[1], "status")[1]["hypotheses"][0]["status"] == "rejected"


FIELDS = {"id", "hypothesis", "round", "check", "group", "outcome", "condition", "held", "reason",
          "sqrt_vt", "delta_stat", "levers", "sobol_index", "best_point", "gp", "delta", "gates",
          "burn_in", "confirmation", "fidelity", "proxy", "prediction", "frozen", "trials", "context"}


def test_verdict_records_hold_every_field_and_the_probe_shows_them(tmp_path, project_python):
    harmful = (spec({"x": lever()}), '    term += 2 * lever("H1.x")')
    run_dir, [out] = verdict_run(tmp_path / "r", project_python, 1, [harmful])
    repo = run_dir.parents[1]
    vs = records(run_dir)
    assert vs and out["verdicts"] == vs
    for v in vs:
        assert set(v) == FIELDS
        assert set(v["gates"]) == {"trials", "coverage", "fit", "homogeneity"}
        assert all("passed" in g for g in v["gates"].values())
        assert set(v["gates"]["trials"]) == {"sampler", "agent", "passed"}
        for stat in ("sqrt_vt", "delta_stat"):
            assert v[stat]["lower"] <= v[stat]["estimate"] <= v[stat]["upper"]
        assert v["fidelity"] == {"epochs": 4} and v["proxy"] == "reference"
        assert v["context"] == {"retained": [], "co_active": []}
        assert v["prediction"]["flag"] is None
    # harmful: the lever matters (√V_T above δ) but only hurts, so Δ's upper bound is below δ
    [rej] = of_type(run_dir, "hypothesis_rejected")
    assert rej["condition"] == "no-improvement" and rej["frozen"] == ["H1.x"]
    assert vs[-1]["sqrt_vt"]["lower"] > DELTA and vs[-1]["delta_stat"]["upper"] < DELTA
    code, probe = bo(repo, "verdict", "H1")
    assert code == 0 and probe["status"] == "rejected" and probe["records"] == vs
    assert bo(repo, "verdict", "H9")[1]["refused"]


def test_no_verdict_before_burn_in(tmp_path, project_python):
    useless = (spec({"x": lever()}), '    term += 0 * lever("H1.x")')
    # slow trials and a small budget: the round cap ends R1 well short of 20 sampler trials
    run_dir, [out] = verdict_run(tmp_path / "r", project_python, 5, [useless], budget=16, sleep=0.3)
    assert out["trigger"] == "cap" and out["verdicts"] == [] and not of_type(run_dir, "verdict")
    code, probe = bo(run_dir.parents[1], "verdict", "H1.v1")
    fresh = sum(t["kind"] == "sampler" for t in started_in(run_dir, 1))
    assert code == 0 and probe["verdict"] == "burn-in" and probe["records"] == []
    assert probe["burn_in"] == {"fresh": fresh, "needed": 20, "spacing": 10, "cap": 80} and fresh < 20


def test_a_harmful_lever_is_rejected_no_improvement(tmp_path, project_python):
    harmful = (spec({"x": lever()}), '    term += 2 * lever("H1.x")')
    runs = repeat(tmp_path, project_python, [harmful])
    conditions = [[p["condition"] for p in of_type(run_dir, "hypothesis_rejected")] for run_dir, _ in runs]
    assert conditions.count(["no-improvement"]) >= 2, conditions
    assert all(c in ([], ["no-improvement"]) for c in conditions), conditions


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
                assert v["sqrt_vt"]["lower"] > DELTA and v["delta_stat"]["lower"] > DELTA
                assert v["prediction"] == {"flag": None, "contradicted": []}


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


def test_a_lever_that_helps_only_with_a_co_active_one_is_never_rejected(tmp_path, project_python):
    # H1.a does nothing alone; with H2.b both up the loss drops by 2
    alone = (spec({"a": lever()}), '    term += 0 * lever("H1.a")')
    together = (spec({"b": lever()}), '    term += -2 * lever("H1.a") * lever("H2.b")')
    runs = repeat(tmp_path, project_python, [alone, together], rounds=2)
    for run_dir, outs in runs:
        assert len(outs) == 2
        assert not of_type(run_dir, "hypothesis_rejected")
        vs = records(run_dir, "H1.v1") + records(run_dir, "H2.v1")
        assert vs and all(v["outcome"] in ("active", "retained") for v in vs), vs
        for v in records(run_dir, "H1.v1"):
            assert v["context"]["co_active"] == ["H2.v1"]
            assert v["delta_stat"]["lower"] > DELTA  # B optimised in both terms: A's Δ is real


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


def test_a_stuck_hypothesis_escalates_then_is_inconclusive_at_the_evidence_cap(tmp_path, project_python):
    # √V_T of a linear effect of 0.346 over the range is 0.1 = δ: with σ = 0.15 no check decides it
    stuck = (spec({"x": lever()}), '    term += -0.346 * lever("H1.x")')
    run_dir, outs = verdict_run(tmp_path / "r", project_python, 1, [stuck], rounds=8, sigma=0.15)
    assert outs[0]["trigger"] == "stall"
    esc = of_type(run_dir, "hypothesis_escalated")
    assert esc[0] == {"id": "H1.v1", "round": 1, "step": "replicates", "replicate_share": 0.3}
    assert len(esc) == 1  # at the reference fidelity there is no rung to move up to
    r2 = started_in(run_dir, 2)
    assert sum("replicate_of" in t for t in r2) >= 0.3 * len(r2) - 1
    r1 = started_in(run_dir, 1)
    assert sum(t["kind"] == "replicate" for t in r2) / len(r2) > sum(t["kind"] == "replicate" for t in r1) / len(r1)
    [inc] = of_type(run_dir, "hypothesis_inconclusive")
    last = records(run_dir)[-1]
    assert inc == {"id": "H1.v1", "verdict": last["id"], "reason": "no verdict after 80 fresh sampler trials"}
    assert last["burn_in"]["fresh"] == 80 and last["outcome"] == "inconclusive"
    assert outs[-1]["trigger"] == "search_space" and not of_type(run_dir, "hypothesis_rejected")
    assert bo(run_dir.parents[1], "status")[1]["next"] == [
        "propose and register a hypothesis",
        "record interplay: review H1.v1's removal against the untested list",
        "generate (the queue holds 0, below 2x the 1 slot(s) per round)"]


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


def test_a_broken_proxy_downgrades_its_rounds_rejects_to_inconclusive(tmp_path, project_python):
    # H1 is useless everywhere; H2's bowl ranks backwards at the proxy, so the drift check breaks it
    useless = (spec({"x": lever()}), '    term += 0 * lever("H1.x")')
    bowl = (spec({"y": lever(0.5)}), '    term += 4 * (lever("H2.y") - 0.7) ** 2')
    run_dir, [out] = verdict_run(tmp_path / "r", project_python, 1, [useless, bowl], proxy=True,
                                 extra="deterministic: true\n", scramble=1)
    assert [v["outcome"] for v in records(run_dir)][-2:] == ["pending-reject", "reject"]
    assert out["trigger"] == "search_space" and out["drift"]["broken"] is True
    assert not of_type(run_dir, "hypothesis_rejected")
    assert of_type(run_dir, "hypothesis_inconclusive") == [
        {"id": "H1.v1", "verdict": records(run_dir)[-1]["id"], "reason": "broken proxy fidelity"}]


def test_a_single_lever_below_delta_is_frozen_and_leaves_the_search(tmp_path, project_python):
    two = (spec({"x": lever(0.2), "z": lever()}),
           '    term += 4 * (lever("H1.x") - 0.7) ** 2 + 0 * lever("H1.z")')
    runs = repeat(tmp_path, project_python, [two], rounds=2)
    frozen = [of_type(run_dir, "lever_frozen") for run_dir, _ in runs]
    assert sum(f == [{"id": "H1.v1", "lever": "H1.z", "verdict": f[0]["verdict"]}] if f else False
               for f in frozen) >= 2, frozen
    for (run_dir, outs), f in zip(runs, frozen):
        if not f:
            continue
        v = next(v for v in records(run_dir) if v["id"] == f[0]["verdict"])
        assert v["levers"]["H1.z"]["upper"] < DELTA and v["frozen"] == ["H1.z"]
        assert outs[0]["trigger"] == "search_space"  # a freeze changes the search space
        [r2] = [s for s in of_type(run_dir, "round_started") if s["round"] == 2]
        assert set(r2["search_space"]) == {"H1.x"}
        # frozen is credibly irrelevant: H1.z's key is dropped, so every R1 trial seeds R2
        r1 = [t for t in started_in(run_dir, 1) if t["status"] == "finished"]
        assert any(t["levers"]["H1.z"] != 0.0 for t in r1)
        assert {t["trial"] for t in r1} <= set(r2["seeded"])
        assert all(t["levers"]["H1.z"] == 0.0 for t in started_in(run_dir, 2))
        assert not of_type(run_dir, "hypothesis_rejected")
