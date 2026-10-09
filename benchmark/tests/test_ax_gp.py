"""#47's Ax variant: OptunaBackend's methods on Ax's Client, and the runs it makes. Skipped without Ax."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("ax")

import ax_gp
import bench
import boar
from boar import schema, warmstart

SPACE = {lever["name"]: schema.normalize_lever(lever) for p in bench.PROPOSALS.values() for lever in p["levers"]}


def test_the_harness_is_the_working_tree_not_a_copy_made_at_sync():
    # uv rebuilds a non-editable path dependency only when its pyproject.toml changes, not its source.
    assert Path(boar.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2] / "plugin/harness/src")


def test_queued_configs_come_first_in_order_then_ax_samples_the_space():
    backend = ax_gp.AxBackend(SPACE, 0, 6)
    queued = [dict(bench.DEFAULTS), {**bench.DEFAULTS, "dedup_set": True}]
    backend.enqueue("baseline", queued[0])
    backend.enqueue("try dedup", queued[1])
    asked = []
    for n in range(1, 4):
        asked.append(backend.ask(n))
        backend.tell("complete", 1.0)
    assert asked[:2] == [(queued[0], "baseline"), (queued[1], "try dedup")]
    params, label = asked[2]
    assert label is None and set(params) == set(SPACE)
    assert all(warmstart.allowed(SPACE[name], value) for name, value in params.items())


def test_ax_gets_each_result_with_its_guard_and_a_crash_as_a_failed_trial():
    backend = ax_gp.AxBackend(SPACE, 0, 6)
    # Ax minimises the metric, and a trial holds its guards only if `guards` stays at most 0.5.
    assert str(backend._client._experiment.optimization_config) == (
        'OptimizationConfig(objective=Objective(expression="-metric"), '
        'outcome_constraints=[OutcomeConstraint(guards <= 0.5)])')
    for n, (state, metric) in enumerate((("complete", 3.0), ("infeasible", 2.0), ("failed", None)), 1):
        backend.ask(n)
        backend.tell(state, metric)
    summary = backend._client.summarize()
    assert summary["trial_status"].tolist() == ["COMPLETED", "COMPLETED", "FAILED"]
    assert summary["metric"].tolist()[:2] == [3.0, 2.0] and summary["guards"].tolist()[:2] == [0.0, 1.0]


def test_warm_trials_reach_ax_as_finished_trials_with_levers_new_to_them_at_default():
    backend = ax_gp.AxBackend(SPACE, 0, 6)
    backend.add_warm([{"trial": 1, "config": {"dedup_set": True}, "state": "complete", "metric": 3.0},
                      {"trial": 2, "config": {"batch": 4}, "state": "infeasible", "metric": 2.0}], bench.DEFAULTS)
    summary = backend._client.summarize()
    assert summary["trial_status"].tolist() == ["COMPLETED"] * 2
    assert summary["metric"].tolist() == [3.0, 2.0] and summary["guards"].tolist() == [0.0, 1.0]
    assert summary["dedup_set"].tolist() == [True, False] and summary["batch"].tolist() == [1, 4]


TINY = {"rounds": 2, "trials_per_round": 3, "repeats": 1}


def test_the_gp_takes_over_once_trials_cover_the_startup_budget_and_a_seed_reproduces_its_run():
    variant = ax_gp.Recorded(ax_gp.AxBackend)
    run = bench.simulate(variant, 0, TINY)
    # Round 1: the baseline is queued, then two Sobol draws; round 2 has 3 warm trials, so after the incumbent the
    # GP suggests.
    assert [r["fit"] for r in variant.records] == [None, None, "ok", "ok"]
    assert all(r["seconds"] > 0 for r in variant.records)
    assert bench.simulate(ax_gp.Recorded(ax_gp.AxBackend), 0, TINY) == run


@pytest.mark.parametrize("logger, level, message, fit", [
    ("botorch", logging.DEBUG, "Fit attempt #1 of 5 triggered retry policy; retrying...", "failed"),  # BoTorch refits
    ("botorch", logging.DEBUG, "Fit attempt #1 of 5 failed with exception:\nNotPSDError", "failed"),
    ("ax.generation_strategy.generation_node", logging.WARNING,  # the GP fitted, but its candidates were all measured
     "gen failed with error GenerationStrategy exceeded `MAX_GEN_ATTEMPTS` of 5 while trying to generate a unique "
     "parameterization. This indicates that the search space has likely been fully explored, or that the sweep has "
     "converged., switching to fallback model with generator_enum Generators.SOBOL", "repeated"),
])
def test_a_failed_gp_fit_is_told_apart_from_a_sobol_draw_after_repeated_candidates(monkeypatch, logger, level, message,
                                                                                    fit):
    backend = ax_gp.AxBackend({"batch": SPACE["batch"]}, 0, 2)
    backend.add_warm([{"trial": n, "config": {"batch": n}, "state": "complete", "metric": float(n)} for n in (1, 2)],
                     bench.DEFAULTS)
    suggest = backend._client.get_next_trials

    def falling_back(max_trials):
        logging.getLogger(logger).log(level, message)
        return suggest(max_trials)

    monkeypatch.setattr(backend._client, "get_next_trials", falling_back)
    backend.ask(3)
    assert backend.fit == fit


def test_the_suggestion_table_gives_the_time_per_suggestion_how_its_trials_ended_and_how_its_gp_fits_went():
    def run(*states):  # one round: a queued trial that broke the guard, then the suggested trials
        queued = {"state": "infeasible", "queued": "baseline"}
        return [{"trials": [queued, *({"state": s, "queued": None} for s in states)]}]

    table = ax_gp.suggestions(
        {"harness": [run("complete", "infeasible", "failed")],
         "ax": [run("complete", "infeasible"), run("infeasible", "failed")]},
        {"harness": [{"seconds": s, "fit": None} for s in (0.01, 0.02, 0.03)],
         "ax": [{"seconds": 1, "fit": None}, {"seconds": 2, "fit": "ok"}, {"seconds": 3, "fit": "repeated"},
                {"seconds": 4, "fit": "failed"}]})
    rows = [line.strip("| ").split(" | ") for line in table.splitlines()[2:]]
    # median [IQR] seconds, broke guard, crashed, GP fits, fit failed, Sobol instead
    assert rows == [["harness", "3", "0.02 [0.015, 0.025]", "1", "1", "0", "0", "0"],
                    ["ax", "4", "2.5 [1.75, 3.25]", "2", "1", "3", "1", "1"]]


def _crash_then_ask(backend_class) -> tuple[dict, dict]:
    """The config a backend suggests and sees crash, then its next suggestion."""
    backend = backend_class({"batch": SPACE["batch"]}, 0, 2)
    backend.add_warm([{"trial": n, "config": {"batch": n}, "state": "complete", "metric": 6.0 - n} for n in (1, 2)],
                     bench.DEFAULTS)
    crashed, _ = backend.ask(3)
    backend.tell("failed", None)
    return crashed, backend.ask(4)[0]


def test_ax_suggests_a_crashed_config_again_unless_told_the_crash_as_a_broken_guard():
    crashed, again = _crash_then_ask(ax_gp.AxBackend)  # a failed trial gives the GP no data
    assert again == crashed
    crashed, after = _crash_then_ask(ax_gp.CrashAsGuard)
    assert after != crashed


def test_the_command_runs_each_variant_on_the_same_seeds_and_compares_it_with_the_first(capsys):
    ax_gp.main(["--first-seed", "7", "--seeds", "2", "--rounds", "2", "--trials", "3", "--repeats", "1", "--jobs", "2",
                "--variant", "harness", "--variant", "ax-crash"])
    out = capsys.readouterr().out
    assert "harness: 2 seeds" in out and "ax-crash: 2 seeds" in out and "ax-crash − harness, per seed:" in out
    assert bench.report([bench.simulate(bench.Harness(), seed, TINY) for seed in (7, 8)]) in out
    # Each seed: the baseline, then two startup draws; the incumbent, then two suggestions.
    assert [line.split(" | ")[:2] for line in out.splitlines() if line.startswith(("| harness", "| ax"))] == [
        ["| harness", "8"], ["| ax-crash", "8"]]
