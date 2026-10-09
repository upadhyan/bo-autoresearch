"""The shared hard scenario (training.py): its levers, regret reference, memory limit, noisy guard, removal and
switches, run through bench.simulate as a target."""

from __future__ import annotations

import random

import pytest

import bench
import training
from boar import schema, store
from test_bench import _draw

TARGETS = training.TARGETS


def rounds(target) -> list[tuple[dict, dict]]:
    """Each round's world and search space, as bench.simulate builds them from the target's SCENARIO."""
    hyps: list[dict] = []
    world, out = dict(target.WORLD), []
    for r, step in enumerate(target.SCENARIO, 1):
        bench._change(hyps, step, r, target.PROPOSALS)
        world.update(step.get("world", {}))
        out.append((dict(world), store.search_space(hyps)))
    return out


def test_the_proposals_pass_the_harness_schema_and_most_numeric_defaults_sit_on_a_bound():
    assert schema.validate(list(training.PROPOSALS.values()), set(), set(), allow_supersedes=False) == []
    levers = [lever for p in training.PROPOSALS.values() for lever in p["levers"]]
    numeric = [lever for lever in levers if lever["type"] in ("int", "float")]
    assert len(levers) == 20 and len(numeric) == 14
    assert sum(lever["default"] in (lever["low"], lever["high"]) for lever in numeric) == 10  # the real run: 19 of 23


def test_the_active_levers_grow_to_19_and_aux_weight_is_tuned_in_rounds_2_to_4_only():
    spaces = [space for _, space in rounds(TARGETS["training"])]
    assert [len(space) for space in spaces] == [10, 13, 15, 16, 19, 19, 19, 19, 19]
    assert [r for r, space in enumerate(spaces, 1) if "aux_weight" in space] == [2, 3, 4]


SMALL = {"rounds": 5, "trials_per_round": 3, "repeats": 2}


def test_a_seed_reproduces_its_run():
    def run(seed: int) -> list[dict]:
        return bench.simulate(bench.Harness(), seed, SMALL, TARGETS["training"])

    assert run(3) == run(3)
    assert [t["metric"] for r in run(3) for t in r["trials"]] != [t["metric"] for r in run(4) for t in r["trials"]]


# The best true metric in each round's space, from an independent grid search over every lever (no closed forms),
# so best_config may be a little lower but never higher. The baseline is 9.20528 in every round: no world change
# moves it.
OPTIMUM = {
    "training": [5.91, 5.045415, 4.795415, 4.724956, 4.474956, 4.474956, 4.271002, 4.271002, 4.271002],
    "training-fixed-guard": [5.91, 5.045415, 4.795415, 4.724956, *[4.474956] * 5],
    "training-additive": [5.91, 5.045415, 4.795415, 4.724956, *[4.474956] * 5],
}


@pytest.mark.parametrize("name", TARGETS)
def test_the_regret_reference_is_the_best_feasible_config_in_each_rounds_space(name):
    target, rng = TARGETS[name], random.Random(0)
    for r, (world, space) in enumerate(rounds(target), 1):
        best = target.best_config(world, space)
        metric, ok = target.true_metric(best, world)
        assert set(best) == set(space) and ok and not training.crashes(best)
        assert OPTIMUM[name][r - 1] - 1e-3 < metric <= OPTIMUM[name][r - 1] + 1e-9
        assert target.true_metric({}, world)[0] == pytest.approx(9.20528, abs=1e-5)
        for _ in range(2000):
            config = {lever: _draw(space[lever], rng) for lever in space}
            other, ok = target.true_metric(config, world)
            assert other >= metric or not ok or training.crashes(config)


def memory(config: dict) -> float:
    c = {**training.DEFAULTS, **config}
    return training._memory(c["batch"], c["width"], c["precision"], c["checkpointing"])


def test_the_memory_limit_sits_just_past_the_optimum_while_only_fp32_exists_and_a_crash_ends_the_trial():
    (_, _), (w2, s2), (w3, s3), (w4, s4), *_ = rounds(TARGETS["training"])
    assert 0.9 < memory(TARGETS["training"].best_config(w2, s2)) / training.MEMORY < 1  # round 2: b192 w474
    assert training.crashes({"batch": 192, "width": 548})  # the best batch and width once dropout follows width
    assert memory(TARGETS["training"].best_config(w3, s3)) / training.MEMORY < 0.6  # from round 3, fp16 halves it
    assert memory(TARGETS["training"].best_config(w4, s4)) / training.MEMORY < 0.6
    trial = training.measure({"batch": 1024, "width": 1024}, w3, "c3", 0, 2)
    assert (trial["state"], trial["runs"], trial["margin"]) == ("failed", 1, None)


def test_the_guard_needs_aug_strength_and_label_smoothing_together_or_much_aux_weight_and_binds_at_the_optimum():
    world = dict(training.WORLD)

    def holds(**config) -> bool:
        return training.true_metric(config, world)[1]

    assert holds(aug_strength=1.0) and holds(label_smoothing=0.3) and not holds(aug_strength=1.0, label_smoothing=0.25)
    assert holds(aux_weight=0.6) and not holds(aux_weight=0.8)
    for name, target in TARGETS.items():
        for world, space in rounds(target)[1:]:
            assert 0 <= training.reg_r2(target.best_config(world, space), world) - training.FLOOR < 1e-6, name


def test_round_7_is_a_new_commit_on_every_target_and_moves_the_guard_only_on_training():
    def worlds_and_commits(name: str) -> tuple[list[float], list[str]]:
        run = bench.simulate(bench.Harness(), 0, {"rounds": 9, "trials_per_round": 1, "repeats": 1}, TARGETS[name])
        return [world["r2_shift"] for world, _ in rounds(TARGETS[name])], [rec["commit"] for rec in run]

    commits = ["c1", "c2", "c3", "c4", "c5", "c5", "c7", "c7", "c7"]
    assert worlds_and_commits("training") == ([0.0] * 6 + [0.06] * 3, commits)
    assert worlds_and_commits("training-fixed-guard") == ([0.0] * 9, commits)
    assert worlds_and_commits("training-additive") == ([0.0] * 9, commits)


def test_the_guard_is_measured_with_noise_on_each_repeat_and_the_margin_says_whether_it_held():
    world = dict(training.WORLD)
    near = {"aug_strength": 0.7, "label_smoothing": 0.142}  # 0.00084 above the floor, inside its noise
    trials = [training.measure(near, world, "c2", seed, 2) for seed in range(200)]
    assert all((t["margin"] >= 0) == (t["state"] == "complete") for t in trials)
    assert 20 < sum(t["state"] == "infeasible" for t in trials) < 180
    assert all(t["runs"] == (2 if t["state"] == "complete" else len(t["repeats"])) for t in trials)
    again, new = (training.measure(near, world, c, 5, 2) for c in ("c2", "c3"))
    assert again == trials[5] and new["margin"] != again["margin"]


def test_h4s_off_state_is_measured_in_round_4_so_its_removal_keeps_the_incumbent():
    run = bench.simulate(bench.Harness(), 2, SMALL, TARGETS["training"])  # a seed whose incumbent set aux_weight
    assert run[3]["trials"][0]["config"]["aux_weight"] > 0
    r4 = {t["queued"]: t for t in run[3]["trials"] if t["queued"]}
    assert r4["investigate H4 (off-state)"]["config"] == {**r4["incumbent"]["config"], "aux_weight": 0.0}
    assert r4["investigate H4 (1/1)"]["config"]["aux_weight"] == 0.1
    sampled = [t for rec in run[1:4] for t in rec["trials"] if t["state"] != "failed" and t["config"]["aux_weight"]]
    assert run[4]["excluded"]["rule2_outside_space"] == len(sampled) > 0
    assert run[4]["trials"][0]["queued"] == "incumbent" and run[4]["trials"][0]["config"]["aux_weight"] == 0


def test_aux_weight_moves_the_lr_optimum_only_where_it_interacts_and_lr_past_3x_its_optimum_diverges():
    def error(world: dict, **config) -> float:
        return training.true_metric({"aux_weight": 0.5, "warmup_frac": 0.07, **config}, world)[0]

    interacting, additive = TARGETS["training"].WORLD, TARGETS["training-additive"].WORLD
    assert error(interacting, lr_scale=1.75) < error(interacting, lr_scale=1.0)  # 1.75 = 1 + 1.5 × aux
    assert error(additive, lr_scale=1.0) < error(additive, lr_scale=1.75)
    below, past = (error(additive, lr_scale=lr) for lr in (2.999, 3.001))
    assert past - below == pytest.approx(2.0, abs=0.01)


def test_dead_levers_change_nothing_and_conditional_levers_matter_only_under_their_parent():
    world = dict(training.WORLD)

    def error(**config) -> float:
        return training.true_metric(config, world)[0]

    assert error(pin_memory=True, num_workers=8, cudnn_benchmark=True, log_every=100) == error()
    assert error(lr_floor=0.25) == error() and error(schedule="cosine", lr_floor=0.25) < error(schedule="step")
    assert error(schedule="cosine") > error(schedule="step")  # cosine pays off only with its floor
    assert error(ema_halflife=300) == error() and error(ema=True, ema_halflife=300) < error() < error(ema=True)
