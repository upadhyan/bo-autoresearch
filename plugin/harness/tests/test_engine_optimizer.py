"""optimizer: the backend a round's study goes through, on a throwaway study.db."""

from __future__ import annotations

import re
from pathlib import Path

import optuna
import pytest
from optuna.study import StudyDirection
from optuna.trial import TrialState

from boar import optimizer, warmstart

A = {"name": "a", "type": "bool", "default": False}
B = {"name": "b", "type": "int", "low": 1, "high": 8, "log": False, "default": 1}
X = {"name": "x", "type": "float", "low": 0.1, "high": 1.0, "log": True, "default": 1.0}
SPACE = {"a": A, "b": B, "x": X}


def db(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/study.db"


def create(tmp_path, direction="min", n_startup=4):
    return optimizer.OptunaBackend.create(db(tmp_path), "round-2", direction, SPACE, 0, n_startup)


def resume(tmp_path, n_startup=4):
    return optimizer.OptunaBackend.resume(db(tmp_path), "round-2", SPACE, 0, n_startup)


def stored(tmp_path) -> optuna.Study:
    """The study read straight from study.db: a round interrupted under older code resumes from this layout."""
    return optuna.load_study(study_name="round-2", storage=db(tmp_path))


def test_queued_configs_are_asked_first_in_order_with_their_labels(tmp_path):
    study = create(tmp_path)
    study.enqueue("incumbent", {"a": True, "b": 4, "x": 0.5})
    study.enqueue("joint H1+H2", {"a": True, "b": 8, "x": 1.0})
    asked = []
    for n in (1, 2, 3):
        asked.append(study.ask(n))
        study.tell("complete", 10.0 - n)
    assert asked[:2] == [({"a": True, "b": 4, "x": 0.5}, "incumbent"), ({"a": True, "b": 8, "x": 1.0}, "joint H1+H2")]
    params, label = asked[2]
    assert label is None and set(params) == set(SPACE)
    assert all(warmstart.allowed(SPACE[name], v) for name, v in params.items())


def test_results_are_stored_in_the_study_db_layout(tmp_path):
    study = create(tmp_path, direction="max")
    study.enqueue("baseline", {"a": False, "b": 1, "x": 1.0})
    for n, (state, metric) in enumerate([("complete", 3.0), ("infeasible", 5.0), ("failed", None)], 1):
        study.ask(n)
        study.tell(state, metric)
    s = stored(tmp_path)
    assert s.direction == StudyDirection.MAXIMIZE
    assert [t.state for t in s.trials] == [TrialState.COMPLETE, TrialState.COMPLETE, TrialState.FAIL]
    assert [t.user_attrs for t in s.trials] == [{"queued": "baseline", "boar_trial": 1}, {"boar_trial": 2}, {"boar_trial": 3}]
    assert [t.values for t in s.trials[:2]] == [[3.0], [5.0]]
    assert [t.system_attrs["constraints:guards"] for t in s.trials[:2]] == [0.0, 1.0], "where TPE reads them"


def test_warm_trials_are_copied_in_with_their_guard_result_and_new_levers_at_default(tmp_path):
    defaults = {"a": False, "b": 1, "x": 1.0}
    earlier = [{"trial": 1, "config": {"a": True}, "state": "complete", "metric": 5.0},
               {"trial": 2, "config": {"a": False, "b": 3}, "state": "infeasible", "metric": 1.0}]
    create(tmp_path).add_warm(earlier, defaults)
    s = stored(tmp_path)
    assert [t.params for t in s.trials] == [{"a": True, "b": 1, "x": 1.0}, {"a": False, "b": 3, "x": 1.0}]
    assert [t.user_attrs for t in s.trials] == [{"boar_trial": 1, "warm": True}, {"boar_trial": 2, "warm": True}]
    assert [(t.values, t.constraints) for t in s.trials] == [([5.0], {"guards": 0.0}), ([1.0], {"guards": 1.0})]


def test_create_replaces_a_half_built_study_of_the_same_name(tmp_path):
    create(tmp_path).enqueue("baseline", {"a": False, "b": 1, "x": 1.0})
    create(tmp_path, direction="max")
    s = stored(tmp_path)
    assert s.trials == [] and s.direction == StudyDirection.MAXIMIZE


def test_resume_fails_the_trial_left_in_flight_and_names_the_queued_configs_still_waiting(tmp_path):
    study = create(tmp_path)
    study.enqueue("incumbent", {"a": True, "b": 4, "x": 0.5})
    study.enqueue("investigate H1 (1/1)", {"a": True, "b": 1, "x": 1.0})
    study.ask(1)  # the worker was killed during this trial
    assert resume(tmp_path).waiting() == {"investigate H1 (1/1)"}
    assert [t.state for t in stored(tmp_path).trials] == [TrialState.FAIL, TrialState.WAITING]
    with pytest.raises(KeyError):
        optimizer.OptunaBackend.resume(db(tmp_path), "round-3", SPACE, 0, 4)


def test_a_resumed_study_does_not_replay_the_random_draws_it_already_made(tmp_path):
    study = create(tmp_path, n_startup=10)
    drawn = []
    for n in (1, 2):
        drawn.append(study.ask(n)[0])
        study.tell("complete", 1.0)
    params, _ = resume(tmp_path, n_startup=10).ask(3)
    assert params not in drawn


def test_only_the_backend_imports_optuna():
    src = Path(optimizer.__file__).parent
    users = [p.name for p in sorted(src.glob("*.py")) if re.search(r"^(import|from) optuna\b", p.read_text(), re.M)]
    assert users == ["optimizer.py"]
