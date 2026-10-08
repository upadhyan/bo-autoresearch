"""optimizer: the backend a round's study goes through, on a throwaway study.db."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from boar import optimizer

SPACE = {"a": {"name": "a", "type": "bool", "default": False}}


def db(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/study.db"


def create(tmp_path):
    return optimizer.OptunaBackend.create(db(tmp_path), "round-2", "min", SPACE, 0, 4)


def test_create_replaces_a_half_built_study_of_the_same_name(tmp_path):
    create(tmp_path).enqueue("baseline", {"a": False})
    assert create(tmp_path).waiting() == set()


def test_resume_names_the_queued_configs_still_waiting(tmp_path):
    study = create(tmp_path)
    study.enqueue("incumbent", {"a": True})
    study.enqueue("investigate H1 (1/1)", {"a": False})
    study.ask(1)  # the worker was killed during this trial
    assert optimizer.OptunaBackend.resume(db(tmp_path), "round-2", SPACE, 0, 4).waiting() == {"investigate H1 (1/1)"}
    with pytest.raises(KeyError):
        optimizer.OptunaBackend.resume(db(tmp_path), "round-3", SPACE, 0, 4)


def test_only_the_backend_imports_optuna():
    src = Path(optimizer.__file__).parent
    users = [p.name for p in sorted(src.glob("*.py")) if re.search(r"^(import|from) optuna\b", p.read_text(), re.M)]
    assert users == ["optimizer.py"]
