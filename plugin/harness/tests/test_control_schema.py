from __future__ import annotations

import pytest

from boar import schema


def lever(name="n_workers", **kw):
    base = {"name": name, "type": "int", "low": 1, "high": 8, "default": 1}
    base.update(kw)
    return base


def prop(*levers, **kw):
    base = {
        "statement": "Parallelising the parse stage cuts wall time",
        "mechanism": "parse is 60% of the profile and has no shared state",
        "source": "research",
        "citations": ["profile.txt"],
        "levers": list(levers) or [lever()],
        "supersedes": None,
    }
    base.update(kw)
    return base


def errors(data, existing=(), active=(), allow_supersedes=False):
    return schema.validate(data, set(existing), set(active), allow_supersedes)


def test_valid_file_of_every_lever_type():
    data = [
        prop(
            lever("workers"),
            {"name": "use_set", "type": "bool", "default": False},
            {"name": "alpha", "type": "float", "low": 1e-4, "high": 1.0, "log": True, "default": 0.01},
            {"name": "mode", "type": "categorical", "choices": ["a", "b", 3, None, True], "default": "a"},
        ),
        prop(lever("batch.size_2", low=16, high=512, log=True, default=64), source="user", citations=[]),
    ]
    assert errors(data) == []


def test_optional_keys_may_be_omitted():
    p = prop()
    del p["citations"], p["supersedes"]
    assert errors([p]) == []


@pytest.mark.parametrize("data, needle", [({"statement": "x"}, "JSON list"), ([], "empty"), (["x"], "JSON object")])
def test_file_shape(data, needle):
    (err,) = errors(data)
    assert needle in err


@pytest.mark.parametrize(
    "change, needle",
    [
        ({"statement": "  "}, "'statement' must be a non-empty string"),
        ({"mechanism": 3}, "'mechanism' must be a non-empty string"),
        ({"source": "web"}, "'source' must be one of user, research, synthesis"),
        ({"citations": "paper"}, "'citations' must be a list of strings"),
        ({"citations": [1]}, "'citations' must be a list of strings"),
        ({"levers": []}, "'levers' must be a non-empty list"),
        ({"id": "H1"}, "unknown key 'id'"),
    ],
)
def test_proposal_field_errors(change, needle):
    (err,) = errors([prop(**change)])
    assert err.startswith("proposal[0]:") and needle in err


def test_missing_required_keys_are_all_reported():
    errs = errors([{"source": "user"}])
    for key in ("statement", "mechanism", "levers"):
        assert any(f"missing {key!r}" in e for e in errs)


@pytest.mark.parametrize(
    "lv, needle",
    [
        (lever(name="1bad"), "'name' must match"),
        (lever(name="has space"), "'name' must match"),
        (lever(type="string"), "'type' must be one of"),
        (lever(step=2), "unknown key 'step' for a int lever"),
        ({"name": "b", "type": "bool", "default": False, "low": 0}, "unknown key 'low' for a bool lever"),
        ({"name": "b", "type": "bool", "default": 0}, "default must be true or false"),
        ({"name": "b", "type": "bool"}, "missing 'default'"),
        (lever(default=None), "default must be an integer"),
        (lever(low=1.5), "'low' must be an integer"),
        (lever(low=True), "'low' must be an integer"),
        (lever(default=True), "default must be an integer"),
        (lever(low=8, high=8), "needs low < high"),
        (lever(default=9), "default 9 is outside [1, 8]"),
        (lever(low=0, high=8, default=0, log=True), "needs low > 0"),
        (lever(log="yes"), "'log' must be true or false"),
        ({"name": "f", "type": "float", "low": 0.0, "high": 1.0, "default": 2.0}, "outside"),
        ({"name": "f", "type": "float", "low": 0.0, "high": float("inf"), "default": 0.0}, "finite number"),
        ({"name": "f", "type": "float", "low": "0", "high": 1.0, "default": 0.5}, "'low' must be a finite number"),
        ({"name": "c", "type": "categorical", "choices": [], "default": "a"}, "'choices' must be a non-empty list"),
        ({"name": "c", "type": "categorical", "choices": ["a", ["b"]], "default": "a"}, "every choice must be"),
        ({"name": "c", "type": "categorical", "choices": ["a", "a"], "default": "a"}, "must be distinct"),
        ({"name": "c", "type": "categorical", "choices": [1, True], "default": 1}, "must be distinct"),
        ({"name": "c", "type": "categorical", "choices": [1, 2], "default": True}, "not one of the choices"),
        ({"name": "c", "type": "categorical", "choices": [1, 2], "default": 1.0}, "not one of the choices"),
        ({"name": "c", "type": "categorical", "choices": ["a"], "default": "a", "log": True}, "unknown key 'log'"),
        ("lever", "must be a JSON object"),
    ],
)
def test_lever_errors(lv, needle):
    errs = errors([prop(lv)])
    assert errs and all(e.startswith("proposal[0].levers[0]") for e in errs)
    assert any(needle in e for e in errs), errs


def test_lever_name_unique_within_file_and_across_run():
    errs = errors([prop(lever("a")), prop(lever("a"), lever("b"))], existing={"b"})
    assert any("proposal[1].levers[0] 'a'" in e and "also used by proposal[0].levers[0]" in e for e in errs)
    assert any("proposal[1].levers[1] 'b'" in e and "already used in this run" in e for e in errs)


def test_supersedes_rules():
    assert "must be null in setup" in errors([prop(supersedes="H1")], active={"H1"})[0]
    assert errors([prop(supersedes="H1")], active={"H1"}, allow_supersedes=True) == []
    (err,) = errors([prop(supersedes="H9")], active={"H1"}, allow_supersedes=True)
    assert "must name an active hypothesis (active: H1)" in err


@pytest.mark.parametrize("sup", [["H1"], {"id": "H1"}, 1])
def test_non_string_supersedes_is_listed_with_the_other_errors(sup):
    errs = errors([prop(supersedes=sup, source="?")], active={"H1"}, allow_supersedes=True)
    assert len(errs) == 2
    assert any("'supersedes' must be null or a single hypothesis id string" in e for e in errs)
    assert any("'source'" in e for e in errs)


def test_all_errors_are_collected_with_index_and_lever_name():
    data = [prop(lever("ok")), prop(lever("bad", default=99), source="?"), prop(lever("x", type="nope"))]
    errs = errors(data)
    assert len(errs) == 3
    assert any(e.startswith("proposal[1]:") and "'source'" in e for e in errs)
    assert any(e.startswith("proposal[1].levers[0] 'bad'") for e in errs)
    assert any(e.startswith("proposal[2].levers[0] 'x'") for e in errs)


def test_normalize_lever():
    assert schema.normalize_lever(lever()) == {
        "name": "n_workers", "type": "int", "low": 1, "high": 8, "log": False, "default": 1,
    }
    f = schema.normalize_lever({"name": "f", "type": "float", "low": 0, "high": 2, "default": 1})
    assert f == {"name": "f", "type": "float", "low": 0.0, "high": 2.0, "log": False, "default": 1.0}
    assert isinstance(f["default"], float)
    assert schema.normalize_lever({"name": "b", "type": "bool", "default": True}) == {
        "name": "b", "type": "bool", "default": True,
    }
    c = schema.normalize_lever({"name": "c", "type": "categorical", "choices": ["x", "y"], "default": "y"})
    assert c == {"name": "c", "type": "categorical", "choices": ["x", "y"], "default": "y"}
