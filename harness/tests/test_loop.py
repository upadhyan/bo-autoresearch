"""The loop (#36): the round analyst is a duty after every round, and the five plugin agent
definitions match what the hooks and `record` expect."""
import json
import re
from pathlib import Path

import pytest
import yaml

from conftest import bo
from boautoresearch import checks, hypotheses, records
from test_rounds import BASE, init, make_repo
from test_start import ready_run

AGENTS = Path(__file__).parents[2] / "agents"
ROLES = {"hypothesis-generator": "proposal", "lever-coder": None, "registration-reviewer": "review",
         "interplay-reviewer": "interplay", "round-analyst": "narrative"}


def test_after_a_round_its_narrative_is_a_duty_and_gates_the_next_round(tmp_path, project_python):
    repo, run_dir = ready_run(tmp_path, project_python)
    code, out = bo(repo, "round-run", "--rationale", "R1")
    assert code == 0, out
    status = bo(repo, "status")[1]
    assert status["narrative_missing"] == [1]
    assert any(d.startswith("record narrative R1") for d in status["next"]), status["next"]
    code, out = bo(repo, "round-run", "--rationale", "R2")
    assert code == 1 and "narrative" in out["failing"], out
    cites = [v["id"] for v in bo(repo, "verdict", "H1.v1")[1]["records"] if v["round"] == 1]
    f = tmp_path / "narrative.json"
    f.write_text(json.dumps({"round": 1, "text": "H1 is still burning in.", "cites": cites,
                             "diagnostics": [], "suggestions": [], "generation": False}))
    code, out = bo(repo, "record", "narrative", "--file", str(f), "--agent-id", "an", "--actor",
                   "round-analyst", "--rationale", "the round's narrative")
    assert code == 0, out
    status = bo(repo, "status")[1]
    assert status["narrative_missing"] == [] and not any("narrative" in d for d in status["next"])


@pytest.fixture(scope="module")
def run_repo(tmp_path_factory, project_python):
    """An initialised run with one prohibited directive (the checks are inert without a run)."""
    repo = make_repo(tmp_path_factory.mktemp("loop"))
    init(repo, project_python, BASE + "directives:\n- {id: D1, severity: prohibited, statement: no huge"
         " steps, reason: stability, scope: mechanism}\n")
    return repo


def check_bash(repo, role, command):
    agent = [] if role == "orchestrator" else ["--agent", f"boautoresearch:{role}", "--agent-id", "a1"]
    code, out = bo(repo, "check", "bash", *agent, "--", command)
    assert code == 0, out
    return out


def test_the_registry_probe_shows_the_current_brief_and_directives_to_a_read_only_reviewer(run_repo):
    assert check_bash(run_repo, "registration-reviewer", "/x/venv/bin/boautoresearch registry")["allow"]
    code, out = bo(run_repo, "registry")
    assert code == 0, out
    reg = out["registry"]
    assert reg["version"] == 1 and reg["brief"]["purpose"] and [d["id"] for d in reg["directives"]] == ["D1"]


def agent(role):
    text = (AGENTS / f"{role}.md").read_text()
    _, front, body = text.split("---\n", 2)
    return yaml.safe_load(front), body


def test_every_role_has_an_agent_on_the_top_tier_whose_name_the_hooks_know():
    assert {p.stem for p in AGENTS.glob("*.md")} == set(ROLES)
    for role, kind in ROLES.items():
        front, body = agent(role)
        assert front["name"] == role and front["description"] and front["model"] == "opus", role
        assert checks.role(f"boautoresearch:{role}") == role
        assert (role in checks.RECORDING) == (kind is not None), role
        tools = {t.strip() for t in front["tools"].split(",")}
        if role in checks.READ_ONLY:
            assert tools <= {"Read", "Bash"}, role
        if kind:
            assert f"record {kind}" in body, role


RECORD = re.compile(r"(BO record (\w+) --file - [^\n]*<<'EOF'\n(.*?)\nEOF)", re.S)


def test_the_record_examples_in_the_prompts_use_their_kinds_fields_and_pass_the_hook(run_repo):
    skill = (AGENTS.parent / "skills" / "start" / "SKILL.md").read_text()
    seen = set()
    for role, text in [*((r, agent(r)[1]) for r in ROLES), ("orchestrator", skill)]:
        for command, kind, body in RECORD.findall(text):
            fields, _ = records.KINDS[kind]
            for example in records.items(kind, json.loads(body)):  # a batch's items, each as its own record
                assert set(example) <= fields, (role, kind, set(example) - fields)
                for spec in example.get("hypotheses", []):
                    assert set(spec) <= hypotheses.SPEC_KEYS, (role, set(spec) - hypotheses.SPEC_KEYS)
            out = check_bash(run_repo, role, command.replace("BO ", "/x/venv/bin/boautoresearch ", 1))
            assert out["allow"], (role, out)
            if role != "orchestrator":  # a subagent's record carries its own id and role
                assert "--agent-id a1 --actor " + role in out["updated_command"], out
            seen.add(kind)
    assert seen >= {"proposal", "review", "interplay", "narrative", "expected"}
