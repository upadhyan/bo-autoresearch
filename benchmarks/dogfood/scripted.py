"""The scripted caller: every role of the loop played by a script through the harness CLI, with the
reviewers' calls from expected.yaml and the lever code for the planted mechanisms written here. It
stands in for `claude -p` where there is no Claude login: it checks check.py, and that the harness
reaches the planted verdicts end to end. The adversarial checklist fires at the same fixed points as
in a real run (run_benchmark.adversary).

    python scripted.py <repo> <run.yaml> <adversarial.json>    (run_benchmark.py calls it)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

import run_benchmark as rb

HERE = Path(__file__).resolve().parent
EXPECTED = yaml.safe_load((HERE / "expected.yaml").read_text())
ANCHOR = "    # research changes to cfg go here\n"


class Caller:
    def __init__(self, repo: Path, env: dict):
        self.repo, self.env, self.tmp = repo, env, Path(tempfile.mkdtemp(prefix="dogfood-records-"))
        self.bo = None  # the run venv's boautoresearch, once init has made it

    def cli(self, *args, check=True):
        p = subprocess.run([*self.bo, *args], cwd=self.repo, capture_output=True, text=True, env=self.env)
        out = json.loads(p.stdout)
        if check and p.returncode:
            raise RuntimeError(f"{args[0]} refused: {out.get('reason')}")
        return out

    def record(self, kind, body, actor, rationale):
        f = self.tmp / f"{kind}-{len(list(self.tmp.iterdir()))}.json"
        f.write_text(json.dumps(body))
        agent = [] if actor == "orchestrator" else ["--agent-id", f"{actor}-{f.stem}", "--actor", actor]
        return self.cli("record", kind, "--file", str(f), *agent, "--rationale", rationale)

    def status(self):
        return self.cli("status")

    def ids(self, st=None):
        """title -> the latest version's id."""
        out = {}
        for h in (st or self.status())["hypotheses"]:
            out[h["title"]] = h["id"]
        return out

    # the roles -----------------------------------------------------------------------------------

    def review(self, hid, title, ids):
        """The registration reviewer: expected.yaml's call for the title, else allow / fits / none."""
        call = dict(EXPECTED["reviews"].get(title, {}))
        call["conflict_with"] = [ids[t] for t in call.get("conflict_with", [])]
        body = {"hypothesis": hid, "directive_verdict": "allow", "intent": "fits", "conflict": "none",
                "conflict_with": [], "rationale": "scripted review from expected.yaml", "strict": True, **call}
        self.record("review", body, "registration-reviewer", "the registration review")

    def lever_code(self, hid, run_dir):
        """The lever coder: each lever sets its CONFIG key (the lever's path) in train_and_eval."""
        spec = self.cli("show", hid)["spec"]
        train = run_dir / "worktree" / "train.py"
        src = train.read_text()
        if "from boautoresearch import lever" not in src:
            src = src.replace("import time\n", "import time\n\nfrom boautoresearch import lever\n", 1)
        lines = "".join(f'    cfg["{lv["path"].split(".", 1)[1]}"] = lever("{name}")\n'
                        for name, lv in spec["levers"].items())
        train.write_text(src.replace(ANCHOR, ANCHOR + lines, 1))
        out = self.cli("smoke", hid, "--rationale", "the planted mechanism, as written")
        assert out["passed"], out
        self.cli("commit-lever", hid, "--rationale", "touches: train.py; smoke passed")

    def interplay(self, m, st):
        """The interplay reviewer: flags exactly the planted partners (expected.yaml)."""
        ids, flags = self.ids(st), []
        who = m.get("removed") or m.get("newcomer")
        title = next(h["title"] for h in st["hypotheses"] if h["id"] == who)
        untested = {h["id"] for h in st["hypotheses"] if h["status"] in ("proposed", "registered")}
        for case in EXPECTED["cases"]:
            main, partner = case["hypotheses"]["main"], case["hypotheses"].get("partner")
            if "removed" in m and case.get("revived") == "removal" and title == main and ids.get(partner) in untested:
                flags.append({"partner": ids[partner], "reason": "the pair acts on one term: the EMA is the teacher",
                              "cites": [self.cli("verdict", who)["records"][-1]["id"]]})
            if "newcomer" in m and case.get("revived") == "newcomer" and title == partner and main in ids:
                removed = ids[main]
                if (v := self.cli("verdict", removed)["records"]):
                    flags.append({"partner": removed, "reason": "clipping only pays when the steps are larger",
                                  "cites": [v[-1]["id"]]})
        self.record("interplay", {**m, "flags": flags}, "interplay-reviewer", "the interplay review")

    def narrative(self, r, st):
        cites = [v["id"] for h in st["hypotheses"] for v in self.cli("verdict", h["id"])["records"]
                 if v["round"] == r]
        self.record("narrative", {"round": r, "text": "The round as its verdict records show it.", "cites": cites,
                                  "diagnostics": [], "suggestions": [], "generation": False},
                    "round-analyst", "the round's narrative")

    # the orchestrator ----------------------------------------------------------------------------

    def duties(self, run_dir, adversary):
        """Do every duty `next` lists until only round-run is left."""
        while True:
            st = self.status()
            ids = self.ids(st)
            titles = {h["id"]: h["title"] for h in st["hypotheses"]}
            if st["narrative_missing"]:
                self.narrative(st["narrative_missing"][0], st)
            elif proposed := [h for h in st["hypotheses"] if h["status"] == "proposed"]:
                if adversary and not adversary.done("register without review"):
                    adversary.register_without_review(next((h["id"] for h in proposed if h["title"] in rb.DOOMED), proposed[-1]["id"]))
                for h in proposed:  # review all, then register those left standing
                    self.review(h["id"], h["title"], ids)
                for h in self.status()["hypotheses"]:
                    if h["status"] == "proposed":
                        self.cli("register", h["id"], "--rationale", "reviewed; worth its test")
            elif st["review_missing"]:
                self.review(st["review_missing"][0], titles[st["review_missing"][0]], ids)
            elif st["interplay_missing"]:
                if adversary and not adversary.done("round-run while an interplay review is missing"):
                    adversary.between_rounds()
                self.interplay(st["interplay_missing"][0], st)
            elif st["generation"]["due"]:
                self.cli("generate", "--rationale", "; ".join(st["generation"]["due"]))
            elif uncoded := [i for i in st["schedule"]["selected"]
                             if not next(h["commit"] for h in st["hypotheses"] if h["id"] == i)]:
                self.lever_code(uncoded[0], run_dir)
            elif st["schedule"]["expected_missing"]:
                h = st["schedule"]["expected_missing"][0]
                self.record("expected", {"hypothesis": h, "verdict": "undecided", "reason": "no view yet"},
                            "orchestrator", "before the round")
            else:
                return st

    def round_run(self, run_dir, adversary):
        """round-run in the background, as the orchestrator does; the first BO round also hosts the
        adversarial actions that fire while a round runs."""
        p = subprocess.Popen([*self.bo, "round-run", "--rationale", "the next round"], cwd=self.repo,
                             stdout=subprocess.PIPE, text=True, env=self.env)
        if adversary and not adversary.done("protected write via Edit"):
            while p.poll() is None and not rb.round_running(run_dir):
                time.sleep(0.2)
            if p.poll() is None:
                adversary.during_round()
        out = json.loads(p.communicate()[0])
        if "refused" in out:
            raise RuntimeError(f"round-run refused: {out['reason']}")
        return out

    def wrapup(self, run_dir):
        """The hand-back: takeaways, the distillation spec and its review, the distilled code."""
        out = self.cli("wrapup", "--rationale", "the run has ended")
        while out.get("result") is None:
            st = self.status()
            w, duty = st["wrapup"], st["next"][0]
            cites = sorted({self.cli("verdict", h["id"])["records"][-1]["id"] for h in st["hypotheses"]
                            if h["status"] == "retained"})
            if duty.startswith("record takeaways"):
                bullets = [f"{v} retained its hypothesis." for v in cites[:5]] or ["Nothing was retained."]
                self.record("takeaways", {"takeaways": bullets, "cites": cites}, "round-analyst", "wrap-up")
            elif duty.startswith("record distill_spec"):
                content = ("Keep every retained mechanism by setting its CONFIG key in train.py to the wrap-up "
                           "config's value; drop the lever() calls, the boautoresearch import and every "
                           "hypothesis that was not retained.")
                changes = [{"commit": c["commit"], "decision": "keep", "reason": "part of the run"}
                           for c in w["changes"]]
                self.record("distill_spec", {"content": content, "cites": cites, "changes": changes},
                            "round-analyst", "wrap-up")
            elif duty.startswith("record distill_review"):
                self.record("distill_review", {"verdict": "approve", "rationale": "small, and it fits the brief"},
                            "registration-reviewer", "wrap-up")
            elif duty.startswith(("write the distilled branch", "fix the distilled branch")):
                self.distil(Path(w["distilled"]["worktree"]), w["config"], st)
                out = self.cli("wrapup", "--rationale", "commit and verify the distilled branch")
                continue
            out = self.cli("wrapup", "--rationale", "the next wrap-up step")
        return out

    def distil(self, wt, config, st):
        """The lever coder in distill mode: the retained levers' values as constants, no harness import."""
        paths = {}
        for h in st["hypotheses"]:
            spec = self.cli("show", h["id"])["spec"]
            paths.update({n: lv["path"].split(".", 1)[1] for n, lv in spec["levers"].items()})
        retained = {n for h in st["hypotheses"] if h["status"] == "retained"
                    for n in self.cli("show", h["id"])["spec"]["levers"]}
        lines = "".join(f'    cfg["{paths[n]}"] = {config[n]!r}\n' for n in sorted(retained))
        train = wt / "train.py"
        src = subprocess.run(["git", "show", "HEAD:train.py"], cwd=wt, capture_output=True, text=True,
                             check=True).stdout
        train.write_text(src.replace(ANCHOR, ANCHOR + lines, 1))

    def run(self, run_yaml: Path, adversary_log: Path | None):
        boot = [sys.executable, "-m", "boautoresearch"]
        env = {**self.env, "PYTHONPATH": str(rb.PLUGIN / "harness")}
        p = subprocess.run([*boot, "init", str(run_yaml), "--rationale", "headless start"], cwd=self.repo,
                           capture_output=True, text=True, env=env)
        out = json.loads(p.stdout)
        if p.returncode:
            raise RuntimeError(f"init refused: {out['reason']}")
        run_dir = Path(out["run_dir"])
        self.bo = [out["venv"] + "/bin/boautoresearch"]
        adversary = rb.Adversary(self.repo, run_dir, self.bo, self.env, adversary_log) if adversary_log else None
        self.round_run(run_dir, None)  # R0
        while True:
            st = self.duties(run_dir, adversary)
            if st["run_ended"]:
                break
            out = self.round_run(run_dir, adversary)
            if out.get("run_ended"):
                break
        st = self.status()
        for r in st["narrative_missing"]:
            self.narrative(r, st)
        return self.wrapup(run_dir)


def main():
    repo, run_yaml = Path(sys.argv[1]), Path(sys.argv[2])
    log = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out = Caller(repo, dict(os.environ)).run(run_yaml, log)
    print(json.dumps({"result": out.get("result")}))


if __name__ == "__main__":
    main()
