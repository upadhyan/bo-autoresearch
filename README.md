# bo-autoresearch

## Check that it works: the dogfood benchmark

The dogfood benchmark runs the plugin end to end on a toy training script whose effects are planted
(`benchmarks/dogfood/`), then scores each run against the planted truth.

You need Python 3.10+, git, and [Claude Code](https://claude.com/claude-code)
logged in (`claude auth login`; a subscription works). [uv](https://docs.astral.sh/uv/) is optional but makes
each run's venv install (torch included) much faster. The `python` below must be 3.10 or newer: macOS's
built-in `python3` is 3.9, so make an environment first, e.g. `uv venv --python 3.12 .venv && source .venv/bin/activate`
(then `uv pip install pyyaml` in place of the pip line).

```bash
git clone https://github.com/upadhyan/bo-autoresearch && cd bo-autoresearch
python -m pip install pyyaml
python benchmarks/dogfood/run_benchmark.py --caller scripted --jobs 2
python benchmarks/dogfood/run_benchmark.py --caller claude --jobs 2
```

- `--caller scripted` checks the harness alone: a script plays every agent, so no Claude usage.
- `--caller claude` checks the harness and the skills: real headless `claude -p "/boautoresearch:start"`
  runs, 3 seeds × 2 noise levels scripted plus 1 free-generation run. Budget roughly 35 minutes per
  scripted run and 1.5–2 hours for the free run, all against your Claude usage. `--jobs` sets how many
  run at once; add `--no-free` to skip the free run, or `--seeds 1 --sigmas 0.025 --no-free` for one
  quick run.

It passes when every hard invariant holds in every run and each planted verdict is right in at least 5
of the 6 scripted runs. Each run's `check.py` report lands under `--work` (default: `dogfood-runs/` in the
system temp dir; it must be outside the repository, so no run can browse to the planted truth).

On GitHub, the `dogfood` workflow does the same on demand (Actions > dogfood > Run workflow). It needs a
`CLAUDE_CODE_OAUTH_TOKEN` repository secret from `claude setup-token`.
