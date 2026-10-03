# BOAR proposals

`boar propose <file.json>` adds hypotheses as pending proposals; the reviewer then accepts or rejects each one. The harness allows it in setup (S3) and in R4. Keep proposal files in `<run dir>/work/` (for example `work/proposals-2.json`), out of the target repo's tracked tree.

## File format

A JSON list of hypotheses. The harness assigns each one its `id` (H1, H2, …), `status` and round, so leave those out.

```json
[
  {
    "statement": "Parallelising the parse stage cuts wall time",
    "mechanism": "parse is 60% of the profile and has no shared state",
    "source": "research",
    "citations": ["work/profile.txt", "research.md#parse-stage"],
    "levers": [{"name": "parse_workers", "type": "int", "low": 1, "high": 8, "default": 1}],
    "supersedes": null
  }
]
```

| Field | Meaning |
|---|---|
| `statement` | The testable claim: mechanism X moves the metric. |
| `mechanism` | Why it should work, in terms of the target system. |
| `source` | `user` (from the spec's User ideas), `research` (from S2 or targeted research), or `synthesis` (from round results). |
| `citations` | Where the evidence lives: research.md sections, files in `work/`, URLs, round summaries, trial ids. |
| `levers` | One or more levers (below). |
| `supersedes` | `null`, or the id of the active hypothesis this one replaces (R4 only; always `null` in setup). |

## Levers

| Type | Keys |
|---|---|
| `bool` | `name`, `type`, `default` |
| `int`, `float` | `name`, `type`, `low`, `high`, `default` (inside `[low, high]`, so the range must include today's value), and optionally `log` (false unless given; true when the range spans orders of magnitude, which needs `low > 0`) |
| `categorical` | `name`, `type`, `choices`, `default` (one of the choices) |

## Rules

- **One mechanism.** A hypothesis tests a single mechanism; its levers turn that mechanism on or scale it. Two independent ideas are two hypotheses, so BO and R3 can judge each on its own.
- **The default reproduces the baseline.** Each lever's `default` is exactly today's behaviour: `false` for a mechanism that is off today, today's value for a numeric setting, today's choice for a categorical one. The baseline trial (every lever at default) must measure the unmodified code.
- **Realistic ranges.** Ranges stay within what the target population and the spec's deployment allow (worker counts up to the cores the deployment has, cache sizes inside the RAM guard). A range that only pays off on the eval's inputs is a cheat.
- **Unique lever names.** Names are unique across the whole run, including rejected, withdrawn and removed hypotheses, and `boar propose` refuses a reused one. Prefix them with the mechanism (`parse_workers`, `parse_chunk_kb`), and give a re-proposal fresh names.
- **Retuning uses `supersedes`.** To widen or shift a range, change a type, or rework a lever, propose a new hypothesis with `"supersedes": "<id>"` and fresh lever names. When it activates at round close, the old hypothesis is removed and its levers stay pinned at their defaults; in R1 the new levers take over the mechanism.
- **Budget.** At most `max_active` hypotheses (default 6) are active when a round starts, sharing `trials_per_round` trials (default 6). A few strong hypotheses beat many weak ones.

`boar propose` refuses the whole file on any error and lists every error: fix them all and resubmit. On success it prints the assigned ids.

`boar propose --none --reason "…"` (R4 only) records that nothing new is worth testing this round.
