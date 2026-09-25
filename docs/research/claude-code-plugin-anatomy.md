# Research: Claude Code plugin anatomy

Resolves issue #4. Sources are the official Claude Code docs at `code.claude.com/docs/en/*` (fetched 2026-09-24, around Claude Code v2.1.28x) and the official example plugins in `anthropics/claude-code/plugins`. Docs abbreviated below as `D/<page>` = `https://code.claude.com/docs/en/<page>`.

## Key facts for Agent roster / Code-change management / enforcement hooks

1. **Layout**: the only manifest is `.claude-plugin/plugin.json`, and only `name` is required. Components live at the plugin root: `skills/<name>/SKILL.md`, `agents/*.md`, `hooks/hooks.json`, `workflows/*.js`, `bin/` (put on the Bash tool's PATH), `.mcp.json`, and `settings.json`. Everything is namespaced as `<plugin>:<name>`. [D/plugins/manifest-reference#standard-layout](https://code.claude.com/docs/en/plugins/manifest-reference#standard-layout)
2. **Per-agent model and tools are supported in plugin agents.** Supported fields: `model` (`haiku`/`sonnet`/`opus`/full ID/`inherit`), `effort`, `maxTurns`, `tools`, `disallowedTools`, `skills`, `memory`, `background`, `isolation: worktree`, `color`. [D/plugins/components#frontmatter-fields-in-plugin-agents](https://code.claude.com/docs/en/plugins/components#frontmatter-fields-in-plugin-agents)
3. **Plugin agents ignore `hooks`, `mcpServers`, `permissionMode` and `initialPrompt`.** Enforcement has to live in the plugin-level `hooks/hooks.json`, where a hook can scope itself to one agent with the `agent_type` / `agent_id` input fields or a `SubagentStart`/`SubagentStop` matcher such as `^bo-autoresearch:hypothesis-judge$`. [same](https://code.claude.com/docs/en/plugins/components#frontmatter-fields-in-plugin-agents), [D/hooks#common-input-fields](https://code.claude.com/docs/en/hooks#common-input-fields)
4. **Model precedence**: the per-call `model` parameter wins, then frontmatter `model`, then the `CLAUDE_CODE_SUBAGENT_MODEL` env var, then the main model. An org's `availableModels` allowlist can substitute a different model. A user-set `CLAUDE_CODE_SUBAGENT_MODEL_FORCE` overrides every agent's `model`. So "cheap model per agent" is the default behaviour, but it isn't guaranteed. [D/sub-agents#choose-a-model](https://code.claude.com/docs/en/sub-agents#choose-a-model)
5. **Structured output**: plain subagents (Agent tool) have no schema option and return free text. The only schema-enforced path is a **workflow** `agent(prompt, {schema, agentType, model, isolation})` call, which validates the output and retries up to 5 times (`MAX_STRUCTURED_OUTPUT_RETRIES`). Plugins can ship workflows in `workflows/`. [D/workflows#what-the-saved-script-looks-like](https://code.claude.com/docs/en/workflows#what-the-saved-script-looks-like)
6. **Fallback schema gate for plain subagents**: a `SubagentStop` hook receives `last_assistant_message`. It can validate it and return `decision:"block"` with a `reason`, which keeps the subagent running with the reason as its next instruction (a loop cap applies). Caveat: in auto mode the report goes through the `SubagentHandback` tool instead, so the hook has to read that tool's `tool_input.message`. [D/hooks#subagentstop](https://code.claude.com/docs/en/hooks#subagentstop)
7. **Parallel and background**: subagents run in the background by default in interactive sessions (fork mode). Up to 20 run concurrently (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`) and they can nest 3 levels deep. Background agents get a reduced built-in tool set, which still includes Read/Edit/Write/Bash/Grep/Glob/Skill. Results arrive as later completion notifications. Workflows run up to 16 agents at once. [D/sub-agents#run-subagents-in-foreground-or-background](https://code.claude.com/docs/en/sub-agents#run-subagents-in-foreground-or-background), [D/sub-agents#concurrent-subagent-limit](https://code.claude.com/docs/en/sub-agents#concurrent-subagent-limit)
8. **Worktrees**: `isolation: worktree` (in frontmatter, on the Agent call, or in the workflow `agent()` options) gives the agent a temporary worktree under `.claude/worktrees/`. It branches from the **default branch** unless `worktree.baseRef: "head"` is set, and it's auto-removed if unchanged. Edits and git commands aimed at the main checkout are blocked. A `WorktreeCreate` hook can replace creation entirely. [D/worktrees#isolate-subagents-with-worktrees](https://code.claude.com/docs/en/worktrees#isolate-subagents-with-worktrees)
9. **Blocking hook events**: `PreToolUse`, `UserPromptSubmit`, `UserPromptExpansion`, `Stop`, `SubagentStop`, `PostToolBatch`, `TaskCreated`, `TaskCompleted`, `TeammateIdle`, `ConfigChange`, `PreCompact`, `PreModelSwitch`, `Elicitation`/`ElicitationResult`, and `WorktreeCreate`/`WorktreeRemove`. `PostToolUse` and `SubagentStart` **cannot** block. Only **exit 2** (or a JSON decision) blocks. Exit 1 and timeouts fail open. [D/hooks#exit-code-2-behavior-per-event](https://code.claude.com/docs/en/hooks#exit-code-2-behavior-per-event)
10. **Hook input**: JSON on stdin with `session_id`, `transcript_path`, `cwd`, `permission_mode`, `hook_event_name`, plus `agent_id`/`agent_type` inside subagents, plus event fields (for example `tool_name`/`tool_input` on PreToolUse). Plugin hooks are active for the whole session once the plugin loads, so their matchers must be narrow. [D/hooks#common-input-fields](https://code.claude.com/docs/en/hooks#common-input-fields), [D/plugins/components#when-plugin-hooks-fire](https://code.claude.com/docs/en/plugins/components#when-plugin-hooks-fire)
11. **Bundled files**: `${CLAUDE_PLUGIN_ROOT}` (versioned, read-only in practice) and `${CLAUDE_PLUGIN_DATA}` (persistent across updates) are substituted inline in skill and agent Markdown and in hook commands. They are **not** set in the Bash tool's environment, so the skill text must carry the expanded path. Plugin state must not be written under `CLAUDE_PLUGIN_ROOT`. [D/plugins/manifest-reference#environment-variables](https://code.claude.com/docs/en/plugins/manifest-reference#environment-variables)
12. **Python has no automatic install.** Only Node `package.json` with a lockfile is auto-installed. The docs send Python dependencies to a hook-driven install into `${CLAUDE_PLUGIN_DATA}`. Anthropic's own `hookify` plugin runs `python3 ${CLAUDE_PLUGIN_ROOT}/...` and uses only the standard library. Runner scripts import the user's ML stack, so the Optuna package realistically has to be installed into the **user's project environment** (see §6). [D/plugins/loading#when-the-dependency-install-fails-or-is-skipped](https://code.claude.com/docs/en/plugins/loading#when-the-dependency-install-fails-or-is-skipped)

---

## 1. Plugin directory layout and manifest

- The manifest is `.claude-plugin/plugin.json` and is optional. Without it, components load from the default locations and the name comes from the marketplace entry or directory. `name` (kebab-case) is the only required key. Every other file goes at the plugin root, **not** inside `.claude-plugin/`. [D/plugins/manifest-reference#manifest-file](https://code.claude.com/docs/en/plugins/manifest-reference#manifest-file)
- Default locations: `skills/` (one `<name>/SKILL.md` each), `commands/` (legacy flat `.md`, and "prefer `skills/` for new plugins"), `agents/` (recursive, with subfolders joining the name as `plugin:sub:name`), `hooks/hooks.json`, `.mcp.json`, `.lsp.json`, `output-styles/`, `workflows/`, `themes/`, `monitors/monitors.json`, `bin/` (on the Bash tool PATH while enabled), and `settings.json` (only `agent` and `subagentStatusLine` take effect). [D/plugins/manifest-reference#standard-layout](https://code.claude.com/docs/en/plugins/manifest-reference#standard-layout), [D/plugins/components#organize-agents-in-subfolders](https://code.claude.com/docs/en/plugins/components#organize-agents-in-subfolders)
- Manifest keys `commands`, `agents`, `outputStyles` and `workflows` **replace** their default scan. `skills` **adds** to it. `hooks`, `mcpServers` and `lspServers` **merge**. Paths must start with `./` and resolve inside the plugin root. `..` and symlinks out of the plugin are rejected. [D/plugins/manifest-reference#path-rules](https://code.claude.com/docs/en/plugins/manifest-reference#path-rules)
- A marketplace-installed plugin is **copied** to `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/`, and files outside the plugin directory aren't copied. A plugin loaded with `--plugin-dir`, or from a local-directory marketplace, loads in place. Old versions are cleaned up 14 days after an update. [D/plugins/loading#in-place-and-copied-plugins](https://code.claude.com/docs/en/plugins/loading#in-place-and-copied-plugins)
- A `CLAUDE.md` at the plugin root is **not** loaded. Instructions have to be put in a skill. [D/plugins/manifest-reference#standard-layout](https://code.claude.com/docs/en/plugins/manifest-reference#standard-layout)
- `userConfig` prompts the user for values when the plugin is enabled. Values are referenced as `${user_config.KEY}` in skill/agent content and exec-form hook args, or as `CLAUDE_PLUGIN_OPTION_<KEY>` in hook env. This fits settings like "checkpoint mode on/off". [D/plugins/manifest-reference#user-configuration](https://code.claude.com/docs/en/plugins/manifest-reference#user-configuration)
- `dependencies` covers **other plugins** only, not packages. [D/plugins/manifest-reference#dependencies](https://code.claude.com/docs/en/plugins/manifest-reference#dependencies)
- Distribution: `.claude-plugin/marketplace.json` (`name`, `owner`, `plugins`) with a relative-path source such as `"./plugin"` works for a plugin living in this same repo. [D/plugins/marketplace-reference#relative-path-plugin-source](https://code.claude.com/docs/en/plugins/marketplace-reference#relative-path-plugin-source)
- Validate with `claude plugin validate ./plugin`, using `--strict` in CI. [D/plugins/manifest-reference#validate-the-manifest](https://code.claude.com/docs/en/plugins/manifest-reference#validate-the-manifest)

## 2. Skills and slash commands

- A plugin skill `skills/review/SKILL.md` runs as `/<plugin>:review`. Frontmatter `name` replaces only the last segment. Commands are "the older format" and skills supersede them. [D/plugins/components#skills](https://code.claude.com/docs/en/plugins/components#skills), [D/plugins/components#commands](https://code.claude.com/docs/en/plugins/components#commands)
- Relevant skill frontmatter [D/skills#frontmatter-reference](https://code.claude.com/docs/en/skills#frontmatter-reference):
  - `description` / `when_to_use`: capped at 1,536 characters combined.
  - `disable-model-invocation`: user-only.
  - `user-invocable: false`: Claude-only.
  - `allowed-tools`: a pre-approval that lasts for the invoking turn.
  - `disallowed-tools`: for example, removing `AskUserQuestion` for an autonomous loop.
  - `model` / `effort`: for the rest of the turn.
  - `context: fork` + `agent`: run the skill as a subagent.
  - `hooks`: registered on invocation and kept for the session, with `once: true` available.
- `context: fork` runs in the background by default, and `background: false` waits for it. A backgrounded forked skill's edits bypass checkpoints, so `/rewind` can't undo them and git has to be used. [D/skills#run-skills-in-a-subagent](https://code.claude.com/docs/en/skills#run-skills-in-a-subagent)
- A skill's supporting files (`reference.md`, `scripts/*.py`) go beside `SKILL.md` and are linked from it. The docs advise keeping `SKILL.md` under 500 lines. [D/skills#add-supporting-files](https://code.claude.com/docs/en/skills#add-supporting-files)

## 3. Referencing bundled files

- Skill substitutions: `${CLAUDE_SKILL_DIR}` (this skill's directory), `${CLAUDE_PLUGIN_ROOT}` (plugin install directory, plugin skills only), `${CLAUDE_PLUGIN_DATA}` (persistent directory), `${CLAUDE_PROJECT_DIR}`, `${CLAUDE_SESSION_ID}`, `$ARGUMENTS`/`$0`. They are substituted in the body **and** in `allowed-tools` Bash rules, so `allowed-tools: Bash(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/x.py *)` pre-approves exactly the command the body tells Claude to run. [D/skills#available-string-substitutions](https://code.claude.com/docs/en/skills#available-string-substitutions)
- Where each variable resolves: hooks get the variables both inline and in the process env. Skill, command and agent content is substituted inline only. The variables **aren't present in the Bash tool's environment**, in the main session or in subagents. [D/plugins/manifest-reference#where-each-variable-resolves](https://code.claude.com/docs/en/plugins/manifest-reference#where-each-variable-resolves)
- `CLAUDE_PLUGIN_ROOT` changes with every version, so durable state goes in `CLAUDE_PLUGIN_DATA` (`~/.claude/plugins/data/<id>/`, deleted on the last uninstall unless `--keep-data` is passed). Per-project research state such as the Optuna SQLite DB and the experiment log belongs in the user's project, not in either of these. [D/plugins/manifest-reference#environment-variables](https://code.claude.com/docs/en/plugins/manifest-reference#environment-variables)
- In hook commands, `"${CLAUDE_PLUGIN_ROOT}"` should be quoted in shell form, or exec form with `args` used instead. [D/plugins/manifest-reference#quoting-and-path-separators](https://code.claude.com/docs/en/plugins/manifest-reference#quoting-and-path-separators)

## 4. Subagents

**Frontmatter.** [D/sub-agents#supported-frontmatter-fields](https://code.claude.com/docs/en/sub-agents#supported-frontmatter-fields)
- `name` and `description` are required.
- `tools` is an allowlist and `disallowedTools` a denylist, applied first.
- `model`, `effort` and `maxTurns` (which returns output marked as partial) are available.
- `skills` preloads full skill content.
- `memory` takes `user`, `project` or `local`.
- `background: true` forces background execution.
- `omitClaudeMd` requires v2.1.271+.
- `isolation: worktree`.
- `experimental.cacheTtl`.

**Plugin-agent restrictions.** Plugin agents drop `permissionMode`, `hooks`, `mcpServers` and `initialPrompt`. [D/plugins/components#frontmatter-fields-in-plugin-agents](https://code.claude.com/docs/en/plugins/components#frontmatter-fields-in-plugin-agents)

**Startup context.** A subagent gets only its own body as the system prompt, plus environment details and CLAUDE.md, and starts with a fresh context. It doesn't see the conversation. Everything it needs must be in the delegation prompt, and any file paths must be passed explicitly. [D/sub-agents#write-subagent-files](https://code.claude.com/docs/en/sub-agents#write-subagent-files), [D/sub-agents#what-loads-at-startup](https://code.claude.com/docs/en/sub-agents#what-loads-at-startup)

**Official example.** `feature-dev/agents/code-architect.md` uses `tools: Glob, Grep, LS, Read, ...` and `model: sonnet` ([github.com/anthropics/claude-code/.../feature-dev/agents](https://github.com/anthropics/claude-code/tree/main/plugins/feature-dev/agents)).

**Background vs foreground.** [D/sub-agents#run-subagents-in-foreground-or-background](https://code.claude.com/docs/en/sub-agents#run-subagents-in-foreground-or-background)
- In an interactive session with fork mode on (the default), every spawned subagent runs in the background and Claude can't ask for the foreground.
- With `-p` or the SDK, fork mode is off. Claude then picks, and `background: true` forces the background.
- `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` forces the foreground.
- Background permission prompts surface in the main session.

**Background tool set.** Background subagents keep only these built-in tools: Read, Grep, Glob, LSP, Bash, PowerShell, Edit, Write, NotebookEdit, WebFetch, WebSearch, TodoWrite, Skill, ToolSearch, EnterWorktree/ExitWorktree, Monitor, TaskStop, SendMessage, Artifact. They also keep all MCP tools. `AskUserQuestion` is never available to subagents. [D/sub-agents#available-tools](https://code.claude.com/docs/en/sub-agents#available-tools)

**Parallelism.**
- Multiple Agent calls run concurrently, with a concurrency cap of 20 (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`).
- Nesting depth defaults to 3 (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`).
- A plain subagent can spawn its own subagents only if `Agent` is in its `tools`.

Sources: [D/sub-agents#concurrent-subagent-limit](https://code.claude.com/docs/en/sub-agents#concurrent-subagent-limit), [D/sub-agents#let-subagents-spawn-their-own-subagents](https://code.claude.com/docs/en/sub-agents#let-subagents-spawn-their-own-subagents)

**Structured output.** There is no frontmatter or Agent-tool schema field. The Agent tool input is `prompt`, `description`, `subagent_type` and `model` ([D/hooks#agent](https://code.claude.com/docs/en/hooks#agent)). There are four ways to get structured output:
1. **Workflow `agent()` with `schema`**: returns JSON matching the schema, validated up to 5 attempts. It's contradiction-checked before start. [D/workflows#what-the-saved-script-looks-like](https://code.claude.com/docs/en/workflows#what-the-saved-script-looks-like)
   - The bundled `/workflow-authoring` reference ([D/workflows#edit-a-saved-script](https://code.claude.com/docs/en/workflows#edit-a-saved-script)) documents `agent()` options `{label, phase, schema, model, effort, isolation: 'worktree', agentType}`. `agentType` accepts a custom (for example plugin) subagent and "composes with schema". It also documents `parallel()` (a barrier) and `pipeline()` (no barrier).
   - The public page confirms `schema` and `label`. The other options were confirmed only in the bundled skill text.
2. **SubagentStop gate**: validate `last_assistant_message` and block with a `reason` to force a retry. [D/hooks#subagentstop](https://code.claude.com/docs/en/hooks#subagentstop)
3. **Headless**: `claude -p --output-format json --json-schema '<schema>'` puts the result in `.structured_output`. [D/headless#get-structured-output](https://code.claude.com/docs/en/headless#get-structured-output)
4. **Agent SDK structured outputs**: [D/agent-sdk/structured-outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs)

**Workflow limits.** [D/workflows#behavior-and-limits](https://code.claude.com/docs/en/workflows#behavior-and-limits)
- No mid-run user input.
- No filesystem or shell access from the script itself.
- No `import()`.
- `Date.now()` and `Math.random()` throw.
- 16 concurrent agents and 1,000 agents per run.
- Resumable in the same session.
- Available on paid plans. It has to be enabled on Pro.

Plugin workflows run as `/<plugin>:<meta.name>` and accept an `args` global. [D/workflows#distribute-a-workflow-in-a-plugin](https://code.claude.com/docs/en/workflows#distribute-a-workflow-in-a-plugin)

**Result telemetry.** For a foreground Agent call, `PostToolUse` on `Agent` gets `tool_response.content`, `resolvedModel`, `totalDurationMs` and similar fields. A background call returns only `status:"async_launched"`. [D/hooks#agent](https://code.claude.com/docs/en/hooks#agent)

## 5. Hooks

**Events.** `SessionStart`, `Setup`, `UserPromptSubmit`, `UserPromptExpansion`, `PreToolUse`, `PermissionRequest`, `PermissionDenied`, `PostToolUse`, `PostToolUseFailure`, `PostToolBatch`, `Notification`, `MessageDisplay`, `SubagentStart`, `SubagentStop`, `TaskCreated`, `TaskCompleted`, `Stop`, `StopFailure`, `TeammateIdle`, `InstructionsLoaded`, `ConfigChange`, `CwdChanged`, `DirectoryAdded`, `FileChanged`, `WorktreeCreate`, `WorktreeRemove`, `PreCompact`, `PostCompact`, `PreModelSwitch`, `PostModelSwitch`, `Elicitation`, `ElicitationResult`, `SessionEnd`. [D/hooks#hook-lifecycle](https://code.claude.com/docs/en/hooks#hook-lifecycle)

**Which events can block.** See the per-event table at [D/hooks#exit-code-2-behavior-per-event](https://code.claude.com/docs/en/hooks#exit-code-2-behavior-per-event).
- Blocking: `PreToolUse` (blocks the call), `UserPromptSubmit`, `UserPromptExpansion`, `Stop`/`SubagentStop` (forces the agent to continue), `PostToolBatch` (stops the loop), `TaskCreated`/`TaskCompleted`, `TeammateIdle`, `ConfigChange`, `PreCompact`, `PreModelSwitch`, `Elicitation`/`ElicitationResult`, and `WorktreeCreate`/`WorktreeRemove` (any non-zero exit).
- Not blocking: `PostToolUse` and `PostToolUseFailure` (stderr is shown to Claude after the fact), `SubagentStart` and `SessionStart` (context only), `PermissionRequest` (deny through JSON instead), and all the notification-style events.

**Blocking semantics.** [D/hooks#exit-code-output](https://code.claude.com/docs/en/hooks#exit-code-output), [D/hooks#timeouts](https://code.claude.com/docs/en/hooks#timeouts)
- Exit 2 blocks, with stderr as the reason. It can't be overridden by JSON.
- Exit 0 with JSON gives finer control: `hookSpecificOutput.permissionDecision: allow|deny|ask|defer` plus `updatedInput` on PreToolUse, or `decision:"block"` plus `reason` on the others.
- Exit 1, a missing script, or a timeout is **non-blocking**, so policy hooks fail open.

**Input.** [D/hooks#common-input-fields](https://code.claude.com/docs/en/hooks#common-input-fields), [D/hooks#pretooluse-input](https://code.claude.com/docs/en/hooks#pretooluse-input)
- Common fields: `session_id`, `prompt_id`, `transcript_path`, `cwd`, `scratchpad_dir`, `permission_mode`, `effort`, `hook_event_name`. Inside subagents, `agent_id` and `agent_type` are added.
- PreToolUse adds `tool_name`, `tool_input` (for example `command` for Bash, or `file_path`/`content` for Write) and `tool_use_id`.
- SubagentStop adds `agent_transcript_path`, `last_assistant_message` and `stop_hook_active`.

**Handler types.** `command`, `http`, `mcp_tool`, `prompt` (a single-turn LLM yes/no) and `agent` (experimental, uses tools to verify). All matching handlers run in parallel. The `if` field (for example `"Write(.bo/**)"`) filters tool events, but it's best-effort. [D/hooks#hook-handler-fields](https://code.claude.com/docs/en/hooks#hook-handler-fields)

**Official example.** `hookify` ships `hooks/hooks.json` with PreToolUse, PostToolUse, Stop and UserPromptSubmit entries, each running `python3 ${CLAUDE_PLUGIN_ROOT}/hooks/<event>.py` with `timeout: 10`. Each script adds `CLAUDE_PLUGIN_ROOT` to `sys.path` to import its own package, and fails open on import error. [github.com/anthropics/claude-code/plugins/hookify/hooks](https://github.com/anthropics/claude-code/tree/main/plugins/hookify/hooks)

**Implications for enforcement hooks (inference).** A "no hypothesis without reject conditions" invariant can be a `PreToolUse` hook matched on `Write|Edit` with `if: "Write(<hypotheses dir>/**)"`. It would parse `tool_input.content` and exit 2 when the conditions are missing. A Bash `echo > file` bypasses that matcher, so the hard gate should also live in the Python package's own "register hypothesis" API, with the hook as defence in depth. "Trial finished without artifact" fits a `SubagentStop` or `Stop` hook that blocks with a reason. Hooks should be written to fail closed deliberately, using exit 2 on internal error, because the default is to fail open.

## 6. Shipping or depending on a Python package

What the docs provide:
- Automatic install only for Node: `package.json` plus a bun or npm lockfile, run with `--ignore-scripts` and a 60 s timeout. [D/plugins/loading#nodejs-package-dependencies](https://code.claude.com/docs/en/plugins/loading#nodejs-package-dependencies)
- For everything else, including Python, the docs say to install from a hook into `${CLAUDE_PLUGIN_DATA}`. [D/plugins/loading#when-the-dependency-install-fails-or-is-skipped](https://code.claude.com/docs/en/plugins/loading#when-the-dependency-install-fails-or-is-skipped)
- Their reference pattern is a `SessionStart` hook that diffs the manifest against a copy in `CLAUDE_PLUGIN_DATA` and reinstalls on change. [D/plugins/components#install-dependencies-into-the-data-directory](https://code.claude.com/docs/en/plugins/components#install-dependencies-into-the-data-directory)
- `bin/` executables are on the Bash tool's PATH. [D/plugins/manifest-reference#standard-layout](https://code.claude.com/docs/en/plugins/manifest-reference#standard-layout)

Options for the runner-harness package (my assessment, not from the docs):

| Option | How | Fit |
|---|---|---|
| A. Publish to PyPI / git URL, install into the user's project env | The skill tells Claude to run `pip install bo-autoresearch` or `uv add ...` in the project venv, checking first with `python -c "import bo_autoresearch"` | **Best.** Runner scripts must import the user's training code and deps (torch and so on), so they have to run in the user's interpreter. Versions can be pinned to the plugin version. |
| B. Vendor the source in the plugin and install it editable/local | `pip install "${CLAUDE_PLUGIN_ROOT}/python"`, with the path substituted in the skill body | Works offline and needs no PyPI. The path goes stale after a plugin update because `CLAUDE_PLUGIN_ROOT` is versioned, so a non-editable install is needed. |
| C. SessionStart hook creates a venv in `${CLAUDE_PLUGIN_DATA}` | The documented pattern | Good for **hook scripts'** own deps. Wrong for runners, since that venv lacks the user's ML stack. |
| D. `PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/python` | Inline in commands | Fragile, and the variable isn't in the Bash env. Avoid. |

Recommendation: A, with B as a fallback. Hook scripts should stay standard-library-only (like hookify) so they need no install.

## 7. Worktrees and code-change management

**Isolation rules.** [D/worktrees#how-claude-code-enforces-isolation](https://code.claude.com/docs/en/worktrees#how-claude-code-enforces-isolation)
- `isolation: worktree` in subagent frontmatter creates a temporary worktree under `.claude/worktrees/`. It's removed if there are no changes. A worktree with changes stays until a periodic sweep removes it, and the sweep keeps worktrees that hold uncommitted or unpushed work.
- Claude Code holds a git worktree lock while the agent runs.
- It blocks Edit/Write into the main checkout, Bash with a cwd in the main checkout, `git -C`/`GIT_DIR` redirects, and unverifiable command shapes.

**Base branch.** Worktrees branch from the remote **default branch** (`"fresh"`), **not** the parent's HEAD. `worktree.baseRef: "head"` branches from the current HEAD instead. A named branch can't be set. [D/worktrees#choose-the-base-branch](https://code.claude.com/docs/en/worktrees#choose-the-base-branch)
- For lever code that must build on the research branch, `baseRef: "head"` is needed, or the orchestrator should create its own branch or worktree with git and use `EnterWorktree` with a `path`. [D/tools-reference](https://code.claude.com/docs/en/tools-reference)

**Untracked files.** Gitignored files such as `.env` or data paths aren't in the worktree unless they're listed in `.worktreeinclude`. [D/worktrees#copy-gitignored-files-into-worktrees](https://code.claude.com/docs/en/worktrees#copy-gitignored-files-into-worktrees)

**Custom creation.** A `WorktreeCreate` hook can replace creation, for example to use a deterministic branch name per round or hypothesis, by printing the worktree path. [D/hooks#worktreecreate](https://code.claude.com/docs/en/hooks#worktreecreate)

**Checkpoints.** Edits by background forked skills and subagents bypass `/rewind` checkpoints, so git is the reversibility mechanism. [D/skills#run-skills-in-a-subagent](https://code.claude.com/docs/en/skills#run-skills-in-a-subagent)

## Open points for downstream tickets

- Whether to run the round loop as a **workflow**, which gives schema-enforced verdicts, deterministic fan-out and resumability but no mid-run user input, or as **orchestrator-skill-driven Agent calls**, which are free-text but interactive. A hybrid is possible: the skill drives rounds, and each round's judge or generator fan-out is a plugin workflow invoked with `args`. Checkpoint mode maps naturally to "one workflow per round".
- `SubagentHandback` in auto mode changes where a subagent's report appears. Any SubagentStop-based validation has to handle both paths.
