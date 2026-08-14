# zsh28code

`zsh28code` is an RLM-powered terminal coding agent for local development and
Terminal-Bench evaluation. It uses an OpenAI-compatible model endpoint, with
`poolside/laguna-s-2.1:free` as the default model through OpenRouter.

The project has two execution paths:

- **Local TUI:** a full-screen Textual terminal interface with filesystem tools,
  bash, web tools, approvals, task isolation, and recursive RLM delegation.
- **Harbor adapter:** a direct `BaseAgent` integration for Terminal-Bench. Model
  inference runs in the host Harbor process, while bash commands execute inside
  each isolated task environment.

Both paths use the same `AgentRuntime` for model turns, tool argument parsing,
repetition guards, completion validation, and runtime events. Mode-specific
adapters only decide how to call the model, where tools execute, and how logs are
persisted. This keeps Terminal-Bench behavior aligned with the local agent.

## Features

### Local terminal agent

The local agent can:

- Read existing files, create new files, and edit existing files.
- Search files with regex patterns.
- List directories and find files by glob.
- Run shell commands with process-group timeout cleanup.
- Track multi-step work with a todo list.
- Fetch and search web content when Firecrawl is configured.
- Ask for directory access before filesystem operations.
- Keep a searchable context store instead of sending the entire transcript to
  the model on every turn.
- Ask focused recursive child agents to analyze large context sections.
- Reset task-local prompt context between sequential TUI tasks.
- Display `YOU`, `AGENT`, tool output, `THINKING`, `RUNNING`, and `RESPONDING`
  states in the TUI.

### Recursive Language Model (RLM)

RLM is the context-management and delegation strategy used by zsh28code. A
traditional agent sends the entire transcript and every tool output back to the
model on every turn. That becomes expensive and unreliable when a command emits
thousands of lines. zsh28code instead stores the complete conversation in a
`ContextStore` and gives the model small navigation tools.

The model can therefore treat its conversation like a searchable external
variable:

```text
Large command output
        │
        ▼
ContextStore (full searchable history)
        │
        ├── rlm_peek   → inspect beginning/end
        ├── rlm_search → find a regex and nearby lines
        ├── rlm_slice  → read a known character range
        └── rlm_agent  → delegate a focused chunk to a child model
```

#### RLM tools

- `rlm_peek`: read the first or last part of the stored context. This is the
  cheapest way to orient on a large transcript.
- `rlm_search`: search all stored messages and tool results with a regular
  expression. This is useful for finding error messages, filenames, symbols,
  or test failures.
- `rlm_slice`: extract a precise character range after an offset has been found.
- `rlm_agent`: send only a selected context chunk and a focused question to a
  child model. The child returns an answer to the root agent.

#### RLM example

For a large build log, the root agent should not ask the model to reread the
whole output. A typical flow is:

```text
1. Run the build and store its output.
2. Use rlm_search("Traceback|error|FAILED") to locate failures.
3. Use rlm_slice(start, end) to inspect the relevant section.
4. Use rlm_agent to ask a child model to explain that section.
5. Apply and verify the fix with bash or file tools.
```

The default maximum recursion depth is `3`:

```bash
.venv/bin/zsh28code --headless --rlm-depth 3 \
  "Inspect the build logs, find the root failure, and fix it"
```

Depth limits prevent recursive calls from exploding into unbounded model usage.
RLM child agents receive focused context, not the complete root conversation.

#### RLM by execution mode

- **TUI:** full local tools plus configured async recursive agents.
- **Headless CLI:** bash plus context navigation and recursive agents.
- **Harbor:** bash executes in the task container; RLM navigation operates on
  the adapter's stored messages, and child analysis runs through the same model
  endpoint without accessing the host filesystem.

### Recursive Self-Improvement (RSI)

RSI is different from ordinary RLM task solving. RLM helps complete the current
task. RSI uses RLM to inspect and improve the agent itself across separate
evaluation cycles.

The `RSIOrchestrator` pipeline is:

```text
1. Run a task suite and collect success/reward data.
2. Store results in MemoryDB.
3. Analyze failed tasks and relevant source files with RLMSelfImprover.
4. Generate small candidate patches.
5. Apply candidates to a checkpointed working tree.
6. Run the task suite again.
7. Keep candidates only when the score improves; otherwise revert them.
8. Repeat up to the configured RSI depth.
```

`RLMSelfImprover` limits analysis to an allowlist of agent source files, extracts
function boundaries with Python's AST parser, and asks the model for minimal
unified patches. It records the issue, confidence, affected functions,
reasoning, depth, and suggested patch.

RSI state is persisted through `MemoryDB`, including:

- Task success and reward values.
- Iteration counts and elapsed time.
- Configuration hashes.
- Candidate improvements and score deltas.
- RLM trajectory references.

RSI is intentionally not automatically invoked by the TUI or Terminal-Bench.
It can modify source code and must be run as a controlled development action on
a clean branch or disposable worktree.

### Reinforcement-Learning Configuration Search (RL)

`RLOptimizer` is a gradient-free configuration search, not weight training. It
searches over agent behavior settings such as:

- System prompt variants.
- Temperature and top-p values.
- Maximum turns.
- Output-token limits.
- Model keyword arguments.

Its loop is:

```text
1. Load the best known configuration from MemoryDB.
2. Ask the model to generate a population of variants.
3. Evaluate variants against an evaluation task list.
4. Keep the highest-reward elite variants.
5. Store them in MemoryDB.
6. Ask the model to synthesize a better prompt policy.
```

The current evaluator is a lightweight heuristic over configuration features,
not a full Terminal-Bench verifier. It should therefore be treated as an
experimental search component, not as a benchmark score.

### Autoresearch

`Autoresearcher` gathers external knowledge that may improve agent behavior for
a task family. It:

1. Builds a search query from the task description and task type.
2. Uses Firecrawl web search to find relevant sources.
3. Fetches and extracts code blocks, command patterns, and relevant sentences.
4. Uses the LLM to synthesize concrete recommendations.
5. Stores recommendations in MemoryDB when explicitly applied.

Example research categories include Terminal-Bench, file manipulation, Git
operations, and system administration. Autoresearch recommendations are
advice records; they do not automatically patch source code.

### How the improvement systems relate

```text
RLM       = navigate context and delegate focused analysis during a task
RSI       = evaluate and improve the agent implementation across task suites
RL search = explore prompts and runtime configurations
Research  = gather external practices and recommendations
MemoryDB  = persist results, configurations, improvements, and rewards
```

These systems are complementary but deliberately separated. A normal task does
not silently rewrite the agent or change benchmark configuration. That separation
keeps local development predictable and Terminal-Bench results reproducible.

## Requirements

- Python `3.10` or newer.
- An OpenRouter API key for model access.
- Docker Desktop for local Harbor execution.
- Harbor for Terminal-Bench runs. It is installed by the `harbor` optional
  dependency.
- A Firecrawl API key only if web search/fetch tools are needed.

## Installation

Create a virtual environment and install the project with development tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install Harbor support as well:

```bash
python -m pip install -e ".[dev,harbor]"
```

Set the model key in the shell. Do not commit keys or paste them into logs:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

Optional configuration:

```bash
export ZSH28CODE_MODEL="poolside/laguna-s-2.1:free"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export FIRECRAWL_API_KEY="your-firecrawl-key"
```

## Running Locally

The helper script changes into the project root before launching the agent.
With no arguments it starts the TUI:

```bash
./run_zsh28code.sh
```

The same command can launch a one-shot headless task:

```bash
./run_zsh28code.sh --headless "Inspect this workspace and summarize its structure"
```

Use the installed CLI directly when preferred:

```bash
.venv/bin/zsh28code --tui
.venv/bin/zsh28code --headless "Create hello.py containing print('hello')"
```

Useful CLI options:

```text
-m, --model MODEL             Model name; defaults to Laguna free.
--reasoning-effort LEVEL      low, medium, high, xhigh, or max.
--max-tokens N                Maximum output tokens per model call.
--max-iterations N            Maximum root-agent loop iterations.
--tui                         Explicitly launch the Textual interface.
--headless                    Disable the UI for scripts and benchmarks.
--output PATH                 Write the internal trajectory JSON to PATH.
--rlm-depth N                 Maximum recursive RLM depth; default 3.
```

For file changes, the agent is instructed to use an actual `write`, `edit`, or
`bash` tool and verify the result. In the TUI, the first filesystem access to a
directory opens an approval dialog. Approval is remembered for that directory
for the current session.

### Reading and Editing Files

File work is not limited to creating new files. The local agent has three
separate file operations:

```text
read  → inspect an existing file with line numbers
write → create a file or replace its complete contents
edit  → replace one exact string inside an existing file
```

Typical workflow:

```text
1. read the relevant file
2. identify the exact change
3. edit the matching text, or write a new complete file
4. read the file again or run a test to verify the result
```

Example requests:

```text
Read src/config.py and explain how the API key is loaded.
Edit src/config.py to use a 30-second timeout instead of 60 seconds.
Create tests/test_config.py covering the new behavior.
```

The `edit` tool requires an exact, case-sensitive `old_string` and refuses
ambiguous matches. This prevents an edit from changing multiple unrelated parts
of a file. If the exact text is not found, the agent receives an error and can
read the file again before retrying.

In Harbor, the same workflow is performed with bash commands inside the isolated
`/app` task environment. For example, the model may use `cat`, `sed`, Python,
or another shell program to read and modify task files. Harbor never edits the
local checkout on the host.

## Tool Modes

### Full local tools

The TUI uses `get_default_tools()` and includes:

```text
read, write, edit, grep, find, ls, bash, todo_write,
rlm_peek, rlm_search, rlm_slice, rlm_agent
```

`web_fetch` and `web_search` are added when `FIRECRAWL_API_KEY` is available.

### Headless CLI tools

The headless path uses the smaller benchmark-friendly set:

```text
bash, rlm_peek, rlm_search, rlm_slice, rlm_agent
```

This keeps the model tool schema focused while bash remains capable of all
filesystem operations.

## Terminal-Bench

The runner evaluates against:

```text
terminal-bench/terminal-bench-2-1
```

Start Docker Desktop, export the API key, and run one smoke task first:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
ZSH28CODE_BENCH_TASKS=1 \
ZSH28CODE_BENCH_CONCURRENT=1 \
./run_terminal_bench.sh
```

Run multiple tasks:

```bash
ZSH28CODE_BENCH_TASKS=10 \
ZSH28CODE_BENCH_CONCURRENT=2 \
./run_terminal_bench.sh
```

Run the complete dataset:

```bash
ZSH28CODE_BENCH_TASKS=all \
ZSH28CODE_BENCH_CONCURRENT=2 \
./run_terminal_bench.sh
```

The runner supports these environment variables:

```text
ZSH28CODE_BENCH_TASKS                  Number of tasks or all; default 1.
ZSH28CODE_BENCH_CONCURRENT             Concurrent trials; default 1.
ZSH28CODE_BENCH_TIMEOUT_MULTIPLIER     Harbor timeout multiplier; default 0.5.
ZSH28CODE_BENCH_JOB_NAME               Custom job name.
ZSH28CODE_BENCH_DRY_RUN=1              Print the resolved Harbor command.
```

The direct Harbor adapter:

1. Checks the task environment with `pwd` and `ls -la`.
2. Sends the task and model messages through OpenRouter.
3. Executes bash calls inside the task container with a 120-second command
   timeout.
4. Stores each model response and tool result by episode.
5. Truncates oversized tool output before returning it to the model.
6. Exposes bounded RLM context tools.
7. Writes trajectory and metadata even when a trial fails.

The adapter does not install this project into each task container. This avoids
the repeated wheel/virtualenv setup that makes installed-agent adapters slow.

## Inspecting Jobs

List jobs by recency:

```bash
ls -lt jobs/
```

Inspect overall progress:

```bash
cat jobs/<job-name>/result.json
```

Inspect a trial's episodes:

```bash
find jobs/<job-name>/<trial-name>/agent -maxdepth 1 -type d -name 'episode-*' | sort
```

Inspect individual model responses and command output:

```bash
cat jobs/<job-name>/<trial-name>/agent/episode-000/response.json
cat jobs/<job-name>/<trial-name>/agent/episode-000/tool-*.txt
```

Start Harbor's local browser viewer:

```bash
.venv/bin/harbor view jobs --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in a browser.

## Architecture

```text
zsh28code/
├── __main__.py              CLI argument parsing and mode selection
├── agent.py                 Local model/context/tool adapter and RLM recursion
├── runtime.py               Shared local/TUI/Harbor orchestration state machine
├── context.py               Searchable context store and recent summaries
├── config.py                Runtime configuration and environment variables
├── llm.py                   Async OpenAI/OpenRouter wrapper
├── tools/
│   ├── base.py              Tool interface and OpenAI schemas
│   ├── file.py              read/write/edit/grep/find/ls
│   ├── shell.py             Process-group-safe bash execution
│   ├── web.py               todo, web fetch/search, legacy task tool
│   └── rlm.py               peek/search/slice/recursive-agent tools
├── ui/
│   ├── app.py               Textual TUI and access approval dialog
│   ├── __init__.py          Headless and interactive entry points
│   ├── headless.py          Non-TUI rendering helpers
│   ├── banner.py            Legacy Rich banner/status renderers
│   └── theme.py             Rich color theme definitions
├── benchmark/
│   ├── harbor_agent.py      Direct BaseAgent Terminal-Bench adapter
│   └── trajectory.py        ATIF trajectory conversion
├── self_improve/
│   ├── memory.py            SQLite memory database
│   ├── rlm.py               Source analysis and patch proposals
│   ├── rsi.py               Score-gated recursive improvement loop
│   ├── rl.py                Prompt/configuration search
│   └── research.py          Web research recommendations
└── utils/                   Retry, token, and subprocess helpers
```

## Testing and Quality Checks

Run the test suite:

```bash
.venv/bin/pytest tests/ -q
```

Compile the main runtime modules:

```bash
.venv/bin/python -m py_compile \
  zsh28code/agent.py \
  zsh28code/ui/app.py \
  zsh28code/benchmark/harbor_agent.py
```

Run Ruff on the source and tests:

```bash
.venv/bin/ruff check zsh28code tests
```

## Security and Operational Notes

- Never commit `OPENROUTER_API_KEY` or `FIRECRAWL_API_KEY`.
- Revoke a key immediately if it is pasted into chat, screenshots, or logs.
- Local TUI filesystem access is approval-gated by directory.
- Headless and Harbor modes automatically execute task commands because they are
  intended for unattended evaluation.
- Harbor bash commands run in isolated task environments, not in the local
  project directory.
- RSI/autoresearch can modify source files and should be run on a clean branch
  or disposable worktree.
- `jobs/`, caches, virtual environments, and local agent state are ignored by
  Git.

## Current Limitations

- Laguna's free endpoint can be rate-limited or slow, especially with concurrent
  Terminal-Bench trials.
- A benchmark result is not valid until the verifier completes; an agent saying
  it finished is not sufficient.
- The self-improvement modules are implemented as opt-in library workflows, not
  an automatic background process.
- Web tools require Firecrawl and are not part of the minimal Harbor tool set.

## License

MIT
