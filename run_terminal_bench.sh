#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Error: .venv is missing. Create it with: python3 -m venv .venv" >&2
  exit 1
fi

N_TASKS="${ZSH28CODE_BENCH_TASKS:-1}"
CONCURRENT="${ZSH28CODE_BENCH_CONCURRENT:-1}"
JOB_NAME="${ZSH28CODE_BENCH_JOB_NAME:-zsh28code-tbench-2-1-$(date +%Y%m%d-%H%M%S)}"

if [[ "$N_TASKS" != "all" && ! "$N_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: ZSH28CODE_BENCH_TASKS must be a positive integer or all." >&2
  exit 1
fi
if [[ ! "$CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: ZSH28CODE_BENCH_CONCURRENT must be a positive integer." >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  read -r -s -p "OpenRouter API Key: " OPENROUTER_API_KEY
  echo
  export OPENROUTER_API_KEY
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Error: OPENROUTER_API_KEY is required." >&2
  exit 1
fi

HARBOR_ARGS=(
  run
  --dataset terminal-bench/terminal-bench-2-1
  --agent zsh28code.benchmark.harbor_agent:Zsh28Code
  --model openrouter/poolside/laguna-s-2.1:free
  --n-concurrent "$CONCURRENT"
  --n-attempts 1
  --yes
  --job-name "$JOB_NAME"
)

if [[ "$N_TASKS" != "all" ]]; then
  HARBOR_ARGS+=(--n-tasks "$N_TASKS")
fi

if [[ "${ZSH28CODE_BENCH_DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' .venv/bin/harbor "${HARBOR_ARGS[@]}"
  printf '\n'
  exit 0
fi

.venv/bin/pip install -e '.[harbor]' --quiet
echo "Running $N_TASKS task(s) with Laguna: jobs/$JOB_NAME"
.venv/bin/harbor "${HARBOR_ARGS[@]}"
echo "Results: jobs/$JOB_NAME/result.json"
