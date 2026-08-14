#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/zsh28code ]]; then
  echo "Error: .venv/bin/zsh28code is missing." >&2
  echo "Run: python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  read -r -s -p "OpenRouter API Key: " OPENROUTER_API_KEY
  echo
  export OPENROUTER_API_KEY
fi

if [[ $# -eq 0 ]]; then
  exec .venv/bin/zsh28code --tui
fi

exec .venv/bin/zsh28code "$@"
