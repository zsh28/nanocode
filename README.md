# zsh28code

RLM-powered terminal coding agent built on `poolside/laguna-s-2.1:free`.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
zsh28code "write a hello world script"
zsh28code  # interactive TUI
```

## Terminal-Bench 2.1

Start Docker Desktop, then run:

```bash
./run_terminal_bench.sh
```

The script installs Harbor, prompts for the OpenRouter key, and runs one task
from `terminal-bench/terminal-bench-2-1` by default. Results are written under
`jobs/`.
