"""CLI entry point for zsh28code."""

import argparse
import sys

from zsh28code.__init__ import __version__
from zsh28code.config import load_config
from zsh28code.ui import run_headless, run_interactive


def main():
    parser = argparse.ArgumentParser(
        prog="zsh28code",
        description="An RLM-powered terminal coding agent for Terminal-Bench.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Initial instruction for the agent (required for headless mode).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"zsh28code {__version__}",
    )
    parser.add_argument(
        "-m", "--model",
        default="poolside/laguna-s-2.1:free",
        help="OpenRouter model to use (default: poolside/laguna-s-2.1:free).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort level for providers that support it.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens per LLM call.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum agent loop iterations.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Run without tools (text-only responses).",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch full-screen Textual TUI (interactive mode).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without any UI (benchmark / pipe mode).",
    )
    parser.add_argument(
        "--output",
        help="Write trajectory JSON to this path (for benchmark mode).",
    )
    parser.add_argument(
        "--rlm-depth",
        type=int,
        default=3,
        help="Max recursion depth for RLM self-improvement (default: 3).",
    )

    args = parser.parse_args()

    config = load_config()
    config.model = args.model
    config.reasoning_effort = args.reasoning_effort
    config.max_tokens = args.max_tokens
    config.max_iterations = args.max_iterations

    if args.tui or (not args.headless and not args.prompt):
        run_interactive(config, args)
    else:
        if not args.prompt:
            print("Error: prompt required for headless mode", file=sys.stderr)
            sys.exit(1)
        run_headless(config, args.prompt, args)


if __name__ == "__main__":
    main()
