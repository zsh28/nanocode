"""Headless and interactive rendering for zsh28code."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console

console = Console(file=sys.stdout)


async def _get_user_approval(app, tool, args: dict) -> bool:
    """Prompt user for approval on potentially destructive actions."""
    tool_name = getattr(tool, "name", str(tool))
    console.print(f"\n[yellow]⚠ Approval required for: {tool_name}[/yellow]")
    console.print(f"  Args: {json.dumps(args, indent=2)[:300]}")

    if not sys.stdin.isatty():
        console.print("[dim]Auto-approved (headless mode)[/dim]")
        return True

    response = input("  Approve? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def run_headless(config, prompt: str, args: argparse.Namespace | None = None) -> str:
    """Run the agent in headless mode (no TUI, plain text output).

    Main entry point for benchmark/CLI usage.
    """
    from zsh28code.agent import Agent
    from zsh28code.context import ContextStore
    from zsh28code.tools import get_headless_tools

    config.max_iterations = getattr(args, "max_iterations", config.max_iterations) if args else config.max_iterations
    config.max_tokens = getattr(args, "max_tokens", config.max_tokens) if args else config.max_tokens

    store = ContextStore()

    async def _spawn_rlm_agent(
        query: str,
        context: str,
        depth: int,
        max_depth: int,
    ) -> str:
        """Spawn a sub-agent for RLM recursion (headless)."""
        from zsh28code.agent import Agent
        from zsh28code.context import ContextStore
        from zsh28code.tools import get_headless_tools

        sub_config = type(config)(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            reasoning_effort="low",
            max_tokens=config.max_tokens,
            max_iterations=min(config.max_iterations, 10),
            auto_approve=True,
            rlm_depth=config.rlm_depth,
            large_output_threshold=config.large_output_threshold,
            recent_summary_tokens=1000,
        )

        sub_store = ContextStore()
        sub_tools = get_headless_tools(context_store=sub_store)
        sub_agent = Agent(
            config=sub_config,
            tools=sub_tools,
            context_store=sub_store,
            headless=True,
        )

        try:
            focused_prompt = (
                f"Task: {query}\n\n"
                f"Focused context:\n{context[:50000]}\n\n"
                f"Recursion depth: {depth}/{max_depth}"
            )
            result = await sub_agent.run_headless(focused_prompt)
        finally:
            await sub_agent.aclose()

        return result.submission if result.completed else result.reason

    tools = [] if (args and getattr(args, "no_tools", False)) else get_headless_tools(
        context_store=store,
        agent_factory=_spawn_rlm_agent,
    )

    agent = Agent(config=config, tools=tools, context_store=store, headless=True)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(agent.run_headless(prompt))
    finally:
        loop.run_until_complete(agent.aclose())
        loop.close()

    # Write trajectory if output path specified
    if args and getattr(args, "output", None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.trajectory, indent=2))

    if result.completed:
        console.print(f"[green]✅ Completed:[/] {result.submission}")
    else:
        console.print(f"[yellow]⚠ Stopped:[/] {result.reason}")

    console.print(f"[dim]Iterations: {result.iterations}[/dim]")

    return result.submission if result.completed else result.reason


def run_interactive(config, args: argparse.Namespace | None = None) -> None:
    """Run the agent in interactive TUI mode.

    Launches the full-screen Textual interface.
    """
    from zsh28code.ui.app import Zsh28App

    app = Zsh28App(task=getattr(args, "prompt", None) if args else None)
    app.run()


__all__ = [
    "run_headless",
    "run_interactive",
    "get_user_approval",
]


# Compatibility alias for imports that expect this name
get_user_approval = _get_user_approval  # type: ignore
