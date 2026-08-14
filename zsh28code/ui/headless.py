"""Headless (non-interactive) rendering for benchmark mode.

Provides clean text output when running without a TTY,
and user approval prompts for potential destructive actions.
"""

import sys

from rich.console import Console

console = Console(file=sys.stdout, force_terminal=False)


async def get_user_approval(action: str, details: str = "") -> bool:
    """Prompt user for approval on potentially destructive actions.

    In headless mode, always approves with a warning log.
    """
    console.print(f"\n[yellow]⚠ Approval required: {action}[/yellow]")
    if details:
        console.print(f"  Details: {details[:200]}")

    # In headless/benchmark mode, auto-approve
    if not sys.stdin.isatty():
        console.print("[dim]Auto-approved (headless mode)[/dim]")
        return True

    # Interactive mode — prompt user
    try:
        response = input("  Approve? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def render_headless_result(task: str, result, success: bool, reward=None) -> None:
    """Render a completed task in headless mode."""
    status_icon = "✅" if success else "❌"
    reward_str = f" | Reward: {reward:.2f}" if reward is not None else ""

    console.print(f"\n{status_icon} Task: {task}{reward_str}")
    console.print(f"Iterations: {result.iterations}")
    if result.submission:
        console.print(f"\n[bold]Submission:[/] {result.submission[:500]}")
    if not result.completed:
        console.print(f"\n[yellow]Reason: {result.reason}[/yellow]")
    console.print()


def run_headless(task: str) -> str:
    """Run the agent in headless mode (no TUI, plain text output).

    This is the main entry point for benchmark/CLI usage.
    """
    import asyncio

    from zsh28code.agent import Agent
    from zsh28code.config import load_config
    from zsh28code.context import ContextStore
    from zsh28code.tools import get_headless_tools

    config = load_config()
    store = ContextStore()
    tools = get_headless_tools(
        context_store=store,
        agent_factory=lambda query, context, depth, max_depth: None,  # placeholder
    )

    agent = Agent(config=config, tools=tools, context_store=store, headless=True)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(agent.run_headless(task))
    finally:
        loop.close()

    return result.submission if result.completed else result.reason


__all__ = [
    "run_headless",
    "get_user_approval",
    "render_headless_result",
]
