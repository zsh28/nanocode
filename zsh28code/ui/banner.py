"""Banner and startup display for the TUI."""

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text


def render_banner() -> RenderableType:
    """Render the zsh28code startup banner."""
    banner_text = Text()
    banner_text.append("  __             __ ", style="bold cyan")
    banner_text.append("__\n", style="bold cyan")
    banner_text.append("  \\ \\           / / ", style="bold cyan")
    banner_text.append("__\n", style="bold cyan")
    banner_text.append("   \\ \\         / /_ ", style="bold cyan")
    banner_text.append("  _   _  ___ _ __\n", style="white")
    banner_text.append("    \\ { } / / ", style="bold cyan")
    banner_text.append(" | | | |/ __| '_ \\\n", style="white")
    banner_text.append("     \\ { } / /  | |_", style="bold cyan")
    banner_text.append("| |_| | (__| |_) |\n", style="white")
    banner_text.append("      \\_ {_}/   |_|\\__", style="bold cyan")
    banner_text.append("| .___/\\___| .__/ \n", style="white")
    banner_text.append("      (zsh28code)", style="bold cyan")
    banner_text.append("            | |\n", style="white")
    banner_text.append(" RL-Powered Agent  |_|", style="bold cyan")
    banner_text.append("\n", style="")

    return Panel(
        banner_text,
        title="[bold]v0.1.0-alpha[/bold]",
        border_style="cyan",
        subtitle="[dim]Recursive Language Model Agent[/dim]",
    )


def render_status_banner(status: str, model: str = "poolside/laguna-s-2.1:free") -> RenderableType:
    """Render a status line at the top of the TUI."""
    text = Text()
    text.append("● ", style="green" if status == "ready" else "yellow")
    text.append(f" [{status.upper()}] ", style="bold")
    text.append(model, style="dim")

    return Panel(text, height=3, border_style="blue", title="zsh28code")
