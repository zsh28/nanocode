"""Color theme for the TUI."""

from rich.theme import Theme

ZSH28_THEME = Theme({
    "banner": "bold cyan",
    "status.ok": "green",
    "status.dirty": "yellow",
    "status.error": "red",
    "task.pending": "dim",
    "task.running": "yellow",
    "task.done": "green",
    "tool.bash": "bold blue",
    "tool.read": "bold green",
    "tool.write": "bold yellow",
    "tool.edit": "bold magenta",
    "tool.rlm": "bold red",
    "tool": "bold white",
    "user.input": "bold cyan",
    "llm.output": "white",
    "system": "dim",
    "error": "bold red",
    "warning": "bold yellow",
    "success": "bold green",
})
