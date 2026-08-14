"""Full-screen Textual TUI for interactive agent mode.

Designed to look like Codex / Pi Code: clean conversation view, minimal chrome,
clear message types, and a slim input at the bottom.
"""

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path

from rich.console import RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Input, RichLog, Static

from zsh28code.agent import Agent
from zsh28code.config import load_config
from zsh28code.context import ContextStore
from zsh28code.tools import get_default_tools


def _clean_bash_output(text: str) -> str:
    """Remove the [Output: N chars] line noise from bash results."""

    text = re.sub(
        r"\[Output:\s*\d+\s*chars\s*—\s*too large for direct context\]\n?",
        "",
        text,
    )
    text = re.sub(r"Head:\n", "", text)
    text = re.sub(r"\n\.\.\.\s*\(\d+ chars elided\)\s*\.\.\.\n", "\n...\n", text)
    text = text.replace("Tail:\n", "")
    text = text.replace("Use rlm_search('pattern') or rlm_peek() to examine full output.", "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


class MessageBubble(Text):
    """A single conversation message, rendered as clean text with role color."""

    def __init__(self, role: str, content: str, metadata: str = ""):
        style = {
            "user": "bold cyan",
            "assistant": "white",
            "system": "dim",
            "tool": "green",
        }.get(role, "white")

        header = ""
        if role in ("assistant", "tool"):
            header = f"{metadata}\n" if metadata else ""

        if role == "user":
            header = "YOU\n"
            prefix = "$ "
            content = content.strip()
        else:
            prefix = ""

        if role == "user":
            header = "YOU\n"

        if role == "tool":
            content = _clean_bash_output(content)

        super().__init__(
            f"{header}{prefix}{content}",
            style=style,
        )


class ConvScroll(RichLog):
    """The conversation scroll area."""

    def __init__(self):
        self._messages: list[MessageBubble] = []
        super().__init__(highlight=False, markup=False)

    def add_message(self, role: str, content: str, metadata: str = ""):
        bubble = MessageBubble(role=role, content=content, metadata=metadata)
        self._messages.append(bubble)
        self.write(bubble)


class StatusStrip(Static):
    """Thin status line at the bottom."""

    def __init__(self, status: str = "ready", model: str = ""):
        dot = "●"
        dot_style = "green" if status == "ready" else "yellow"
        super().__init__()
        self.status = status
        self.model = model

    def on_mount(self) -> None:
        self.update(self._render_status())

    def update_status(self, status: str, model: str | None = None):
        self.status = status
        if model:
            self.model = model
        self.update(self._render_status())

    def _render_status(self) -> RenderableType:
        dot = "●"
        dot_style = "green" if self.status == "ready" else "yellow"
        text = Text()
        text.append(f" {dot} ", style=dot_style)
        text.append(f"{self.status.upper()}", style="bold")
        if self.model:
            text.append(f"  {self.model}", style="dim")
        return text


class AccessPrompt(ModalScreen[bool]):
    """Confirm access to a directory before filesystem tools inspect it."""

    CSS = """
    AccessPrompt {
        align: center middle;
    }
    #access-dialog {
        width: 72;
        height: auto;
        padding: 2 3;
        border: round $primary;
        background: $surface;
    }
    #access-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    #access-actions Button {
        margin-left: 1;
    }
    """

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory

    def compose(self) -> ComposeResult:
        with Vertical(id="access-dialog"):
            yield Static("Can I access these files?", classes="access-title")
            yield Static(self.directory)
            with Horizontal(id="access-actions"):
                yield Button("Deny", id="deny")
                yield Button("Allow", id="allow", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class Zsh28App(App):
    """Full-screen Textual app for zsh28code — Codex/Pi Code style."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
        color: $text;
    }
    #conv-container {
        height: 1fr;
        width: 100%;
        padding: 1 2;
        border: round $panel;
    }
    #user-input {
        height: auto;
        margin: 1 2 0 2;
        border: round $primary;
    }
    Input > .input-prompt {
        color: cyan;
    }
    StatusStrip {
        height: 2;
        margin: 0 2 1 2;
        padding: 0 1;
        border: none;
        background: $surface;
        color: $text-muted;
    }
    #conv-container RichLog {
        scrollbar-size: 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", key_display="Ctrl+C"),
        Binding("ctrl+l", "clear", "Clear", key_display="Ctrl+L"),
        Binding("/", "focus_input", "Focus Input"),
    ]

    def __init__(self, task: str | None = None):
        super().__init__()
        self._task_input = task
        self.config = load_config()
        self.conv = ConvScroll()
        self.input_widget: Input | None = None
        self.status = StatusStrip(model=self.config.model or "poolside/laguna-s-2.1:free")
        self.agent: Agent | None = None
        self._pending_task: asyncio.Task | None = None
        self._pending: bool = False
        self._approved_directories: set[str] = set()

    async def on_mount(self) -> None:
        """Called when the app starts."""
        self.title = "zsh28code"

        # Initialize agent components
        self.context = ContextStore()
        tools = get_default_tools(
            context_store=self.context,
            firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY"),
        )

        if self.config.api_key:
            self.agent = Agent(
                config=self.config,
                tools=tools,
                context_store=self.context,
                headless=False,
                approval_callback=self._approve_tool,
            )
            self.conv.add_message("system", "Ready. Enter a task below.", metadata="zsh28code")
        else:
            self.status.update_status("missing API key")
            self.conv.add_message(
                "system",
                "OPENROUTER_API_KEY not set. Set it and restart.",
                metadata="zsh28code",
            )

        self.input_widget = self.query_one("#user-input", Input)
        self.input_widget.focus()

        if self._task_input:
            self.conv.add_message("user", self._task_input)
            self._pending_task = asyncio.create_task(self.submit_task(self._task_input))

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="conv-container"):
            yield self.conv
        yield Input(placeholder="What can I help with?", id="user-input")
        yield self.status

    def action_quit(self) -> None:
        self.exit()

    def action_focus_input(self) -> None:
        self.query_one("#user-input", Input).focus()

    def _tool_directory(self, tool_name: str, args: dict[str, object]) -> str | None:
        """Map a filesystem tool call to the directory it will access."""
        if tool_name in {"ls", "grep", "find"}:
            raw_path = str(args.get("path", "."))
        elif tool_name in {"read", "write", "edit"}:
            raw_path = str(args.get("path", "."))
            if tool_name in {"read", "write", "edit"}:
                raw_path = str(Path(raw_path).parent)
        else:
            return None
        return str(Path(raw_path).expanduser().resolve())

    async def _approve_tool(self, tool, args: dict[str, object]) -> bool:
        directory = self._tool_directory(tool.name, args)
        if directory is None:
            return True
        if directory in self._approved_directories:
            return True
        loop = asyncio.get_running_loop()
        result: asyncio.Future[bool] = loop.create_future()

        def on_access_decision(allowed: bool | None) -> None:
            if not result.done():
                result.set_result(bool(allowed))

        self.push_screen(AccessPrompt(directory), callback=on_access_decision)
        allowed = await result
        if allowed:
            self._approved_directories.add(directory)
        return bool(allowed)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        value = event.value.strip()
        if not value or not self.agent or self._pending:
            return

        self._pending = True
        self.conv.add_message("user", value)
        self.status.update_status("working")
        self.query_one("#user-input", Input).value = ""

        self._pending_task = asyncio.create_task(self.submit_task(value))

    async def submit_task(self, task: str) -> None:
        """Submit a task to the agent."""
        if not self.agent:
            self.status.update_status("ready")
            return

        try:
            async for partial in self.agent.run_stream(task):
                if partial["role"] == "status":
                    self.status.update_status(partial["content"], partial.get("metadata") or None)
                elif partial["role"] == "assistant":
                    self.conv.add_message("assistant", partial["content"], metadata=partial.get("metadata", "AGENT"))
                elif partial["role"] == "tool":
                    self.conv.add_message("tool", partial["content"], metadata=f"[{datetime.now().strftime('%H:%M:%S')}]")
        except Exception as e:
            self.conv.add_message("system", f"Error: {e}", metadata="zsh28code")
        finally:
            self.status.update_status("ready")
            self._pending = False
            self.query_one("#user-input", Input).focus()


__all__ = ["Zsh28App"]
