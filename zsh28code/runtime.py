"""Shared orchestration runtime for local, TUI, and Harbor agents."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

COMPLETION_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


@dataclass
class RuntimeToolCall:
    """Normalized model tool call."""

    id: str
    name: str
    arguments: str


@dataclass
class ModelTurn:
    """One complete model response."""

    content: str = ""
    tool_calls: list[RuntimeToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolOutcome:
    """Result returned by a tool executor."""

    content: str
    is_error: bool = False
    counts_as_action: bool = True


@dataclass
class RuntimeEvent:
    """Observable event emitted by the shared runtime."""

    kind: str
    iteration: int
    content: str = ""
    tool_call: RuntimeToolCall | None = None
    is_error: bool = False


@dataclass
class RuntimeResult:
    """Final state of a runtime task."""

    completed: bool
    submission: str
    reason: str
    iterations: int


ModelCaller = Callable[[int], Awaitable[ModelTurn]]
ToolExecutor = Callable[[RuntimeToolCall, dict[str, Any]], Awaitable[ToolOutcome]]
TurnObserver = Callable[[ModelTurn], Awaitable[None] | None]
ToolObserver = Callable[[RuntimeToolCall, ToolOutcome], Awaitable[None] | None]
EventObserver = Callable[[RuntimeEvent], Awaitable[None] | None]


def task_requires_action(task: str) -> bool:
    """Return whether a request needs an observable tool action."""
    return bool(
        re.search(
            r"\b(create|write|edit|modify|change|delete|remove|fix|run|test|implement|build)\b",
            task,
            re.IGNORECASE,
        )
    )


def tool_output_is_error(tool_name: str, content: str) -> bool:
    """Normalize common local and Harbor tool failure formats."""
    if content.lstrip().lower().startswith("error"):
        return True
    if tool_name != "bash":
        return False
    local_exit = re.search(r"\[exit=(-?\d+)", content)
    harbor_exit = re.search(r"EXIT CODE:\s*(-?\d+)", content)
    code = local_exit or harbor_exit
    return bool(code and int(code.group(1)) != 0)


class AgentRuntime:
    """Shared model/tool loop independent of UI and execution environment."""

    def __init__(
        self,
        *,
        task: str,
        max_iterations: int,
        call_model: ModelCaller,
        execute_tool: ToolExecutor,
        on_turn: TurnObserver | None = None,
        on_tool: ToolObserver | None = None,
        emit: EventObserver | None = None,
        completion_signal: str = COMPLETION_SIGNAL,
        repeated_call_limit: int = 3,
    ) -> None:
        self.task = task
        self.max_iterations = max(0, max_iterations)
        self.call_model = call_model
        self.execute_tool = execute_tool
        self.on_turn = on_turn
        self.on_tool = on_tool
        self.emit = emit
        self.completion_signal = completion_signal
        self.repeated_call_limit = max(2, repeated_call_limit)
        self._last_signature = ""
        self._repeat_count = 0
        self._action_succeeded = False

    async def _notify(self, callback, *args) -> None:
        if callback is None:
            return
        result = callback(*args)
        if inspect.isawaitable(result):
            await result

    async def _emit(self, event: RuntimeEvent) -> None:
        await self._notify(self.emit, event)

    def _register_call(self, call: RuntimeToolCall) -> bool:
        signature = f"{call.name}:{call.arguments}"
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1
        return self._repeat_count >= self.repeated_call_limit

    def _validate_completion(self, content: str, iteration: int) -> RuntimeResult:
        if task_requires_action(self.task) and not self._action_succeeded:
            return RuntimeResult(
                completed=False,
                submission=content.strip(),
                reason="The task requires a tool action, but no tool completed successfully.",
                iterations=iteration,
            )
        return RuntimeResult(
            completed=True,
            submission=content.strip(),
            reason="Completed",
            iterations=iteration,
        )

    async def run(self) -> RuntimeResult:
        last_content = ""
        for iteration in range(1, self.max_iterations + 1):
            await self._emit(RuntimeEvent("thinking", iteration))
            try:
                turn = await self.call_model(iteration)
            except Exception as error:
                reason = f"LLM error: {error}"
                await self._emit(RuntimeEvent("error", iteration, reason, is_error=True))
                return RuntimeResult(False, "", reason, iteration)

            last_content = turn.content
            await self._notify(self.on_turn, turn)
            if turn.content.strip():
                await self._emit(RuntimeEvent("assistant", iteration, turn.content.strip()))

            if self.completion_signal and self.completion_signal in turn.content:
                submission = turn.content.split(self.completion_signal)[-1].strip()
                result = self._validate_completion(submission, iteration)
                await self._emit(RuntimeEvent("complete", iteration, result.reason))
                return result

            if turn.tool_calls:
                for call in turn.tool_calls:
                    if self._register_call(call):
                        reason = f"Stopped after repeating {call.name} three times."
                        await self._emit(RuntimeEvent("error", iteration, reason, call, True))
                        return RuntimeResult(False, turn.content, reason, iteration)

                    await self._emit(RuntimeEvent("tool_start", iteration, tool_call=call))
                    try:
                        arguments = json.loads(call.arguments or "{}")
                        if not isinstance(arguments, dict):
                            raise TypeError("tool arguments must decode to an object")
                    except (json.JSONDecodeError, TypeError) as error:
                        outcome = ToolOutcome(
                            f"Error parsing arguments for {call.name}: {error}",
                            is_error=True,
                        )
                    else:
                        try:
                            outcome = await self.execute_tool(call, arguments)
                        except Exception as error:
                            outcome = ToolOutcome(
                                f"Error executing {call.name}: {error}",
                                is_error=True,
                            )

                    if not outcome.is_error and outcome.counts_as_action:
                        self._action_succeeded = True
                    await self._notify(self.on_tool, call, outcome)
                    await self._emit(
                        RuntimeEvent(
                            "tool_result",
                            iteration,
                            outcome.content,
                            call,
                            outcome.is_error,
                        )
                    )

                    if call.name == "bash" and self.completion_signal in outcome.content:
                        submission = outcome.content.split(self.completion_signal)[-1].strip()
                        result = self._validate_completion(submission, iteration)
                        await self._emit(RuntimeEvent("complete", iteration, result.reason))
                        return result
                continue

            if turn.finish_reason == "length":
                await self._emit(RuntimeEvent("truncated", iteration, turn.content))
                continue
            if turn.content.strip():
                result = self._validate_completion(turn.content, iteration)
                await self._emit(RuntimeEvent("complete", iteration, result.reason))
                return result

        return RuntimeResult(
            completed=False,
            submission=last_content,
            reason="Max iterations reached",
            iterations=self.max_iterations,
        )


__all__ = [
    "AgentRuntime",
    "COMPLETION_SIGNAL",
    "ModelTurn",
    "RuntimeEvent",
    "RuntimeResult",
    "RuntimeToolCall",
    "ToolOutcome",
    "task_requires_action",
    "tool_output_is_error",
]
