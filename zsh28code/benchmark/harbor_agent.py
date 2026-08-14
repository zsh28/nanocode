"""Direct Harbor adapter for Terminal-Bench.

The benchmark adapter intentionally does not install the local application into
the task container. It keeps the model loop on the host and sends only bash
commands to Harbor's task environment, which makes timeouts and trajectories
observable at episode granularity.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from openai import OpenAI

from zsh28code.benchmark.trajectory import convert_to_atif
from zsh28code.context import ContextStore, EntryType

DEFAULT_MODEL = "poolside/laguna-s-2.1:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_EPISODES = 80
COMMAND_TIMEOUT_SECONDS = 120
MAX_OUTPUT_BYTES = 12000

SYSTEM_PROMPT = """You are zsh28code, a terminal agent solving a Terminal-Bench task.
Use the bash tool to do the work in the task environment.

Workflow:
1. Inspect the environment with pwd, ls, and relevant files.
2. Implement the requested solution using bash commands.
3. Run focused verification and tests.
4. Only stop after the requested files or behavior are actually complete.

Never claim a file was created or changed unless a bash command performed it and
you verified the result. Keep commands focused and recover from command errors.
When finished, respond with a concise summary and do not call bash again."""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the Terminal-Bench task environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["command"],
        },
    },
}

RLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rlm_peek",
            "description": "Read the beginning or end of the stored conversation context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chars": {"type": "integer", "default": 2000},
                    "from_end": {"type": "boolean", "default": True},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rlm_search",
            "description": "Search the stored conversation context with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "limit": {"type": "integer", "default": 20}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rlm_slice",
            "description": "Extract a character range from stored conversation context.",
            "parameters": {
                "type": "object",
                "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rlm_agent",
            "description": "Ask a bounded child model to analyze a focused context chunk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "context_chunk": {"type": "string"},
                    "depth": {"type": "integer", "default": 1},
                },
                "required": ["query", "context_chunk"],
            },
        },
    },
]


class Zsh28Code(BaseAgent):
    """Run Laguna directly and execute its bash calls in Harbor."""

    SUPPORTS_ATIF = True
    SUPPORTS_RESUME = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_episodes: int = MAX_EPISODES,
        command_timeout: int = COMMAND_TIMEOUT_SECONDS,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._model = (model_name or os.environ.get("NANOCODE_MODEL") or DEFAULT_MODEL).removeprefix("openrouter/")
        self._base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        self._api_key = self.extra_env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self._max_episodes = max(1, int(max_episodes))
        self._command_timeout = max(1, int(command_timeout))
        self._max_output_bytes = max(1000, int(max_output_bytes))
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        self._messages: list[dict[str, Any]] = []
        self._context = ContextStore()
        self._trajectory: list[dict[str, Any]] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._episodes_run = 0

    @staticmethod
    @override
    def name() -> str:
        return "zsh28code"

    @override
    def version(self) -> str | None:
        return "0.2.0-direct-harbor"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Verify the task container is reachable before model inference."""
        result = await environment.exec(command="pwd && printf '\\n' && ls -la", timeout_sec=30)
        self.logger.info("Environment ready:\n%s", result.stdout[:2000])

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        self._context = ContextStore()
        self._context.add(EntryType.SYSTEM, "system", SYSTEM_PROMPT)
        self._context.add(EntryType.USER, "user", instruction)
        self._trajectory = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        self._input_tokens = 0
        self._output_tokens = 0
        self._episodes_run = 0
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            for episode in range(self._max_episodes):
                self._episodes_run = episode + 1
                episode_dir = self.logs_dir / f"episode-{episode:03d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                (episode_dir / "messages.json").write_text(
                    json.dumps(self._messages, indent=2), encoding="utf-8"
                )

                self.logger.info("Episode %d/%d: querying %s", episode + 1, self._max_episodes, self._model)
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=self._messages,
                    tools=[BASH_TOOL, *RLM_TOOLS],
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=16384,
                )
                usage = response.usage
                if usage:
                    self._input_tokens += usage.prompt_tokens or 0
                    self._output_tokens += usage.completion_tokens or 0
                self._sync_context(context)

                message = response.choices[0].message
                response_log = {
                    "content": message.content,
                    "tool_calls": [
                        {"name": call.function.name, "arguments": call.function.arguments}
                        for call in (message.tool_calls or [])
                    ],
                }
                (episode_dir / "response.json").write_text(
                    json.dumps(response_log, indent=2), encoding="utf-8"
                )

                if not message.tool_calls:
                    content = message.content or ""
                    self._messages.append({"role": "assistant", "content": content})
                    self._trajectory.append({"role": "assistant", "content": content})
                    self.logger.info("Episode %d: finished without tool calls", episode)
                    break

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
                self._messages.append(assistant_message)
                self._context.add(EntryType.ASSISTANT, "assistant", message.content or "")
                self._trajectory.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": assistant_message["tool_calls"],
                })

                for call in message.tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        command = str(args.get("command", ""))
                    except (json.JSONDecodeError, TypeError) as error:
                        tool_output = f"Error: invalid bash arguments: {error}"
                        command = ""
                        args = {}
                    else:
                        self.logger.info("Episode %d executing: %s", episode, command[:240])
                        if call.function.name == "bash":
                            tool_output = await self._exec_bash(environment, command)
                        else:
                            tool_output = await self._execute_rlm(call.function.name, args)

                    tool_output = self._truncate(tool_output)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_output,
                    })
                    self._context.add(
                        EntryType.TOOL_RESULT,
                        "tool",
                        tool_output,
                        tool_name=call.function.name,
                        tool_id=call.id,
                    )
                    self._trajectory.append({
                        "role": "tool_result",
                        "content": tool_output,
                        "tool_name": "bash",
                        "tool_id": call.id,
                    })
                    (episode_dir / f"tool-{call.id}.txt").write_text(tool_output, encoding="utf-8")
                    self._sync_context(context)
        finally:
            self._sync_context(context)
            (self.logs_dir / "messages.json").write_text(
                json.dumps(self._messages, indent=2), encoding="utf-8"
            )
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(convert_to_atif(self._trajectory, self.session_id or "unknown"), indent=2),
                encoding="utf-8",
            )

    async def _exec_bash(self, environment: BaseEnvironment, command: str) -> str:
        if not command.strip():
            return "Error: empty bash command"
        try:
            result = await environment.exec(command=command, timeout_sec=self._command_timeout)
            output = []
            if result.stdout:
                output.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output.append(f"STDERR:\n{result.stderr}")
            output.append(f"EXIT CODE: {result.return_code}")
            return "\n".join(output)
        except TimeoutError:
            return f"Error: command timed out after {self._command_timeout}s"
        except Exception as error:
            return f"Error executing command: {error}"

    async def _execute_rlm(self, name: str, args: dict[str, Any]) -> str:
        """Execute RLM context tools without touching the task container."""
        if name == "rlm_peek":
            result = self._context.peek(args.get("chars", 2000), args.get("from_end", True))
            return result.excerpt
        if name == "rlm_search":
            result = self._context.search(args["pattern"], args.get("limit", 20))
            return result.excerpt
        if name == "rlm_slice":
            return self._context.slice(args["start"], args["end"]).excerpt
        if name == "rlm_agent":
            depth = int(args.get("depth", 1))
            if depth >= 3:
                return "RLM maximum recursion depth reached."
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a bounded RLM child agent. Analyze only the supplied context."},
                    {"role": "user", "content": f"Question: {args['query']}\n\nContext:\n{args['context_chunk']}"},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        return f"Error: unknown RLM tool {name}"

    def _truncate(self, text: str) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= self._max_output_bytes:
            return text
        half = self._max_output_bytes // 2
        first = encoded[:half].decode("utf-8", errors="ignore")
        last = encoded[-half:].decode("utf-8", errors="ignore")
        return f"{first}\n[... truncated ...]\n{last}"

    def _sync_context(self, context: AgentContext) -> None:
        context.n_input_tokens = self._input_tokens
        context.n_output_tokens = self._output_tokens
        context.metadata = {
            **(context.metadata or {}),
            "episodes": self._episodes_run,
            "model": self._model,
        }
