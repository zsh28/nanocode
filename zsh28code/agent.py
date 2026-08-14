"""Core RLM agent loop.

Unlike traditional agents that pass the entire conversation to the LLM,
zsh28code stores all messages in a ContextStore and sends only a focused
input (system + task + recent summary) to the model. The model uses RLM
tools (rlm_peek, rlm_search, rlm_slice, rlm_agent) to selectively access
the full context.
"""

import json
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from zsh28code.config import Config
from zsh28code.context import ContextStore, EntryType
from zsh28code.tools import get_headless_tools
from zsh28code.tools.base import Tool

# Maximum characters of tool output to include in LLM input directly.
# Larger outputs are stored in ContextStore and summarized.
_LARGE_OUTPUT_THRESHOLD = 4000

# Maximum characters of rlm_search/slice results returned to LLM.
_RLM_RESULT_MAX_CHARS = 8000


@dataclass
class AgentResult:
    """Result of an agent run."""
    completed: bool = False
    submission: str = ""
    reason: str = ""
    iterations: int = 0
    trajectory: list[dict] = field(default_factory=list)


class Agent:
    """RLM-powered agent with selective context access."""

    def __init__(
        self,
        config: Config,
        tools: list[Tool] | None = None,
        context_store: ContextStore | None = None,
        headless: bool = False,
        approval_callback: Callable[[Tool, dict[str, Any]], Awaitable[bool]] | None = None,
    ):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.context = context_store or ContextStore()
        self.headless = headless
        self.approval_callback = approval_callback

        if tools is not None:
            self.tools = tools
        else:
            self.tools = get_headless_tools(
                context_store=self.context,
                agent_factory=self._spawn_rlm_agent,
                max_rlm_depth=config.rlm_depth,
                model=config.model,
                base_url=config.base_url,
                api_key=config.api_key,
            )

        self._tools_by_name = {t.name: t for t in self.tools}
        self._iteration = 0
        self.trajectory: list[dict] = []

    async def aclose(self) -> None:
        """Close the underlying HTTP client and its streaming resources."""
        await self.client.close()

    async def _spawn_rlm_agent(
        self, query: str, context: str, depth: int, max_depth: int
    ) -> str:
        """Spawn a recursive sub-agent on a focused context chunk.

        This implements the RLM recursion primitive.
        """
        from zsh28code.agent import Agent as RlmSubAgent

        sub_config = Config(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            reasoning_effort="low",
            max_tokens=4096,
            max_iterations=10,
            auto_approve=True,
            rlm_depth=self.config.rlm_depth,
            large_output_threshold=_LARGE_OUTPUT_THRESHOLD,
            recent_summary_tokens=1000,
        )

        sub_tools = [
            t for t in self.tools if t.name not in ("rlm_agent",)
        ]

        sub_agent = RlmSubAgent(
            config=sub_config,
            tools=sub_tools,
            context_store=None,  # fresh context for sub-agent
            headless=True,
        )

        prompt = (
            f"You are a Sub-LLM (RLM recursion level {depth}).\n\n"
            f"Your task: {query}\n\n"
            f"Focus ONLY on the following context — do not invent information outside it:\n\n"
            f"```\n{context[:50000]}\n```\n\n"
            f"Recursion depth: {depth}/{max_depth}"
        )

        result = await sub_agent._run_single(prompt, auto_approve=True)
        return result

    def _build_system_prompt(self, task: str) -> str:
        """Build the system prompt for this agent run."""
        # Load AGENTS.md if present
        instructions = ""
        agents_path = os.path.join(os.getcwd(), "AGENTS.md")
        if os.path.exists(agents_path):
            try:
                with open(agents_path, encoding="utf-8") as f:
                    instructions = f.read()
            except Exception:
                pass

        workspace = os.path.abspath(os.getcwd())
        system = f"""You are zsh28code, an RLM-powered terminal coding agent.

## Active Workspace
The active workspace is exactly:
`{workspace}`

All relative paths refer to this directory. Start every task by running `pwd`
and `ls` in this workspace before deciding where files are. Do not search for a
directory named `terminal-bench` unless the user explicitly asks for it. Work
inside the active workspace and its descendants; do not silently switch to
another directory.

You store conversation context in a searchable variable. Use rlm_peek, rlm_search,
and rlm_slice to navigate large outputs efficiently — don't blindly stuff the context window.

## Your Task
{task}

## RLM Context Navigation
When bash outputs are large (>4000 chars), they are stored in a context variable
and summarized. Use these RLM tools to access them:
- rlm_peek(n, from_end): Read the first/last N chars of the full conversation
- rlm_search(pattern): Regex search across ALL messages and tool outputs
- rlm_slice(start, end): Extract a char range from the conversation
- rlm_agent(query, chunk): Recurse: spawn a sub-LLM on a focused chunk

## Tool Result Handling
Small results (< 4000 chars) appear directly in your context.
Large results are stored and summarized — use rlm_search/rlm_peek to examine them.

## Completion — VERY IMPORTANT
CRITICAL: The ONLY way to signal task completion is to run this exact command as
your final bash call — do NOT add any other text or instructions alongside it:

  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

This tells the system you are done. If the task requires a specific output
(e.g., a file content or command result), include it in the same echo:

  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

If you are NOT done yet, continue working. You can verify your work first,
but the completion signal must be on its own line as the last thing you do.

## Philosophy
- Agentic search: Use bash, grep, find to explore actively. The model DECIDES the search strategy.
- Test changes immediately after making them.
- Be terse. Prefer action over explanation.
- One bash call per response. When you want to end a task, output echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.

## Project Instructions
{instructions if instructions else "(No AGENTS.md found)"}
"""
        return system

    async def _run_single(
        self,
        prompt: str,
        auto_approve: bool = False,
    ) -> str:
        """Run a single agent session (prompt → tools → response).

        Used for rlm_agent recursion and headless benchmark mode.
        """
        system_prompt = self._build_system_prompt(prompt)
        self.context.add(EntryType.SYSTEM, "system", system_prompt)
        self.context.add(EntryType.USER, "user", prompt)
        self._record_trajectory(role="system", content=system_prompt)
        self._record_trajectory(role="user", content=prompt)

        result = await self._loop(auto_approve=auto_approve)
        return result.submission if result.completed else result.reason

    def _build_llm_input(self) -> list[dict[str, Any]]:
        """Build the focused input for the LLM.

        Instead of sending the full conversation, we send:
        1. System prompt (first entry)
        2. Task instruction (second entry)
        3. Recent summary (compressed recent tool results)
        """
        entries = self.context._entries

        messages: list[dict[str, Any]] = []

        # System prompt
        system_entries = [e for e in entries if e.type == EntryType.SYSTEM]
        if system_entries:
            messages.append({"role": "system", "content": system_entries[-1].content})

        # User instruction (first user message)
        user_entries = [e for e in entries if e.type == EntryType.USER]
        if user_entries:
            messages.append({"role": "user", "content": user_entries[0].content})

        # Recent summary (compressed assistant + tool entries)
        recent = self.context.recent_summary(
            max_tokens=self.config.recent_summary_tokens
        )
        if recent:
            messages.append({
                "role": "user",
                "content": f"## Recent activity summary:\n{recent}",
            })

        return messages

    def _should_summarize_tool_result(self, content: str, tool_name: str) -> str:
        """Determine if a tool result should be summarized or included directly.

        Returns a summary string if the result is large, or the full content
        if it's small enough.
        """
        if len(content) <= _LARGE_OUTPUT_THRESHOLD:
            return content

        # Summarize: show head and tail, note total size
        head = content[:500]
        tail = content[-500:] if len(content) > 1000 else ""
        summary = (
            f"[Output: {len(content)} chars — too large for direct context]\n"
            f"Head:\n{head}\n"
        )
        if tail:
            summary += f"\n... ({(len(content) - 1000)} chars elided) ...\n"
            summary += f"\nTail:\n{tail}\n"
        summary += "\nUse rlm_search('pattern') or rlm_peek() to examine full output."
        return summary

    async def _call_llm(self, messages: list[dict[str, Any]]) -> tuple[str, list, str]:
        """Call the LLM and parse response + tool calls.

        Returns (reply_text, tool_calls, finish_reason).
        """
        tool_schemas = [t.to_schema() for t in self.tools]

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tool_schemas,
            "stream": True,
            "max_tokens": self.config.max_tokens,
        }

        # Add reasoning eff
        if self.config.reasoning_effort:
            kwargs["extra_headers"] = {
                "HTTP-Referer": "https://github.com/zeeshanaligulamhusein/nanocode",
                "X-Title": "zsh28code",
            }
            # OpenRouter supports reasoning via extra body
            kwargs["extra_body"] = {
                "reasoning": {"effort": self.config.reasoning_effort},
            }

        stream = await self.client.chat.completions.create(**kwargs)

        reply = ""
        tool_calls: list = []
        finish_reason = None

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            if choice.delta.content:
                if not self.headless:
                    print(choice.delta.content, end="", flush=True)
                reply += choice.delta.content

            for tc in choice.delta.tool_calls or []:
                if tc.index >= len(tool_calls):
                    tool_calls.append({
                        "id": "",
                        "name": "",
                        "arguments": "",
                    })
                tc_entry = tool_calls[tc.index]
                tc_entry["id"] += tc.id or ""
                tc_entry["name"] += tc.function.name or ""
                tc_entry["arguments"] += tc.function.arguments or ""

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if not self.headless:
            print()  # newline after streaming

        return reply, tool_calls, finish_reason

    async def _execute_tool_call(self, tc: dict, tool: Tool) -> tuple[str, bool]:
        """Execute a single tool call and return (result, is_error)."""
        try:
            args = json.loads(tc["arguments"])
        except json.JSONDecodeError as e:
            return f"Error parsing arguments: {e}", True

        try:
            result = await tool.execute(args)
            return result, False
        except Exception as e:
            return f"Error executing {tc['name']}: {e}", True

    async def _approve_tool_call(self, tool: Tool, args: dict[str, Any]) -> bool:
        """Request approval for interactive filesystem access and mutations."""
        if self.headless or self.approval_callback is None:
            return True
        if tool.name in {"read", "ls", "grep", "find"} or not tool.is_read_only:
            return await self.approval_callback(tool, args)
        return True

    def _record_trajectory(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
        tool_result: str | None = None,
        tool_name: str | None = None,
    ):
        """Record an entry in the trajectory for logging/ATIF conversion."""
        entry = {
            "role": role,
            "content": content,
        }
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if tool_result is not None:
            entry["tool_result"] = tool_result
            entry["tool_name"] = tool_name or ""
        self.trajectory.append(entry)

    async def _loop(self, auto_approve: bool = False) -> AgentResult:
        """Main agent loop.

        1. Build focused LLM input (system + task + recent summary)
        2. LLM responds with text + tool calls
        3. Execute tools, store results in ContextStore
        4. For large tool results: store in context, summarize for LLM
        5. RLM tools search/peek/slice the context store
        6. Repeat until completion or max iterations
        """
        last_tool_signature = ""
        repeated_tool_calls = 0
        reply = ""

        for _ in range(self.config.max_iterations):
            self._iteration += 1

            # Build focused input
            messages = self._build_llm_input()

            # Record for trajectory
            self._record_trajectory(
                role="assistant_input",
                content=f"[LLM input length: {sum(len(m.get('content','')) for m in messages if isinstance(m.get('content'), str))} chars]",
            )

            # Call LLM
            try:
                reply, tool_calls, finish_reason = await self._call_llm(messages)
            except Exception as e:
                print(f"\nError calling LLM: {e}", file=sys.stderr)
                return AgentResult(
                    completed=False,
                    submission="",
                    reason=f"LLM error: {e}",
                    iterations=self._iteration,
                    trajectory=self.trajectory,
                )

            # Record assistant message
            tc_info = tool_calls or None
            self._record_trajectory(
                role="assistant",
                content=reply,
                tool_calls=tc_info,
            )

            # Preserve tool arguments in the searchable context so the next
            # focused LLM call can tell exactly which actions already ran.
            assistant_context = reply
            if tool_calls:
                calls = "\n".join(
                    f"Tool call: {tc['name']}({tc['arguments']})"
                    for tc in tool_calls
                )
                assistant_context = f"{reply}\n{calls}".strip()
            self.context.add(EntryType.ASSISTANT, "assistant", assistant_context)

            # Check for completion signal in LLM text response
            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in reply:
                submission = reply.split("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")[-1].strip()
                self._record_trajectory(role="exit", content="Completed successfully.")
                return AgentResult(
                    completed=True,
                    submission=submission,
                    reason="Completed",
                    iterations=self._iteration,
                    trajectory=self.trajectory,
                )

            # Execute tool calls
            if finish_reason == "tool_calls" and tool_calls:
                for tc in tool_calls:
                    signature = f"{tc['name']}:{tc['arguments']}"
                    if signature == last_tool_signature:
                        repeated_tool_calls += 1
                    else:
                        last_tool_signature = signature
                        repeated_tool_calls = 1

                    if repeated_tool_calls >= 3:
                        reason = f"Stopped after repeating {tc['name']} three times."
                        self._record_trajectory(role="exit", content=reason)
                        return AgentResult(
                            completed=False,
                            submission=reply,
                            reason=reason,
                            iterations=self._iteration,
                            trajectory=self.trajectory,
                        )

                    tool = self._tools_by_name.get(tc["name"])
                    if not tool:
                        result = f"Error: Unknown tool {tc['name']}"
                        is_error = True
                    else:
                        # Headless and explicitly approved runs bypass prompts.
                        if auto_approve:
                            result, is_error = await self._execute_tool_call(tc, tool)
                        else:
                            arguments = json.loads(tc["arguments"])
                            approved = await self._approve_tool_call(tool, arguments)
                            if approved:
                                result, is_error = await self._execute_tool_call(tc, tool)
                            else:
                                result = "User denied the tool call."
                                is_error = True

                    # Store in context
                    self.context.add(
                        EntryType.TOOL_RESULT,
                        "tool",
                        result,
                        tool_name=tc["name"],
                        tool_id=tc["id"],
                    )

                    # Record in trajectory
                    self._record_trajectory(
                        role="tool_result",
                        content=result,
                        tool_name=tc["name"],
                    )

                    # In headless mode, print tool execution info
                    if self.headless:
                        # Truncate display of large outputs
                        display = self._should_summarize_tool_result(result, tc["name"])
                        if len(display) > 2000:
                            display = display[:1000] + "\n...\n" + display[-1000:]
                        print(f"\n[{tc['name']}] {display[:500]}", file=sys.stderr)

                    # Check for completion signal in bash output
                    if tc["name"] == "bash" and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in result:
                        submission = result.split("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")[-1].strip()
                        if not submission:
                            submission = "Task completed successfully."
                        self._record_trajectory(role="exit", content="Completed successfully (bash signal).")
                        return AgentResult(
                            completed=True,
                            submission=submission,
                            reason="Completed via bash signal",
                            iterations=self._iteration,
                            trajectory=self.trajectory,
                        )

            else:
                # finish_reason is "stop", "length", or None
                if finish_reason == "length":
                    # Output was truncated — continue looping instead of exiting
                    self._record_trajectory(
                        role="assistant",
                        content=(reply + "\n[CUT_OFF: output truncated, continuing]") if reply else "[CUT_OFF: output truncated, continuing]",
                    )
                    continue
                # A normal text response is the model's final answer. Requiring
                if reply.strip():
                    self._record_trajectory(role="exit", content="Completed with final response.")
                    return AgentResult(
                        completed=True,
                        submission=reply.strip(),
                        reason="Completed",
                        iterations=self._iteration,
                        trajectory=self.trajectory,
                    )

        return AgentResult(
            completed=False,
            submission=reply,
            reason="Max iterations reached",
            iterations=self._iteration,
            trajectory=self.trajectory,
        )

    async def run(self, task: str, auto_approve: bool = False) -> AgentResult:
        """Run the agent on a task in interactive mode."""
        system_prompt = self._build_system_prompt(task)

        self.context.add(EntryType.SYSTEM, "system", system_prompt)
        self.context.add(EntryType.USER, "user", task)
        self._record_trajectory(role="system", content=system_prompt)
        self._record_trajectory(role="user", content=task)

        print(f"\nzsh28code: {task}\n", flush=True)

        return await self._loop(auto_approve=auto_approve)

    async def run_headless(self, task: str) -> AgentResult:
        """Run the agent in headless mode (no TUI, plain text output)."""
        system_prompt = self._build_system_prompt(task)

        self.context.add(EntryType.SYSTEM, "system", system_prompt)
        self.context.add(EntryType.USER, "user", task)
        self._record_trajectory(role="system", content=system_prompt)
        self._record_trajectory(role="user", content=task)

        return await self._loop(auto_approve=True)

    async def run_stream(self, task: str):
        """Run the agent and yield messages incrementally for the TUI.

        Yields dicts: {"role": "assistant"|"tool", "content": str, "metadata": str}
        """
        system_prompt = self._build_system_prompt(task)
        self.context.add(EntryType.SYSTEM, "system", system_prompt)
        self.context.add(EntryType.USER, "user", task)
        self._record_trajectory(role="system", content=system_prompt)
        self._record_trajectory(role="user", content=task)

        last_tool_signature = ""
        repeated_tool_calls = 0
        import datetime as _dt

        for _ in range(self.config.max_iterations):
            self._iteration += 1
            messages = self._build_llm_input()

            yield {
                "role": "status",
                "content": "thinking",
                "metadata": f"iteration {self._iteration}",
            }

            self._record_trajectory(
                role="assistant_input",
                content=f"[LLM input length: {sum(len(m.get('content','')) for m in messages if isinstance(m.get('content'), str))} chars]",
            )

            try:
                reply, tool_calls, finish_reason = await self._call_llm(messages)
            except Exception as e:
                print(f"\nError calling LLM: {e}", file=sys.stderr)
                yield {"role": "system", "content": f"LLM error: {e}", "metadata": "zsh28code"}
                return

            self._record_trajectory(role="assistant", content=reply, tool_calls=tool_calls or None)
            assistant_context = reply
            if tool_calls:
                calls = "\n".join(
                    f"Tool call: {tc['name']}({tc['arguments']})"
                    for tc in tool_calls
                )
                assistant_context = f"{reply}\n{calls}".strip()
            self.context.add(EntryType.ASSISTANT, "assistant", assistant_context)

            if reply.strip():
                yield {
                    "role": "assistant",
                    "content": reply.strip(),
                    "metadata": "AGENT",
                }

            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in reply:
                yield {"role": "status", "content": "responding", "metadata": ""}
                return

            if finish_reason == "tool_calls" and tool_calls:
                for tc in tool_calls:
                    signature = f"{tc['name']}:{tc['arguments']}"
                    if signature == last_tool_signature:
                        repeated_tool_calls += 1
                    else:
                        last_tool_signature = signature
                        repeated_tool_calls = 1

                    if repeated_tool_calls >= 3:
                        reason = f"Stopped after repeating {tc['name']} three times."
                        yield {"role": "system", "content": reason, "metadata": "zsh28code"}
                        return

                    tool = self._tools_by_name.get(tc["name"])
                    if not tool:
                        result = f"Error: Unknown tool {tc['name']}"
                        is_error = True
                    else:
                        arguments = json.loads(tc["arguments"])
                        yield {
                            "role": "status",
                            "content": f"running {tc['name']}",
                            "metadata": "",
                        }
                        approved = await self._approve_tool_call(tool, arguments)
                        if approved:
                            result, is_error = await self._execute_tool_call(tc, tool)
                        else:
                            result, is_error = "User denied the tool call.", True

                    ts = _dt.datetime.now().strftime("%H:%M:%S")
                    yield {"role": "tool", "content": result, "metadata": f"[{tc['name']}] {ts}"}

                    self.context.add(
                        EntryType.TOOL_RESULT, "tool", result,
                        tool_name=tc["name"], tool_id=tc["id"],
                    )
                    self._record_trajectory(role="tool_result", content=result, tool_name=tc["name"])

                    if tc["name"] == "bash" and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in result:
                        submission = result.split("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")[-1].strip()
                        if not submission:
                            submission = "Task completed successfully."
                        yield {"role": "assistant", "content": f"✅ {submission}", "metadata": "zsh28code"}
                        return

            else:
                if finish_reason == "length":
                    self._record_trajectory(
                        role="assistant",
                        content=(reply + "\n[CUT_OFF: output truncated, continuing]") if reply else "[CUT_OFF: output truncated, continuing]",
                    )
                    continue
                if reply.strip():
                    yield {"role": "status", "content": "responding", "metadata": ""}
                    return

        yield {"role": "system", "content": "Max iterations reached", "metadata": "zsh28code"}
