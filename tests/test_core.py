"""Tests for zsh28code core components."""

import json
import tempfile
from pathlib import Path

import pytest

from zsh28code.agent import Agent
from zsh28code.config import Config
from zsh28code.context import ContextStore, EntryType
from zsh28code.self_improve.memory import MemoryDB
from zsh28code.tools.shell import BashTool


class ScriptedAgent(Agent):
    """Agent with deterministic LLM responses for loop tests."""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self.responses = iter(responses)
        self.seen_messages = []

    async def _call_llm(self, messages):
        self.seen_messages.append(messages)
        return next(self.responses)


class TestContextStore:
    """Tests for the RLM ContextStore."""

    def test_add_and_serialize(self):
        store = ContextStore()
        store.add(EntryType.SYSTEM, "system", "Hello world")
        store.add(EntryType.USER, "user", "How are you?")
        serialized = store.serialize()
        assert "Hello world" in serialized
        assert "How are you?" in serialized
        assert "system" in serialized
        assert "user" in serialized

    def test_peek_from_end(self):
        store = ContextStore()
        for i in range(100):
            store.add(EntryType.SYSTEM, "system", f"Line {i}")
        result = store.peek(chars=100, from_end=True)
        assert result.excerpt.endswith("Line 99")

    def test_peek_from_start(self):
        store = ContextStore()
        for i in range(100):
            store.add(EntryType.SYSTEM, "system", f"Line {i}")
        result = store.peek(chars=100, from_end=False)
        assert result.excerpt.startswith("[system]")
        assert "Line 0" in result.excerpt

    def test_search(self):
        store = ContextStore()
        store.add(EntryType.SYSTEM, "system", "The quick brown fox")
        store.add(EntryType.USER, "user", "jumps over the lazy dog")
        result = store.search(pattern="fox")
        assert result.matched_count > 0
        assert "fox" in result.excerpt

    def test_slice(self):
        store = ContextStore()
        store.add(EntryType.SYSTEM, "system", "Hello world test message")
        result = store.slice(0, 20)
        assert "Hello" in result.excerpt or result.excerpt != ""

    def test_recent_summary(self):
        store = ContextStore()
        store.add(EntryType.SYSTEM, "system", "System prompt here")
        store.add(EntryType.TOOL_RESULT, "tool", "x" * 5000, tool_name="bash")
        store.add(EntryType.ASSISTANT, "assistant", "I'll run a test")

        summary = store.recent_summary(max_tokens=500)
        assert "x" * 5000 not in summary  # Large output should be summarized
        assert "bash" in summary


class TestTools:
    """Tests for agent tools."""

    @pytest.mark.asyncio
    async def test_bash_tool(self):
        from zsh28code.tools.shell import BashTool
        tool = BashTool()
        result = await tool.execute({"command": "echo hello"})
        assert "hello" in result
        assert "exit=0" in result

    @pytest.mark.asyncio
    async def test_read_write_tools(self):
        from zsh28code.tools.file import ReadFileTool, WriteFileTool
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            write = WriteFileTool()
            result = await write.execute({"path": str(filepath), "content": "test content"})
            assert "Wrote" in result

            read = ReadFileTool()
            result = await read.execute({"path": str(filepath)})
            assert "test content" in result

    @pytest.mark.asyncio
    async def test_todo_tool(self):
        from zsh28code.tools.web import TodoWriteTool
        tool = TodoWriteTool()
        result = await tool.execute({
            "items": [
                {"content": "Task 1", "status": "done"},
                {"content": "Task 2", "status": "pending"},
            ]
        })
        assert "Task 1" in result
        assert "[x]" in result
        assert "[ ]" in result

    @pytest.mark.asyncio
    async def test_rlm_peek_tool(self):
        from zsh28code.tools.rlm import RlmPeekTool
        store = ContextStore()
        store.add(EntryType.SYSTEM, "system", "Hello world")
        tool = RlmPeekTool()
        tool.configure(store)
        result = await tool.execute({"chars": 50})
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_rlm_search_tool(self):
        from zsh28code.tools.rlm import RlmSearchTool
        store = ContextStore()
        store.add(EntryType.SYSTEM, "system", "find this pattern")
        tool = RlmSearchTool()
        tool.configure(store)
        result = await tool.execute({"pattern": "this pattern"})
        assert "matches" in result.lower() or "1 matches" in result


class TestAgentLoop:
    """Regression tests for completion and repeated-call handling."""

    def test_system_prompt_names_active_workspace(self):
        agent = Agent(
            config=Config(api_key="test"),
            tools=[],
            context_store=ContextStore(),
            headless=True,
        )
        try:
            prompt = agent._build_system_prompt("Inspect the workspace")
        finally:
            import asyncio
            asyncio.run(agent.aclose())

        assert "## Active Workspace" in prompt
        assert str(Path.cwd()) in prompt
        assert "Start every task by running `pwd`" in prompt
        assert "MUST use" in prompt
        assert "Do not merely explain" in prompt

    @pytest.mark.asyncio
    async def test_text_response_completes(self):
        agent = ScriptedAgent(
            [("Finished successfully.", [], "stop")],
            config=Config(api_key="test", max_iterations=5),
            tools=[],
            context_store=ContextStore(),
            headless=True,
        )
        try:
            result = await agent.run_headless("Do the task")
        finally:
            await agent.aclose()

        assert result.completed
        assert result.submission == "Finished successfully."
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_tool_arguments_are_visible_next_turn(self):
        tool_call = {
            "id": "call-1",
            "name": "bash",
            "arguments": json.dumps({"command": "printf hello"}),
        }
        agent = ScriptedAgent(
            [
                ("", [tool_call], "tool_calls"),
                ("The command succeeded.", [], "stop"),
            ],
            config=Config(api_key="test", max_iterations=5),
            tools=[BashTool()],
            context_store=ContextStore(),
            headless=True,
        )
        try:
            result = await agent.run_headless("Print hello")
        finally:
            await agent.aclose()

        assert result.completed
        assert "printf hello" in str(agent.seen_messages[1])

    @pytest.mark.asyncio
    async def test_sequential_tasks_use_current_prompt(self):
        agent = ScriptedAgent(
            [
                ("First complete", [], "stop"),
                ("Second complete", [], "stop"),
            ],
            config=Config(api_key="test", max_iterations=2),
            tools=[],
            context_store=ContextStore(),
            headless=True,
        )
        try:
            await agent.run_headless("first task")
            await agent.run_headless("second task")
        finally:
            await agent.aclose()

        second_input = str(agent.seen_messages[1])
        assert "second task" in second_input
        assert "first task" not in second_input
        assert "First complete" not in second_input

    @pytest.mark.asyncio
    async def test_bash_completion_signal_completes(self):
        tool_call = {
            "id": "call-1",
            "name": "bash",
            "arguments": json.dumps({
                "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            }),
        }
        agent = ScriptedAgent(
            [("", [tool_call], "tool_calls")],
            config=Config(api_key="test", max_iterations=5),
            tools=[BashTool()],
            context_store=ContextStore(),
            headless=True,
        )
        try:
            result = await agent.run_headless("Finish")
        finally:
            await agent.aclose()

        assert result.completed
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_three_identical_calls_stop_loop(self):
        tool_call = {
            "id": "call-1",
            "name": "bash",
            "arguments": json.dumps({"command": "printf hello"}),
        }
        agent = ScriptedAgent(
            [("", [tool_call], "tool_calls")] * 3,
            config=Config(api_key="test", max_iterations=5),
            tools=[BashTool()],
            context_store=ContextStore(),
            headless=True,
        )
        try:
            result = await agent.run_headless("Print hello")
        finally:
            await agent.aclose()

        assert not result.completed
        assert "repeating bash three times" in result.reason
        assert result.iterations == 3

    @pytest.mark.asyncio
    async def test_interactive_approval_can_deny_tool_call(self):
        tool_call = {
            "id": "call-1",
            "name": "bash",
            "arguments": json.dumps({"command": "echo should-not-run"}),
        }

        async def deny(_tool, _args):
            return False

        agent = ScriptedAgent(
            [
                ("", [tool_call], "tool_calls"),
                ("The action was denied.", [], "stop"),
            ],
            config=Config(api_key="test", max_iterations=5),
            tools=[BashTool()],
            context_store=ContextStore(),
            headless=False,
            approval_callback=deny,
        )
        try:
            result = await agent.run("Run the command")
        finally:
            await agent.aclose()

        assert result.completed
        assert any("User denied the tool call" in entry["content"] for entry in agent.trajectory)


class TestMemoryDB:
    """Tests for the self-improvement memory database."""

    def test_add_and_retrieve_task_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDB(db_path=str(Path(tmpdir) / "test.db"))
            db.add_task_result("task1", True, 0.8, "cfg1")
            results = db.get_task_results()
            assert len(results) == 1
            assert results[0].task_name == "task1"
            assert results[0].success
            assert results[0].reward == 0.8

    def test_success_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDB(db_path=str(Path(tmpdir) / "test2.db"))
            db.add_task_result("task1", True, 0.9, "cfg1")
            db.add_task_result("task2", False, 0.1, "cfg1")
            assert 0.4 < db.get_success_rate() < 0.6

    def test_add_and_retrieve_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDB(db_path=str(Path(tmpdir) / "test3.db"))
            imp = db.add_improvement(
                "agent.py", "+1 -1", "Fixed a bug"
            )
            assert imp.id is not None
            assert imp.source_file == "agent.py"
            assert imp.description == "Fixed a bug"

            improvements = db.get_improvements()
            assert len(improvements) == 1
            assert improvements[0].source_file == "agent.py"

    def test_get_improvements_for_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDB(db_path=str(Path(tmpdir) / "test4.db"))
            db.add_improvement("agent.py", "patch1", "fix1")
            db.add_improvement("agent.py", "patch2", "fix2")
            db.add_improvement("config.py", "patch3", "fix3")

            agent_imps = db.get_improvements_for_file("agent.py")
            assert len(agent_imps) == 2


class TestTrajectory:
    """Tests for trajectory conversion."""

    def test_convert_to_atif(self):
        from zsh28code.benchmark.trajectory import convert_to_atif

        traj = [
            {"role": "user", "content": "test task"},
            {"role": "assistant", "content": "doing it"},
            {"role": "tool_result", "content": "output", "tool_name": "bash"},
            {"role": "exit", "content": "done"},
        ]
        atif = convert_to_atif(traj, "test-session")
        assert atif["session_id"] == "test-session"
        assert len(atif["steps"]) >= 2
