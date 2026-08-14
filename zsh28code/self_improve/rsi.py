"""RSI: Recursive Self-Improvement loop.

Runs the agent on a task suite, analyzes results, triggers RLM
self-improvement, applies patches, re-tests, and manages git checkpoints.
Depth-limited to max_depth=3.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import git

from zsh28code.agent import Agent
from zsh28code.config import Config
from zsh28code.context import ContextStore
from zsh28code.llm import LLMClient
from zsh28code.self_improve.memory import MemoryDB
from zsh28code.self_improve.rlm import MAX_RSI_DEPTH, RLMSelfImprover
from zsh28code.tools import get_headless_tools

logger = logging.getLogger(__name__)

SRC_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root


@dataclass
class RSIResult:
    """Result of one RSI iteration."""
    depth: int
    successes: int
    total: int
    success_rate: float
    avg_reward: float
    improvements_applied: list[str]
    improvements_reverted: list[str]
    config_hash: str
    elapsed_seconds: float


class RSIOrchestrator:
    """Orchestrates the recursive self-improvement loop.

    Pipeline:
    1. Run task suite → collect rewards
    2. Trigger RLM analysis → generate patches
    3. Apply patches to working tree
    4. Re-run task suite → measure improvement
    5. If improvement → commit; if regression → revert
    6. Recurse with depth+1 (up to MAX_RSI_DEPTH)
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryDB,
        task_suite: list[dict[str, Any]],
        max_depth: int = MAX_RSI_DEPTH,
    ):
        self.llm = llm
        self.memory = memory
        self.task_suite = task_suite
        self.max_depth = max_depth
        self.rlm = RLMSelfImprover(llm, memory, max_depth)

        # Track git state
        self.repo: git.Repo | None = None
        try:
            self.repo = git.Repo(SRC_ROOT)
        except git.InvalidGitError:
            logger.warning("Not a git repo — RSI will run without checkpoints")

    async def run(self) -> list[RSIResult]:
        """Execute the full RSI loop, returning results per depth."""
        results: list[RSIResult] = []

        for depth in range(self.max_depth):
            logger.info(f"RSI iteration at depth {depth}")
            result = await self._run_iteration(depth)
            results.append(result)

            # If success rate improved, continue; otherwise stop
            if depth > 0 and result.success_rate <= results[-2].success_rate:
                logger.info(f"No improvement at depth {depth}, stopping RSI")
                break

        return results

    async def _run_iteration(self, depth: int) -> RSIResult:
        """Run one full PSI → improve → test → commit cycle."""
        import time
        start_time = time.time()

        # Step 1: Run task suite
        task_results = await self._run_task_suite(depth)
        successes = sum(1 for r in task_results if r["success"])
        rewards = [r["reward"] for r in task_results]

        successes / len(task_results) if task_results else 0.0
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Record baseline config hash
        config_hash = hashlib.sha256(
            json.dumps({"depth": depth}).encode()
        ).hexdigest()[:16]

        # Store results in memory
        for r in task_results:
            self.memory.add_task_result(
                task_name=r["task"],
                success=r["success"],
                reward=r["reward"],
                config_hash=config_hash,
                iterations=r.get("iterations", 0),
                elapsed_seconds=r.get("elapsed_seconds", 0.0),
            )

        logger.info(
            f"Depth {depth} baseline: {successes}/{len(task_results)} passed, "
            f"avg_reward={avg_reward:.2f}"
        )

        # Step 2: Create git checkpoint (if available)
        if self.repo:
            # Stage current state
            self.repo.git.add(A=True)

        # Step 3: Trigger RLM self-improvement
        improvements = await self.rlm.recursive_improve(
            [r for r in task_results if not r["success"]],
            depth=depth,
        )

        # Step 4: Apply improvements
        applied: list[str] = []
        reverted: list[str] = []

        for imp in improvements:
            if self._try_apply_patch(imp):
                applied.append(imp.file_path)
                logger.info(f"Applied improvement to {imp.file_path}")
            else:
                reverted.append(imp.file_path)
                logger.warning(f"Failed to apply improvement to {imp.file_path}")

        # Step 5: Re-run task suite to measure improvement
        post_results = await self._run_task_suite(depth)
        post_successes = sum(1 for r in post_results if r["success"])
        post_rewards = [r["reward"] for r in post_results]
        post_success_rate = post_successes / len(post_results) if post_results else 0.0
        post_avg_reward = sum(post_rewards) / len(post_rewards) if post_rewards else 0.0

        logger.info(
            f"Depth {depth} post-improvement: {post_successes}/{len(post_results)} passed, "
            f"avg_reward={post_avg_reward:.2f}"
        )

        # Step 6: Decide whether to keep or revert
        if post_avg_reward <= avg_reward:
            # Regression — revert all changes
            logger.warning("Improvement caused regression, reverting changes")
            for imp in improvements:
                if imp.file_path in applied:
                    self.memory.add_improvement(
                        source_file=imp.file_path,
                        patch="",
                        description=f"REVERTED: {imp.issue_description}",
                        depth=depth,
                    )
                    reverted.append(imp.file_path)
            if self.repo:
                # Restore from checkpoint
                self.repo.git.checkout("--", ".")
            applied = []
        else:
            # Improvement accepted — record it
            for imp in improvements:
                if imp.file_path in applied:
                    score_delta = post_avg_reward - avg_reward
                    self.memory.add_improvement(
                        source_file=imp.file_path,
                        patch=imp.suggested_patch,
                        description=imp.issue_description,
                        depth=depth,
                    )
                    self.memory.mark_improvement_applied(imp.id, score_delta=score_delta)

        elapsed = time.time() - start_time

        return RSIResult(
            depth=depth,
            successes=post_successes,
            total=len(post_results),
            success_rate=post_success_rate,
            avg_reward=post_avg_reward,
            improvements_applied=applied,
            improvements_reverted=reverted,
            config_hash=config_hash,
            elapsed_seconds=elapsed,
        )

    async def _run_task_suite(self, depth: int) -> list[dict]:
        """Run the agent on each task in the suite."""
        import time
        results: list[dict] = []

        for task in self.task_suite:
            task_start = time.time()
            task_name = task.get("task", "unknown")
            task_prompt = task.get("prompt", "")

            # Create agent instance
            store = ContextStore()
            config = Config(
                model=self.llm.model,
                api_key=self.llm.api_key,
                base_url=self.llm.base_url,
            )
            tools = get_headless_tools(
                context_store=store,
                agent_factory=self._spawn_rlm_agent,
            )
            agent = Agent(config=config, tools=tools, context_store=store, headless=True)

            try:
                # Run agent with task prompt
                result = await agent.run_headless(task_prompt)
                task_elapsed = time.time() - task_start

                # Determine reward from result
                response = result.submission if result.completed else ""
                reward = self._compute_reward(task, response)
                success = reward > 0.5

                results.append({
                    "task": task_name,
                    "prompt": task_prompt,
                    "response": response[:500] if isinstance(response, str) else str(response)[:500],
                    "reward": reward,
                    "success": success,
                    "iterations": result.iterations,
                    "elapsed_seconds": task_elapsed,
                })
            except Exception as e:
                task_elapsed = time.time() - task_start
                logger.error(f"Task '{task_name}' failed with error: {e}")
                results.append({
                    "task": task_name,
                    "prompt": task_prompt,
                    "response": str(e)[:500],
                    "reward": 0.0,
                    "success": False,
                    "iterations": 0,
                    "elapsed_seconds": task_elapsed,
                    "error": str(e),
                })

        return results

    def _compute_reward(self, task: dict, response: str) -> float:
        """Compute a reward based on task success criteria."""
        reward = 0.0

        # Check for required output patterns
        expected_patterns = task.get("expected_patterns", [])
        if expected_patterns:
            found = sum(1 for p in expected_patterns if p in response)
            reward += 0.3 * (found / len(expected_patterns))

        # Check for forbidden patterns (deductions)
        forbidden_patterns = task.get("forbidden_patterns", [])
        for p in forbidden_patterns:
            if p in response:
                reward -= 0.1

        # Bonus for concise responses (terminal output should be short)
        if isinstance(response, str) and len(response) < 1000:
            reward += 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, reward))

    def _try_apply_patch(self, analysis) -> bool:
        """Attempt to apply a patch from an RLM analysis.

        Returns True if patch was successfully applied.
        """
        try:
            from zap import Patcher
            patcher = Patcher()
            patcher.apply(analysis.file_path, analysis.suggested_patch)
            return True
        except ImportError:
            # Fall back to manual diff application
            return self._apply_patch_manual(analysis)
        except Exception as e:
            logger.warning(f"Failed to apply patch: {e}")
            return False

    def _apply_patch_manual(self, analysis) -> bool:
        """Apply a unified diff patch manually."""
        try:
            import re

            diffs = analysis.suggested_patch.split("\n")
            source_path = SRC_ROOT / "zsh28code" / analysis.file_path

            if not source_path.exists():
                logger.warning(f"Source file not found: {source_path}")
                return False

            original = source_path.read_text()
            lines = original.splitlines(keepends=True)

            # Simple patch applier for unified diff
            new_lines = list(lines)
            current_line = 0
            offset = 0

            for line in diffs:
                # Parse hunk headers: @@ -start,count +start,count @@
                match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    current_line = int(match.group(1)) - 1 + offset
                    continue

                if line.startswith("---") or line.startswith("+++"):
                    continue

                if line.startswith("-"):
                    if current_line < len(new_lines):
                        new_lines.pop(current_line)
                        offset -= 1
                elif line.startswith("+"):
                    new_lines.insert(current_line, line[1:])
                    offset += 1
                    current_line += 1
                elif not line.startswith("\\"):
                    current_line += 1

            new_source = "".join(new_lines)
            source_path.write_text(new_source)
            return True

        except Exception as e:
            logger.warning(f"Manual patch failed: {e}")
            return False


__all__ = [
    "RSIResult",
    "RSIOrchestrator",
]
