"""RLM: Recursive Language Model-powered self-improvement.

The RLM agent inspects its own source code by loading it into a ContextStore,
then uses rlm_search/rlm_slice to find relevant sections, analyzes them with
the LLM, and generates targeted patches. This process recurses (max depth 3).
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

from zsh28code.context import ContextStore, EntryType
from zsh28code.llm import LLMClient
from zsh28code.self_improve.memory import MemoryDB

logger = logging.getLogger(__name__)

MAX_RSI_DEPTH = 3
SRC_ROOT = Path(__file__).resolve().parent.parent  # zsh28code/


@dataclass
class RLMAnalysis:
    """Output of an RLM analysis pass on a source file."""
    file_path: str
    issue_description: str
    suggested_patch: str
    confidence: float  # 0.0 to 1.0
    depth: int
    affected_functions: list[str]
    reasoning: str


class RLMSelfImprover:
    """Recursive self-improvement via RLM (Recursive Language Model).

    Works by:
    1. Loading source files into a ContextStore
    2. Using rlm_search to find code patterns related to failures
    3. Using rlm_agent to analyze and propose patches
    4. Recursively applying improvements up to max_depth
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryDB,
        max_depth: int = MAX_RSI_DEPTH,
    ):
        self.llm = llm
        self.memory = memory
        self.max_depth = max_depth

        # Files that are safe to modify for self-improvement
        self.improvable_files = [
            "agent.py",
            "context.py",
            "config.py",
            "tools/base.py",
            "tools/file.py",
            "tools/shell.py",
            "tools/todo.py",
            "tools/web.py",
            "tools/rlm.py",
            "self_improve/rlm.py",
            "self_improve/rsi.py",
            "self_improve/rl.py",
            "self_improve/research.py",
            "self_improve/memory.py",
        ]

    def build_context_store(self, source_files: list[str] | None = None) -> ContextStore:
        """Load source files into a ContextStore for RLM navigation."""
        if source_files is None:
            source_files = self.improvable_files

        store = ContextStore()

        for rel_path in source_files:
            file_path = SRC_ROOT / rel_path
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        store.add(EntryType.RLMMETA, "rlm_source", f.read(),
                                  tool_name=f"src/{rel_path}")
                except Exception as e:
                    logger.warning(f"Could not load {rel_path}: {e}")

        return store

    async def analyze_file(
        self,
        file_path: str,
        task_results: list[dict],
        depth: int = 0,
    ) -> RLMAnalysis | None:
        """Analyze a source file for potential improvements.

        Args:
            file_path: Relative path within zsh28code/ package
            task_results: Recent task results with failure patterns
            depth: Current recursion depth

        Returns:
            RLMAnalysis if an improvement is found, None otherwise
        """
        if depth >= self.max_depth:
            logger.debug(f"Max RLM depth reached ({depth}), skipping further analysis")
            return None

        full_path = SRC_ROOT / file_path

        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return None

        source = full_path.read_text()

        # Build context store with the file and recent task results
        store = ContextStore()
        store.add(EntryType.RLMMETA, "rlm_source", source,
                  tool_name=f"src/{file_path}")

        # Add task results as context
        failures = "\n".join(
            f"- {r.get('task', 'unknown')}: reward={r.get('reward', 0)}, "
            f"success={r.get('success', False)}, "
            f"error={r.get('error', 'none')}"
            for r in task_results
        )
        store.add(EntryType.RLMMETA, "rlm_failures", failures,
                  tool_name="recent_failures")

        # Use rlm_search to find functions and their purposes
        functions = self._extract_functions(source)
        for fname, signature, body_start, body_end in functions:
            store.add(EntryType.RLMMETA, "rlm_func", source[body_start:body_end],
                      tool_name=f"func:{fname}")

        # Use the LLM to analyze with RLM
        prompt = f"""
You are zsh28code doing recursive self-improvement (RLM).
You are inspecting your own source code to find bugs and performance issues.

File: src/{file_path}

Recent task results (failures first):
{failures}

Functions in this file:
{chr(10).join(f'- {f[0]}: {f[1]}' for f in functions)}

Task: Analyze the code for potential improvements. Focus on:
1. Logic errors that could cause task failures
2. Missing edge case handling
3. Inefficiencies in tool usage
4. Prompt engineering issues for terminal tasks

If you find a concrete improvement, output a patch in this JSON format:
{{
  "issue": "Brief description of the issue",
  "confidence": 0.85,
  "affected_functions": ["function_name"],
  "reasoning": "Why this fix helps",
  "patch": "Unified diff patch starting with ```diff"
}}

If no improvement is needed, output: {{"no_change": true}}

IMPORTANT: Only suggest changes to the source code, never to data or configs.
Keep patches minimal and surgical.
"""

        response = await self.llm.chat(prompt, max_tokens=2000)

        try:
            import json
            data = json.loads(response.strip().removeprefix("```json").removesuffix("```").strip())

            if data.get("no_change"):
                return None

            # Validate and parse
            patch = self._extract_patch(data.get("patch", ""))
            if not patch:
                return None

            return RLMAnalysis(
                file_path=file_path,
                issue_description=data.get("issue", "Unknown issue"),
                suggested_patch=patch,
                confidence=data.get("confidence", 0.0),
                depth=depth,
                affected_functions=data.get("affected_functions", []),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse RLM analysis response: {e}")
            return None

    def _extract_functions(self, source: str) -> list[tuple[str, str, int, int]]:
        """Extract function definitions with their source ranges."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        functions = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = node.end_lineno or start
                # Get signature
                args = [a.arg for a in node.args.args]
                sig = f"{node.name}({', '.join(args)})"
                functions.append((node.name, sig, start - 1, end))

        return functions

    def _extract_patch(self, text: str) -> str:
        """Extract a diff patch from text that may contain code blocks."""
        if "```diff" in text:
            parts = text.split("```diff")
            if len(parts) >= 2:
                return parts[1].split("```")[0].strip()
        elif "---" in text and "+++" in text:
            return text.strip()
        return ""

    async def recursive_improve(
        self,
        task_results: list[dict],
        depth: int = 0,
    ) -> list[RLMAnalysis]:
        """Run recursive self-improvement pass.

        Analyzes each improvable file, and for each improvement found,
        recursively analyzes the patched code again (up to max_depth).
        """
        if depth >= self.max_depth:
            return []

        improvements: list[RLMAnalysis] = []
        failures = [r for r in task_results if not r.get("success", True)]

        for file_path in self.improvable_files:
            analysis = await self.analyze_file(file_path, failures, depth=depth)
            if analysis and analysis.confidence > 0.6:
                improvements.append(analysis)
                logger.info(
                    f"RLM (depth {depth}) found improvement in {file_path}: "
                    f"{analysis.issue_description}"
                )

                # Recurse: re-analyze with the improvement in context
                sub_improvements = await self.recursive_improve(
                    task_results,
                    depth=depth + 1,
                )
                improvements.extend(sub_improvements)

        return improvements


__all__ = [
    "RLMAnalysis",
    "RLMSelfImprover",
]
