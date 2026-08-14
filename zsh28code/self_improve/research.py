"""Autoresearch: Automatically research terminal-bench tasks and solutions.

Uses web search to find solutions, tutorials, and patterns for specific
terminal-bench tasks, then extracts improvements for the agent.
"""

import logging
import re
from dataclasses import dataclass

from zsh28code.llm import LLMClient
from zsh28code.self_improve.memory import MemoryDB
from zsh28code.tools.web import WebFetchTool, WebSearchTool

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Result of researching a task or topic."""
    query: str
    key_findings: list[str]
    relevant_urls: list[str]
    recommended_improvements: list[str]
    applied: bool = False


class Autoresearcher:
    """Automatically researches coding tasks and patterns.

    Searches the web for solutions to specific task types, extracts
    best practices, and converts findings into actionable improvements
    for the agent.
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryDB,
        web_search: WebSearchTool | None = None,
        web_fetch: WebFetchTool | None = None,
    ):
        self.llm = llm
        self.memory = memory
        self.web_search = web_search or WebSearchTool()
        self.web_fetch = web_fetch or WebFetchTool()

    async def research_task(
        self,
        task_description: str,
        task_type: str = "terminal-bench",
    ) -> ResearchResult:
        """Research a specific task type.

        Args:
            task_description: The task prompt or description
            task_type: Type of task (terminal-bench, coding, system-admin, etc.)

        Returns:
            ResearchResult with findings and recommendations
        """
        query = self._build_search_query(task_description, task_type)
        logger.info(f"Autoresearch query: {query}")

        # Step 1: Web search
        search_results = await self.web_search.search(query, limit=5)
        if isinstance(search_results, str):
            search_results = []

        relevant_urls: list[str] = []
        key_findings: list[str] = []

        # Step 2: Fetch and analyze top results
        for result in search_results[:3]:
            url = result.get("url", "")
            result.get("title", "")

            if not url:
                continue

            relevant_urls.append(url)

            try:
                content = await self.web_fetch.fetch(url)
                # Extract key info from content
                findings = self._extract_findings(content, task_description)
                key_findings.extend(findings[:3])  # Top 3 per source
            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")

        # Step 3: Synthesize findings using LLM
        recs = await self._synthesize_recommendations(
            task_description,
            key_findings,
            relevant_urls,
        )

        result = ResearchResult(
            query=query,
            key_findings=key_findings,
            relevant_urls=relevant_urls,
            recommended_improvements=recs,
        )

        return result

    async def research_task_type(
        self,
        task_type: str,
        keywords: list[str],
    ) -> ResearchResult:
        """Research an entire category of tasks.

        Args:
            task_type: The type of task (e.g., 'file manipulation', 'git operations')
            keywords: Relevant keywords to search for

        Returns:
            ResearchResult with findings and recommendations
        """
        query = f"{task_type} task best practices {' '.join(keywords)}"
        search_results = await self.web_search.search(query, limit=5)
        if isinstance(search_results, str):
            search_results = []

        relevant_urls: list[str] = []
        key_findings: list[str] = []

        for result in search_results[:3]:
            url = result.get("url", "")
            if url:
                relevant_urls.append(url)
                try:
                    content = await self.web_fetch.fetch(url)
                    findings = self._extract_findings(content, task_type)
                    key_findings.extend(findings[:3])
                except Exception:
                    pass

        recs = await self._synthesize_recommendations(
            f"Task type: {task_type}",
            key_findings,
            relevant_urls,
        )

        return ResearchResult(
            query=query,
            key_findings=key_findings,
            relevant_urls=relevant_urls,
            recommended_improvements=recs,
        )

    def _build_search_query(self, task_description: str, task_type: str) -> str:
        """Build a search query from task description."""
        # Extract key terms
        keywords = re.findall(r'\b[a-z_]+', task_description.lower())
        keywords = [k for k in keywords if len(k) > 4 and k not in ("this", "with", "from", "have", "they")]

        if task_type == "terminal-bench":
            return f"terminal-bench solution {' '.join(keywords[:5])}"
        else:
            return f"{task_type} best practices {' '.join(keywords[:5])}"

    def _extract_findings(self, content: str, task_description: str) -> list[str]:
        """Extract key findings from fetched content."""
        findings = []

        # Look for code blocks
        code_blocks = re.findall(r'```(?:bash|sh)?\n(.*?)```', content, re.DOTALL)
        for block in code_blocks[:3]:
            # Check if code block is relevant to the task
            if any(kw in block.lower() for kw in task_description.lower().split()):
                findings.append(f"Code: {block.strip()[:200]}")

        # Look for command patterns
        commands = re.findall(r'(?:^|\n)\$(\s*\S.*?)(?:\n|$)', content)
        for cmd in commands[:5]:
            findings.append(f"Command: {cmd.strip()}")

        # Look for key insights
        sentences = re.split(r'[.!?]+', content)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 50 and len(sent) < 300:
                if any(kw in sent.lower() for kw in task_description.lower().split()):
                    findings.append(sent)

        return findings[:5]

    async def _synthesize_recommendations(
        self,
        task_description: str,
        findings: list[str],
        urls: list[str],
    ) -> list[str]:
        """Use LLM to synthesize actionable recommendations."""
        prompt = f"""
Given these research findings about a coding task, synthesize specific,
actionable recommendations for improving the agent's approach.

Task: {task_description}

Research findings:
{chr(10).join(f'- {f}' for f in findings[:10])}

Sources:
{chr(10).join(f'- {url}' for url in urls[:5])}

Generate 3-5 specific recommendations in this format:
- RECOMMENDATION: [specific action]
- Rationale: [why this helps]

Be concrete and actionable. Focus on what the agent should do differently.
"""

        response = await self.llm.chat(prompt, max_tokens=1500)

        recommendations = []
        for line in response.strip().split("\n"):
            if line.startswith("- RECOMMENDATION:") or line.startswith("RECOMMENDATION:"):
                recommendations.append(line.replace("- ", "").strip())

        return recommendations

    async def apply_research_to_improvements(self, results: list[ResearchResult]) -> int:
        """Apply research findings to the improvement memory.

        Returns the number of improvements recorded.
        """
        count = 0
        for result in results:
            if result.applied:
                continue

            for rec in result.recommended_improvements:
                self.memory.add_improvement(
                    source_file="research",
                    patch=rec,
                    description=f"Research-finding: {rec[:100]}",
                    depth=0,
                )
                count += 1

            result.applied = True

        return count


__all__ = [
    "ResearchResult",
    "Autoresearcher",
]
