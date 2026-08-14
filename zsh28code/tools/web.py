"""Todo write and web tools."""

from typing import Any

import requests

from zsh28code.tools.base import Tool


class TodoWriteTool(Tool):
    """Manage task lists."""

    name = "todo_write"
    description = (
        "Write the current task list. Replaces the whole list each call. "
        "Use this to plan tasks with more than a couple of steps, then mark "
        "items in_progress/done as you work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["items"],
    }
    is_read_only = True

    def __init__(self):
        self.items: list[dict[str, str]] = []

    async def execute(self, args: dict[str, Any]) -> str:
        self.items = args["items"]
        marks = {"pending": " ", "in_progress": "~", "done": "x"}
        lines = [
            f"[{marks.get(item['status'], ' ')}] {item['content']}"
            for item in self.items
        ]
        return "\n".join(lines) if lines else "Todo list is empty."


class WebFetchTool(Tool):
    """Fetch a URL and return its content."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return its content as readable markdown text. "
        "Use this to read documentation, blog posts, or any web page."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch."},
        },
        "required": ["url"],
    }
    is_read_only = True

    def __init__(self):
        self._api_key: str | None = None
        self._max_content_length: int = 5000

    def configure(self, api_key: str, max_content_length: int = 5000):
        self._api_key = api_key
        self._max_content_length = max_content_length

    async def execute(self, args: dict[str, Any]) -> str:
        url = args["url"]
        if not self._api_key:
            return "Error: FIRECRAWL_API_KEY not configured. Set OPENROUTER_API_KEY and FIRECRAWL_API_KEY."

        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"url": url, "formats": ["markdown"]},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            content = data["data"]["markdown"][:self._max_content_length]
            return content
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {e}"


class WebSearchTool(Tool):
    """Search the web for information."""

    name = "web_search"
    description = (
        "Search the web and return the top results (title, URL, description). "
        "Use this for research, finding documentation, or discovering current info."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Number of results. Default: 5.", "default": 5},
        },
        "required": ["query"],
    }
    is_read_only = True

    def __init__(self):
        self._api_key: str | None = None

    def configure(self, api_key: str):
        self._api_key = api_key

    async def execute(self, args: dict[str, Any]) -> str:
        query = args["query"]
        limit = args.get("limit", 5)
        if not self._api_key:
            return "Error: FIRECRAWL_API_KEY not configured."

        try:
            response = requests.post(
                "https://api.firecrawl.dev/v2/search",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": query, "limit": limit, "sources": ["web"]},
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()["data"]["web"]
            if not results:
                return "No results found."

            lines = []
            for r in results:
                title = r.get("title", "N/A")
                url = r.get("url", "N/A")
                desc = r.get("description", "")
                lines.append(f"{title}\n{url}\n{desc}")
            return "\n\n".join(lines)
        except requests.RequestException as e:
            return f"Error searching web: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing search results: {e}"


class SpawnAgentTool(Tool):
    """Spawn a sub-agent with a focused context (RLM recursion)."""

    name = "task"
    description = (
        "Spawn a sub-agent with a focused context to do a sub-task; returns its final answer. "
        "This is the RLM recursion primitive — use it to delegate a focused question "
        "to a fresh sub-agent that only sees the given context."
    )
    parameters = {
        "type": "object",
        "property": {
            "name": {},
        },
    } if False else {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short description of the sub-task."},
            "prompt": {"type": "string", "description": "Full instructions for the sub-agent."},
        },
        "required": ["description", "prompt"],
    }
    is_read_only = False

    def __init__(self):
        self._tool_factory = None  # set by agent

    def configure(self, tool_factory):
        self._tool_factory = tool_factory

    async def execute(self, args: dict[str, Any]) -> str:
        if self._tool_factory is None:
            return "Error: SpawnAgentTool not configured with agent runner."
        # The agent will handle spawning via the factory
        return "Spawning sub-agent..."
