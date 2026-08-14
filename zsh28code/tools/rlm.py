"""RLM context tools — give the LLM selective access to the full conversation.

Instead of stuffing the entire conversation history into the LLM prompt (which
causes context rot), these tools let the LLM search, peek, and slice the stored
context variable at specific positions. The LLM only sees focused fragments.
"""

from typing import Any

from zsh28code.tools.base import Tool


class RlmPeekTool(Tool):
    """RLM peek — read the first/last N characters of context.

    This is the cheapest RLM operation. The LLM uses it to understand the
    overall structure of the conversation without searching.
    """

    name = "rlm_peek"
    description = (
        "Read the first or last N characters of the conversation context. "
        "Use this to understand the overall structure, check the task "
        "instruction, or see what happened at the beginning/end. "
        "This is cheaper than rlm_search and should be the first tool you "
        "reach for when you need to orient yourself."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chars": {"type": "integer", "description": "Number of characters to read. Default: 2000.", "default": 2000},
            "from_end": {"type": "boolean", "description": "If true, read from the end. If false, from the start. Default: true.", "default": True},
        },
        "required": [],
    }
    is_read_only = True

    def __init__(self):
        self._context_store = None

    def configure(self, context_store):
        self._context_store = context_store

    async def execute(self, args: dict[str, Any]) -> str:
        chars = args.get("chars", 2000)
        from_end = args.get("from_end", True)

        result = self._context_store.peek(chars=chars, from_end=from_end)
        direction = "end" if from_end else "start"
        header = f"[rlm_peek] Last {len(result.excerpt)} chars (from {direction} of {result.total_chars} total chars)"
        return f"{header}\n---\n{result.excerpt}"


class RlmSearchTool(Tool):
    """RLM search — regex search across the entire conversation context.

    Equivalent to grep over the conversation history. Returns matching lines
    with surrounding context and line numbers.
    """

    name = "rlm_search"
    description = (
        "Search the entire conversation context (all previous messages and "
        "tool outputs) for a regex pattern. Returns matching lines with "
        "file paths, line numbers, and surrounding context. "
        "Use this when you need to find specific information in large outputs "
        "that were stored in the context, like error messages, specific values, "
        "or patterns in bash output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "limit": {"type": "integer", "description": "Maximum matches. Default: 20.", "default": 20},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    def __init__(self):
        self._context_store = None

    def configure(self, context_store):
        self._context_store = context_store

    async def execute(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        limit = args.get("limit", 20)

        result = self._context_store.search(pattern=pattern, limit=limit)
        header = f"[rlm_search] Pattern: '{pattern}' — {result.matched_count} matches (in {result.total_chars} total chars)"
        return f"{header}\n---\n{result.excerpt}"


class RlmSliceTool(Tool):
    """RLM slice — extract a specific byte range from context.

    When the LLM needs to examine a specific section of a large output
    (e.g., lines 45000-52000 of a build log), it uses this tool.
    """

    name = "rlm_slice"
    description = (
        "Extract a specific char range from the conversation context. "
        "Use this when you know roughly where the information you need is "
        "located (e.g., after finding the offset via rlm_search). "
        "Returns at most ~8000 chars of content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "start": {"type": "integer", "description": "Start character offset."},
            "end": {"type": "integer", "description": "End character offset."},
        },
        "required": ["start", "end"],
    }
    is_read_only = True

    def __init__(self):
        self._context_store = None

    def configure(self, context_store):
        self._context_store = context_store

    async def execute(self, args: dict[str, Any]) -> str:
        start = args["start"]
        end = args["end"]

        result = self._context_store.slice(start=start, end=end)
        header = f"[rlm_slice] Chars {start}-{end} of {result.total_chars} total"
        return f"{header}\n---\n{result.excerpt}"


class RlmAgentTool(Tool):
    """RLM recursive agent — spawn a sub-LLM on a focused context chunk.

    This is the recursion primitive: the Root LLM delegates a sub-task to a
    Sub-LLM that only sees a focused slice of the context. The Sub-LLM can
    itself recurse (up to rlm_depth levels).

    This is the RLM paradigm: the model interacts with its own context via code.
    """

    name = "rlm_agent"
    description = (
        "Spawn a recursive sub-agent (Sub-LLM) that only sees a focused slice "
        "of the conversation context. The sub-agent can use the same tools "
        "as you. This is the RLM recursion primitive — use it to deeply analyze "
        "a specific section of a large output, extract structured information, "
        "or solve a sub-problem isolated to a focused context.\n\n"
        "The sub-agent receives only the provided context_chunk as its world. "
        "It returns its final answer as a string.\n\n"
        "Maximum recursion depth: 3 levels."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question or task for the sub-agent."},
            "context_chunk": {"type": "string", "description": "The focused context for the sub-agent. Typically extracted via rlm_slice or rlm_search."},
            "depth": {"type": "integer", "description": "Current recursion depth (0 = root). Max 3. Default: 1.", "default": 1},
        },
        "required": ["query", "context_chunk"],
    }
    is_read_only = False

    def __init__(self):
        self._agent_factory = None
        self._max_depth = 3
        self._model = None
        self._base_url = None
        self._api_key = None

    def configure(self, agent_factory, max_depth: int = 3, model=None, base_url=None, api_key=None):
        self._agent_factory = agent_factory
        self._max_depth = max_depth
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

    async def execute(self, args: dict[str, Any]) -> str:
        query = args["query"]
        context_chunk = args["context_chunk"]
        depth = args.get("depth", 1)

        if depth >= self._max_depth:
            return f"[rlm_agent] Max recursion depth ({self._max_depth}) reached. Cannot recurse further."

        if self._agent_factory is None:
            # Fallback: use the LLM directly on the focused chunk
            return await self._fallback_recursive_call(query, context_chunk, depth)

        try:
            result = await self._agent_factory(
                query=query,
                context=context_chunk,
                depth=depth,
                max_depth=self._max_depth,
            )
            return f"[rlm_agent depth={depth}]\n{result}"
        except Exception as e:
            return f"[rlm_agent] Error: {e}"

    async def _fallback_recursive_call(
        self, query: str, context_chunk: str, depth: int
    ) -> str:
        """Fallback: call the LLM directly with the focused chunk."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)

        messages = [
            {"role": "system", "content": (
                f"You are a Sub-LLM (RLM recursion level {depth}). "
                f"You only see a focused context chunk. "
                f"Your task: {query}\n\n"
                f"Focus ONLY on the following context — do not invent information outside it."
            )},
            {"role": "user", "content": f"Context chunk:\n```\n{context_chunk}\n```\n\nQuestion: {query}"},
        ]

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=4096,
                temperature=0.3,
            )
            result = response.choices[0].message.content or ""
            return f"[rlm_agent depth={depth}]\n{result}"
        except Exception as e:
            return f"[rlm_agent] Error calling LLM: {e}"
