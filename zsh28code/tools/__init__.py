"""Tool registry and factory functions."""

from zsh28code.tools.base import Tool, ToolCall
from zsh28code.tools.file import (
    EditFileTool,
    FindTool,
    GrepTool,
    LsTool,
    ReadFileTool,
    WriteFileTool,
)
from zsh28code.tools.rlm import (
    RlmAgentTool,
    RlmPeekTool,
    RlmSearchTool,
    RlmSliceTool,
)
from zsh28code.tools.shell import BashTool
from zsh28code.tools.web import (
    TodoWriteTool,
    WebFetchTool,
    WebSearchTool,
)


def get_default_tools(
    context_store=None,
    firecrawl_api_key: str | None = None,
    max_web_content_length: int = 5000,
    agent_factory=None,
    max_rlm_depth: int = 3,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[Tool]:
    """Get the default set of tools, configured with runtime dependencies."""

    tools = [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GrepTool(),
        LsTool(),
        FindTool(),
        BashTool(),
        TodoWriteTool(),
    ]

    # Web tools (optional, only if API key provided)
    if firecrawl_api_key:
        web_fetch = WebFetchTool()
        web_fetch.configure(firecrawl_api_key, max_web_content_length)
        tools.append(web_fetch)

        web_search = WebSearchTool()
        web_search.configure(firecrawl_api_key)
        tools.append(web_search)

    # RLM context tools (only if context store is provided)
    if context_store is not None:
        peek = RlmPeekTool()
        peek.configure(context_store)
        tools.append(peek)

        search = RlmSearchTool()
        search.configure(context_store)
        tools.append(search)

        slc = RlmSliceTool()
        slc.configure(context_store)
        tools.append(slc)

        rlm_agent = RlmAgentTool()
        if agent_factory is not None:
            rlm_agent.configure(
                agent_factory=agent_factory,
                max_depth=max_rlm_depth,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        tools.append(rlm_agent)

    return tools


def get_readonly_tools(
    context_store=None,
    firecrawl_api_key: str | None = None,
    max_web_content_length: int = 5000,
) -> list[Tool]:
    """Get only read-only tools (for plan mode)."""
    tools = [ReadFileTool(), GrepTool(), LsTool(), FindTool()]

    if context_store is not None:
        peek = RlmPeekTool()
        peek.configure(context_store)
        search = RlmSearchTool()
        search.configure(context_store)
        tools.append(peek)
        tools.append(search)

    return tools


def get_headless_tools(
    context_store=None,
    agent_factory=None,
    max_rlm_depth: int = 3,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> list[Tool]:
    """Get tools for headless/benchmark mode (bash + rlm context tools only).

    This mirrors the mini-swe-agent / Terminus-2 philosophy: minimize the
    number of tools to reduce parsing overhead. The agent uses bash for
    most operations and rlm_* tools for navigating large context outputs.
    """
    tools = [BashTool()]

    if context_store is not None:
        peek = RlmPeekTool()
        peek.configure(context_store)
        tools.append(peek)

        search = RlmSearchTool()
        search.configure(context_store)
        tools.append(search)

        slc = RlmSliceTool()
        slc.configure(context_store)
        tools.append(slc)

        rlm_agent = RlmAgentTool()
        if agent_factory is not None:
            rlm_agent.configure(
                agent_factory=agent_factory,
                max_depth=max_rlm_depth,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        tools.append(rlm_agent)

    return tools


__all__ = [
    "Tool",
    "ToolCall",
    "get_default_tools",
    "get_readonly_tools",
    "get_headless_tools",
]
