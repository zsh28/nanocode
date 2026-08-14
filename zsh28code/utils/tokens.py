"""Token counting utilities."""

import json


def estimate_tokens(text: str) -> int:
    """Rough token estimate using the 4 chars/token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a list of chat messages."""
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    total += estimate_tokens(part["text"])
        # Add fixed overhead for tool calls
        for tc in msg.get("tool_calls", []) or []:
            total += estimate_tokens(json.dumps(tc))
    return total


def count_context_tokens(context_str: str) -> int:
    """Estimate tokens for a context variable string."""
    return estimate_tokens(context_str)


def truncate_to_token_budget(text: str, budget: int, reserve: int = 200) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = (budget - reserve) * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...(truncated)..."
