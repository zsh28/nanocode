"""RLM ContextStore — stores conversation as a searchable Python variable.

Instead of passing the entire conversation to the LLM, we store it in a
ContextStore. The LLM receives a focused input (system + task + recent summary)
and uses RLM tools (rlm_peek, rlm_search, rlm_slice, rlm_agent) to access
specific parts of the context as needed.
"""

import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum


class EntryType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    RLMMETA = "rlm_meta"  # metadata from RLM operations


@dataclass
class ContextEntry:
    """A single entry in the context store."""

    type: EntryType
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_name: str | None = None
    tool_id: str | None = None
    token_count: int = 0
    timestamp: float = 0.0

    def format(self) -> str:
        """Format entry as a string for serialization."""
        prefix = f"[{self.type.value}]"
        if self.tool_name:
            prefix += f" {self.tool_name}"
        return f"{prefix}\n{self.content}"


@dataclass
class RLMResult:
    """Result from an RLM context search/peek/slice operation."""

    matches: list[str] = field(default_factory=list)
    excerpt: str = ""
    total_chars: int = 0
    matched_count: int = 0


class ContextStore:
    """RLM context variable — stores all conversation history as a searchable structure.

    The full conversation is materialized as a single string via ``serialize()``
    so that RLM search/slice tools can operate efficiently with regex.
    """

    def __init__(self):
        self._entries: list[ContextEntry] = []
        self._serialized: str = ""
        self._dirty: bool = True

    def add(
        self,
        type: EntryType,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_id: str | None = None,
    ) -> ContextEntry:
        """Add a new entry to the context store."""
        import time

        entry = ContextEntry(
            type=type,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_id=tool_id,
            token_count=self._estimate_tokens(content),
            timestamp=time.time(),
        )
        self._entries.append(entry)
        self._dirty = True
        return entry

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return max(1, len(text) // 4)

    def serialize(self) -> str:
        """Materialize the full context as a single string (cached)."""
        if not self._dirty and self._serialized:
            return self._serialized

        parts: list[str] = []
        for entry in self._entries:
            parts.append(entry.format())
        self._serialized = "\n---\n".join(parts)
        self._dirty = False
        return self._serialized

    @property
    def total_chars(self) -> int:
        return len(self.serialize())

    @property
    def total_tokens(self) -> int:
        return sum(e.token_count for e in self._entries)

    def peek(self, chars: int = 2000, from_end: bool = True) -> RLMResult:
        """RLM peek: return the first/last N characters of context.

        This is the cheapest RLM operation — the model uses it to understand
        the overall structure without searching.
        """
        full = self.serialize()
        if not full:
            return RLMResult(excerpt="", total_chars=0, matched_count=0)

        if from_end:
            excerpt = full[-chars:] if len(full) > chars else full
        else:
            excerpt = full[:chars] if len(full) > chars else full

        return RLMResult(
            excerpt=excerpt,
            total_chars=len(full),
            matched_count=1,
        )

    def search(self, pattern: str, limit: int = 20) -> RLMResult:
        """RLM search: regex search across the full context.

        Returns matching lines with surrounding context (line numbers).
        Equivalent to grep over the conversation history.
        """
        full = self.serialize()
        if not full:
            return RLMResult(excerpt="", total_chars=0)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return RLMResult(excerpt=f"Error: Invalid regex: {e}", total_chars=0)

        lines = full.split("\n")
        matches: list[str] = []
        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context_lines = lines[start:end]
                matches.append(
                    f"Line {i} (match):\n" + "\n".join(context_lines)
                )
                if len(matches) >= limit:
                    break

        return RLMResult(
            matches=matches,
            excerpt="\n\n".join(matches) if matches else "No matches found.",
            total_chars=len(full),
            matched_count=len(matches),
        )

    def slice(self, start: int, end: int) -> RLMResult:
        """RLM slice: extract context[start:end] as a focused chunk.

        Characters are counted across the serialized context string.
        """
        full = self.serialize()
        if not full:
            return RLMResult(excerpt="", total_chars=0)

        start = max(0, min(start, len(full)))
        end = max(start, min(end, len(full)))
        excerpt = full[start:end]

        return RLMResult(
            excerpt=excerpt,
            total_chars=len(full),
            matched_count=1 if excerpt else 0,
        )

    def recent_summary(self, max_tokens: int = 2000, start_index: int = 0) -> str:
        """Generate a compressed summary of recent entries for the LLM input.

        This is what gets sent to the LLM alongside the system prompt and task.
        Recent tool outputs that exceed the threshold are summarized rather
        than included verbatim.
        """
        threshold = 4000  # chars — outputs above this are summarized
        parts: list[str] = []
        token_budget = max_tokens * 4  # rough char budget

        entries = self._entries[max(0, start_index):]
        for entry in reversed(entries):
            if token_budget <= 0:
                break

            if entry.type == EntryType.SYSTEM:
                # System prompt is sent separately
                continue

            if entry.type == EntryType.TOOL_RESULT and entry.tool_name:
                content = entry.content
                if len(content) > threshold:
                    # Summarize large outputs
                    head = textwrap.shorten(content[:200], width=200, placeholder="...")
                    summary = (
                        f"[{entry.tool_name}] Result: {len(content)} chars total. "
                        f"Summary: {head}\n"
                        f"(Use rlm_peek/rlm_search/rlm_slice to examine full output.)"
                    )
                    parts.append(summary)
                    token_budget -= len(summary)
                else:
                    parts.append(f"[{entry.tool_name}]\n{content}")
                    token_budget -= len(content)
            elif entry.type == EntryType.ASSISTANT and entry.content:
                shortened = textwrap.shorten(entry.content, width=500, placeholder="...")
                parts.append(f"[assistant]\n{shortened}")
                token_budget -= len(shortened)

        parts.reverse()
        return "\n\n".join(parts)

    def entries_by_tool(self, tool_name: str) -> list[ContextEntry]:
        """Return all entries where tool_name matches."""
        return [e for e in self._entries if e.tool_name == tool_name]

    def get(self, index: int) -> ContextEntry | None:
        """Get entry at index."""
        return self._entries[index] if 0 <= index < len(self._entries) else None

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<ContextStore entries={len(self._entries)} chars={self.total_chars} tokens≈{self.total_tokens}>"
