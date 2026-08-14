"""Convert zsh28code events into a small ATIF-compatible trajectory."""

from typing import Any


def convert_to_atif(trajectory: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    """Convert internal trajectory events to Harbor's step representation."""
    steps: list[dict[str, Any]] = []
    for index, entry in enumerate(trajectory, start=1):
        role = entry.get("role", "unknown")
        content = entry.get("content", "") or ""
        step: dict[str, Any] = {
            "step_id": index,
            "source": role,
            "message": {"role": role, "content": content},
        }
        if entry.get("tool_name"):
            step["tool"] = {"name": entry["tool_name"], "result": content}
        if entry.get("tool_calls"):
            step["tool_calls"] = entry["tool_calls"]
        steps.append(step)

    return {
        "schema_version": "ATIF-v1.0",
        "session_id": session_id,
        "steps": steps,
    }
