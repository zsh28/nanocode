"""File system tools: read, write, edit, grep, ls, find."""

import os
import re
from pathlib import Path
from typing import Any

from zsh28code.tools.base import Tool


class ReadFileTool(Tool):
    """Read a file from disk."""

    name = "read"
    description = (
        "Read a file from disk. Returns the file content with line numbers. "
        "Use this for viewing project files, reading code, checking config."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "start_line": {"type": "integer", "description": "Starting line number (1-indexed). Defaults to 1."},
            "end_line": {"type": "integer", "description": "Ending line number. Defaults to entire file."},
        },
        "required": ["path"],
    }
    is_read_only = True

    async def execute(self, args: dict[str, Any]) -> str:
        path = args["path"]
        start_line = args.get("start_line", 1)
        end_line = args.get("end_line")

        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return f"Error: File not found: {path}"

            if path_obj.is_dir():
                return f"Error: Path is a directory, not a file: {path}"

            content = path_obj.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")

            if start_line < 1:
                start_line = 1
            start_idx = start_line - 1

            if end_line is not None and end_line < len(lines):
                end_idx = end_line
            else:
                end_idx = len(lines)

            result_lines = lines[start_idx:end_idx]
            return "\n".join(
                f"{start_line + i}: {line}" for i, line in enumerate(result_lines)
            )
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write"
    description = (
        "Write content to a file, creating or overwriting it. "
        "Use this to create new files or save generated content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    async def execute(self, args: dict[str, Any]) -> str:
        path = args["path"]
        content = args["content"]

        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileTool(Tool):
    """Replace an exact string in a file."""

    name = "edit"
    description = (
        "Replace an exact string in a file with a new string. "
        "The old_string must match exactly (case-sensitive). Returns success or error. "
        "Use this for targeted edits to existing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact string to replace."},
            "new_string": {"type": "string", "description": "Replacement string."},
        },
        "required": ["path", "old_string", "new_string"],
    }
    is_read_only = False

    async def execute(self, args: dict[str, Any]) -> str:
        path = args["path"]
        old_string = args["old_string"]
        new_string = args["new_string"]

        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return f"Error: File not found: {path}"

            content = path_obj.read_text(encoding="utf-8")

            if old_string not in content:
                return f"Error: old_string not found in {path}"

            count = content.count(old_string)
            if count > 1:
                return f"Error: old_string found {count} times in {path}. Make the match more specific."

            updated = content.replace(old_string, new_string, 1)
            path_obj.write_text(updated, encoding="utf-8")
            return f"Edited {path} (1 replacement)"
        except Exception as e:
            return f"Error editing file: {e}"


class GrepTool(Tool):
    """Search files under a directory for lines matching a regex."""

    name = "grep"
    description = (
        "Search files under a directory for lines matching a regex pattern. "
        "Returns matching lines with file path and line number. "
        "Use ripgrep-style syntax. Case-insensitive by default."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "Directory to search in. Defaults to cwd.", "default": "."},
            "max_results": {"type": "integer", "description": "Maximum number of results to return. Default: 50.", "default": 50},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    async def execute(self, args: dict[str, Any]) -> str:
        pattern = args["pattern"]
        search_path = args.get("path", ".")
        max_results = args.get("max_results", 50)

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches: list[str] = []

        try:
            for dirpath, dirnames, filenames in os.walk(search_path):
                # skip hidden dirs and common non-source dirs
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}]
                for filename in filenames:
                    if filename.startswith("."):
                        continue
                    filepath = os.path.join(dirpath, filename)
                    try:
                        with open(filepath, encoding="utf-8", errors="replace") as f:
                            for lineno, line in enumerate(f, 1):
                                if regex.search(line):
                                    matches.append(f"{filepath}:{lineno}: {line.rstrip()}")
                                    if len(matches) >= max_results:
                                        return "\n".join(matches) + f"\n\n(Stopped at {max_results} results)"
                    except (UnicodeDecodeError, OSError, PermissionError):
                        continue
        except FileNotFoundError:
            return f"Error: Path not found: {search_path}"

        return "\n".join(matches) if matches else "No matches found."


class LsTool(Tool):
    """List files in a directory."""

    name = "ls"
    description = (
        "List files and directories in a path. Returns names, types, and sizes. "
        "Use this to explore the filesystem structure."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Defaults to cwd.", "default": "."},
            "recursive": {"type": "boolean", "description": "List recursively.", "default": False},
        },
        "required": [],
    }
    is_read_only = True

    async def execute(self, args: dict[str, Any]) -> str:
        path = args.get("path", ".")
        recursive = args.get("recursive", False)

        path_obj = Path(path)
        if not path_obj.exists():
            return f"Error: Path not found: {path}"

        lines: list[str] = []

        if recursive:
            for root, dirs, files in os.walk(path_obj):
                dirs[:] = sorted([d for d in dirs if not d.startswith("._")])
                files = sorted(files)
                rel = os.path.relpath(root, path_obj)
                prefix = "" if rel == "." else rel + "/"
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        size = os.path.getsize(fp)
                        lines.append(f"{prefix}{f}  ({size} bytes)")
                    except OSError:
                        lines.append(f"{prefix}{f}")
        else:
            try:
                entries = sorted(os.listdir(path_obj))
            except NotADirectoryError:
                return f"Error: Not a directory: {path}"

            for entry in entries:
                full = os.path.join(path_obj, entry)
                try:
                    if os.path.isdir(full):
                        lines.append(f"{entry}/")
                    else:
                        size = os.path.getsize(full)
                        lines.append(f"{entry}  ({size} bytes)")
                except OSError:
                    lines.append(entry)

        return "\n".join(lines) if lines else f"Directory is empty: {path}"


class FindTool(Tool):
    """Find files matching a glob pattern."""

    name = "find"
    description = (
        "Find files matching a glob pattern (e.g., '*.py'). "
        "Searches recursively from the given path. "
        "Use glob patterns, not regex."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root directory to search. Defaults to cwd.", "default": "."},
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', 'src/**/*.ts')."},
            "max_results": {"type": "integer", "description": "Maximum results. Default: 100.", "default": 100},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    async def execute(self, args: dict[str, Any]) -> str:
        path = args.get("path", ".")
        pattern = args["pattern"]
        max_results = args.get("max_results", 100)

        path_obj = Path(path)
        if not path_obj.exists():
            return f"Error: Path not found: {path}"

        matches = sorted(path_obj.rglob(pattern))[:max_results]
        return "\n".join(str(m) for m in matches) if matches else "No matches found."
