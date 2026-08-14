"""Shell execution tool."""

import os
import signal
import subprocess
import time
from typing import Any

from zsh28code.tools.base import Tool


class BashTool(Tool):
    """Execute a shell command."""

    name = "bash"
    description = (
        "Execute a shell command in the active workspace directory. "
        "The active workspace is the process cwd; begin with pwd when location matters. "
        "Returns stdout + stderr combined. "
        "Commands run with `set -o pipefail` to catch pipeline failures. "
        "Timeout is 60 seconds by default. "
        "Each command runs in a fresh subshell — cd/path changes are NOT persistent. "
        "To persist state, write to files or export env vars to files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds. Default: 60.", "default": 60},
        },
        "required": ["command"],
    }
    is_read_only = False

    async def execute(self, args: dict[str, Any]) -> str:
        command = f"set -o pipefail; {args['command']}"
        timeout = args.get("timeout", 60)

        start = time.time()
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                cwd=os.getcwd(),
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                else:
                    process.kill()
                stdout, _ = process.communicate()
                elapsed = time.time() - start
                return f"Error: Command timed out after {timeout}s (elapsed: {elapsed:.1f}s)\nOutput so far:\n{stdout}"

            elapsed = time.time() - start

            header = f"[exit={process.returncode} elapsed={elapsed:.1f}s]"
            result = f"{header}\n{stdout}" if stdout else header
            return result.strip()

        except Exception as e:
            return f"Error running command: {e}"
