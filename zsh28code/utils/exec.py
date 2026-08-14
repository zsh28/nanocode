"""Subprocess execution with process-group kill."""

import os
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass
class ExecResult:
    stdout: str
    returncode: int
    elapsed: float
    timed_out: bool = False


def run_command(
    command: str,
    cwd: str | None = None,
    env: dict | None = None,
    timeout: int = 60,
) -> ExecResult:
    """Execute a shell command with process-group kill on timeout.

    Unlike subprocess.run with timeout, this kills the entire process group,
    ensuring no child processes are orphaned.

    Args:
        command: Shell command to execute
        cwd: Working directory (default: current dir)
        env: Environment variables (default: os.environ)
        timeout: Timeout in seconds (default: 60)

    Returns:
        ExecResult with stdout, returncode, elapsed time
    """
    start = time.time()

    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        cwd=cwd or os.getcwd(),
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env or os.environ.copy(),
        start_new_session=True,  # creates new process group on POSIX
    )

    try:
        stdout, _ = process.communicate(timeout=timeout)
        elapsed = time.time() - start
        return ExecResult(
            stdout=stdout or "",
            returncode=process.returncode,
            elapsed=elapsed,
        )
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        if os.name == "posix" and process.pid:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    process.kill()
                except Exception:
                    pass
        else:
            try:
                process.kill()
            except Exception:
                pass

        stdout, _ = process.communicate()
        elapsed = time.time() - start
        return ExecResult(
            stdout=stdout or "",
            returncode=-1,
            elapsed=elapsed,
            timed_out=True,
        )
