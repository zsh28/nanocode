"""Harbor adapter for running zsh28code on Terminal-Bench."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class Zsh28Code(BaseInstalledAgent):
    """Install the local wheel and run the headless zsh28code CLI."""

    SUPPORTS_ATIF = True
    SUPPORTS_RESUME = False
    SUPPORTS_CONFIG = True

    def __init__(
        self,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens

    @staticmethod
    @override
    def name() -> str:
        return "zsh28code"

    @override
    def version(self) -> str | None:
        return getattr(self, "_version", None)

    @override
    def get_version_command(self) -> str | None:
        return "zsh28code --version"

    @override
    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as wheel_dir:
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", wheel_dir, str(project_root)],
                check=True,
            )
            wheel = next(Path(wheel_dir).glob("zsh28code-*.whl"))
            remote_wheel = f"/tmp/{wheel.name}"
            await environment.upload_file(wheel, remote_wheel)

        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && apt-get install -y python3 python3-pip python3-venv && "
                "python3 -m venv /opt/zsh28code && "
                f"/opt/zsh28code/bin/pip install {remote_wheel} && "
                "ln -sf /opt/zsh28code/bin/zsh28code /usr/local/bin/zsh28code && "
                "zsh28code --version"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

    @property
    def _trajectory_path(self) -> str:
        from harbor.models.trial.paths import EnvironmentPaths

        return (EnvironmentPaths.agent_dir / "zsh28code.trajectory.json").as_posix()

    @with_prompt_template
    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        env = {
            "OPENROUTER_API_KEY": self._get_env("OPENROUTER_API_KEY") or "",
            "OPENROUTER_BASE_URL": self._get_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
        }
        if self._get_env("FIRECRAWL_API_KEY"):
            env["FIRECRAWL_API_KEY"] = self._get_env("FIRECRAWL_API_KEY") or ""

        model = self.model_name or "poolside/laguna-s-2.1:free"
        model = model.removeprefix("openrouter/")
        args = ["--headless", "--model", shlex.quote(model), "--max-iterations", "50", "--max-tokens", str(self._max_tokens or 16384), "--output", shlex.quote(self._trajectory_path)]
        if self._reasoning_effort:
            args.extend(["--reasoning-effort", shlex.quote(self._reasoning_effort)])

        from harbor.models.trial.paths import EnvironmentPaths

        command = f"zsh28code {' '.join(args)} {shlex.quote(instruction)} 2>&1 | tee {EnvironmentPaths.agent_dir.as_posix()}/zsh28code.txt"
        await self.exec_as_agent(environment, command=command, env=env)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        path = self.logs_dir / "zsh28code.trajectory.json"
        if not path.exists():
            return
        try:
            trajectory = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        context.n_input_tokens = sum(len(str(e.get("content", ""))) // 4 for e in trajectory if e.get("role") in ("system", "user"))
        context.n_output_tokens = sum(len(str(e.get("content", ""))) // 4 for e in trajectory if e.get("role") == "assistant")
        context.n_cache_tokens = 0
        from zsh28code.benchmark.trajectory import convert_to_atif

        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(convert_to_atif(trajectory, self.session_id or str(uuid.uuid4())), indent=2)
        )
