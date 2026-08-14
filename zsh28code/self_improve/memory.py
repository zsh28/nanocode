"""SQLite-backed memory for self-improvement tracking.

Stores improvements, task results, configurations, and scores so the
agent can learn from past performance and recursively improve itself.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Improvement:
    """A recorded code/prompt improvement."""
    id: str
    source_file: str
    patch: str
    description: str
    created_at: str
    applied_at: str | None = None
    parent_id: str | None = None
    depth: int = 0
    score_delta: float = 0.0


@dataclass
class TaskResult:
    """Result of running a task."""
    task_name: str
    success: bool
    reward: float
    config_hash: str
    trajectory_path: str | None
    created_at: str
    iterations: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class AgentConfig:
    """A configuration variant."""
    hash: str
    description: str
    system_prompt: str
    tool_config: dict[str, Any]
    model_kwargs: dict[str, Any]


class MemoryDB:
    """Persistent memory database for self-improvement."""

    def __init__(self, db_path: str = "~/.zsh28code/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS improvements (
                id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                patch TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                parent_id TEXT,
                depth INTEGER DEFAULT 0,
                score_delta REAL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS task_results (
                task_name TEXT NOT NULL,
                success INTEGER NOT NULL,
                reward REAL NOT NULL,
                config_hash TEXT NOT NULL,
                trajectory_path TEXT,
                created_at TEXT NOT NULL,
                iterations INTEGER DEFAULT 0,
                elapsed_seconds REAL DEFAULT 0.0,
                PRIMARY KEY (task_name, created_at)
            );
            CREATE TABLE IF NOT EXISTS configs (
                hash TEXT PRIMARY KEY,
                description TEXT,
                system_prompt TEXT,
                tool_config TEXT,
                model_kwargs TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS improvement_scores (
                improvement_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                score_delta REAL NOT NULL,
                FOREIGN KEY (improvement_id) REFERENCES improvements(id)
            );
            CREATE TABLE IF NOT EXISTS rlm_trajectories (
                id TEXT PRIMARY KEY,
                task_name TEXT,
                config_hash TEXT,
                trajectory_path TEXT,
                reward REAL,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def _hash_config(system_prompt: str, tool_config: dict, model_kwargs: dict) -> str:
        """Generate a deterministic hash for a configuration."""
        key = json.dumps({
            "system_prompt": system_prompt,
            "tool_config": tool_config,
            "model_kwargs": model_kwargs,
        }, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # --- Improvements ---

    def add_improvement(
        self,
        source_file: str,
        patch: str,
        description: str,
        parent_id: str | None = None,
        depth: int = 0,
    ) -> Improvement:
        """Record a new improvement."""
        imp_id = hashlib.sha256(
            f"{source_file}:{patch[:100]}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        now = datetime.now(timezone.utc).isoformat()

        improvement = Improvement(
            id=imp_id,
            source_file=source_file,
            patch=patch,
            description=description,
            created_at=now,
            parent_id=parent_id,
            depth=depth,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO improvements (id, source_file, patch, description, created_at, parent_id, depth)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (imp_id, source_file, patch, description, now, parent_id, depth))
        conn.commit()
        conn.close()

        return improvement

    def get_improvements(self, limit: int = 100) -> list[Improvement]:
        """Retrieve improvements, most recent first."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM improvements ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        return [Improvement(**dict(row)) for row in rows]

    def get_improvements_for_file(self, source_file: str) -> list[Improvement]:
        """Get all improvements applied to a specific file."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM improvements WHERE source_file = ? ORDER BY created_at DESC
        """, (source_file,)).fetchall()
        conn.close()

        return [Improvement(**dict(row)) for row in rows]

    def mark_improvement_applied(self, imp_id: str, score_delta: float = 0.0):
        """Mark an improvement as applied with a score delta."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE improvements
            SET applied_at = ?, score_delta = ?
            WHERE id = ?
        """, (now, score_delta, imp_id))
        conn.commit()
        conn.close()

    # --- Task Results ---

    def add_task_result(
        self,
        task_name: str,
        success: bool,
        reward: float,
        config_hash: str,
        trajectory_path: str | None = None,
        iterations: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> None:
        """Record a task result."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO task_results (task_name, success, reward, config_hash, trajectory_path, created_at, iterations, elapsed_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (task_name, int(success), reward, config_hash, trajectory_path, now, iterations, elapsed_seconds))
        conn.commit()
        conn.close()

    def get_task_results(self, limit: int = 100) -> list[TaskResult]:
        """Get task results, most recent first."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM task_results ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        return [TaskResult(**dict(row)) for row in rows]

    def get_task_performance(self, task_name: str) -> list[TaskResult]:
        """Get all results for a specific task."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM task_results WHERE task_name = ? ORDER BY created_at DESC
        """, (task_name,)).fetchall()
        conn.close()

        return [TaskResult(**dict(row)) for row in rows]

    def get_success_rate(self) -> float:
        """Get overall success rate."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT AVG(CAST(success AS REAL)) as rate FROM task_results
        """).fetchone()
        conn.close()
        return row["rate"] if row and row["rate"] is not None else 0.0

    # --- Configs ---

    def add_config(
        self,
        description: str,
        system_prompt: str,
        tool_config: dict[str, Any],
        model_kwargs: dict[str, Any],
    ) -> AgentConfig:
        """Record a configuration variant."""
        config_hash = self._hash_config(system_prompt, tool_config, model_kwargs)
        now = datetime.now(timezone.utc).isoformat()

        config = AgentConfig(
            hash=config_hash,
            description=description,
            system_prompt=system_prompt,
            tool_config=tool_config,
            model_kwargs=model_kwargs,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO configs (hash, description, system_prompt, tool_config, model_kwargs, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (config_hash, description, system_prompt,
              json.dumps(tool_config), json.dumps(model_kwargs), now))
        conn.commit()
        conn.close()

        return config

    def get_best_config(self) -> AgentConfig | None:
        """Get the configuration with the highest average reward."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT c.*, AVG(tr.reward) as avg_reward
            FROM configs c
            JOIN task_results tr ON c.hash = tr.config_hash
            GROUP BY c.hash
            ORDER BY avg_reward DESC
            LIMIT 1
        """).fetchone()
        conn.close()

        if not row:
            return None

        return AgentConfig(
            hash=row["hash"],
            description=row["description"],
            system_prompt=row["system_prompt"],
            tool_config=json.loads(row["tool_config"]),
            model_kwargs=json.loads(row["model_kwargs"]),
        )

    def get_recent_rewards(self, limit: int = 20) -> list[float]:
        """Get recent task rewards as a list."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT reward FROM task_results ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [row[0] for row in rows]


__all__ = [
    "Improvement",
    "TaskResult",
    "AgentConfig",
    "MemoryDB",
]
