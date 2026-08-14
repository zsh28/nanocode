"""Configuration management for zsh28code."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for the agent."""

    model: str = "poolside/laguna-s-2.1:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    reasoning_effort: str = "high"
    max_tokens: int = 8192
    max_iterations: int = 100
    auto_approve: bool = False
    rlm_depth: int = 3

    # Large output threshold: outputs >= N chars go to ContextStore, not LLM input
    large_output_threshold: int = 4000

    # Token budget for summarizing recent context
    recent_summary_tokens: int = 2000

    # Project instructions file
    instructions_file: str = "AGENTS.md"

    # Self-improvement settings
    enable_self_improve: bool = True
    self_improve_after_tasks: int = 3  # run self-improvement every N tasks
    rl_batch_size: int = 5  # tasks per RL optimization cycle
    rl_min_reward_improvement: float = 0.05  # min improvement to accept change


def load_config() -> Config:
    """Load configuration from environment variables."""
    model = os.environ.get("ZSH28CODE_MODEL", Config().model)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENROUTER_BASE_URL", Config().base_url)

    return Config(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
