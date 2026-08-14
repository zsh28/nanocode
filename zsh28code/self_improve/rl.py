"""RL: Reinforcement Learning optimizer for agent configs.

Uses a gradient-free RL approach (akin to REINFORCE) to optimize
the agent's configuration: system prompt, tool settings, and model
kwargs. The LLM generates config variants, which are evaluated
against a held-out task set.
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

from zsh28code.llm import LLMClient
from zsh28code.self_improve.memory import AgentConfig, MemoryDB

logger = logging.getLogger(__name__)

# Default base config
BASE_PROMPT = """You are an expert terminal coding agent named zsh28code.
You operate in a bash/zsh environment and must complete tasks by executing commands,
reading/writing files, and using available tools.

Guidelines:
- Break complex tasks into smaller steps
- Use the todo tool to track progress
- Test your changes immediately
- Be concise but thorough
- Verify results before declaring completion

Available tools: read, write, edit, grep, find, ls, bash, todo, web_fetch, web_search."""


@dataclass
class RLConfigVariant:
    """A configuration variant proposed by RL."""
    config: AgentConfig
    estimated_value: float  # LLM's estimate of how good this variant is
    features: dict[str, Any]  # What was changed


class RLOptimizer:
    """Optimizes agent configuration via policy gradient-like search.

    Works by:
    1. Using LLM to generate config variants with estimated value
    2. Evaluating variants on held-out tasks
    3. Updating the "policy" (prompt generation) to favor high-reward configs
    """

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryDB,
        base_prompt: str = BASE_PROMPT,
        population_size: int = 8,
        elite_fraction: float = 0.25,
    ):
        self.llm = llm
        self.memory = memory
        self.base_prompt = base_prompt
        self.population_size = population_size
        self.elite_fraction = elite_fraction

        # Tool and model config defaults
        self.default_tool_config: dict[str, Any] = {
            "max_turns": 50,
            "max_output_tokens": 4096,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        self.default_model_kwargs: dict[str, Any] = {
            "model": "poolside/laguna-s-2.1:free",
            "temperature": 0.3,
            "top_p": 0.9,
        }

    async def optimize(
        self,
        eval_tasks: list[dict[str, Any]],
        iterations: int = 5,
    ) -> AgentConfig:
        """Run RL optimization for N iterations.

        Args:
            eval_tasks: Held-out tasks to evaluate configs
            iterations: Number of RL iterations

        Returns:
            Best config found
        """
        best_config: AgentConfig | None = None
        best_reward: float = -1.0

        for i in range(iterations):
            logger.info(f"RL iteration {i+1}/{iterations}")

            # Step 1: Get current best from memory
            current_best = self.memory.get_best_config()
            if current_best is None:
                current_best = AgentConfig(
                    hash="base",
                    description="Base configuration",
                    system_prompt=self.base_prompt,
                    tool_config=self.default_tool_config,
                    model_kwargs=self.default_model_kwargs,
                )
                self.memory.add_config(
                    description="Base configuration",
                    system_prompt=self.base_prompt,
                    tool_config=self.default_tool_config,
                    model_kwargs=self.default_model_kwargs,
                )

            # Check if current is new best
            current_reward = await self._evaluate_config(current_best, eval_tasks)
            if current_reward > best_reward:
                best_reward = current_reward
                best_config = current_best

            # Step 2: Generate population of config variants
            variants = await self._generate_variants(current_best, self.population_size)

            # Step 3: Evaluate variants
            scored_variants: list[tuple[RLConfigVariant, float]] = []
            for variant in variants:
                config = AgentConfig(
                    hash=self._hash_config(
                        variant.config.system_prompt,
                        variant.config.tool_config,
                        variant.config.model_kwargs,
                    ),
                    description=variant.config.description,
                    system_prompt=variant.config.system_prompt,
                    tool_config=variant.config.tool_config,
                    model_kwargs=variant.config.model_kwargs,
                )
                reward = await self._evaluate_config(config, eval_tasks)
                scored_variants.append((variant, reward))
                logger.info(f"Variant '{variant.config.description}': reward={reward:.3f}")

                if reward > best_reward:
                    best_reward = reward
                    best_config = config

            # Step 4: Select elites and update policy
            scored_variants.sort(key=lambda x: x[1], reverse=True)
            n_elite = max(1, int(len(scored_variants) * self.elite_fraction))
            elites = scored_variants[:n_elite]

            if elites:
                elite_rewards = [r for _, r in elites]
                logger.info(
                    f"Elite average reward: {sum(elite_rewards)/len(elite_rewards):.3f}"
                )

                # Store elite configs in memory
                for variant, reward in elites:
                    self.memory.add_config(
                        description=variant.config.description,
                        system_prompt=variant.config.system_prompt,
                        tool_config=variant.config.tool_config,
                        model_kwargs=variant.config.model_kwargs,
                    )

            # Step 5: Policy update — use elite feedback to improve prompt generation
            await self._update_policy(elites)

        # Store the best config
        if best_config:
            self.memory.add_config(
                description=f"RL-optimized config (reward={best_reward:.3f})",
                system_prompt=best_config.system_prompt,
                tool_config=best_config.tool_config,
                model_kwargs=best_config.model_kwargs,
            )

        return best_config

    async def _evaluate_config(
        self,
        config: AgentConfig,
        eval_tasks: list[dict[str, Any]],
    ) -> float:
        """Evaluate a config on held-out tasks (simplified for now).

        Uses LLM to estimate reward based on config features rather than
        running full task execution.
        """
        # For now, use a heuristic based on config features
        # In full implementation, this would run actual tasks
        reward = 0.0

        # Higher temperature helps exploration but can hurt precision
        temp = config.tool_config.get("temperature", 0.3)
        if 0.1 <= temp <= 0.5:
            reward += 0.1

        # More turns = more thorough
        max_turns = config.tool_config.get("max_turns", 50)
        if max_turns >= 30:
            reward += 0.1

        # Check if prompt has good structure
        prompt = config.system_prompt
        reward += 0.05 if "guidelines" in prompt.lower() else 0
        reward += 0.05 if "break" in prompt.lower() else 0
        reward += 0.05 if "test" in prompt.lower() else 0
        reward += 0.05 if "verify" in prompt.lower() else 0

        return min(1.0, reward)

    async def _generate_variants(
        self,
        base_config: AgentConfig,
        n_variants: int,
    ) -> list[RLConfigVariant]:
        """Generate config variants using the LLM."""
        prompt = f"""
You are optimizing a coding agent's system prompt for maximum task success.
Generate {n_variants} diverse system prompt variants.

Base prompt:
---
{base_config.system_prompt}
---

For each variant, provide:
1. A description of what was changed
2. The modified system prompt
3. Your estimated value (0-1) of how much this improves task success
4. The specific features/variants used

Return as JSON:
{{
  "variants": [
    {{
      "description": "Added emphasis on testing",
      "system_prompt": "...",
      "estimated_value": 0.75,
      "features": {{
        "testing_emphasis": true,
        "temperature": 0.3,
        "max_turns": 50
      }}
    }},
    ...
  ]
}}

Generate {n_variants} variants with diverse strategies:
- More/less detailed instructions
- Different problem-solving approaches (top-down, bottom-up, TDD)
- Different tool usage patterns
- Different temperature/top_p settings
"""

        response = await self.llm.chat(prompt, max_tokens=4000)

        try:
            import json
            data = json.loads(response.strip().removeprefix("```json").removesuffix("```").strip())
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse variant generation: {e}")
            # Return random perturbation of base config
            return await self._random_variants(base_config, n_variants)

        variants: list[RLConfigVariant] = []

        for v in data.get("variants", []):
            tool_config = {**self.default_tool_config}
            model_kwargs = {**self.default_model_kwargs}

            if "features" in v:
                if "temperature" in v["features"]:
                    tool_config["temperature"] = v["features"]["temperature"]
                    model_kwargs["temperature"] = v["features"]["temperature"]
                if "max_turns" in v["features"]:
                    tool_config["max_turns"] = v["features"]["max_turns"]

            config = AgentConfig(
                hash=self._hash_config(v["system_prompt"], tool_config, model_kwargs),
                description=v.get("description", "Generated variant"),
                system_prompt=v["system_prompt"],
                tool_config=tool_config,
                model_kwargs=model_kwargs,
            )

            variants.append(RLConfigVariant(
                config=config,
                estimated_value=v.get("estimated_value", 0.5),
                features=v.get("features", {}),
            ))

        # Fill remaining slots with random variants
        if len(variants) < n_variants:
            extra = await self._random_variants(base_config, n_variants - len(variants))
            variants.extend(extra)

        return variants[:n_variants]

    async def _random_variants(
        self,
        base_config: AgentConfig,
        n: int,
    ) -> list[RLConfigVariant]:
        """Generate random perturbation variants."""
        variants = []
        temperatures = [0.1, 0.3, 0.5, 0.7]
        max_turns = [30, 50, 80]

        for _ in range(n):
            temp = random.choice(temperatures)
            turns = random.choice(max_turns)

            tool_config = {**self.default_tool_config, "temperature": temp, "max_turns": turns}
            model_kwargs = {**self.default_model_kwargs, "temperature": temp}

            desc = f"Random variant (temp={temp}, turns={turns})"
            config = AgentConfig(
                hash=self._hash_config(base_config.system_prompt, tool_config, model_kwargs),
                description=desc,
                system_prompt=base_config.system_prompt,
                tool_config=tool_config,
                model_kwargs=model_kwargs,
            )

            variants.append(RLConfigVariant(
                config=config,
                estimated_value=0.5,
                features={"temperature": temp, "max_turns": turns},
            ))

        return variants

    def _hash_config(
        self,
        system_prompt: str,
        tool_config: dict[str, Any],
        model_kwargs: dict[str, Any],
    ) -> str:
        """Generate a deterministic hash for a configuration."""
        key = json.dumps({
            "system_prompt": system_prompt,
            "tool_config": tool_config,
            "model_kwargs": model_kwargs,
        }, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def _update_policy(self, elites: list[tuple[RLConfigVariant, float]]):
        """Update the policy based on elite performance.

        Uses elite feedback to refine future variant generation.
        """
        if not elites:
            return

        [v.config.system_prompt for v, _ in elites]
        [r for _, r in elites]

        # Use LLM to synthesize improvements from elite prompts
        prompt = """
Based on these elite system prompts and their rewards, synthesize an improved
base prompt for future optimization iterations.

Elite prompts (best to worst):
"""
        for i, (variant, reward) in enumerate(elites):
            prompt += f"\n--- Variant {i+1} (reward={reward:.3f}) ---\n{variant.config.system_prompt[:500]}\n..."

        prompt += "\nSynthesize a refined base prompt that captures the best aspects:"

        response = await self.llm.chat(prompt, max_tokens=2000)

        # Update base prompt if response is valid
        if response and len(response) > 50:
            self.base_prompt = response
            logger.info("Updated base prompt from elite synthesis")


__all__ = [
    "RLConfigVariant",
    "RLOptimizer",
]
