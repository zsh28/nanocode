"""LLM client wrapper for OpenRouter API.

Provides a unified interface for LLM calls used by the RLM tools
and self-improvement modules.
"""

import logging
from typing import Any

from openai import AsyncOpenAI

from zsh28code.config import Config, load_config

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client wrapping OpenRouter's OpenAI-compatible API.

    Used by RLM tools (rlm_agent) and self-improvement modules
    for direct LLM calls that don't require the full agent loop.
    """

    def __init__(
        self,
        model: str = "poolside/laguna-s-2.1:free",
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key or ""
        self.base_url = base_url
        self.max_tokens = max_tokens

        self._client: AsyncOpenAI | None = None

    @classmethod
    def from_config(cls, config: Config | None = None) -> "LLMClient":
        """Create an LLMClient from a Config object."""
        if config is None:
            config = load_config()
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
        )

    def _get_client(self) -> AsyncOpenAI:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            headers = {
                "HTTP-Referer": "https://github.com/zeeshanaligulamhusein/nanocode",
                "X-Title": "zsh28code",
            }
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers=headers,
            )
        return self._client

    async def chat(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        system_prompt: str = "",
        stream: bool = False,
    ) -> str:
        """Send a chat request and return the response text.

        Args:
            prompt: The user message
            max_tokens: Max tokens in response (default: self.max_tokens)
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
            system_prompt: Optional system prompt
            stream: If True, returns streaming response

        Returns:
            The model's response text
        """
        client = self._get_client()
        tokens = max_tokens or self.max_tokens

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stream:
            return await self._chat_stream(client, kwargs)
        else:
            return await self._chat_complete(client, kwargs)

    async def _chat_complete(self, client: AsyncOpenAI, kwargs: dict) -> str:
        """Non-streaming chat completion."""
        try:
            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            raise

    async def _chat_stream(self, client: AsyncOpenAI, kwargs: dict) -> str:
        """Streaming chat completion — collects and returns full response."""
        full_response = ""
        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Print for interactive use
                    print(content, end="", flush=True)
            print()  # newline after streaming
            return full_response
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            raise

    @staticmethod
    def count_tokens(text: str) -> int:
        """Rough token count estimation."""
        return max(1, len(text) // 4)


__all__ = ["LLMClient"]
