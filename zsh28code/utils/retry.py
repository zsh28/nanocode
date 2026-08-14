"""Retry utilities for API calls."""

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


async def with_retry(
    coro_func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    **kwargs,
) -> any:
    """Execute a coroutine with exponential backoff retry.

    Args:
        coro_func: The async function to call
        *args: Positional arguments to pass to coro_func
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exception types to retry on (default: all)
        **kwargs: Keyword arguments to pass to coro_func

    Returns:
        The result of coro_func
    """
    delay = base_delay
    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except exceptions as e:
            last_exc = e
            if attempt < max_retries:
                actual_delay = min(delay, max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {actual_delay:.1f}s..."
                )
                await asyncio.sleep(actual_delay)
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

    raise last_exc  # type: ignore[misc]
