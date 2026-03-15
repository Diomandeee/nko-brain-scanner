#!/usr/bin/env python3
"""
Retry Utilities for N'Ko Video Analyzer

Provides robust retry logic with exponential backoff and jitter
for handling transient API failures and network issues.
"""

import asyncio
import random
import logging
from typing import TypeVar, Callable, Optional, Tuple, Type
from functools import wraps

T = TypeVar('T')

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, message: str, attempts: int, last_exception: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Execute an async function with exponential backoff retry.
    
    Args:
        func: Async function to execute (no arguments, use lambda for params)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)
        max_delay: Maximum delay between retries (seconds)
        jitter: Random jitter to add to delay (0 to jitter seconds)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(attempt, exception, delay) on each retry
        
    Returns:
        Result of the function call
        
    Raises:
        RetryError: If all retry attempts are exhausted
        
    Example:
        async def fetch_data():
            async with session.get(url) as resp:
                return await resp.json()
        
        result = await retry_with_backoff(fetch_data, max_retries=3)
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                # Final attempt failed
                raise RetryError(
                    f"Failed after {max_retries + 1} attempts: {e}",
                    attempts=attempt + 1,
                    last_exception=e,
                )
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter > 0:
                delay += random.uniform(0, jitter)
            
            # Log and callback
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            
            if on_retry:
                on_retry(attempt + 1, e, delay)
            
            await asyncio.sleep(delay)
    
    # Should never reach here, but just in case
    raise RetryError(
        f"Unexpected retry loop exit",
        attempts=max_retries + 1,
        last_exception=last_exception,
    )


def retry_decorator(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for adding retry logic to async functions.
    
    Example:
        @retry_decorator(max_retries=3, base_delay=1.0)
        async def fetch_api_data(url: str):
            async with session.get(url) as resp:
                return await resp.json()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exceptions=exceptions,
            )
        return wrapper
    return decorator


class RetryConfig:
    """Configuration class for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.exceptions = exceptions
    
    @classmethod
    def from_dict(cls, config: dict) -> 'RetryConfig':
        """Create RetryConfig from dictionary (e.g., from YAML)."""
        return cls(
            max_retries=config.get('max_retries', 3),
            base_delay=config.get('retry_base_delay', 2.0),
            max_delay=config.get('retry_max_delay', 60.0),
            jitter=config.get('retry_jitter', 1.0),
        )
    
    async def execute(self, func: Callable) -> T:
        """Execute a function with this retry configuration."""
        return await retry_with_backoff(
            func,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            jitter=self.jitter,
            exceptions=self.exceptions,
        )


# Predefined retry configurations for common scenarios
GEMINI_API_RETRY = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=60.0,
    jitter=1.0,
)

SUPABASE_RETRY = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
    jitter=0.5,
)

NETWORK_RETRY = RetryConfig(
    max_retries=3,
    base_delay=5.0,
    max_delay=120.0,
    jitter=2.0,
)


# Test
if __name__ == "__main__":
    import aiohttp
    
    async def test_retry():
        call_count = 0
        
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Simulated failure {call_count}")
            return f"Success on attempt {call_count}"
        
        try:
            result = await retry_with_backoff(
                flaky_function,
                max_retries=3,
                base_delay=0.1,  # Fast for testing
            )
            print(f"Result: {result}")
        except RetryError as e:
            print(f"All retries failed: {e}")
    
    asyncio.run(test_retry())

