from typing import Dict, Optional
import asyncio
import time
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    concurrent_requests: int = 10
    burst_size: int = 5

@dataclass
class RateLimiter:
    """Rate limiter implementation using token bucket algorithm."""
    config: RateLimitConfig
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _semaphore: asyncio.Semaphore = field(init=False)
    
    def __post_init__(self):
        self._tokens = float(self.config.burst_size)
        self._last_update = time.time()
        self._semaphore = asyncio.Semaphore(self.config.concurrent_requests)
    
    async def acquire(self) -> bool:
        """
        Acquire a rate limit token.
        
        Returns:
            bool: True if token acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            time_passed = now - self._last_update
            self._tokens = min(
                self.config.burst_size,
                self._tokens + time_passed * (self.config.requests_per_minute / 60.0)
            )
            self._last_update = now
            
            if self._tokens < 1.0:
                logger.warning("Rate limit exceeded")
                return False
            
            self._tokens -= 1.0
            return True
    
    async def __aenter__(self):
        """Acquire both rate limit token and concurrency semaphore."""
        while True:
            if await self.acquire():
                await self._semaphore.acquire()
                return self
            await asyncio.sleep(1.0)  # Wait before retrying
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release concurrency semaphore."""
        self._semaphore.release()

# Global rate limiters for different providers
RATE_LIMITERS: Dict[str, RateLimiter] = {
    "openai": RateLimiter(RateLimitConfig(
        requests_per_minute=60,
        concurrent_requests=10,
        burst_size=5
    )),
    "anthropic": RateLimiter(RateLimitConfig(
        requests_per_minute=120,
        concurrent_requests=20,
        burst_size=10
    )),
    "openrouter": RateLimiter(RateLimitConfig(
        requests_per_minute=90,
        concurrent_requests=15,
        burst_size=8
    ))
}

async def with_rate_limit(provider: str):
    """Get rate limiter for specific provider."""
    limiter = RATE_LIMITERS.get(provider)
    if not limiter:
        raise ValueError(f"No rate limiter configured for provider: {provider}")
    return limiter
