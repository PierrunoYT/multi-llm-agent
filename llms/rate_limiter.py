from typing import Dict, Optional
import asyncio
import time
from dataclasses import dataclass

@dataclass
class RateLimit:
    requests_per_minute: int
    concurrent_requests: int

class RateLimiter:
    """Rate limiter for API requests with concurrent request limiting."""
    
    def __init__(self, rate_limits: Dict[str, RateLimit]):
        self._rate_limits = rate_limits
        self._request_times: Dict[str, list] = {model: [] for model in rate_limits}
        self._locks: Dict[str, asyncio.Semaphore] = {
            model: asyncio.Semaphore(limit.concurrent_requests)
            for model, limit in rate_limits.items()
        }
    
    async def acquire(self, model: str):
        """Acquire permission to make a request."""
        rate_limit = self._rate_limits.get(model)
        if not rate_limit:
            return  # No rate limit for this model
            
        # Wait for concurrent request slot
        async with self._locks[model]:
            # Clean old request times
            current_time = time.time()
            self._request_times[model] = [
                t for t in self._request_times[model]
                if current_time - t < 60  # Keep last minute
            ]
            
            # If at rate limit, wait until we can make another request
            if len(self._request_times[model]) >= rate_limit.requests_per_minute:
                wait_time = 60 - (current_time - self._request_times[model][0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Add current request
            self._request_times[model].append(time.time())
    
    async def release(self, model: str):
        """Release the rate limit lock."""
        if model in self._locks:
            self._locks[model].release()

# Default rate limits for different models
DEFAULT_RATE_LIMITS = {
    "openai/o1-preview": RateLimit(
        requests_per_minute=60,
        concurrent_requests=5
    ),
    "anthropic/claude-3.5-sonnet:beta": RateLimit(
        requests_per_minute=50,
        concurrent_requests=3
    ),
    "anthropic/claude-3-5-haiku:beta": RateLimit(
        requests_per_minute=50,
        concurrent_requests=3
    )
}

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(DEFAULT_RATE_LIMITS)
    return _rate_limiter
