from typing import Dict, Any, Optional, List
import asyncio
import logging
from openai import OpenAI
import anthropic
from config import LLMConfig
from .errors import (
    handle_openrouter_error,
    should_retry_error,
    is_warmup_error,
    BaseLLMError
)
from .cache_control import (
    create_cacheable_message,
    should_enable_caching
)
from .rate_limiter import with_rate_limit  # Import the rate limiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLMModule:
    """Base class for LLM modules with common functionality."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        self._validate_config()
        self.client = self._initialize_client()
        
    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not self.config.api_key:
            raise ValueError("API key is required")
        if self.config.provider not in ["openai", "openrouter", "anthropic"]:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
            
    def _initialize_client(self) -> Any:
        """Initialize the appropriate client based on provider."""
        if self.config.provider == "openai":
            return OpenAI(api_key=self.config.api_key)
        elif self.config.provider == "openrouter":
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.config.api_key,
                default_headers={
                    "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                    "X-Title": self.config.extra_config.get("app_name", "")
                }
            )
        elif self.config.provider == "anthropic":
            return anthropic.Anthropic(api_key=self.config.api_key)
        
    def add_context(self, context: Dict[str, str]) -> None:
        """Add additional context."""
        self.context.update(context)
        
    async def _make_api_call(
        self,
        request_kwargs: Dict[str, Any],
        error_prefix: str,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> Any:
        """Make an API call with retry logic and rate limiting."""
        async with await with_rate_limit(self.config.provider) as limiter:
            for attempt in range(max_retries + 1):
                try:
                    logger.debug(f"Making API call attempt {attempt + 1}/{max_retries + 1}")
                    # Use sync client since OpenAI's async client is not working properly
                    response = self.client.chat.completions.create(**request_kwargs)
                    
                    # Check for generation errors
                    if hasattr(response, "error"):
                        error_message = handle_openrouter_error(
                            response_data={"error": response.error},
                            status_code=200
                        )
                        if should_retry_error({"error": response.error}, 200):
                            await self._handle_retry(attempt, retry_delay, is_warmup=is_warmup_error({"error": response.error}))
                            continue
                        raise BaseLLMError(f"{error_prefix}: {error_message}")
                    
                    return response
                    
                except Exception as e:
                    status_code = getattr(e, "status_code", None)
                    response_data = self._extract_error_response(e)
                    error_message = handle_openrouter_error(response_data, status_code)
                    
                    if attempt < max_retries and should_retry_error(response_data, status_code):
                        await self._handle_retry(attempt, retry_delay, is_warmup=is_warmup_error(response_data))
                        continue
                    
                    raise BaseLLMError(f"{error_prefix}: {error_message}") from e
            
            raise BaseLLMError(f"{error_prefix}: Maximum retries exceeded")
    
    async def _handle_retry(self, attempt: int, retry_delay: float, is_warmup: bool = False) -> None:
        """Handle retry delay with logging."""
        delay = retry_delay * (attempt + 1) * (2 if is_warmup else 1)
        logger.info(f"Retrying after {delay:.1f}s (attempt {attempt + 1})")
        await asyncio.sleep(delay)
    
    def _extract_error_response(self, error: Exception) -> Dict[str, Any]:
        """Extract error response data from an exception."""
        if hasattr(error, "response"):
            if hasattr(error.response, "json"):
                return error.response.json()
            elif isinstance(error.response, dict):
                return error.response
        return {}
