from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from config import LLMConfig
from .base import BaseLLMModule
from .cache_control import (
    create_cacheable_message,
    should_enable_caching,
    cache_response,
    get_cached_response
)
from .errors import ExecutorError, raise_for_status_code
from .rate_limiter import get_rate_limiter

class ExecutorModule(BaseLLMModule):
    """Module for executing plans and generating specific actions."""
    
    async def execute(
        self,
        plan: List[str],
        context: str,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> str:
        """
        Execute the plan and generate specific actions or responses.
        
        Args:
            plan: List of plan steps to execute
            context: Additional context and analysis
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            str: Execution details or response
            
        Raises:
            ExecutorError: If execution fails after retries
        """
        # Validate plan
        if not plan or not isinstance(plan, list):
            raise ExecutorError("Invalid plan: Must be a non-empty list of steps")
        
        cache_enabled = (
            self.config.cache_config.enabled and 
            should_enable_caching(self.config.model)
        )
        
        # Create messages with caching
        messages = []
        if cache_enabled:
            system_msg = await create_cacheable_message(
                role="system",
                content="You are an execution engine focused on taking concrete actions based on plans.",
                cache_large_content=self.config.cache_config.cache_system_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            user_msg = await create_cacheable_message(
                role="user",
                content=self._create_execution_prompt(plan, context),
                cache_large_content=self.config.cache_config.cache_user_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            messages.extend([system_msg, user_msg])
        else:
            messages.extend([
                {"role": "system", "content": "You are an execution engine focused on taking concrete actions based on plans."},
                {"role": "user", "content": self._create_execution_prompt(plan, context)}
            ])
        
        # Check cache first
        cached_response = await get_cached_response(
            provider=self.config.provider,
            model=self.config.model,
            messages=messages
        )
        if cached_response:
            return self._validate_execution_response(cached_response)
        
        # Prepare request parameters
        request_kwargs = {
            **self.config.to_request_params(),
            "messages": messages,
            "extra_headers": {
                "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                "X-Title": self.config.extra_config.get("app_name", "")
            }
        }
        
        try:
            # Apply rate limiting
            rate_limiter = get_rate_limiter()
            await rate_limiter.acquire(self.config.model)
            
            try:
                response = await self._make_api_call_with_backoff(
                    request_kwargs=request_kwargs,
                    error_prefix="Execution failed",
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
            finally:
                await rate_limiter.release(self.config.model)
            
            result = response.choices[0]['message'].content
            validated_result = self._validate_execution_response(result)
            
            # Cache the response if appropriate
            if cache_enabled:
                await cache_response(
                    provider=self.config.provider,
                    model=self.config.model,
                    messages=messages,
                    response=result
                )
            
            return validated_result
            
        except Exception as e:
            raise ExecutorError(f"Execution failed: {str(e)}") from e
    
    def _create_execution_prompt(self, plan: List[str], context: str) -> str:
        """Create a detailed prompt for execution."""
        context_str = "\n".join([
            context,
            *[f"{k}: {v}" for k, v in self.context.items()]
        ])
        
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        return f"""Generate specific actions or responses based on this plan and context.
Consider:
1. Required resources and dependencies
2. Error handling and edge cases
3. Success criteria and validation
4. User experience and clarity

Context:
{context_str}

Plan:
{plan_str}

Generate detailed execution steps or response:"""

    def _validate_execution_response(self, response: str) -> str:
        """Validate and clean up execution response."""
        if not response or not isinstance(response, str):
            raise ExecutorError("Invalid execution response: Must be a non-empty string")
        
        # Remove any potential harmful characters or sequences
        response = response.replace('\x00', '')  # Remove null bytes
        
        # Ensure response isn't too short
        if len(response.strip()) < 10:
            raise ExecutorError("Invalid execution response: Response too short")
            
        return response.strip()
    
    async def _execute_api_call(self, request_kwargs: dict) -> Any:
        """Execute API call to OpenRouter."""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                **request_kwargs.get("extra_headers", {})
            }
            
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": request_kwargs["model"],
                    "messages": request_kwargs["messages"],
                    "stream": request_kwargs.get("stream", False),
                    **{k: v for k, v in request_kwargs.items() if k not in ["extra_headers", "model", "messages", "stream"]}
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise_for_status_code(response.status, error_text)
                    
                return await response.json()
    
    async def _make_api_call_with_backoff(self, request_kwargs: dict, error_prefix: str, max_retries: int, retry_delay: float):
        """Make API call with exponential backoff retry strategy."""
        for attempt in range(max_retries):
            try:
                return await self._make_api_call(
                    request_kwargs=request_kwargs,
                    error_prefix=error_prefix,
                    max_retries=1,  # We handle retries here
                    retry_delay=0  # No delay in inner retry
                )
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise
                
                # Exponential backoff
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)