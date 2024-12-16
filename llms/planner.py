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
from .errors import PlannerError, raise_for_status_code
from .rate_limiter import get_rate_limiter

class PlannerModule(BaseLLMModule):
    """Module for strategic planning and task breakdown."""
    
    async def create_plan(
        self,
        input_text: str,
        context: str,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> List[str]:
        """
        Create a structured plan based on the input and reasoning context.
        
        Args:
            input_text: The task to create a plan for
            context: Additional context and analysis
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            List[str]: Ordered list of plan steps
            
        Raises:
            PlannerError: If plan creation fails after retries
        """
        cache_enabled = (
            self.config.cache_config.enabled and 
            should_enable_caching(self.config.model)
        )
        
        # Create messages with caching
        messages = []
        if cache_enabled:
            system_msg = await create_cacheable_message(
                role="system",
                content="You are a strategic planner focused on breaking down tasks into actionable steps.",
                cache_large_content=self.config.cache_config.cache_system_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            user_msg = await create_cacheable_message(
                role="user",
                content=self._create_planning_prompt(input_text, context),
                cache_large_content=self.config.cache_config.cache_user_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            messages.extend([system_msg, user_msg])
        else:
            messages.extend([
                {"role": "system", "content": "You are a strategic planner focused on breaking down tasks into actionable steps."},
                {"role": "user", "content": self._create_planning_prompt(input_text, context)}
            ])
        
        # Check cache first
        cached_response = await get_cached_response(
            provider=self.config.provider,
            model=self.config.model,
            messages=messages
        )
        if cached_response:
            return self._parse_plan(cached_response)
        
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
                    error_prefix="Plan creation failed",
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
            finally:
                await rate_limiter.release(self.config.model)
            
            result = response.choices[0]['message'].content
            
            # Cache the response if appropriate
            if cache_enabled:
                await cache_response(
                    provider=self.config.provider,
                    model=self.config.model,
                    messages=messages,
                    response=result
                )
            
            return self._parse_plan(result)
            
        except Exception as e:
            raise PlannerError(f"Plan creation failed: {str(e)}") from e
    
    def _create_planning_prompt(self, input_text: str, context: str) -> str:
        """Create a detailed prompt for planning."""
        context_str = "\n".join([
            context,
            *[f"{k}: {v}" for k, v in self.context.items()]
        ])
        
        return f"""Create a detailed, step-by-step plan for the following task.
Consider:
1. Dependencies and prerequisites
2. Resource requirements
3. Potential challenges
4. Success criteria

Context:
{context_str}

Task:
{input_text}

Provide a numbered list of concrete steps:"""
    
    def _parse_plan(self, content: str) -> List[str]:
        """Parse the response content into a list of plan steps."""
        # Split into lines and clean up
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Extract numbered steps (1. Step one, 2. Step two, etc.)
        steps = []
        current_step = []
        
        for line in lines:
            # Check if line starts a new step
            if line[0].isdigit() or line[0] == '-':
                # Save previous step if exists
                if current_step:
                    steps.append(' '.join(current_step))
                    current_step = []
                
                # Add new step content
                step = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                current_step.append(step)
            else:
                # Continue previous step
                current_step.append(line)
        
        # Add last step if exists
        if current_step:
            steps.append(' '.join(current_step))
        
        # Validate steps
        if not steps:
            raise PlannerError("Failed to parse plan: No valid steps found")
        
        return steps
    
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