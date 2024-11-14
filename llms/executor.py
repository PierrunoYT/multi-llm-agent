from typing import Dict, List, Optional, Any
import asyncio
from config import LLMConfig
from .base import BaseLLMModule
from .cache_control import (
    create_cacheable_message,
    should_enable_caching,
    cache_response,
    get_cached_response
)
from .errors import ExecutorError

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
            return cached_response
        
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
            response = await self._make_api_call(
                request_kwargs=request_kwargs,
                error_prefix="Execution failed",
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            result = response.choices[0].message.content
            
            # Cache the response if appropriate
            if cache_enabled:
                await cache_response(
                    provider=self.config.provider,
                    model=self.config.model,
                    messages=messages,
                    response=result
                )
            
            return result
            
        except Exception as e:
            raise ExecutorError(str(e)) from e
    
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
