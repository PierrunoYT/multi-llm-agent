from typing import Dict, List, Optional, Any
import asyncio
from config import LLMConfig
from .base import BaseLLMModule
from .cache_control import (
    create_cacheable_message,
    should_enable_caching
)
from .errors import PlannerError

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
                error_prefix="Plan creation failed",
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            return self._parse_plan(response.choices[0].message.content)
            
        except Exception as e:
            raise PlannerError(str(e)) from e
    
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
        for line in lines:
            # Skip lines that don't look like steps
            if not line[0].isdigit() and line[0] != '-':
                continue
            
            # Remove number/bullet and clean up
            step = line.split('.', 1)[-1].split(')', 1)[-1].strip()
            if step:
                steps.append(step)
        
        return steps
