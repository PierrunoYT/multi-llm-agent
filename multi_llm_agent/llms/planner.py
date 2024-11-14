from typing import Dict, List, Optional, Any
from openai import OpenAI
import asyncio
from ..config import LLMConfig
from .cache_control import (
    create_cacheable_message,
    should_enable_caching,
    get_cache_pricing
)
from .errors import (
    handle_openrouter_error,
    should_retry_error,
    is_warmup_error
)

class PlannerError(Exception):
    """Custom exception for planner module errors."""
    pass

class PlannerModule:
    """
    Planner module responsible for breaking down tasks and creating action plans.
    Uses Claude-3 through OpenRouter for detailed planning.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        
        if config.provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key,
                default_headers={
                    "HTTP-Referer": config.extra_config.get("site_url", ""),
                    "X-Title": config.extra_config.get("app_name", "")
                }
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    async def create_plan(
        self,
        input_text: str,
        context: str,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> List[str]:
        """
        Create a structured plan based on the input and reasoning context.
        Returns a list of steps to execute.
        
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
            system_msg = create_cacheable_message(
                role="system",
                content="You are a strategic planner focused on breaking down tasks into actionable steps.",
                cache_large_content=self.config.cache_config.cache_system_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            user_msg = create_cacheable_message(
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
        
        # Make API call with retry logic
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**request_kwargs)
                
                # Check for generation errors
                if hasattr(response, "error"):
                    error_message = handle_openrouter_error(
                        response_data={"error": response.error},
                        status_code=200
                    )
                    if should_retry_error({"error": response.error}, 200):
                        if is_warmup_error({"error": response.error}):
                            await asyncio.sleep(retry_delay * (attempt + 1) * 2)
                        else:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    raise PlannerError(f"Plan creation failed: {error_message}")
                
                return self._parse_plan(response.choices[0].message.content)
                
            except Exception as e:
                # Extract error details
                status_code = getattr(e, "status_code", None)
                response_data = {}
                
                if hasattr(e, "response"):
                    if hasattr(e.response, "json"):
                        response_data = e.response.json()
                    elif isinstance(e.response, dict):
                        response_data = e.response
                
                # Format error message
                error_message = handle_openrouter_error(response_data, status_code)
                
                # Check if we should retry
                if attempt < max_retries and should_retry_error(response_data, status_code):
                    if is_warmup_error(response_data):
                        await asyncio.sleep(retry_delay * (attempt + 1) * 2)
                    else:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                
                raise PlannerError(f"Plan creation failed: {error_message}") from e
        
        raise PlannerError("Plan creation failed: Maximum retries exceeded")
    
    def _create_planning_prompt(self, input_text: str, context: str) -> str:
        """Create a detailed prompt for the planning task."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        
        return f"""Create a detailed, step-by-step plan for the following task.
Consider the analysis provided and break down the task into clear, actionable steps.

Context:
{context_str}

Additional Analysis:
{context}

Task:
{input_text}

Create a numbered list of steps to accomplish this task:"""
    
    def _parse_plan(self, response: str) -> List[str]:
        """Parse the response into a list of steps."""
        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Remove numbering and any other formatting
        steps = []
        for line in lines:
            # Remove common numbering patterns (1., 1), Step 1:, etc.)
            step = line.lstrip('0123456789.) ').strip()
            if step and not step.lower().startswith(('here', 'step')):
                steps.append(step)
        
        return steps
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add or update context information."""
        self.context.update(context)
