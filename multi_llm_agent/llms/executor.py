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

class ExecutorError(Exception):
    """Custom exception for executor module errors."""
    pass

class ExecutorModule:
    """
    Executor module responsible for generating specific actions and responses.
    Uses Claude-3.5-haiku through OpenRouter for efficient execution.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        
        if config.provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key,
                default_headers={
                    "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                    "X-Title": self.config.extra_config.get("app_name", "")
                }
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
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
            system_msg = create_cacheable_message(
                role="system",
                content="You are an execution engine focused on taking concrete actions based on plans.",
                cache_large_content=self.config.cache_config.cache_system_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            user_msg = create_cacheable_message(
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
                    raise ExecutorError(f"Execution failed: {error_message}")
                
                return response.choices[0].message.content
                
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
                
                raise ExecutorError(f"Execution failed: {error_message}") from e
        
        raise ExecutorError("Execution failed: Maximum retries exceeded")
    
    def _create_execution_prompt(self, plan: List[str], context: str) -> str:
        """Create a detailed prompt for the execution task."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        
        return f"""Execute the following plan based on the provided context and analysis.
Focus on generating specific, actionable outputs for each step.

Context:
{context_str}

Analysis:
{context}

Plan:
{plan_str}

Generate detailed execution steps or responses for this plan:"""
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add or update context information."""
        self.context.update(context)
