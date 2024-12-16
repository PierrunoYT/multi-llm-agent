from typing import Dict, List, Optional, Any, Union
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
from .errors import ReasoningError, raise_for_status_code
from .image_handler import ImageHandler
from .rate_limiter import get_rate_limiter

class ReasoningModule(BaseLLMModule):
    """Module for deep analysis and reasoning using LLMs."""
    
    async def analyze(
        self, 
        input_text: str, 
        image_paths: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> Union[str, Any]:
        """
        Analyze the input and provide detailed reasoning about it.
        Supports text, images, tool calls, and prompt caching.
        
        Args:
            input_text: The text to analyze
            image_paths: Optional list of paths to images to include
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            max_retries: Maximum number of retry attempts
            retry_delay: Delay in seconds between retries
            
        Returns:
            Either a string response, stream of chunks, or tool calls response
            
        Raises:
            ReasoningError: If the analysis fails after retries
        """
        messages = []
        cache_enabled = (
            self.config.cache_config.enabled and 
            should_enable_caching(self.config.model)
        )
        
        # Add system message with optional caching
        if self.config.extra_config.get("system_prompt"):
            system_message = await create_cacheable_message(
                role="system",
                content=self.config.extra_config["system_prompt"],
                cache_large_content=cache_enabled and self.config.cache_config.cache_system_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            messages.append(system_message)
        
        # Build user message with text and optional images
        if image_paths:
            try:
                content = [{"type": "text", "text": self._create_reasoning_prompt(input_text)}]
                for img_path in image_paths:
                    # Process and validate image
                    encoded_image = ImageHandler.encode_image(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_image
                        }
                    })
                
                if cache_enabled and self.config.cache_config.cache_user_messages:
                    content[0] = (await create_cacheable_message(
                        role="user",
                        content=content[0]["text"],
                        cache_large_content=True,
                        min_cache_size=self.config.cache_config.min_cache_size
                    ))["content"][0]
                messages.append({"role": "user", "content": content})
            except ValueError as e:
                raise ReasoningError(f"Image processing failed: {str(e)}") from e
        else:
            user_message = await create_cacheable_message(
                role="user",
                content=self._create_reasoning_prompt(input_text),
                cache_large_content=cache_enabled and self.config.cache_config.cache_user_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            messages.append(user_message)

        # Check cache first
        if not stream:
            cached_response = await get_cached_response(
                provider=self.config.provider,
                model=self.config.model,
                messages=messages
            )
            if cached_response:
                return cached_response

        # Prepare the completion request
        request_kwargs = {
            **self.config.to_request_params(),
            "messages": messages,
            "stream": stream,
            "extra_headers": {
                "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                "X-Title": self.config.extra_config.get("app_name", "")
            }
        }

        # Add tools if provided and using OpenRouter
        if tools and self.config.provider == "openrouter":
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        try:
            # Apply rate limiting
            rate_limiter = get_rate_limiter()
            await rate_limiter.acquire(self.config.model)
            
            try:
                response = await self._make_api_call_with_backoff(
                    request_kwargs=request_kwargs,
                    error_prefix="Analysis failed",
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
            finally:
                await rate_limiter.release(self.config.model)
            
            if stream:
                return response
            
            # Access the message content correctly from the validated response structure
            result = response.choices[0]['message'].content
            
            # Cache the response if appropriate
            if cache_enabled:
                await cache_response(
                    provider=self.config.provider,
                    model=self.config.model,
                    messages=messages,
                    response=result
                )
            
            # Handle tool calls if present
            if hasattr(response.choices[0]['message'], 'tool_calls') and response.choices[0]['message'].tool_calls:
                return {
                    'tool_calls': response.choices[0]['message'].tool_calls,
                    'content': result
                }
            
            return result
            
        except Exception as e:
            raise ReasoningError(f"Analysis failed: {str(e)}") from e
            
    def _create_reasoning_prompt(self, input_text: str) -> str:
        """Create the reasoning prompt with context."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        return f"Context:\n{context_str}\n\nAnalyze this: {input_text}"
    
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