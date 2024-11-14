from typing import Dict, List, Optional, Any, Union
import asyncio
from config import LLMConfig
from .base import BaseLLMModule
from .cache_control import (
    create_cacheable_message,
    should_enable_caching
)
from .errors import ReasoningError

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
            content = [{"type": "text", "text": self._create_reasoning_prompt(input_text)}]
            for img_path in image_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image(img_path)
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
        else:
            user_message = await create_cacheable_message(
                role="user",
                content=self._create_reasoning_prompt(input_text),
                cache_large_content=cache_enabled and self.config.cache_config.cache_user_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            messages.append(user_message)

        # Prepare the completion request
        request_kwargs = {
            **self.config.to_request_params(),
            "messages": messages,
            "stream": stream
        }

        # Add tools if provided and using OpenRouter
        if tools and self.config.provider == "openrouter":
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        try:
            response = await self._make_api_call(
                request_kwargs=request_kwargs,
                error_prefix="Analysis failed",
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            if stream:
                return response
            
            # Handle tool calls if present
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                return {
                    'tool_calls': response.choices[0].message.tool_calls,
                    'content': response.choices[0].message.content
                }
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise ReasoningError(str(e)) from e
            
    def _create_reasoning_prompt(self, input_text: str) -> str:
        """Create the reasoning prompt with context."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        return f"Context:\n{context_str}\n\nAnalyze this: {input_text}"
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image for API request."""
        # Implementation depends on your image handling needs
        raise NotImplementedError("Image encoding not implemented")
