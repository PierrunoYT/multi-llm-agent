from typing import Dict, Optional, List, Union, Any
import openai
import anthropic
import base64
from pathlib import Path
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
import asyncio

class ReasoningError(Exception):
    """Custom exception for reasoning module errors."""
    pass

class ReasoningModule:
    """
    Reasoning module responsible for deep understanding and analysis.
    Uses sophisticated models through OpenRouter or direct provider APIs.
    Supports multimodal inputs, tool calling, and prompt caching.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.context: Dict[str, str] = {}
        
        if config.provider == "openai":
            self.client = openai.Client(api_key=config.api_key)
        elif config.provider == "openrouter":
            self.client = openai.Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=config.api_key,
                default_headers={
                    "HTTP-Referer": config.extra_config.get("site_url", ""),
                    "X-Title": config.extra_config.get("app_name", "MultiLLMAgent")
                }
            )
        elif config.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Convert an image file to base64 string with content type."""
        image_path = Path(image_path)
        content_type = f"image/{image_path.suffix[1:]}"  # Remove the dot from suffix
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            return f"data:{content_type};base64,{encoded}"

    async def analyze(
        self, 
        input_text: str, 
        image_paths: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> Union[str, openai.Stream, Dict[str, Any]]:
        """
        Analyze the input and provide detailed reasoning about it.
        Supports text, images, tool calls, and prompt caching.
        
        Args:
            input_text: The text to analyze
            image_paths: Optional list of paths to images to include
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            max_retries: Maximum number of retry attempts for retryable errors
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
            system_message = create_cacheable_message(
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
            # Cache the text part if it's large enough
            if cache_enabled and self.config.cache_config.cache_user_messages:
                content[0] = create_cacheable_message(
                    role="user",
                    content=content[0]["text"],
                    cache_large_content=True,
                    min_cache_size=self.config.cache_config.min_cache_size
                )["content"][0]
            messages.append({"role": "user", "content": content})
        else:
            # Create cacheable user message
            user_message = create_cacheable_message(
                role="user",
                content=self._create_reasoning_prompt(input_text),
                cache_large_content=cache_enabled and self.config.cache_config.cache_user_messages,
                min_cache_size=self.config.cache_config.min_cache_size
            )
            messages.append(user_message)

        # Prepare the completion request
        request_kwargs = {
            **self.config.to_request_params(),  # Include all configured parameters
            "messages": messages,
            "stream": stream
        }

        # Add tools if provided and using OpenRouter
        if tools and self.config.provider == "openrouter":
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        # Add provider-specific headers for OpenRouter
        if self.config.provider == "openrouter":
            request_kwargs["extra_headers"] = {
                "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                "X-Title": self.config.extra_config.get("app_name", "")
            }

        # Make the API call based on provider with retry logic
        for attempt in range(max_retries + 1):
            try:
                if self.config.provider in ["openai", "openrouter"]:
                    response = await self.client.chat.completions.create(**request_kwargs)
                    
                    # Check for generation errors (status 200 but error in body)
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
                        raise ReasoningError(f"Analysis failed: {error_message}")
                    
                    if stream:
                        return response
                    
                    # Handle tool calls if present
                    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                        return {
                            "tool_calls": response.choices[0].message.tool_calls,
                            "message_id": response.id,
                            "message": response.choices[0].message
                        }
                    return response.choices[0].message.content
                    
                elif self.config.provider == "anthropic":
                    response = await self.client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{
                            "role": "user",
                            "content": self._create_reasoning_prompt(input_text)
                        }]
                    )
                    return response.content[0].text
                    
            except Exception as e:
                # Extract error details if available
                status_code = getattr(e, "status_code", None)
                response_data = {}
                
                # Try to get response data from different error formats
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
                        # Increase delay for warmup errors
                        await asyncio.sleep(retry_delay * (attempt + 1) * 2)
                    else:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                
                raise ReasoningError(f"Analysis failed: {error_message}") from e
        
        raise ReasoningError("Analysis failed: Maximum retries exceeded")

    async def continue_tool_conversation(
        self,
        messages: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        stream: bool = False
    ) -> Union[str, openai.Stream]:
        """
        Continue a conversation after tool calls have been executed.
        
        Args:
            messages: Previous conversation messages
            tool_results: Results from tool executions
            stream: Whether to stream the response
            
        Returns:
            Either a string response or a stream of response chunks
        """
        if self.config.provider != "openrouter":
            raise ValueError("Tool conversation only supported with OpenRouter provider")

        # Add tool results to the conversation
        for result in tool_results:
            messages.append({
                "role": "tool",
                "content": str(result["content"]),
                "tool_call_id": result["tool_call_id"]
            })
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream,
            extra_headers={
                "HTTP-Referer": self.config.extra_config.get("site_url", ""),
                "X-Title": self.config.extra_config.get("app_name", "")
            }
        )

        if stream:
            return response
        return response.choices[0].message.content

    def _create_reasoning_prompt(self, input_text: str) -> str:
        """Create a detailed prompt for the reasoning task."""
        context_str = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        
        return f"""Analyze the following input in detail. Consider:
1. Key concepts and their relationships
2. Underlying assumptions and implications
3. Potential challenges or considerations
4. Relevant context and background information

Context:
{context_str}

Input:
{input_text}

Provide a detailed analysis:"""
    
    def add_context(self, context: Dict[str, str]) -> None:
        """Add or update context information."""
        self.context.update(context)
