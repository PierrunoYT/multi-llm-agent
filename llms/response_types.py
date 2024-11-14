from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

class ResponseUsage(BaseModel):
    """Token usage information for a response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Error(BaseModel):
    """Error information in a response."""
    code: int
    message: str

class FunctionCall(BaseModel):
    """Function call information."""
    name: str
    arguments: str  # JSON format arguments

class ToolCall(BaseModel):
    """Tool call information."""
    id: str
    type: str = "function"
    function: FunctionCall

class ResponseMessage(BaseModel):
    """Message content in a non-streaming response."""
    content: Optional[str]
    role: str
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None  # Deprecated, kept for compatibility

class ResponseDelta(BaseModel):
    """Delta content in a streaming response."""
    content: Optional[str]
    role: Optional[str]
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None  # Deprecated, kept for compatibility

class NonStreamingChoice(BaseModel):
    """Choice in a non-streaming response."""
    finish_reason: Optional[str]
    message: ResponseMessage
    error: Optional[Error] = None

class StreamingChoice(BaseModel):
    """Choice in a streaming response."""
    finish_reason: Optional[str]
    delta: ResponseDelta
    error: Optional[Error] = None

class NonChatChoice(BaseModel):
    """Choice in a non-chat response."""
    finish_reason: Optional[str]
    text: str
    error: Optional[Error] = None

Choice = Union[NonStreamingChoice, StreamingChoice, NonChatChoice]

class OpenRouterResponse(BaseModel):
    """Complete response from OpenRouter API."""
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: str = Field(..., pattern='^(chat.completion|chat.completion.chunk)$')
    system_fingerprint: Optional[str] = None
    usage: Optional[ResponseUsage] = None

class GenerationStats(BaseModel):
    """Statistics for a completed generation."""
    id: str
    model: str
    streamed: bool
    generation_time: int
    created_at: str
    tokens_prompt: int
    tokens_completion: int
    native_tokens_prompt: int
    native_tokens_completion: int
    num_media_prompt: Optional[int]
    num_media_completion: Optional[int]
    origin: str
    total_cost: float
    cache_discount: Optional[float]

async def parse_openrouter_response(response: Dict[str, Any]) -> OpenRouterResponse:
    """
    Parse and validate an OpenRouter API response.
    
    Args:
        response: Raw response dictionary from OpenRouter API
        
    Returns:
        Validated OpenRouterResponse object
    
    Raises:
        ValidationError: If response doesn't match expected schema
    """
    return OpenRouterResponse(**response)

async def parse_generation_stats(stats: Dict[str, Any]) -> GenerationStats:
    """
    Parse and validate generation statistics from OpenRouter API.
    
    Args:
        stats: Raw statistics dictionary from OpenRouter API
        
    Returns:
        Validated GenerationStats object
    
    Raises:
        ValidationError: If stats don't match expected schema
    """
    return GenerationStats(**stats)

async def handle_stream_chunk(chunk: str) -> Optional[OpenRouterResponse]:
    """
    Handle a chunk from an SSE stream.
    
    Args:
        chunk: Raw SSE chunk string
        
    Returns:
        OpenRouterResponse if chunk is valid JSON, None if it's a processing comment
        
    Raises:
        ValidationError: If chunk is invalid JSON or doesn't match schema
        ValueError: If chunk is neither valid JSON nor a processing comment
    """
    # Skip OpenRouter processing comments
    if chunk.startswith(": OPENROUTER PROCESSING"):
        return None
        
    # Parse and validate JSON chunk
    try:
        response_data = OpenRouterResponse.model_validate_json(chunk)
        return response_data
    except Exception as e:
        raise ValueError(f"Invalid stream chunk: {chunk}") from e
