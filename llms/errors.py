from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class BaseLLMError(Exception):
    """Base exception class for LLM module errors."""
    pass

class ReasoningError(BaseLLMError):
    """Exception raised for errors in the reasoning module."""
    pass

class PlannerError(BaseLLMError):
    """Exception raised for errors in the planner module."""
    pass

class ExecutorError(BaseLLMError):
    """Exception raised for errors in the executor module."""
    pass

class ModerationErrorMetadata(BaseModel):
    """Metadata for moderation errors from OpenRouter."""
    reasons: List[str] = Field(description="Why the input was flagged")
    flagged_input: str = Field(description="The text segment that was flagged (truncated to 100 chars)")

class OpenRouterError(BaseModel):
    """OpenRouter API error response model."""
    code: int = Field(description="HTTP status code for the error")
    message: str = Field(description="Descriptive error message")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error metadata, used for moderation errors"
    )

class ErrorResponse(BaseModel):
    """Full error response from OpenRouter."""
    error: OpenRouterError

def handle_openrouter_error(response_data: Dict[str, Any], status_code: Optional[int] = None) -> str:
    """
    Handle OpenRouter API errors and return a user-friendly error message.
    
    Args:
        response_data: JSON response data from the API
        status_code: Optional HTTP status code from the response
        
    Returns:
        str: User-friendly error message
    """
    # Check if this is a content generation error (status 200 but error in body)
    is_generation_error = status_code == 200 and "error" in response_data
    
    # Parse the error response
    try:
        error_response = ErrorResponse(error=response_data.get("error", {}))
    except Exception:
        # Fallback for unexpected error format
        if is_generation_error:
            return f"Error during content generation: {str(response_data)}"
        return f"Unexpected error (Status {status_code}): {str(response_data)}"
    
    # Get the error code, using response body code for generation errors
    error_code = error_response.error.code if is_generation_error else (status_code or 500)
    
    # Handle specific error codes
    error_messages = {
        400: "Invalid request parameters or CORS issue",
        401: "Invalid API credentials or expired OAuth session",
        402: "Insufficient credits - please add more credits to continue",
        403: _format_moderation_error(error_response.error.metadata),
        408: "Request timed out - the model took too long to respond",
        429: "Rate limited - please wait before making more requests",
        502: "Model provider is currently unavailable or returned invalid response",
        503: "No available model provider meets the routing requirements"
    }
    
    base_message = error_messages.get(
        error_code,
        "Error during content generation" if is_generation_error else f"Unexpected error (Code {error_code})"
    )
    
    # Add the API's error message if available
    if error_response.error.message:
        return f"{base_message}: {error_response.error.message}"
    return base_message

def _format_moderation_error(metadata: Optional[Dict[str, Any]]) -> str:
    """Format moderation error with metadata if available."""
    if not metadata:
        return "Content was flagged by moderation"
        
    try:
        mod_data = ModerationErrorMetadata(**metadata)
        reasons = ", ".join(mod_data.reasons)
        return (
            f"Content was flagged by moderation for: {reasons}\n"
            f"Flagged content: {mod_data.flagged_input}"
        )
    except Exception:
        return "Content was flagged by moderation (details unavailable)"

def is_warmup_error(response_data: Dict[str, Any]) -> bool:
    """
    Check if the error indicates a model warmup issue.
    
    Args:
        response_data: JSON response data from the API
        
    Returns:
        bool: True if this appears to be a warmup-related error
    """
    # Check both error message and status 200 with no content
    if not response_data.get("error"):
        return False
        
    error_msg = response_data["error"].get("message", "").lower()
    warmup_indicators = [
        "warming up",
        "cold start",
        "scaling up",
        "no content generated",
        "try again",
        "retry"
    ]
    
    return any(indicator in error_msg for indicator in warmup_indicators)

def should_retry_error(response_data: Dict[str, Any], status_code: Optional[int] = None) -> bool:
    """
    Determine if an error should trigger a retry attempt.
    
    Args:
        response_data: JSON response data from the API
        status_code: Optional HTTP status code from the response
        
    Returns:
        bool: True if the request should be retried
    """
    # Always retry warmup errors
    if is_warmup_error(response_data):
        return True
        
    # Don't retry if no status code (unexpected error)
    if status_code is None:
        return False
        
    # Don't retry client errors except timeout
    if status_code < 500 and status_code != 408:
        return False
        
    # Retry server errors
    return status_code in [502, 503]
