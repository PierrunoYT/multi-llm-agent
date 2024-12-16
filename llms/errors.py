class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class ValidationError(LLMError):
    """Exception raised for validation errors."""
    pass

class RateLimitError(LLMError):
    """Exception raised when rate limits are exceeded."""
    pass

class APIError(LLMError):
    """Exception raised for API-related errors."""
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class CacheError(LLMError):
    """Exception raised for caching-related errors."""
    pass

class ConfigurationError(LLMError):
    """Exception raised for configuration-related errors."""
    pass

class ReasoningError(LLMError):
    """Exception raised for reasoning module errors."""
    pass

class PlannerError(LLMError):
    """Exception raised for planner module errors."""
    pass

class ExecutorError(LLMError):
    """Exception raised for executor module errors."""
    pass

class ImageProcessingError(LLMError):
    """Exception raised for image processing errors."""
    pass

class RetryError(LLMError):
    """Exception raised when all retry attempts fail."""
    def __init__(self, message: str, attempts: int, last_error: Exception = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error

class AuthenticationError(APIError):
    """Exception raised for authentication failures."""
    pass

class QuotaExceededError(APIError):
    """Exception raised when API quota is exceeded."""
    pass

class InvalidRequestError(APIError):
    """Exception raised for malformed requests."""
    pass

class ServiceUnavailableError(APIError):
    """Exception raised when the API service is unavailable."""
    pass

def raise_for_status_code(status_code: int, response_text: str = None) -> None:
    """
    Raise appropriate exception based on HTTP status code.
    
    Args:
        status_code: HTTP status code
        response_text: Optional response text for error details
        
    Raises:
        Appropriate APIError subclass based on status code
    """
    error_map = {
        400: InvalidRequestError("Invalid request", status_code, response_text),
        401: AuthenticationError("Authentication failed", status_code, response_text),
        403: AuthenticationError("Permission denied", status_code, response_text),
        429: QuotaExceededError("Rate limit exceeded", status_code, response_text),
        500: ServiceUnavailableError("Internal server error", status_code, response_text),
        502: ServiceUnavailableError("Bad gateway", status_code, response_text),
        503: ServiceUnavailableError("Service unavailable", status_code, response_text),
        504: ServiceUnavailableError("Gateway timeout", status_code, response_text)
    }
    
    if status_code >= 400:
        error = error_map.get(
            status_code,
            APIError(f"HTTP {status_code} error", status_code, response_text)
        )
        raise error
