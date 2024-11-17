from typing import Dict, Any, Optional
import asyncio
import json
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError
from config import LLMConfig
from .errors import LLMError, ValidationError as LLMValidationError

class APIResponse(BaseModel):
    """Validated API response structure."""
    choices: list
    model: str
    usage: Optional[Dict[str, int]]

class MessageContent(BaseModel):
    """Validated message content structure."""
    content: str
    role: str
    tool_calls: Optional[list] = None

class BaseLLMModule(ABC):
    """Base class for LLM modules with shared functionality."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM module with configuration."""
        self.config = config
        self.context: Dict[str, str] = {}
        self._validate_config()
    
    def _validate_config(self):
        """Validate the module configuration."""
        if not self.config.api_key:
            raise LLMValidationError("API key is required")
        if not self.config.model:
            raise LLMValidationError("Model name is required")
            
        # Validate rate limits
        if hasattr(self.config, 'rate_limit'):
            if self.config.rate_limit.requests_per_minute < 1:
                raise LLMValidationError("Requests per minute must be at least 1")
            if self.config.rate_limit.concurrent_requests < 1:
                raise LLMValidationError("Concurrent requests must be at least 1")
    
    async def _make_api_call(
        self,
        request_kwargs: dict,
        error_prefix: str,
        max_retries: int = 1,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Make an API call with validation and error handling.
        
        Args:
            request_kwargs: API request parameters
            error_prefix: Prefix for error messages
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Validated API response
            
        Raises:
            LLMError: If the API call fails
        """
        try:
            # Validate request parameters
            self._validate_request_params(request_kwargs)
            
            # Make the API call (implementation specific to provider)
            response = await self._execute_api_call(request_kwargs)
            
            # Validate response
            return self._validate_response(response)
            
        except ValidationError as e:
            raise LLMValidationError(f"{error_prefix}: {str(e)}")
        except Exception as e:
            raise LLMError(f"{error_prefix}: {str(e)}")
    
    def _validate_request_params(self, params: dict):
        """Validate API request parameters."""
        required_fields = {'messages', 'model'}
        missing_fields = required_fields - set(params.keys())
        if missing_fields:
            raise LLMValidationError(f"Missing required fields: {missing_fields}")
            
        # Validate messages structure
        if not isinstance(params['messages'], list):
            raise LLMValidationError("Messages must be a list")
        if not params['messages']:
            raise LLMValidationError("Messages list cannot be empty")
            
        # Validate each message
        for msg in params['messages']:
            if not isinstance(msg, dict):
                raise LLMValidationError("Each message must be a dictionary")
            if 'role' not in msg or 'content' not in msg:
                raise LLMValidationError("Messages must have 'role' and 'content'")
    
    def _validate_response(self, response: Any) -> APIResponse:
        """Validate API response."""
        try:
            if isinstance(response, str):
                response = json.loads(response)
                
            validated = APIResponse(
                choices=[{'message': MessageContent(**choice['message'])} for choice in response['choices']],
                model=response['model'],
                usage=response.get('usage')
            )
            
            # Additional validation
            if not validated.choices:
                raise LLMValidationError("Response contains no choices")
            if not validated.choices[0]['message'].content:
                raise LLMValidationError("Response message content is empty")
                
            return validated
            
        except (json.JSONDecodeError, ValidationError) as e:
            raise LLMValidationError(f"Invalid API response format: {str(e)}")
    
    @abstractmethod
    async def _execute_api_call(self, request_kwargs: dict) -> Any:
        """
        Execute the actual API call. Must be implemented by provider-specific classes.
        
        Args:
            request_kwargs: API request parameters
            
        Returns:
            Raw API response
        """
        raise NotImplementedError("API call execution not implemented")
    
    def add_context(self, context: Dict[str, str]):
        """Add context for the module."""
        if not isinstance(context, dict):
            raise LLMValidationError("Context must be a dictionary")
        self.context.update(context)
