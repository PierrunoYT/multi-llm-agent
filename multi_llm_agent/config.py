from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator
import os
from dotenv import load_dotenv

load_dotenv()

class CacheConfig(BaseModel):
    """Configuration for prompt caching."""
    enabled: bool = True
    min_cache_size: int = Field(
        default=1024,
        description="Minimum content size in characters to trigger caching"
    )
    cache_system_messages: bool = Field(
        default=True,
        description="Whether to cache system messages"
    )
    cache_user_messages: bool = Field(
        default=True,
        description="Whether to cache user messages"
    )

class LLMConfig(BaseModel):
    """Configuration for individual LLM modules with comprehensive parameter support."""
    # Required parameters
    provider: str  # 'openai', 'openrouter', 'anthropic'
    model: str
    api_key: Optional[str] = None
    
    # Generation parameters
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=2.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_a: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: Optional[int] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    
    # Advanced parameters
    logit_bias: Dict[int, float] = Field(
        default_factory=dict,
        description="Token ID to bias value mapping (-100 to 100)"
    )
    logprobs: bool = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    response_format: Optional[Dict[str, str]] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    # Caching configuration
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    
    # Extra configuration
    extra_config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('tool_choice')
    def validate_tool_choice(cls, v):
        if isinstance(v, str) and v not in ['none', 'auto', 'required']:
            raise ValueError("tool_choice string must be 'none', 'auto', or 'required'")
        return v

    def to_request_params(self) -> Dict[str, Any]:
        """Convert config to API request parameters."""
        params = {
            k: v for k, v in self.model_dump().items()
            if v is not None and k not in ['provider', 'api_key', 'extra_config', 'cache_config']
        }
        
        # Remove empty optional parameters
        if not params.get('logit_bias'):
            params.pop('logit_bias', None)
        if not params.get('tools'):
            params.pop('tools', None)
            params.pop('tool_choice', None)
        if not params.get('response_format'):
            params.pop('response_format', None)
        if not params.get('stop'):
            params.pop('stop', None)
            
        return params

class AgentConfig(BaseModel):
    """Configuration for the entire agent system."""
    reasoning_config: LLMConfig
    planner_config: LLMConfig
    executor_config: LLMConfig

def create_default_config() -> AgentConfig:
    """Create a default configuration using environment variables."""
    # Common OpenRouter settings
    openrouter_settings = {
        "provider": "openrouter",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "extra_config": {
            "site_url": os.getenv("SITE_URL"),
            "app_name": os.getenv("APP_NAME", "MultiLLMAgent")
        }
    }

    return AgentConfig(
        reasoning_config=LLMConfig(
            **openrouter_settings,
            model="openai/gpt-4-vision-preview",  # Supports multimodal
            temperature=0.7,  # Balanced creativity
            max_tokens=1000,
            presence_penalty=0.1,  # Slight penalty for repetition
            frequency_penalty=0.1,  # Slight penalty for frequency
        ),
        planner_config=LLMConfig(
            **openrouter_settings,
            model="anthropic/claude-3-sonnet",  # Good for planning
            temperature=0.5,  # More focused
            max_tokens=2000,
            top_p=0.9,  # Slightly more constrained
            presence_penalty=0.2,  # Higher penalty for repetition in plans
        ),
        executor_config=LLMConfig(
            **openrouter_settings,
            model="anthropic/claude-3-haiku",  # Fast and efficient
            temperature=0.3,  # More deterministic
            max_tokens=1500,
            top_k=40,  # Limit token choices for more focused output
            repetition_penalty=1.2,  # Higher penalty for repetition in execution
        )
    )
