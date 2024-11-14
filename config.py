from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class CacheConfig:
    enabled: bool = True
    cache_system_messages: bool = True
    cache_user_messages: bool = True
    min_cache_size: int = 100
    ttl_seconds: int = 300  # 5 minutes

@dataclass
class LLMConfig:
    model: str
    api_key: str
    provider: str = "openrouter"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    top_k: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        # Validate top_p
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0")
        
        # Validate penalties
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("Presence penalty must be between -2.0 and 2.0")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("Frequency penalty must be between -2.0 and 2.0")

    def to_request_params(self) -> Dict[str, Any]:
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.presence_penalty != 0:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty != 0:
            params["frequency_penalty"] = self.frequency_penalty
            
        return params

class AgentConfig(BaseModel):
    """Configuration for the entire agent system."""
    reasoning_config: LLMConfig
    planner_config: LLMConfig
    executor_config: LLMConfig

# Default configurations for each module
REASONING_CONFIG = LLMConfig(
    model="openai/o1-preview",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    temperature=0.7,
    cache_config=CacheConfig(
        enabled=True,
        cache_system_messages=True,
        cache_user_messages=True,
        min_cache_size=100
    ),
    extra_config={
        "site_url": os.getenv("SITE_URL", "https://example.com"),
        "app_name": "Multi-LLM Agent"
    }
)

PLANNING_CONFIG = LLMConfig(
    model="anthropic/claude-3.5-sonnet:beta",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    temperature=0.7,
    cache_config=CacheConfig(
        enabled=True,
        cache_system_messages=True,
        cache_user_messages=True,
        min_cache_size=100
    ),
    extra_config={
        "site_url": os.getenv("SITE_URL", "https://example.com"),
        "app_name": "Multi-LLM Agent"
    }
)

EXECUTOR_CONFIG = LLMConfig(
    model="anthropic/claude-3-5-haiku:beta",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    temperature=0.5,  # Lower temperature for more deterministic execution
    cache_config=CacheConfig(
        enabled=True,
        cache_system_messages=True,
        cache_user_messages=True,
        min_cache_size=100
    ),
    extra_config={
        "site_url": os.getenv("SITE_URL", "https://example.com"),
        "app_name": "Multi-LLM Agent"
    }
)

def create_default_config() -> AgentConfig:
    """Create a default configuration using environment variables."""
    return AgentConfig(
        reasoning_config=REASONING_CONFIG,
        planner_config=PLANNING_CONFIG,
        executor_config=EXECUTOR_CONFIG
    )
