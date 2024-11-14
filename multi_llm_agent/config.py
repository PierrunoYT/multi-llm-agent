from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for individual LLM modules."""
    provider: str  # 'openai', 'openrouter', 'anthropic'
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    extra_config: Dict[str, Any] = Field(default_factory=dict)

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
            temperature=0.7,
            max_tokens=1000,
        ),
        planner_config=LLMConfig(
            **openrouter_settings,
            model="anthropic/claude-3-sonnet",  # Good for planning
            temperature=0.5,
            max_tokens=2000,
        ),
        executor_config=LLMConfig(
            **openrouter_settings,
            model="anthropic/claude-3-haiku",  # Fast and efficient
            temperature=0.3,
            max_tokens=1500,
        )
    )
