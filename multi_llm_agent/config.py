from typing import Optional, Dict
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for individual LLM modules."""
    provider: str  # 'openai', 'openrouter'
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    api_key: Optional[str] = None
    extra_config: Dict[str, str] = {}

class AgentConfig(BaseModel):
    """Configuration for the entire agent system."""
    reasoning_config: LLMConfig
    planner_config: LLMConfig
    executor_config: LLMConfig

def create_default_config() -> AgentConfig:
    """Create a default configuration using environment variables."""
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    site_url = os.getenv("SITE_URL", "")
    app_name = os.getenv("APP_NAME", "MultiLLMAgent")
    
    # Common OpenRouter config
    openrouter_extra = {
        "site_url": site_url,
        "app_name": app_name
    }
    
    # Reasoning module uses OpenRouter with GPT-4 preview
    reasoning_config = LLMConfig(
        provider="openrouter",
        model="openai/o1-preview-2024-09-12",
        temperature=0.7,
        api_key=openrouter_key,
        extra_config=openrouter_extra
    )
    
    # Planner uses Claude-3.5-sonnet through OpenRouter
    planner_config = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet:beta",
        temperature=0.5,
        api_key=openrouter_key,
        extra_config=openrouter_extra
    )
    
    # Executor uses Claude-3.5-haiku through OpenRouter
    executor_config = LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3-5-haiku:beta",
        temperature=0.8,
        api_key=openrouter_key,
        extra_config=openrouter_extra
    )
    
    return AgentConfig(
        reasoning_config=reasoning_config,
        planner_config=planner_config,
        executor_config=executor_config
    )
