from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import hashlib
import json
from datetime import datetime, timedelta

# Cache pricing multipliers for different providers
PROVIDER_PRICING = {
    "openai": {
        "gpt-4": 1.0,
        "gpt-4-vision-preview": 1.2,
        "gpt-3.5-turbo": 0.2
    },
    "anthropic": {
        "claude-3-opus": 1.5,
        "claude-3-sonnet": 1.0,
        "claude-3-haiku": 0.5
    }
}

# Cache expiration times (in minutes)
CACHE_EXPIRATION = {
    "openai": 60,  # 1 hour
    "anthropic": 5,  # 5 minutes
    "default": 30  # 30 minutes
}

class CacheControl(BaseModel):
    """Cache control configuration for message parts."""
    type: str = "ephemeral"  # Currently only 'ephemeral' is supported

class MessageContent(BaseModel):
    """Content part of a message with optional cache control."""
    type: str
    text: str
    cache_control: Optional[CacheControl] = None
    image_url: Optional[Dict[str, str]] = None

def should_enable_caching(model: str) -> bool:
    """
    Determine if caching should be enabled for a given model.
    
    Args:
        model: The model identifier
        
    Returns:
        bool: True if caching should be enabled
    """
    # Enable caching for all supported models
    for provider, models in PROVIDER_PRICING.items():
        if model in models:
            return True
    return False

def get_cache_pricing(model: str) -> float:
    """
    Get the pricing multiplier for a model's cache storage.
    
    Args:
        model: The model identifier
        
    Returns:
        float: The pricing multiplier
    """
    for provider, models in PROVIDER_PRICING.items():
        if model in models:
            return models[model]
    return 1.0  # Default multiplier

def get_cache_expiration(provider: str) -> timedelta:
    """
    Get the cache expiration time for a provider.
    
    Args:
        provider: The provider name
        
    Returns:
        timedelta: The cache expiration time
    """
    minutes = CACHE_EXPIRATION.get(provider, CACHE_EXPIRATION["default"])
    return timedelta(minutes=minutes)

def create_cache_key(content: str, role: str, model: str) -> str:
    """
    Create a unique cache key for a message.
    
    Args:
        content: The message content
        role: The message role
        model: The model being used
        
    Returns:
        str: The cache key
    """
    key_data = {
        "content": content,
        "role": role,
        "model": model
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()

def create_cacheable_message(
    role: str,
    content: Union[str, List[Dict[str, Any]]],
    cache_large_content: bool = True,
    min_cache_size: int = 1024
) -> Dict[str, Any]:
    """
    Create a message with cache control for large content parts.
    
    Args:
        role: Message role ('system', 'user', 'assistant')
        content: Message content (string or list of content parts)
        cache_large_content: Whether to enable caching for large content
        min_cache_size: Minimum content size to trigger caching
        
    Returns:
        Message dictionary with cache control where appropriate
    """
    if isinstance(content, str):
        # For string content, wrap in a list if it's large enough to cache
        if cache_large_content and len(content) >= min_cache_size:
            return {
                "role": role,
                "content": [
                    MessageContent(
                        type="text",
                        text=content,
                        cache_control=CacheControl()
                    ).model_dump()
                ]
            }
        return {"role": role, "content": content}
    
    # For multipart content, add cache control to large text parts
    processed_content = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text" and cache_large_content:
                text = part.get("text", "")
                if len(text) >= min_cache_size:
                    processed_content.append(
                        MessageContent(
                            type="text",
                            text=text,
                            cache_control=CacheControl()
                        ).model_dump()
                    )
                    continue
            processed_content.append(part)
    
    return {"role": role, "content": processed_content}

def get_cache_pricing_for_model(model: str) -> Dict[str, float]:
    """
    Get cache write and read cost multipliers for a model.
    
    Args:
        model: Full model identifier
        
    Returns:
        Dictionary with write and read cost multipliers
    """
    if model.startswith("openai/"):
        return {
            "write_multiplier": 1.0,  # No additional cost
            "read_multiplier": 0.5    # Half price
        }
    elif model.startswith("anthropic/"):
        return {
            "write_multiplier": 1.25,  # 25% more expensive
            "read_multiplier": 0.1     # 90% cheaper
        }
    elif model.startswith("deepseek/"):
        return {
            "write_multiplier": 1.0,   # Same price
            "read_multiplier": 0.1     # 90% cheaper
        }
    else:
        return {
            "write_multiplier": 1.0,
            "read_multiplier": 1.0
        }
