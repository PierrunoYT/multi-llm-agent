from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import hashlib
import json
from datetime import datetime, timedelta
import logging
from .cache_sync import cache_manager

logger = logging.getLogger(__name__)

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
    """Determine if caching should be enabled for a model."""
    # Extract provider from model name
    provider = "openai" if "gpt" in model.lower() else "anthropic" if "claude" in model.lower() else None
    
    if not provider:
        return False
    
    # Get pricing multiplier
    pricing = PROVIDER_PRICING.get(provider, {}).get(model, 0)
    
    # Enable caching for more expensive models
    return pricing >= 0.5

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

def create_cache_key(content: str, role: str, model: str, **kwargs: Any) -> str:
    """
    Create a unique cache key for a message.
    
    Args:
        content: The message content
        role: The message role
        model: The model being used
        **kwargs: Additional key components
        
    Returns:
        str: The cache key
    """
    # Normalize content by removing extra whitespace
    normalized_content = " ".join(content.split())
    
    key_data = {
        "content": normalized_content,
        "role": role.lower(),
        "model": model.lower(),
        **kwargs
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return f"{role}_{hashlib.sha256(key_string.encode()).hexdigest()}"

def _calculate_cache_key(content: str, role: str) -> str:
    """Calculate a cache key for the content."""
    # Create a unique hash of the content
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return f"{role}_{content_hash[:16]}"

async def create_cacheable_message(
    role: str,
    content: str,
    cache_large_content: bool = True,
    min_cache_size: int = 1000
) -> Dict[str, Any]:
    """
    Create a message that may be cached if it meets the criteria.
    
    Args:
        role: Message role (system, user, assistant)
        content: Message content
        cache_large_content: Whether to cache large content
        min_cache_size: Minimum content size to cache
        
    Returns:
        Dict containing the message with possible cached content
    """
    if not cache_large_content or len(content) < min_cache_size:
        return {"role": role, "content": content}
    
    cache_key = _calculate_cache_key(content, role)
    
    # Try to get from cache
    cached_content = await cache_manager.get(cache_key)
    if cached_content:
        logger.debug(f"Cache hit for {role} message")
        return {"role": role, "content": cached_content}
    
    # Cache the content
    await cache_manager.set(
        key=cache_key,
        value=content,
        expires_in=CACHE_EXPIRATION.get("default")
    )
    logger.debug(f"Cached {role} message")
    
    return {"role": role, "content": content}

async def cache_response(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    response: str
) -> None:
    """
    Cache an API response if appropriate.
    
    Args:
        provider: API provider name
        model: Model name
        messages: Input messages
        response: Response content to cache
    """
    if not should_enable_caching(model):
        return
    
    # Create cache key from messages
    message_str = json.dumps(messages, sort_keys=True)
    cache_key = f"response_{hashlib.sha256(message_str.encode()).hexdigest()[:16]}"
    
    # Get expiration time
    expires_in = CACHE_EXPIRATION.get(provider, CACHE_EXPIRATION["default"])
    
    # Cache the response
    await cache_manager.set(
        key=cache_key,
        value=response,
        expires_in=expires_in,
        provider=provider
    )
    logger.debug(f"Cached response for {provider}/{model}")

async def get_cached_response(
    provider: str,
    model: str,
    messages: List[Dict[str, str]]
) -> Optional[str]:
    """
    Try to get a cached response for the input messages.
    
    Args:
        provider: API provider name
        model: Model name
        messages: Input messages
        
    Returns:
        Cached response if available, None otherwise
    """
    if not should_enable_caching(model):
        return None
    
    # Create cache key from messages
    message_str = json.dumps(messages, sort_keys=True)
    cache_key = f"response_{hashlib.sha256(message_str.encode()).hexdigest()[:16]}"
    
    # Try to get from cache
    cached_response = await cache_manager.get(cache_key)
    if cached_response:
        logger.debug(f"Cache hit for {provider}/{model} response")
        return cached_response
    
    return None

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
