from typing import Dict, Any, Optional
import asyncio
import aiofiles
import json
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """Thread-safe cache manager with file persistence."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        # Ensure lock exists for this key
        await self._ensure_lock(key)
        
        async with self._locks[key]:
            # Check memory cache first
            if key in self._cache:
                value = self._cache[key]
                if not self._is_expired(value):
                    logger.debug(f"Cache hit for key: {key}")
                    return value["data"]
            
            # Try loading from file
            try:
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists():
                    async with aiofiles.open(cache_file, mode='r') as f:
                        content = await f.read()
                        value = json.loads(content)
                        if not self._is_expired(value):
                            self._cache[key] = value
                            logger.debug(f"Loaded from file cache: {key}")
                            return value["data"]
            except Exception as e:
                logger.warning(f"Error reading cache file for key {key}: {e}")
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        expires_in: Optional[int] = None,
        provider: Optional[str] = None
    ) -> None:
        """Set a value in the cache with optional expiration."""
        await self._ensure_lock(key)
        
        async with self._locks[key]:
            cache_entry = {
                "data": value,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "expires_in": expires_in
            }
            
            # Update memory cache
            self._cache[key] = cache_entry
            
            # Persist to file
            try:
                cache_file = self.cache_dir / f"{key}.json"
                async with aiofiles.open(cache_file, mode='w') as f:
                    await f.write(json.dumps(cache_entry, indent=2))
                logger.debug(f"Cached value for key: {key}")
            except Exception as e:
                logger.error(f"Error writing cache file for key {key}: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        await self._ensure_lock(key)
        
        async with self._locks[key]:
            # Remove from memory cache
            self._cache.pop(key, None)
            
            # Remove cache file
            try:
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists():
                    os.remove(cache_file)
                logger.debug(f"Deleted cache for key: {key}")
            except Exception as e:
                logger.warning(f"Error deleting cache file for key {key}: {e}")
    
    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._global_lock:
            # Clear memory cache
            self._cache.clear()
            
            # Clear all cache files
            try:
                for cache_file in self.cache_dir.glob("*.json"):
                    os.remove(cache_file)
                logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
    
    async def _ensure_lock(self, key: str) -> None:
        """Ensure a lock exists for the given key."""
        async with self._global_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        if not cache_entry.get("expires_in"):
            return False
            
        timestamp = datetime.fromisoformat(cache_entry["timestamp"])
        expiration = timedelta(minutes=cache_entry["expires_in"])
        return datetime.now() - timestamp > expiration

# Global cache manager instance
cache_manager = CacheManager()
