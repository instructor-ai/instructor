"""Example of context manager approach for caching."""

import threading
from contextlib import contextmanager
import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


# Thread-local storage for cache
_cache_context = threading.local()


@contextmanager
def cached_instructor(cache):
    """Context manager to set cache for all instructor calls in this context."""
    old_cache = getattr(_cache_context, 'cache', None)
    _cache_context.cache = cache
    try:
        yield
    finally:
        _cache_context.cache = old_cache


def get_current_cache():
    """Get the current cache from thread-local storage."""
    return getattr(_cache_context, 'cache', None)


def example_with_context_manager():
    """Demo using context manager for caching."""
    print("=== Context Manager Example ===")
    
    cache = AutoCache(maxsize=100)
    
    # All instructor calls within this context use the cache
    with cached_instructor(cache):
        client = instructor.from_provider("openai/gpt-3.5-turbo")
        
        messages = [{"role": "user", "content": "Create user Alice age 25"}]
        
        # First call - hits API
        user1 = client.create(messages=messages, response_model=User)
        print(f"First: {user1}")
        
        # Second call - cached (if we modify patch.py to check get_current_cache())
        user2 = client.create(messages=messages, response_model=User)
        print(f"Second: {user2}")


if __name__ == "__main__":
    example_with_context_manager()