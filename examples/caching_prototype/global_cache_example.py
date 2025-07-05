"""Example of global cache registry approach."""

import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


class CacheRegistry:
    """Global registry for default cache settings."""
    
    def __init__(self):
        self._default_cache = None
    
    def set_default_cache(self, cache):
        """Set the default cache for all new instructor clients."""
        self._default_cache = cache
    
    def get_default_cache(self):
        """Get the default cache."""
        return self._default_cache
    
    def clear_default_cache(self):
        """Clear the default cache."""
        self._default_cache = None


# Global registry instance
cache_registry = CacheRegistry()


def example_with_global_cache():
    """Demo using global cache registry."""
    print("=== Global Cache Registry Example ===")
    
    # Set global default cache
    cache = AutoCache(maxsize=100)
    cache_registry.set_default_cache(cache)
    
    # All new clients automatically use the default cache
    client1 = instructor.from_provider("openai/gpt-3.5-turbo")
    client2 = instructor.from_provider("anthropic/claude-3-haiku-20240307") 
    
    messages = [{"role": "user", "content": "Create user Bob age 30"}]
    
    # Both clients would use the same cache (if we modify from_provider to check registry)
    user1 = client1.create(messages=messages, response_model=User)
    user2 = client2.create(messages=messages, response_model=User)  # Different provider = different cache key
    
    print(f"OpenAI: {user1}")
    print(f"Anthropic: {user2}")
    
    # Clear default
    cache_registry.clear_default_cache()


if __name__ == "__main__":
    example_with_global_cache()