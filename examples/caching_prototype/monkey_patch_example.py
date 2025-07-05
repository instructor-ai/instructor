"""Example of monkey patching approach for global cache."""

import instructor
from instructor.cache import AutoCache
from instructor.client import Instructor, AsyncInstructor
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


# Store original methods
_original_create = Instructor.create
_original_create_async = AsyncInstructor.create


def create_with_global_cache(self, *args, **kwargs):
    """Enhanced create method that checks for global cache."""
    # Check if cache is already provided
    if 'cache' not in kwargs:
        # Check for global cache
        global_cache = getattr(instructor, '_global_cache', None)
        if global_cache:
            kwargs['cache'] = global_cache
    
    return _original_create(self, *args, **kwargs)


async def create_with_global_cache_async(self, *args, **kwargs):
    """Enhanced async create method that checks for global cache."""
    # Check if cache is already provided
    if 'cache' not in kwargs:
        # Check for global cache
        global_cache = getattr(instructor, '_global_cache', None)
        if global_cache:
            kwargs['cache'] = global_cache
    
    return await _original_create_async(self, *args, **kwargs)


def enable_global_cache(cache):
    """Enable global caching by monkey patching the base classes."""
    instructor._global_cache = cache
    
    # Monkey patch the methods
    Instructor.create = create_with_global_cache
    AsyncInstructor.create = create_with_global_cache_async
    
    print(f"Global cache enabled: {type(cache).__name__}")


def disable_global_cache():
    """Disable global caching and restore original methods."""
    if hasattr(instructor, '_global_cache'):
        delattr(instructor, '_global_cache')
    
    # Restore original methods
    Instructor.create = _original_create
    AsyncInstructor.create = _original_create_async
    
    print("Global cache disabled")


def example_with_monkey_patch():
    """Demo using monkey patch approach."""
    print("=== Monkey Patch Example ===")
    
    # Enable global cache
    cache = AutoCache(maxsize=100)
    enable_global_cache(cache)
    
    try:
        # Now ALL instructor clients automatically use the global cache
        client1 = instructor.from_provider("openai/gpt-3.5-turbo")
        client2 = instructor.from_openai(instructor.openai.OpenAI())
        
        messages = [{"role": "user", "content": "Create user Charlie age 35"}]
        
        # Both use global cache automatically
        user1 = client1.create(messages=messages, response_model=User)
        user2 = client2.create(model="gpt-3.5-turbo", messages=messages, response_model=User)
        
        print(f"First client: {user1}")
        print(f"Second client: {user2}")
        
        # Can still override with explicit cache=None
        user3 = client1.create(messages=messages, response_model=User, cache=None)
        print(f"No cache: {user3}")
        
    finally:
        # Clean up
        disable_global_cache()


if __name__ == "__main__":
    example_with_monkey_patch()