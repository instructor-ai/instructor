"""Debug script to check if cache is being passed through kwargs."""

import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


def debug_openai():
    """Debug OpenAI kwargs flow."""
    print("=== Debugging OpenAI kwargs flow ===")
    
    cache = AutoCache(maxsize=100)
    
    # Let's check if the cache parameter is in kwargs
    import instructor
    original_from_openai = instructor.from_openai
    
    def debug_from_openai(*args, **kwargs):
        print(f"from_openai called with kwargs: {list(kwargs.keys())}")
        print(f"cache in kwargs: {'cache' in kwargs}")
        if 'cache' in kwargs:
            print(f"cache value: {kwargs['cache']}")
        return original_from_openai(*args, **kwargs)
    
    instructor.from_openai = debug_from_openai
    
    try:
        client = instructor.from_provider("openai/gpt-3.5-turbo", cache=cache)
        print(f"Client created: {type(client)}")
        
        messages = [{"role": "user", "content": "Create user Debug age 42"}]
        user = client.create(messages=messages, response_model=User)
        print(f"Result: {user}")
        
    finally:
        instructor.from_openai = original_from_openai


def debug_anthropic():
    """Debug Anthropic kwargs flow."""
    print("\n=== Debugging Anthropic kwargs flow ===")
    
    cache = AutoCache(maxsize=100)
    
    # Let's check if the cache parameter is in kwargs
    import instructor
    original_from_anthropic = instructor.from_anthropic
    
    def debug_from_anthropic(*args, **kwargs):
        print(f"from_anthropic called with kwargs: {list(kwargs.keys())}")
        print(f"cache in kwargs: {'cache' in kwargs}")
        if 'cache' in kwargs:
            print(f"cache value: {kwargs['cache']}")
        return original_from_anthropic(*args, **kwargs)
    
    instructor.from_anthropic = debug_from_anthropic
    
    try:
        client = instructor.from_provider("anthropic/claude-3-haiku-20240307", cache=cache)
        print(f"Client created: {type(client)}")
        
        messages = [{"role": "user", "content": "Create user Debug age 42"}]
        user = client.create(messages=messages, response_model=User)
        print(f"Result: {user}")
        
    finally:
        instructor.from_anthropic = original_from_anthropic


if __name__ == "__main__":
    debug_openai()
    debug_anthropic()