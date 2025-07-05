"""Example showing how cache flows through **kwargs automatically."""

import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


def test_kwargs_flow():
    """Demo that cache flows through **kwargs without explicit passing."""
    print("=== Testing **kwargs Flow ===")
    
    cache = AutoCache(maxsize=100)
    
    # The cache parameter flows through **kwargs to the underlying provider functions
    client = instructor.from_provider(
        "openai/gpt-3.5-turbo",
        cache=cache,  # This goes into **kwargs and flows to from_openai()
    )
    
    messages = [{"role": "user", "content": "Create user Test age 99"}]
    
    # Test that caching works
    user1 = client.create(messages=messages, response_model=User)
    print(f"First call: {user1}")
    
    user2 = client.create(messages=messages, response_model=User)
    print(f"Second call (cached): {user2}")
    
    assert user1.name == user2.name
    print("✓ Cache working through **kwargs flow")


def test_anthropic_kwargs():
    """Test that **kwargs flow works with Anthropic too."""
    print("\n=== Testing Anthropic **kwargs Flow ===")
    
    cache = AutoCache(maxsize=100)
    
    client = instructor.from_provider(
        "anthropic/claude-3-haiku-20240307",
        cache=cache,  # Flows through **kwargs to from_anthropic()
    )
    
    messages = [{"role": "user", "content": "Create user Anthropic age 42"}]
    
    user1 = client.create(messages=messages, response_model=User)
    print(f"First call: {user1}")
    
    user2 = client.create(messages=messages, response_model=User)
    print(f"Second call (cached): {user2}")
    
    print("✓ Anthropic cache working through **kwargs")


if __name__ == "__main__":
    test_kwargs_flow()
    test_anthropic_kwargs()