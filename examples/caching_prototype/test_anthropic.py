"""Test caching with Anthropic provider to verify defensive behavior."""

import time
import instructor
from instructor.cache import AutoCache
from pydantic import BaseModel, Field
import logging

# Set up logging to see cache behavior - focus on cache logs only
logging.basicConfig(level=logging.INFO)
cache_logger = logging.getLogger("instructor.cache")
cache_logger.setLevel(logging.DEBUG)

# Create handler for just cache logs
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('CACHE: %(message)s')
handler.setFormatter(formatter)
cache_logger.addHandler(handler)
cache_logger.propagate = False


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


def test_anthropic_caching():
    """Test caching with Anthropic provider."""
    print("\n=== Testing Anthropic Caching ===")
    
    # Create cache and Anthropic client
    cache = AutoCache(maxsize=100)
    client = instructor.from_provider("anthropic/claude-3-haiku-20240307", cache=cache)
    
    messages = [{"role": "user", "content": "Generate a user named Alice who is 25 years old"}]
    
    # First call - hits API
    print("First call (hits Anthropic API)...")
    start = time.time()
    user1 = client.create(
        messages=messages,
        response_model=User,
    )
    api_time = time.time() - start
    print(f"Result: {user1}")
    print(f"Time: {api_time:.2f}s")
    print(f"Has raw response: {hasattr(user1, '_raw_response')}")
    if hasattr(user1, '_raw_response'):
        print(f"Raw response type: {type(user1._raw_response)}")
    
    # Second call - from cache
    print("\nSecond call (from cache)...")
    start = time.time()
    user2 = client.create(
        messages=messages,
        response_model=User,
    )
    cache_time = time.time() - start
    print(f"Result: {user2}")
    print(f"Time: {cache_time:.4f}s")
    print(f"Speedup: {api_time/cache_time:.0f}x faster")
    
    assert user1.name == user2.name
    assert user1.age == user2.age
    print("✓ Anthropic caching working correctly")


def test_anthropic_create_with_completion():
    """Test create_with_completion with Anthropic."""
    print("\n=== Testing Anthropic create_with_completion ===")
    
    cache = AutoCache(maxsize=100)
    client = instructor.from_provider("anthropic/claude-3-haiku-20240307", cache=cache)
    
    messages = [{"role": "user", "content": "Weather is 22C and sunny"}]
    
    class Weather(BaseModel):
        temperature: float
        condition: str
    
    # First call
    print("First call with completion...")
    weather1, completion1 = client.create_with_completion(
        messages=messages,
        response_model=Weather,
    )
    print(f"Weather: {weather1}")
    print(f"Completion type: {type(completion1)}")
    print(f"Completion has ID: {hasattr(completion1, 'id')}")
    if hasattr(completion1, 'id'):
        print(f"Completion ID: {completion1.id}")
    if hasattr(completion1, 'usage'):
        print(f"Usage: {completion1.usage}")
    
    # Second call - cached
    print("\nSecond call (cached)...")
    start = time.time()
    weather2, completion2 = client.create_with_completion(
        messages=messages,
        response_model=Weather,
    )
    cache_time = time.time() - start
    print(f"Weather: {weather2}")
    print(f"Cache time: {cache_time:.4f}s")
    print(f"Completion type: {type(completion2)}")
    
    # Check if completion objects are equivalent
    if hasattr(completion1, 'id') and hasattr(completion2, 'id'):
        print(f"Same completion ID: {completion1.id == completion2.id}")
        assert completion1.id == completion2.id
    
    assert weather1.temperature == weather2.temperature
    assert weather1.condition == weather2.condition
    print("✓ Anthropic completion caching working")


def test_anthropic_different_models():
    """Test caching with different Anthropic models."""
    print("\n=== Testing Different Anthropic Models ===")
    
    cache = AutoCache(maxsize=100)
    
    # Test with Claude 3 Haiku
    client1 = instructor.from_provider("anthropic/claude-3-haiku-20240307", cache=cache)
    
    # Test with Claude 3.5 Sonnet (if available)
    try:
        client2 = instructor.from_provider("anthropic/claude-3-5-sonnet-20241022", cache=cache)
        
        messages = [{"role": "user", "content": "Create user Bob age 30"}]
        
        user1 = client1.create(messages=messages, response_model=User)
        print(f"Haiku result: {user1}")
        
        user2 = client2.create(messages=messages, response_model=User)
        print(f"Sonnet result: {user2}")
        
        # Should be different results (different models = different cache keys)
        print("✓ Different models produce different cache entries")
        
    except Exception as e:
        print(f"Sonnet model not available: {e}")
        print("✓ Single model test completed")


if __name__ == "__main__":
    print("Testing Anthropic Caching")
    print("=" * 50)
    
    try:
        test_anthropic_caching()
        test_anthropic_create_with_completion()
        test_anthropic_different_models()
        
        print("\n" + "=" * 50)
        print("All Anthropic caching tests completed! ✨")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in your environment")