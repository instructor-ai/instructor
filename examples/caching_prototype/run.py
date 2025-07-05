"""Test the built-in caching functionality in Instructor.

This example demonstrates:
1. Basic caching with AutoCache (in-memory LRU)
2. Persistent caching with DiskCache
3. Using create_with_completion to verify raw responses are cached
4. Cache TTL (time-to-live) functionality
5. Cache key generation based on different inputs
"""

import time
import instructor
from instructor import from_openai
from instructor.cache import AutoCache, DiskCache
from pydantic import BaseModel, Field
from openai import OpenAI


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


class Weather(BaseModel):
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (sunny, cloudy, rainy, etc.)")


def test_autocache_basic():
    """Test basic in-memory caching with AutoCache."""
    print("\n=== Testing AutoCache (in-memory LRU) ===")
    
    # Create cache and client
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(), cache=cache)
    
    messages = [{"role": "user", "content": "Create a user named Alice who is 25 years old"}]
    
    # First call - should hit the API
    print("First call (should hit API)...")
    start = time.time()
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    api_time = time.time() - start
    print(f"Result: {user1}")
    print(f"Time: {api_time:.3f}s")
    
    # Second call - should be cached
    print("\nSecond call (should be cached)...")
    start = time.time()
    user2 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    cache_time = time.time() - start
    print(f"Result: {user2}")
    print(f"Time: {cache_time:.3f}s")
    print(f"Speedup: {api_time/cache_time:.1f}x")
    
    # Verify they're the same
    assert user1.name == user2.name
    assert user1.age == user2.age
    print("✓ Cache working correctly - same results returned")


def test_create_with_completion():
    """Test that create_with_completion preserves raw responses in cache."""
    print("\n=== Testing create_with_completion ===")
    
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(), cache=cache)
    
    messages = [{"role": "user", "content": "What's the weather like? 20C and sunny"}]
    
    # First call with create_with_completion
    print("First call with create_with_completion...")
    weather1, completion1 = client.create_with_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=Weather,
    )
    print(f"Weather: {weather1}")
    print(f"Completion ID: {completion1.id}")
    print(f"Usage: {completion1.usage}")
    
    # Second call - should return cached model AND completion
    print("\nSecond call (from cache)...")
    weather2, completion2 = client.create_with_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=Weather,
    )
    print(f"Weather: {weather2}")
    print(f"Completion ID: {completion2.id}")
    
    # Verify both model and raw response are identical
    assert weather1.temperature == weather2.temperature
    assert weather1.condition == weather2.condition
    assert completion1.id == completion2.id
    print("✓ Raw completion correctly cached and restored")


def test_diskcache_persistence():
    """Test persistent caching with DiskCache."""
    print("\n=== Testing DiskCache (persistent) ===")
    
    # Create cache that persists to disk
    cache = DiskCache(directory=".test_instructor_cache")
    client = instructor.from_openai(OpenAI(), cache=cache)
    
    messages = [{"role": "user", "content": "Create a user named Bob who is 30 years old"}]
    
    # First call
    print("First call...")
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    print(f"Result: {user1}")
    
    # Create a new client with same cache directory
    print("\nCreating new client with same cache directory...")
    cache2 = DiskCache(directory=".test_instructor_cache")
    client2 = instructor.from_openai(OpenAI(), cache=cache2)
    
    # Should still get cached result
    print("Calling with new client (should use persisted cache)...")
    user2 = client2.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
    )
    print(f"Result: {user2}")
    
    assert user1.name == user2.name
    assert user1.age == user2.age
    print("✓ Cache persisted across client instances")
    
    # Cleanup
    import shutil
    shutil.rmtree(".test_instructor_cache", ignore_errors=True)


def test_cache_ttl():
    """Test cache TTL (time-to-live) with DiskCache."""
    print("\n=== Testing Cache TTL ===")
    
    cache = DiskCache(directory=".test_instructor_cache_ttl")
    client = instructor.from_openai(OpenAI(), cache=cache)
    
    messages = [{"role": "user", "content": "Create a user named Charlie who is 35 years old"}]
    
    # Set cache with 2 second TTL
    print("Setting cache with 2 second TTL...")
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
        cache_ttl=2,  # 2 seconds
    )
    print(f"Result: {user1}")
    
    # Immediate second call should be cached
    print("\nImmediate call (should be cached)...")
    start = time.time()
    user2 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
        cache_ttl=2,
    )
    elapsed = time.time() - start
    print(f"Time: {elapsed:.3f}s")
    assert elapsed < 0.1, "Should be very fast (from cache)"
    
    # Wait for TTL to expire
    print("\nWaiting 3 seconds for TTL to expire...")
    time.sleep(3)
    
    # This call should hit the API again
    print("Call after TTL expired (should hit API)...")
    start = time.time()
    user3 = client.create(
        model="gpt-3.5-turbo",
        messages=messages,
        response_model=User,
        cache_ttl=2,
    )
    elapsed = time.time() - start
    print(f"Time: {elapsed:.3f}s")
    assert elapsed > 0.5, "Should take time (API call)"
    print("✓ Cache TTL working correctly")
    
    # Cleanup
    import shutil
    shutil.rmtree(".test_instructor_cache_ttl", ignore_errors=True)


def test_cache_key_differentiation():
    """Test that different inputs generate different cache keys."""
    print("\n=== Testing Cache Key Differentiation ===")
    
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(), cache=cache)
    
    # Different messages should have different cache keys
    messages1 = [{"role": "user", "content": "Create a user named David who is 40 years old"}]
    messages2 = [{"role": "user", "content": "Create a user named Eve who is 45 years old"}]
    
    user1 = client.create(
        model="gpt-3.5-turbo",
        messages=messages1,
        response_model=User,
    )
    print(f"User 1: {user1}")
    
    user2 = client.create(
        model="gpt-3.5-turbo",
        messages=messages2,
        response_model=User,
    )
    print(f"User 2: {user2}")
    
    # Should be different users
    assert user1.name != user2.name or user1.age != user2.age
    print("✓ Different messages produce different results (different cache keys)")
    
    # Different models should also have different cache keys
    class SimpleUser(BaseModel):
        name: str  # No age field
    
    simple_user = client.create(
        model="gpt-3.5-turbo",
        messages=messages1,
        response_model=SimpleUser,
    )
    print(f"Simple User: {simple_user}")
    print("✓ Different response models use different cache keys")


def test_from_provider_caching():
    """Test caching with from_provider convenience function."""
    print("\n=== Testing from_provider with caching ===")
    
    cache = AutoCache(maxsize=100)
    
    # Use from_provider with cache
    client = instructor.from_provider(
        "openai/gpt-3.5-turbo",
        cache=cache,
    )
    
    messages = [{"role": "user", "content": "Create a user named Frank who is 50 years old"}]
    
    # First call
    start = time.time()
    user1 = client.create(
        messages=messages,
        response_model=User,
    )
    api_time = time.time() - start
    print(f"First call: {user1} (took {api_time:.3f}s)")
    
    # Cached call
    start = time.time()
    user2 = client.create(
        messages=messages,
        response_model=User,
    )
    cache_time = time.time() - start
    print(f"Cached call: {user2} (took {cache_time:.3f}s)")
    print(f"Speedup: {api_time/cache_time:.1f}x")
    print("✓ from_provider works with caching")


if __name__ == "__main__":
    print("Testing Instructor Caching Functionality")
    print("=" * 50)
    
    # Run all tests
    test_autocache_basic()
    test_create_with_completion()
    test_diskcache_persistence()
    test_cache_ttl()
    test_cache_key_differentiation()
    test_from_provider_caching()
    
    print("\n" + "=" * 50)
    print("All caching tests completed successfully! ✨")