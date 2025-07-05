"""Test caching with mocked OpenAI responses to demonstrate functionality."""

import time
import types
from unittest.mock import Mock, patch

import instructor
from instructor.cache import AutoCache, DiskCache
from pydantic import BaseModel, Field
from openai import OpenAI


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


class Weather(BaseModel):
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition")


def create_mock_completion(content: str, completion_id: str = "chatcmpl-123"):
    """Create a mock ChatCompletion response."""
    return types.SimpleNamespace(
        id=completion_id,
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=content,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
                index=0,
            )
        ],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion",
        usage=types.SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )


def test_basic_caching():
    """Test basic caching functionality."""
    print("\n=== Testing Basic Caching ===")
    
    # Create cache and client
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(api_key="fake"), cache=cache)
    
    # Mock the API call
    user_json = User(name="Alice", age=25).model_dump_json()
    mock_response = create_mock_completion(user_json)
    
    with patch.object(client.client.chat.completions, 'create', return_value=mock_response) as mock_create:
        messages = [{"role": "user", "content": "Create a user named Alice who is 25 years old"}]
        
        # First call - should hit the API
        print("First call (should hit mock API)...")
        start = time.time()
        user1 = client.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=User,
        )
        api_time = time.time() - start
        print(f"Result: {user1}")
        print(f"Time: {api_time:.3f}s")
        print(f"API calls made: {mock_create.call_count}")
        
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
        print(f"API calls made: {mock_create.call_count}")
        
        # Verify
        assert mock_create.call_count == 1, "API should only be called once"
        assert user1.name == user2.name
        assert user1.age == user2.age
        print("✓ Cache working correctly - API called only once")


def test_create_with_completion():
    """Test that create_with_completion preserves raw responses."""
    print("\n=== Testing create_with_completion ===")
    
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(api_key="fake"), cache=cache)
    
    # Mock response
    weather_json = Weather(temperature=20.0, condition="sunny").model_dump_json()
    mock_response = create_mock_completion(weather_json, "chatcmpl-weather-123")
    
    with patch.object(client.client.chat.completions, 'create', return_value=mock_response) as mock_create:
        messages = [{"role": "user", "content": "What's the weather like?"}]
        
        # First call
        print("First call with create_with_completion...")
        weather1, completion1 = client.create_with_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=Weather,
        )
        print(f"Weather: {weather1}")
        print(f"Completion ID: {completion1.id}")
        print(f"Usage: {completion1.usage.total_tokens} tokens")
        
        # Second call - should return cached
        print("\nSecond call (from cache)...")
        weather2, completion2 = client.create_with_completion(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=Weather,
        )
        print(f"Weather: {weather2}")
        print(f"Completion ID: {completion2.id}")
        
        # Verify
        assert mock_create.call_count == 1
        assert weather1.temperature == weather2.temperature
        assert weather1.condition == weather2.condition
        assert completion1.id == completion2.id
        assert completion1.usage.total_tokens == completion2.usage.total_tokens
        print("✓ Raw completion correctly cached and restored")


def test_cache_key_differentiation():
    """Test that different inputs use different cache keys."""
    print("\n=== Testing Cache Key Differentiation ===")
    
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(api_key="fake"), cache=cache)
    
    # Different responses for different messages
    responses = {
        "Alice": create_mock_completion(User(name="Alice", age=25).model_dump_json()),
        "Bob": create_mock_completion(User(name="Bob", age=30).model_dump_json()),
    }
    
    def mock_create(*args, **kwargs):
        content = kwargs.get('messages', [{}])[0].get('content', '')
        if "Alice" in content:
            return responses["Alice"]
        return responses["Bob"]
    
    with patch.object(client.client.chat.completions, 'create', side_effect=mock_create) as mock_api:
        # First message
        user1 = client.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Create user Alice"}],
            response_model=User,
        )
        print(f"User 1: {user1}")
        
        # Different message
        user2 = client.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Create user Bob"}],
            response_model=User,
        )
        print(f"User 2: {user2}")
        
        # Verify different results
        assert user1.name != user2.name
        assert mock_api.call_count == 2
        print("✓ Different messages use different cache keys")
        
        # Same message as first - should be cached
        user3 = client.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Create user Alice"}],
            response_model=User,
        )
        assert user3.name == user1.name
        assert mock_api.call_count == 2  # No new API call
        print("✓ Repeated message uses cached result")


def test_cache_with_different_models():
    """Test that different response models use different cache keys."""
    print("\n=== Testing Different Response Models ===")
    
    cache = AutoCache(maxsize=100)
    client = instructor.from_openai(OpenAI(api_key="fake"), cache=cache)
    
    class SimpleUser(BaseModel):
        name: str
    
    class DetailedUser(BaseModel):
        name: str
        age: int
        email: str = Field(default="user@example.com")
    
    # Mock responses
    simple_response = create_mock_completion(SimpleUser(name="Alice").model_dump_json())
    detailed_response = create_mock_completion(
        DetailedUser(name="Alice", age=25, email="alice@example.com").model_dump_json()
    )
    
    call_count = 0
    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get('response_model') == SimpleUser:
            return simple_response
        return detailed_response
    
    with patch.object(client.client.chat.completions, 'create', side_effect=mock_create):
        messages = [{"role": "user", "content": "Create user Alice"}]
        
        # First call with SimpleUser
        simple = client.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=SimpleUser,
        )
        print(f"Simple user: {simple}")
        assert call_count == 1
        
        # Same message but DetailedUser - should NOT use cache
        detailed = client.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=DetailedUser,
        )
        print(f"Detailed user: {detailed}")
        assert call_count == 2
        
        # Repeat SimpleUser - should use cache
        simple2 = client.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_model=SimpleUser,
        )
        assert simple2.name == simple.name
        assert call_count == 2  # No new call
        
        print("✓ Different response models use different cache keys")


def test_diskcache_persistence():
    """Test that DiskCache persists across client instances."""
    print("\n=== Testing DiskCache Persistence ===")
    
    import tempfile
    import shutil
    
    tmpdir = tempfile.mkdtemp()
    
    try:
        # First client
        cache1 = DiskCache(directory=tmpdir)
        client1 = instructor.from_openai(OpenAI(api_key="fake"), cache=cache1)
        
        user_json = User(name="Persistent Pete", age=42).model_dump_json()
        mock_response = create_mock_completion(user_json)
        
        with patch.object(client1.client.chat.completions, 'create', return_value=mock_response) as mock1:
            messages = [{"role": "user", "content": "Create persistent user"}]
            
            user1 = client1.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_model=User,
            )
            print(f"First client result: {user1}")
            assert mock1.call_count == 1
        
        # Second client with same cache directory
        cache2 = DiskCache(directory=tmpdir)
        client2 = instructor.from_openai(OpenAI(api_key="fake"), cache=cache2)
        
        with patch.object(client2.client.chat.completions, 'create', return_value=mock_response) as mock2:
            # Same call - should use persisted cache
            user2 = client2.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_model=User,
            )
            print(f"Second client result: {user2}")
            assert mock2.call_count == 0  # No API call!
            assert user2.name == user1.name
            assert user2.age == user1.age
            
        print("✓ Cache persisted across client instances")
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("Testing Instructor Caching with Mocked Responses")
    print("=" * 50)
    
    test_basic_caching()
    test_create_with_completion()
    test_cache_key_differentiation()
    test_cache_with_different_models()
    test_diskcache_persistence()
    
    print("\n" + "=" * 50)
    print("All caching tests completed successfully! ✨")