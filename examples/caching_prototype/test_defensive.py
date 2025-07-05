"""Test defensive caching behavior with different response types."""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from instructor.cache import AutoCache, store_cached_response, load_cached_response
from pydantic import BaseModel, Field
from types import SimpleNamespace


# Set up logging to see the defensive warnings
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("instructor.cache")


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


def test_pydantic_response():
    """Test caching with a proper Pydantic response (like OpenAI)."""
    print("\n=== Testing Pydantic Response ===")
    
    cache = AutoCache(maxsize=10)
    
    # Mock OpenAI-style response with model_dump_json
    class MockCompletion:
        def __init__(self):
            self.id = "chatcmpl-123"
            self.object = "chat.completion"
            self.model = "gpt-3.5-turbo"
            self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            
        def model_dump_json(self):
            return '{"id": "chatcmpl-123", "object": "chat.completion", "model": "gpt-3.5-turbo", "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}'
    
    user = User(name="Alice", age=25)
    user._raw_response = MockCompletion()  # type: ignore
    
    # Store in cache
    store_cached_response(cache, "test_key", user)
    
    # Load from cache
    cached_user = load_cached_response(cache, "test_key", User)
    
    print(f"Original: {user}")
    print(f"Cached: {cached_user}")
    print(f"Raw response ID: {cached_user._raw_response.id}")  # type: ignore
    print(f"Raw response usage: {cached_user._raw_response.usage.total_tokens}")  # type: ignore
    
    assert cached_user.name == user.name
    assert cached_user._raw_response.id == "chatcmpl-123"  # type: ignore
    print("✓ Pydantic response cached and restored correctly")


def test_dict_response():
    """Test caching with a plain dict response."""
    print("\n=== Testing Dict Response ===")
    
    cache = AutoCache(maxsize=10)
    
    user = User(name="Bob", age=30)
    # Plain dict response (some custom provider)
    user._raw_response = {  # type: ignore
        "status": "success",
        "data": {"result": "generated"},
        "metadata": {"provider": "custom"}
    }
    
    # Store in cache
    store_cached_response(cache, "test_key", user)
    
    # Load from cache
    cached_user = load_cached_response(cache, "test_key", User)
    
    print(f"Original: {user}")
    print(f"Cached: {cached_user}")
    print(f"Raw response: {cached_user._raw_response}")  # type: ignore
    
    assert cached_user.name == user.name
    assert cached_user._raw_response["status"] == "success"  # type: ignore
    print("✓ Dict response cached and restored correctly")


def test_unpickleable_response():
    """Test caching with an unpickleable response object."""
    print("\n=== Testing Unpickleable Response ===")
    
    cache = AutoCache(maxsize=10)
    
    # Custom object that can't be JSON serialized
    class UnpickleableResponse:
        def __init__(self):
            self.data = "some data"
            self.file_handle = open(__file__, 'r')  # Can't serialize file handles
            
        def __str__(self):
            return f"UnpickleableResponse(data='{self.data}')"
    
    user = User(name="Charlie", age=35)
    user._raw_response = UnpickleableResponse()  # type: ignore
    
    # Store in cache - should fall back to string
    store_cached_response(cache, "test_key", user)
    
    # Load from cache
    cached_user = load_cached_response(cache, "test_key", User)
    
    print(f"Original: {user}")
    print(f"Cached: {cached_user}")
    print(f"Raw response (as string): {cached_user._raw_response}")  # type: ignore
    
    assert cached_user.name == user.name
    assert "UnpickleableResponse" in str(cached_user._raw_response)  # type: ignore
    print("✓ Unpickleable response fell back to string representation")
    
    # Cleanup
    user._raw_response.file_handle.close()  # type: ignore


def test_no_raw_response():
    """Test caching when there's no raw response."""
    print("\n=== Testing No Raw Response ===")
    
    cache = AutoCache(maxsize=10)
    
    user = User(name="David", age=40)
    # No _raw_response attribute
    
    # Store in cache
    store_cached_response(cache, "test_key", user)
    
    # Load from cache
    cached_user = load_cached_response(cache, "test_key", User)
    
    print(f"Original: {user}")
    print(f"Cached: {cached_user}")
    print(f"Has raw response: {hasattr(cached_user, '_raw_response')}")
    
    assert cached_user.name == user.name
    print("✓ Model without raw response cached correctly")


def test_malformed_completion_response():
    """Test with a response that has model_dump_json but returns invalid JSON."""
    print("\n=== Testing Malformed Completion Response ===")
    
    cache = AutoCache(maxsize=10)
    
    class MalformedCompletion:
        def model_dump_json(self):
            return "{ invalid json"  # Intentionally malformed
    
    user = User(name="Eve", age=45)
    user._raw_response = MalformedCompletion()  # type: ignore
    
    # Store in cache - should fall back to string
    store_cached_response(cache, "test_key", user)
    
    # Load from cache
    cached_user = load_cached_response(cache, "test_key", User)
    
    print(f"Original: {user}")
    print(f"Cached: {cached_user}")
    print(f"Raw response: {cached_user._raw_response}")  # type: ignore
    
    assert cached_user.name == user.name
    print("✓ Malformed JSON fell back gracefully")


if __name__ == "__main__":
    print("Testing Defensive Caching Behavior")
    print("=" * 50)
    
    test_pydantic_response()
    test_dict_response()
    test_unpickleable_response()
    test_no_raw_response()
    test_malformed_completion_response()
    
    print("\n" + "=" * 50)
    print("All defensive caching tests passed! ✨")