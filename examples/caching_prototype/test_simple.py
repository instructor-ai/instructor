"""Simple test to verify caching functionality without API calls."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from instructor.cache import AutoCache, DiskCache, make_cache_key
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The user's name")
    age: int = Field(description="The user's age")


def test_autocache():
    """Test AutoCache basic functionality."""
    print("Testing AutoCache...")
    cache = AutoCache(maxsize=10)
    
    # Test set and get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    print("✓ Basic set/get works")
    
    # Test cache miss
    assert cache.get("nonexistent") is None
    print("✓ Cache miss returns None")
    
    # Test LRU eviction
    for i in range(12):
        cache.set(f"key{i}", f"value{i}")
    
    # First keys should be evicted
    assert cache.get("key0") is None
    assert cache.get("key1") is None
    # Recent keys should still be there
    assert cache.get("key11") == "value11"
    print("✓ LRU eviction works")


def test_diskcache():
    """Test DiskCache basic functionality."""
    print("\nTesting DiskCache...")
    import tempfile
    import shutil
    
    tmpdir = tempfile.mkdtemp()
    try:
        cache = DiskCache(directory=tmpdir)
        
        # Test set and get
        cache.set("key1", {"data": "value1"})
        assert cache.get("key1") == {"data": "value1"}
        print("✓ Basic set/get works")
        
        # Test persistence
        cache2 = DiskCache(directory=tmpdir)
        assert cache2.get("key1") == {"data": "value1"}
        print("✓ Persistence works")
        
        # Test TTL
        cache.set("key2", "value2", ttl=1)
        assert cache.get("key2") == "value2"
        import time
        time.sleep(2)
        assert cache.get("key2") is None
        print("✓ TTL expiration works")
        
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_cache_key():
    """Test cache key generation."""
    print("\nTesting cache key generation...")
    
    # Same inputs should produce same key
    key1 = make_cache_key(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-3.5-turbo",
        response_model=User,
        mode="TOOLS"
    )
    key2 = make_cache_key(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-3.5-turbo",
        response_model=User,
        mode="TOOLS"
    )
    assert key1 == key2
    print("✓ Same inputs produce same key")
    
    # Different messages should produce different keys
    key3 = make_cache_key(
        messages=[{"role": "user", "content": "goodbye"}],
        model="gpt-3.5-turbo",
        response_model=User,
        mode="TOOLS"
    )
    assert key1 != key3
    print("✓ Different messages produce different keys")
    
    # Different models should produce different keys
    key4 = make_cache_key(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4",
        response_model=User,
        mode="TOOLS"
    )
    assert key1 != key4
    print("✓ Different models produce different keys")
    
    # Key should be a hex string (SHA-256)
    assert len(key1) == 64
    assert all(c in "0123456789abcdef" for c in key1)
    print("✓ Key is valid SHA-256 hex string")


def test_pydantic_serialization():
    """Test that Pydantic models can be cached properly."""
    print("\nTesting Pydantic model caching...")
    
    cache = AutoCache(maxsize=10)
    
    # Create a user instance
    user = User(name="Alice", age=30)
    
    # Cache the JSON representation
    cache.set("user1", user.model_dump_json())
    
    # Retrieve and reconstruct
    cached_json = cache.get("user1")
    reconstructed = User.model_validate_json(cached_json)
    
    assert reconstructed.name == user.name
    assert reconstructed.age == user.age
    print("✓ Pydantic model serialization/deserialization works")


if __name__ == "__main__":
    print("Testing Instructor Cache Components")
    print("=" * 50)
    
    test_autocache()
    test_diskcache()
    test_cache_key()
    test_pydantic_serialization()
    
    print("\n" + "=" * 50)
    print("All cache component tests passed! ✨")