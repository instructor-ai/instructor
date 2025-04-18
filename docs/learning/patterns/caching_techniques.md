# Caching Techniques

This guide covers implementing caching with Instructor to reduce API calls, lower costs, and improve response times.

## Basic Example with functools.cache

```python
import functools
from pydantic import BaseModel
import instructor
from openai import OpenAI

# Define the structure you want to extract
class Person(BaseModel):
    name: str
    age: int

# Create OpenAI client with Instructor
client = instructor.from_openai(OpenAI())

# Simple caching with functools.cache
@functools.cache
def extract_person(text: str) -> Person:
    """Extract person information with caching"""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ],
        response_model=Person
    )

# Example usage
result1 = extract_person("My name is John and I am 30 years old")
print(f"First call: {result1}")

# Second call with same input - will use cache
result2 = extract_person("My name is John and I am 30 years old")
print(f"Second call (cached): {result2}")

# Different input - will make a new API call
result3 = extract_person("My name is Maria and I am 25 years old")
print(f"Third call (new): {result3}")
```

```
┌───────────────┐            ┌───────────────┐
│ Request with  │    Check   │ Cache Hit:    │
│ Caching       │ ─────────> │ Return Cached │
│               │            │ Response      │
└───────────────┘            └───────────────┘
        │                           ↑
        │ Cache Miss                │
        ↓                           │
┌───────────────┐            ┌───────────────┐
│ Make API Call │    Then    │ Store Result  │
│               │ ─────────> │ in Cache      │
└───────────────┘            └───────────────┘
```

## Persistent Caching with diskcache

For caching that persists between program runs:

```python
import functools
import inspect
import diskcache
from pydantic import BaseModel
import instructor
from openai import OpenAI

# Define the structure you want to extract
class Person(BaseModel):
    name: str
    age: int

# Create OpenAI client with Instructor
client = instructor.from_openai(OpenAI())

# Create disk cache
cache = diskcache.Cache('./instructor_cache')

def instructor_cache(func):
    """Cache a function that returns a Pydantic model"""
    # Get the return type from function signature
    return_type = inspect.signature(func).return_annotation
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from the function arguments
        key = f"{func.__name__}-{functools._make_key(args, kwargs, typed=False)}"
        
        # Check if the result is already cached
        if (cached := cache.get(key)) is not None:
            # Deserialize from JSON based on the return type
            return return_type.model_validate_json(cached)

        # Call the function and cache its result
        result = func(*args, **kwargs)
        serialized_result = result.model_dump_json()
        cache.set(key, serialized_result)

        return result

    return wrapper

# Apply persistent cache decorator
@instructor_cache
def extract_person(text: str) -> Person:
    """Extract person information with persistent caching"""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ],
        response_model=Person
    )

# Usage
result = extract_person("My name is John and I am 30 years old")
print(f"Result: {result}")
```

## Provider-Specific Caching (Anthropic)

Anthropic provides a built-in prompt caching mechanism:

```python
from instructor import Instructor, Mode, patch
from anthropic import Anthropic
from pydantic import BaseModel

# Define the structure to extract
class Person(BaseModel):
    name: str
    age: int

# Setup Anthropic client with prompt caching
client = Instructor(
    client=Anthropic(),
    create=patch(
        create=Anthropic().beta.prompt_caching.messages.create,
        mode=Mode.ANTHROPIC_TOOLS,
    ),
    mode=Mode.ANTHROPIC_TOOLS,
)

# Context that might be large and reused
context = "This is a large document containing information..."

# Using prompt caching with Anthropic
def extract_with_cached_context(query: str) -> Person:
    response = client.chat.completions.create(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<context>{context}</context>",
                        "cache_control": {"type": "ephemeral"},  # Cache this content
                    },
                    {
                        "type": "text",
                        "text": query,  # This part changes with each request
                    },
                ],
            },
        ],
        response_model=Person,
    )
    return response
```

## Next Steps

- Try [Async Processing](async_processing.md) for concurrent operations
- Learn about [Batch Processing](batch_processing.md) to handle multiple items
- Explore [Validation Basics](../validation/basics.md) for ensuring data quality