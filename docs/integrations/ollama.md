---
title: "Structured outputs with Ollama, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Ollama for local LLM deployment. Learn how to generate structured, type-safe outputs with locally hosted models."
---

# Structured outputs with Ollama, a complete guide w/ instructor

Ollama provides an easy way to run large language models locally. This guide shows you how to use Instructor with Ollama for type-safe, validated responses while maintaining complete control over your data and infrastructure.

## Important Limitations

Before getting started, please note these important limitations when using Instructor with Ollama:

1. **No Function Calling/Tools Support**: Ollama does not support OpenAI's function calling or tools mode. You'll need to use JSON mode instead.
2. **Limited Streaming Support**: Streaming features like `create_partial` are not available.
3. **Mode Restrictions**: Only JSON mode is supported. Tools, MD_JSON, and other modes are not available.
4. **Memory Requirements**: Different models have varying memory requirements:
   - Llama 2 (default): Requires 8.4GB+ system memory
   - Mistral-7B: Requires 4.5GB+ system memory
   - For memory-constrained systems (< 8GB RAM), use quantized models like `mistral-7b-instruct-v0.2-q4`

## Quick Start

Install Instructor with OpenAI compatibility (Ollama uses OpenAI-compatible endpoints):

```bash
pip install "instructor[openai]"
```

Make sure you have Ollama installed and running locally. Visit [Ollama's installation guide](https://ollama.ai/download) for setup instructions.

## Simple User Example (Sync)

```python
import openai
import instructor
from pydantic import BaseModel

# Configure OpenAI client with Ollama endpoint
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require an API key
)

# Enable instructor patches with JSON mode
client = instructor.patch(client, mode=instructor.Mode.JSON)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.completions.create(
    model="mistral-7b-instruct-v0.2-q4",  # Recommended for memory-constrained systems
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import openai
import instructor
from pydantic import BaseModel
import asyncio

# Configure async OpenAI client with Ollama endpoint
client = openai.AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Enable instructor patches with JSON mode
client = instructor.patch(client, mode=instructor.Mode.JSON)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.chat.completions.create(
        model="llama2",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user

# Run async function
user = asyncio.run(extract_user())
print(user)  # User(name='Jason', age=25)
```

## Nested Example

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]

# Create structured output with nested objects
user = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """},
    ],
    response_model=User,
)

print(user)  # User with nested Address objects
```

## Alternative to Streaming

Since Ollama doesn't support streaming with `create_partial`, you can achieve similar results by breaking down your requests into smaller chunks:

```python
class User(BaseModel):
    name: str
    age: int
    bio: Optional[str] = None

# First, extract basic information
user = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": "Extract basic info: Jason is 25 years old"},
    ],
    response_model=User,
)

# Then, add additional information in separate requests
user_with_bio = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": f"Generate a short bio for {user.name}, who is {user.age} years old"},
    ],
    response_model=User,
)
```

## Multiple Items Extraction

Instead of using `create_iterable`, which relies on streaming, you can extract multiple items using a list:

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

class UserList(BaseModel):
    users: List[User]

# Extract multiple users from text
response = client.chat.completions.create(
    model="llama2",
    messages=[
        {"role": "user", "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """},
    ],
    response_model=UserList,
)

for user in response.users:
    print(user)  # Prints each user
```

## Instructor Hooks

Instructor provides several hooks to customize behavior:

### Validation Hook

```python
from instructor import Instructor

def validation_hook(value, retry_count, exception):
    print(f"Validation failed {retry_count} times: {exception}")
    return retry_count < 3  # Retry up to 3 times

instructor.patch(client, validation_hook=validation_hook)
```

### Mode Selection

```python
from instructor import Mode

# Ollama only supports JSON mode
client = instructor.patch(client, mode=Mode.JSON)
```

### Custom Retrying

```python
from instructor import RetryConfig

client = instructor.patch(
    client,
    retry_config=RetryConfig(
        max_retries=3,
        on_retry=lambda *args: print("Retrying..."),
    )
)
```

## Available Models

Ollama supports various models:
- Llama 2 (all variants)
- CodeLlama
- Mistral
- Custom models
- And many more via `ollama pull`

## Best Practices

1. **Model Selection**
   - Choose model size based on hardware capabilities
   - Consider memory constraints
   - Balance speed and accuracy needs

2. **Local Deployment**
   - Monitor system resources
   - Implement proper error handling
   - Consider GPU acceleration

3. **Performance Optimization**
   - Use appropriate quantization
   - Implement caching
   - Monitor memory usage

4. **Working with Limitations**
   - Always use JSON mode
   - Break down complex requests into smaller parts
   - Implement your own batching for multiple items
   - Use proper error handling for unsupported features

## Common Use Cases

- Local Data Processing
- Offline Development
- Privacy-Sensitive Applications
- Rapid Prototyping
- Edge Computing

## Related Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Ollama's latest releases. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Always verify model-specific features and limitations before implementation.
