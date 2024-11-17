---
title: "Cerebras Integration with Instructor | Structured Output Guide"
description: "Complete guide to using Instructor with Cerebras's hardware-accelerated AI models. Learn how to generate structured, type-safe outputs with high-performance computing."
---

# Cerebras Integration with Instructor

Cerebras provides hardware-accelerated AI models optimized for high-performance computing environments. This guide shows you how to use Instructor with Cerebras's models for type-safe, validated responses.

## Quick Start

Install Instructor with Cerebras support:

```bash
pip install "instructor[cerebras]"
```

## Simple User Example (Sync)

```python
from cerebras.client import Client
import instructor
from pydantic import BaseModel

# Initialize the client
client = Client(api_key='your_api_key')

# Enable instructor patches
client = instructor.from_cerebras(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.generate(
    prompt="Extract: Jason is 25 years old",
    model='cerebras/btlm-3b-8k',  # or other available models
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from cerebras.client import AsyncClient
import instructor
from pydantic import BaseModel
import asyncio

# Initialize async client
client = AsyncClient(api_key='your_api_key')

# Enable instructor patches
client = instructor.from_cerebras(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.generate(
        prompt="Extract: Jason is 25 years old",
        model='cerebras/btlm-3b-8k',
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
user = client.generate(
    prompt="""
        Extract: Jason is 25 years old.
        He lives at 123 Main St, New York, USA
        and has a summer house at 456 Beach Rd, Miami, USA
    """,
    model='cerebras/btlm-3b-8k',
    response_model=User,
)

print(user)  # User with nested Address objects
```

## Partial Streaming Example

Note: Cerebras's current API does not support partial streaming of structured responses. The streaming functionality returns complete text chunks rather than partial objects. We recommend using the standard synchronous or asynchronous methods for structured output generation.

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.generate_iterable(
    prompt="""
        Extract users:
        1. Jason is 25 years old
        2. Sarah is 30 years old
        3. Mike is 28 years old
    """,
    model='cerebras/btlm-3b-8k',
    response_model=User,
)

for user in users:
    print(user)  # Prints each user as it's extracted
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

### Mode Hooks

```python
from instructor import Mode

# Use different modes for different scenarios
client = instructor.patch(client, mode=Mode.JSON)  # JSON mode
client = instructor.patch(client, mode=Mode.TOOLS)  # Tools mode
client = instructor.patch(client, mode=Mode.MD_JSON)  # Markdown JSON mode
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

Cerebras offers several model options:
- BTLM-3B-8K
- BTLM-7B-8K
- Custom-trained models
- Enterprise deployments

## Best Practices

1. **Model Selection**
   - Choose model based on performance needs
   - Consider hardware requirements
   - Monitor resource usage
   - Use appropriate model sizes

2. **Optimization Tips**
   - Leverage hardware acceleration
   - Optimize batch processing
   - Implement caching strategies
   - Monitor system resources

3. **Error Handling**
   - Implement proper validation
   - Handle hardware-specific errors
   - Monitor model responses
   - Use appropriate timeout settings

## Common Use Cases

- High-Performance Computing
- Large-Scale Processing
- Enterprise Deployments
- Research Applications
- Batch Processing

## Related Resources

- [Cerebras Documentation](https://docs.cerebras.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Cerebras's latest API versions. Check the [changelog](../../CHANGELOG.md) for updates.

Note: Some features like partial streaming may not be available due to API limitations. Always check the latest documentation for feature availability.
