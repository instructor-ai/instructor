---
title: "Cohere Integration with Instructor | Structured Output Guide"
description: "Complete guide to using Instructor with Cohere's language models. Learn how to generate structured, type-safe outputs with enterprise-ready AI capabilities."
---

# Cohere Integration with Instructor

Cohere provides powerful language models optimized for enterprise use cases. This guide shows you how to use Instructor with Cohere's models for type-safe, validated responses.

## Quick Start

Install Instructor with Cohere support:

```bash
pip install "instructor[cohere]"
```

## Simple User Example (Sync)

```python
import cohere
import instructor
from pydantic import BaseModel

# Initialize the client
client = cohere.Client('your_api_key')

# Enable instructor patches
client = instructor.from_cohere(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.generate(
    prompt="Extract: Jason is 25 years old",
    model='command',  # or other available models
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import cohere
import instructor
from pydantic import BaseModel
import asyncio

# Initialize async client
client = cohere.AsyncClient('your_api_key')

# Enable instructor patches
client = instructor.from_cohere(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.generate(
        prompt="Extract: Jason is 25 years old",
        model='command',
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
    model='command',
    response_model=User,
)

print(user)  # User with nested Address objects
```

## Partial Streaming Example

Note: Cohere's current API does not support partial streaming of structured responses. The streaming functionality returns complete text chunks rather than partial objects. We recommend using the standard synchronous or asynchronous methods for structured output generation.

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
    model='command',
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

Cohere offers several model options:
- Command (Latest generation)
- Command-Light (Faster, more efficient)
- Command-Nightly (Experimental features)
- Custom-trained models (Enterprise)

## Best Practices

1. **Model Selection**
   - Choose model based on task complexity
   - Consider latency requirements
   - Monitor token usage
   - Use appropriate model versions

2. **Optimization Tips**
   - Structure prompts effectively
   - Use appropriate temperature settings
   - Implement caching strategies
   - Monitor API usage

3. **Error Handling**
   - Implement proper validation
   - Handle rate limits gracefully
   - Monitor model responses
   - Use appropriate timeout settings

## Common Use Cases

- Enterprise Data Processing
- Content Generation
- Document Analysis
- Semantic Search Integration
- Classification Tasks

## Troubleshooting

Common issues and solutions:
1. API Authentication
2. Rate Limiting
3. Response Validation
4. Model Selection

## Related Resources

- [Cohere API Documentation](https://docs.cohere.com/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Cohere's latest API versions. Check the [changelog](../../CHANGELOG.md) for updates.

Note: Some features like partial streaming may not be available due to API limitations. Always check the latest documentation for feature availability.
