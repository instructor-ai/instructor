---
title: "Structured outputs with Anthropic, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Anthropic's Claude models. Learn how to generate structured, type-safe outputs with state-of-the-art AI capabilities."
---

# Structured outputs with Anthropic

Anthropic's Claude models offer powerful language capabilities with a focus on safety and reliability. This guide shows you how to use Instructor with Anthropic's models for type-safe, validated responses.

## Quick Start

Install Instructor with Anthropic support:

```bash
pip install "instructor[anthropic]"
```

## Simple User Example (Sync)

```python
from anthropic import Anthropic
import instructor
from pydantic import BaseModel

# Initialize the client
client = Anthropic(api_key="your_anthropic_api_key")

# Enable instructor patches
client = instructor.from_anthropic(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.messages.create(
    model="claude-3-opus-20240229",  # or other available models
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from anthropic import AsyncAnthropic
import instructor
from pydantic import BaseModel
import asyncio

# Initialize async client
client = AsyncAnthropic(api_key="your_anthropic_api_key")

# Enable instructor patches
client = instructor.from_anthropic(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.messages.create(
        model="claude-3-opus-20240229",
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
user = client.messages.create(
    model="claude-3-opus-20240229",
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

## Streaming Support

Anthropic's Claude models provide comprehensive streaming support through Instructor:

### Available Streaming Methods

1. **Basic Streaming**: ✅ Fully supported
2. **Iterable Streaming**: ✅ Fully supported
3. **Async Support**: ✅ Available for all streaming operations

```python
from typing import List
import asyncio
from anthropic import AsyncAnthropic
import instructor

class User(BaseModel):
    name: str
    age: int

async def process_users():
    client = AsyncAnthropic(api_key="your_anthropic_api_key")
    client = instructor.from_anthropic(client)

    # Example of basic streaming
    async for partial_user in client.messages.create_partial(
        model="claude-3-opus-20240229",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    ):
        print(f"Partial result: {partial_user}")

    # Example of iterable streaming
    users = client.messages.create_iterable(
        model="claude-3-opus-20240229",
        messages=[
            {"role": "user", "content": """
                Extract users:
                1. Jason is 25 years old
                2. Sarah is 30 years old
                3. Mike is 28 years old
            """},
        ],
        response_model=User,
    )

    async for user in users:
        print(f"User: {user}")

# Run the async function
asyncio.run(process_users())
```

This implementation provides efficient streaming capabilities for both single and multiple object extraction tasks.

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

Anthropic offers several Claude models:
- Claude 3 Opus (Most capable)
- Claude 3 Sonnet (Balanced performance)
- Claude 3 Haiku (Fast and efficient)
- Claude 2.1
- Claude 2.0
- Claude Instant

## Best Practices

1. **Model Selection**
   - Choose model based on task complexity
   - Consider latency requirements
   - Monitor token usage and costs
   - Use appropriate context lengths

2. **Optimization Tips**
   - Structure prompts effectively
   - Use system messages appropriately
   - Implement caching strategies
   - Monitor API usage

3. **Error Handling**
   - Implement proper validation
   - Handle rate limits gracefully
   - Monitor model responses
   - Use appropriate timeout settings

## Common Use Cases

- Data Extraction
- Content Generation
- Document Analysis
- Complex Reasoning Tasks
- Multi-step Processing

## Related Resources

- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Anthropic's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
