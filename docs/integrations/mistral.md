---
title: "Structured outputs with Mistral, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Mistral and Mixtral models. Learn how to generate structured, type-safe outputs with these powerful open-source models."
---

# Mistral & Mixtral Integration with Instructor

Mistral AI's models, including Mistral and Mixtral, offer powerful open-source alternatives for structured output generation. This guide shows you how to leverage these models with Instructor for type-safe, validated responses.

## Quick Start

Install Instructor with Mistral support:

```bash
pip install "instructor[mistralai]"
```

## Simple User Example (Sync)

```python
from mistralai.client import MistralClient
import instructor
from pydantic import BaseModel

# Enable instructor patches for Mistral client
client = instructor.from_mistral(MistralClient(), mode=instructor.Mode.MISTRAL_TOOLS)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.complete(
    model="mistral-large-latest",  # or "mixtral-8x7b-instruct"
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from mistralai.async_client import MistralAsyncClient
import instructor
from pydantic import BaseModel
import asyncio

# Enable instructor patches for async Mistral client
client = instructor.from_mistral(MistralAsyncClient(), mode=instructor.Mode.MISTRAL_TOOLS, use_async=True)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.chat.complete(
        model="mistral-large-latest",
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
user = client.chat.complete(
    model="mixtral-8x7b-instruct",
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

Mistral models have limited streaming support through Instructor. Here are the current capabilities and limitations:

1. **Full Streaming**: Not currently supported
2. **Partial Streaming**: Not currently supported
3. **Iterable Streaming**: Limited support for multiple object extraction
4. **Async Support**: Available for non-streaming operations

### Streaming Limitations
- Full streaming is not currently implemented
- Partial streaming is not available
- Iterable responses must be processed as complete responses
- Use async client for better performance with large responses

### Performance Considerations
- Use batch processing for multiple extractions
- Implement proper error handling
- Consider response size limitations
- Set appropriate timeouts for large responses

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.chat.complete(
    model="mixtral-8x7b-instruct",
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

print(users)  # Prints complete response
```

## Instructor Hooks

Instructor provides several hooks to customize behavior:

### Validation Hook

```python
from instructor import Instructor

def validation_hook(value, retry_count, exception):
    print(f"Validation failed {retry_count} times: {exception}")
    return retry_count < 3  # Retry up to 3 times

client = instructor.from_mistral(client, validation_hook=validation_hook)
```

### Mode Selection

```python
from instructor import Mode

# Use MISTRAL_TOOLS mode for best results
client = instructor.from_mistral(client, mode=Mode.MISTRAL_TOOLS)
```

### Custom Retrying

```python
from instructor import RetryConfig

client = instructor.from_mistral(
    client,
    retry_config=RetryConfig(
        max_retries=3,
        on_retry=lambda *args: print("Retrying..."),
    )
)
```

## Model Options

Mistral AI provides several powerful models:
- Mistral-7B
- Mixtral-8x7B
- Custom fine-tuned variants
- Hosted API options

## Best Practices

1. **Model Selection**
   - Use Mixtral-8x7B for complex tasks
   - Mistral-7B for simpler extractions
   - Consider latency requirements

2. **Optimization Tips**
   - Use async client for better performance
   - Implement proper error handling
   - Monitor token usage

3. **Deployment Considerations**
   - Self-hosted vs. API options
   - Resource requirements
   - Scaling strategies

## Common Use Cases

- Data Extraction
- Content Structuring
- API Response Formatting
- Document Analysis
- Configuration Generation

## Troubleshooting

Common issues and solutions:
1. Model Loading Issues
2. Memory Management
3. Response Validation
4. API Rate Limits

## Related Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest Mistral AI releases. Check the [changelog](../../CHANGELOG.md) for updates.
