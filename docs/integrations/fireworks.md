---
title: "Structured outputs with Fireworks, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Fireworks AI models. Learn how to generate structured, type-safe outputs with high-performance, cost-effective AI capabilities."
---

# Structured outputs with Fireworks, a complete guide w/ instructor

Fireworks provides efficient and cost-effective AI models with enterprise-grade reliability. This guide shows you how to use Instructor with Fireworks's models for type-safe, validated responses.

## Quick Start

Install Instructor with Fireworks support:

```bash
pip install "instructor[fireworks]"
```

## Simple User Example (Sync)

```python
from fireworks.client import Client
import instructor
from pydantic import BaseModel

# Initialize the client
client = Client(api_key='your_api_key')

# Enable instructor patches
client = instructor.from_fireworks(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.generate(
    prompt="Extract: Jason is 25 years old",
    model='accounts/fireworks/models/llama-v2-7b',  # or other available models
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from fireworks.client import AsyncClient
import instructor
from pydantic import BaseModel
import asyncio

# Initialize async client
client = AsyncClient(api_key='your_api_key')

# Enable instructor patches
client = instructor.from_fireworks(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.generate(
        prompt="Extract: Jason is 25 years old",
        model='accounts/fireworks/models/llama-v2-7b',
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
    model='accounts/fireworks/models/llama-v2-7b',
    response_model=User,
)

print(user)  # User with nested Address objects
```

## Streaming Support and Limitations

Fireworks provides streaming capabilities with some limitations:

- **Full Streaming**: ⚠️ Limited support (model-dependent)
- **Partial Streaming**: ⚠️ Limited support (may experience inconsistent behavior)
- **Iterable Streaming**: ✅ Supported
- **Async Support**: ✅ Supported

### Partial Streaming Example

```python
class User(BaseModel):
    name: str
    age: int
    bio: str

# Stream partial objects as they're generated
for partial_user in client.stream_generate(
    prompt="Create a user profile for Jason, age 25",
    model='accounts/fireworks/models/llama-v2-7b',
    response_model=User,
):
    print(f"Current state: {partial_user}")
    # Fields will populate gradually as they're generated
```

**Important Notes on Streaming:**
- Full streaming support varies by model and configuration
- Partial streaming has limited support and may require additional error handling
- Some models may not support streaming at all
- Consider implementing fallback mechanisms for streaming scenarios
- Test streaming capabilities with your specific model before deployment
- Monitor streaming performance and implement appropriate error handling
- For production use, implement non-streaming fallbacks

### Model-Specific Streaming Support

1. **Llama-2 Models**
   - Basic streaming support
   - May experience chunked responses
   - Recommended for non-critical streaming use cases

2. **Mistral Models**
   - Limited streaming support
   - Better suited for non-streaming operations
   - Use with appropriate fallback mechanisms

3. **Custom Models**
   - Streaming capabilities vary
   - Requires thorough testing
   - May need model-specific optimizations

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
    model='accounts/fireworks/models/llama-v2-7b',
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

Fireworks offers several model options:
- Llama-2 (various sizes)
- Mistral (various configurations)
- Custom fine-tuned models
- Enterprise deployments

## Best Practices

1. **Model Selection**
   - Choose models with known streaming support
   - Consider cost-performance ratio
   - Monitor usage and costs
   - Use appropriate context lengths

2. **Optimization Tips**
   - Implement proper caching
   - Use non-streaming fallbacks
   - Monitor token usage
   - Use appropriate temperature settings

3. **Error Handling**
   - Implement streaming-specific error handling
   - Handle rate limits
   - Monitor model responses
   - Use appropriate timeout settings

## Common Use Cases

- Enterprise Applications
- Cost-Effective Processing
- High-Performance Computing
- Research Applications
- Production Deployments

## Related Resources

- [Fireworks Documentation](https://docs.fireworks.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Fireworks's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Always verify model-specific features and limitations before implementing streaming functionality in production environments.
