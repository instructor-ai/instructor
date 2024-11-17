---
title: "Structured outputs with LiteLLM, a complete guide w/ instructor"
description: "Complete guide to using Instructor with LiteLLM's unified interface. Learn how to generate structured, type-safe outputs across multiple LLM providers."
---

# Structured outputs with LiteLLM, a complete guide w/ instructor

LiteLLM provides a unified interface for multiple LLM providers, making it easy to switch between different models and providers. This guide shows you how to use Instructor with LiteLLM for type-safe, validated responses across various LLM providers.

## Quick Start

Install Instructor with LiteLLM support:

```bash
pip install "instructor[litellm]"
```

## Simple User Example (Sync)

```python
from litellm import completion
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_litellm()

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.completion(
    model="gpt-3.5-turbo",  # Can use any supported model
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from litellm import acompletion
import instructor
from pydantic import BaseModel
import asyncio

# Enable instructor patches for async
client = instructor.from_litellm()

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.acompletion(
        model="gpt-3.5-turbo",
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
user = client.completion(
    model="gpt-3.5-turbo",
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

## Streaming Support and Limitations

LiteLLM's streaming capabilities vary by provider. Here's a comprehensive breakdown:

### Provider-Specific Streaming Support

| Provider | Full Streaming | Partial Streaming | Iterable Streaming | Async Support |
|----------|---------------|-------------------|-------------------|---------------|
| OpenAI   | ✅ Full       | ✅ Full           | ✅ Full           | ✅ Full       |
| Anthropic| ✅ Full       | ✅ Full           | ✅ Full           | ✅ Full       |
| Azure    | ✅ Full       | ✅ Full           | ✅ Full           | ✅ Full       |
| Google   | ✅ Full       | ⚠️ Limited        | ✅ Full           | ✅ Full       |
| Cohere   | ❌ None       | ❌ None           | ✅ Full           | ✅ Full       |
| AWS      | ⚠️ Limited    | ⚠️ Limited        | ✅ Full           | ✅ Full       |
| Mistral  | ❌ None       | ❌ None           | ✅ Full           | ✅ Full       |

### Partial Streaming Example

```python
class User(BaseModel):
    name: str
    age: int
    bio: str

# Stream partial objects as they're generated
for partial_user in client.stream_completion(
    model="gpt-3.5-turbo",  # Choose a provider with streaming support
    messages=[
        {"role": "user", "content": "Create a user profile for Jason, age 25"},
    ],
    response_model=User,
):
    print(f"Current state: {partial_user}")
    # Fields will populate gradually as they're generated
```

**Important Notes on Streaming:**
- Streaming capabilities depend entirely on the chosen provider
- Some providers may not support streaming at all
- Partial streaming behavior varies significantly between providers
- Always implement fallback mechanisms for providers without streaming
- Test streaming functionality with your specific provider before deployment
- Consider implementing provider-specific error handling
- Monitor streaming performance across different providers

### Provider-Specific Considerations

1. **OpenAI/Azure/Anthropic**
   - Full streaming support
   - Reliable partial streaming
   - Consistent performance

2. **Google/AWS**
   - Limited partial streaming
   - May require additional error handling
   - Consider implementing fallbacks

3. **Cohere/Mistral**
   - No streaming support
   - Use non-streaming alternatives
   - Implement appropriate fallbacks

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.completion_iterable(
    model="gpt-3.5-turbo",
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

## Supported Providers

LiteLLM supports multiple providers:
- OpenAI
- Anthropic
- Azure
- AWS Bedrock
- Google Vertex AI
- Cohere
- Hugging Face
- And many more

## Best Practices

1. **Provider Selection**
   - Choose providers based on streaming requirements
   - Consider cost and performance
   - Monitor usage across providers
   - Implement provider-specific fallback strategies

2. **Optimization Tips**
   - Use provider-specific features
   - Implement proper caching
   - Monitor costs across providers
   - Handle provider-specific errors

3. **Error Handling**
   - Implement provider-specific handling
   - Use proper fallback logic
   - Monitor provider availability
   - Handle rate limits properly

## Common Use Cases

- Multi-Provider Applications
- Provider Fallback Systems
- Cost Optimization
- Cross-Provider Testing
- Unified API Integration

## Troubleshooting

Common issues and solutions:
1. Provider Authentication
2. Model Availability
3. Provider-Specific Errors
4. Rate Limiting
5. Streaming Compatibility

## Related Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with LiteLLM's latest releases. Check the [changelog](../../CHANGELOG.md) for updates.

Note: Always verify provider-specific features and limitations in their respective documentation before implementation.
