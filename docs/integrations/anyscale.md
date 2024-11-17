---
title: "Structured outputs with Anyscale, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Anyscale's LLM endpoints. Learn how to generate structured, type-safe outputs with Anyscale's powerful hosted models."
---

# Structured outputs with Anyscale, a complete guide w/ instructor

Anyscale provides hosted endpoints for various open-source models, offering a reliable platform for structured output generation. This guide shows you how to use Instructor with Anyscale's endpoints for type-safe, validated responses.

## Quick Start

Install Instructor with OpenAI compatibility (Anyscale uses OpenAI-compatible endpoints):

```bash
pip install "instructor[openai]"
```

⚠️ **Important**: You must set your Anyscale API key before using the client. You can do this in two ways:

1. Set the environment variable:
```bash
export ANYSCALE_API_KEY='your_anyscale_api_key'
```

2. Or provide it directly to the client:
```python
import os
from openai import OpenAI

# Configure OpenAI client with Anyscale endpoint
client = OpenAI(
    api_key=os.getenv('ANYSCALE_API_KEY', 'your_anyscale_api_key'),
    base_url="https://api.endpoints.anyscale.com/v1"
)
```

## Simple User Example (Sync)

```python
import openai
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.completions.create(
    model="meta-llama/Llama-2-70b-chat-hf",  # or other available models
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

# Configure async OpenAI client with Anyscale endpoint
client = openai.AsyncOpenAI(
    api_key="your_anyscale_api_key",
    base_url="https://api.endpoints.anyscale.com/v1"
)

# Enable instructor patches
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.chat.completions.create(
        model="meta-llama/Llama-2-70b-chat-hf",
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
    model="meta-llama/Llama-2-70b-chat-hf",
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

Anyscale provides streaming support through their OpenAI-compatible endpoints, with some limitations:

- **Full Streaming**: ✅ Supported
- **Partial Streaming**: ⚠️ Limited support (may experience inconsistent behavior)
- **Iterable Streaming**: ✅ Supported
- **Async Support**: ✅ Supported

### Error Handling for Streaming

```python
from openai import OpenAIError
import os

class User(BaseModel):
    name: str
    age: int
    bio: str

try:
    # Stream partial objects as they're generated
    for partial_user in client.chat.completions.create_partial(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[
            {"role": "user", "content": "Create a user profile for Jason, age 25"},
        ],
        response_model=User,
    ):
        print(f"Current state: {partial_user}")
except OpenAIError as e:
    if "api_key" in str(e).lower():
        print("Error: Invalid or missing Anyscale API key. Please check your ANYSCALE_API_KEY.")
    elif "rate_limit" in str(e).lower():
        print("Error: Rate limit exceeded. Please wait before retrying.")
    else:
        print(f"OpenAI API error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

**Important Notes on Streaming:**
- Full streaming is supported for complete response generation
- Partial streaming has limited support and may not work consistently across all models
- Some models may exhibit slower streaming performance
- For production use, thoroughly test streaming capabilities with your specific model
- Consider implementing fallback mechanisms for partial streaming scenarios
- Monitor streaming performance and implement appropriate error handling
- Handle API key and rate limit errors appropriately

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.chat.completions.create_iterable(
    model="meta-llama/Llama-2-70b-chat-hf",
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

## Available Models

Anyscale provides access to various open-source models:
- Llama 2 (7B, 13B, 70B variants)
- CodeLlama
- Mistral
- Other open-source models

## Best Practices

1. **Model Selection**
   - Choose model size based on task complexity
   - Consider latency requirements
   - Monitor token usage and costs

2. **Optimization Tips**
   - Use appropriate batch sizes
   - Implement caching strategies
   - Monitor API usage

3. **Error Handling**
   - Implement proper validation
   - Handle rate limits gracefully
   - Monitor model responses

## Common Use Cases

- Data Extraction
- Content Generation
- Document Analysis
- API Response Formatting
- Configuration Generation

## Troubleshooting

Common issues and solutions:

### 1. API Key Issues
- **Missing API Key**: Ensure `ANYSCALE_API_KEY` environment variable is set
- **Invalid API Key**: Verify the key is valid and has not expired
- **Permission Issues**: Check if your API key has access to the required models
- **Rate Limiting**: Monitor your API usage and implement proper rate limiting

### 2. Streaming Issues
- **Connection Timeouts**: Implement proper timeout handling
- **Partial Response Errors**: Handle incomplete responses gracefully
- **Memory Issues**: Monitor memory usage with large streaming responses
- **Rate Limits**: Implement backoff strategies for streaming requests

### 3. Model-Specific Issues
- **Model Access**: Ensure your account has access to required models
- **Context Length**: Monitor and handle context length limits
- **Token Usage**: Track token usage to avoid quota issues
- **Response Format**: Handle model-specific response formats

### 4. Integration Issues
- **Version Compatibility**: Keep OpenAI and Instructor versions in sync
- **Type Validation**: Handle validation errors with proper retry logic
- **Schema Complexity**: Simplify complex schemas if needed
- **Async/Sync Usage**: Use appropriate client for your use case

## Related Resources

- [Anyscale Endpoints Documentation](https://docs.endpoints.anyscale.com/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Anyscale's OpenAI-compatible endpoints. Check the [changelog](../../CHANGELOG.md) for updates.
