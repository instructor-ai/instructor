---
title: "Structured outputs with OpenAI, a complete guide w/ instructor"
description: "Learn how to use Instructor with OpenAI's models for type-safe, structured outputs. Complete guide with examples and best practices for GPT-4 and other OpenAI models."
---

# OpenAI Integration with Instructor

OpenAI is the primary integration for Instructor, offering robust support for structured outputs with GPT-3.5, GPT-4, and future models. This guide covers everything you need to know about using OpenAI with Instructor for type-safe, validated responses.

## Quick Start

Install Instructor with OpenAI support:

```bash
pip install "instructor[openai]"
```

⚠️ **Important**: You must set your OpenAI API key before using the client. You can do this in two ways:

1. Set the environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

2. Or provide it directly to the client:
```python
import os
from openai import OpenAI
client = OpenAI(api_key='your-api-key-here')
```

## Simple User Example (Sync)

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Enable instructor patches for OpenAI client
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import os
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel
import asyncio

# Initialize with API key
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Enable instructor patches for async OpenAI client
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
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
    model="gpt-4-turbo-preview",
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

OpenAI provides comprehensive streaming support through multiple methods, but proper setup and error handling are essential:

### Prerequisites
- Valid OpenAI API key must be set
- Appropriate model access (GPT-4, GPT-3.5-turbo)
- Proper error handling implementation

### Available Streaming Methods

1. **Full Streaming**: ✅ Available through standard streaming mode
2. **Partial Streaming**: ✅ Supports field-by-field streaming
3. **Iterable Streaming**: ✅ Enables streaming of multiple objects
4. **Async Streaming**: ✅ Full async/await support

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
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user profile for Jason, age 25"},
        ],
        response_model=User,
    ):
        print(f"Current state: {partial_user}")
except OpenAIError as e:
    if "api_key" in str(e).lower():
        print("Error: Invalid or missing API key. Please check your OPENAI_API_KEY environment variable.")
    else:
        print(f"OpenAI API error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

### Iterable Example with Error Handling

```python
from typing import List
from openai import OpenAIError

class User(BaseModel):
    name: str
    age: int

try:
    # Extract multiple users from text
    users = client.chat.completions.create_iterable(
        model="gpt-4-turbo-preview",
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
except OpenAIError as e:
    print(f"OpenAI API error: {str(e)}")
    if "api_key" in str(e).lower():
        print("Please ensure your API key is set correctly.")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
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

## Best Practices

1. **Model Selection**
   - Use GPT-4 for complex structured outputs
   - GPT-3.5-turbo for simpler schemas
   - Always specify temperature=0 for consistent outputs

2. **Error Handling**
   - Implement proper validation
   - Use try-except blocks for graceful failure
   - Monitor validation retries

3. **Performance Optimization**
   - Use streaming for large responses
   - Implement caching where appropriate
   - Batch requests when possible

## Common Use Cases

- Data Extraction
- Form Parsing
- API Response Structuring
- Document Analysis
- Configuration Generation

## Related Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest OpenAI API versions and models. Check the [changelog](../../CHANGELOG.md) for updates.

### Environment Setup

For production use, we recommend:
1. Using environment variables for API keys
2. Implementing proper error handling
3. Setting up monitoring for API usage
4. Regular updates of both OpenAI and Instructor packages
