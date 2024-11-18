---
title: "Structured outputs with Google/Gemini, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Google's Gemini models. Learn how to generate structured, type-safe outputs with Google's advanced AI capabilities."
---

# Structured outputs with Google/Gemini, a complete guide w/ instructor

Google's Gemini models provide powerful AI capabilities with multimodal support. This guide shows you how to use Instructor with Google's Gemini models for type-safe, validated responses.

## Quick Start

Install Instructor with Google support:

```bash
pip install "instructor[google]"
```

## Simple User Example (Sync)

```python
from google.generativeai import GenerativeModel
import instructor
from pydantic import BaseModel

# Initialize the client
model = GenerativeModel('gemini-pro')

# Enable instructor patches
client = instructor.from_google(model)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.generate_content(
    prompt="Extract: Jason is 25 years old",
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from google.generativeai import GenerativeModel
import instructor
from pydantic import BaseModel
import asyncio

# Initialize async client
model = GenerativeModel('gemini-pro')

# Enable instructor patches
client = instructor.from_google(model)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.generate_content_async(
        prompt="Extract: Jason is 25 years old",
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
user = client.generate_content(
    prompt="""
        Extract: Jason is 25 years old.
        He lives at 123 Main St, New York, USA
        and has a summer house at 456 Beach Rd, Miami, USA
    """,
    response_model=User,
)

print(user)  # User with nested Address objects
```

## Streaming Support and Limitations

Google's Gemini models provide streaming capabilities with some limitations:

- **Full Streaming**: ✅ Supported
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
for partial_user in client.generate_content_stream(
    prompt="Create a user profile for Jason, age 25",
    response_model=User,
):
    print(f"Current state: {partial_user}")
    # Fields will populate gradually as they're generated
```

**Important Notes on Streaming:**
- Full streaming is well-supported for complete response generation
- Partial streaming has limited support and may require additional error handling
- Some responses may arrive in larger chunks rather than field-by-field
- Consider implementing fallback mechanisms for partial streaming scenarios
- Monitor streaming performance and implement appropriate error handling
- Test thoroughly with your specific use case before deploying to production

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.generate_content_iterable(
    prompt="""
        Extract users:
        1. Jason is 25 years old
        2. Sarah is 30 years old
        3. Mike is 28 years old
    """,
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

Google offers several Gemini models:
- Gemini Pro (General purpose)
- Gemini Pro Vision (Multimodal)
- Gemini Ultra (Coming soon)

## Best Practices

1. **Model Selection**
   - Choose model based on task requirements
   - Consider multimodal needs
   - Monitor quota usage
   - Use appropriate context lengths

2. **Optimization Tips**
   - Structure prompts effectively
   - Use appropriate temperature settings
   - Implement caching strategies
   - Monitor API usage

3. **Error Handling**
   - Implement proper validation
   - Handle quota limits gracefully
   - Monitor model responses
   - Use appropriate timeout settings

## Common Use Cases

- Data Extraction
- Content Generation
- Document Analysis
- Multimodal Processing
- Complex Reasoning Tasks

## Related Resources

- [Google AI Documentation](https://ai.google.dev/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Google's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
