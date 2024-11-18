---
title: "Vertex AI Integration with Instructor | Structured Output Guide"
description: "Complete guide to using Instructor with Google Cloud's Vertex AI. Learn how to generate structured, type-safe outputs with enterprise-grade AI capabilities."
---

# Vertex AI Integration with Instructor

Google Cloud's Vertex AI provides enterprise-grade AI capabilities with robust scaling and security features. This guide shows you how to use Instructor with Vertex AI for type-safe, validated responses.

## Quick Start

Install Instructor with Vertex AI support:

```bash
pip install "instructor[vertex]"
```

You'll also need the Google Cloud SDK and proper authentication:

```bash
pip install google-cloud-aiplatform
```

## Simple User Example (Sync)

```python
from vertexai.language_models import TextGenerationModel
import instructor
from pydantic import BaseModel

# Initialize the model
model = TextGenerationModel.from_pretrained("text-bison@001")

# Enable instructor patches
client = instructor.from_vertex(model)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.predict(
    prompt="Extract: Jason is 25 years old",
    response_model=User,
)

print(user)  # User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
from vertexai.language_models import TextGenerationModel
import instructor
from pydantic import BaseModel
import asyncio

# Initialize the model
model = TextGenerationModel.from_pretrained("text-bison@001")

# Enable instructor patches
client = instructor.from_vertex(model)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.predict_async(
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
user = client.predict(
    prompt="""
        Extract: Jason is 25 years old.
        He lives at 123 Main St, New York, USA
        and has a summer house at 456 Beach Rd, Miami, USA
    """,
    response_model=User,
)

print(user)  # User with nested Address objects
```

## Partial Streaming Example

Note: Vertex AI's current API does not support partial streaming of responses. The streaming functionality returns complete responses in chunks rather than partial objects. We recommend using the standard synchronous or asynchronous methods for structured output generation.

## Iterable Example

```python
from typing import List

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.predict_iterable(
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

Vertex AI offers several model options:
- PaLM 2 for Text (text-bison)
- PaLM 2 for Chat (chat-bison)
- Codey for Code Generation
- Enterprise-specific models
- Custom-trained models

## Best Practices

1. **Model Selection**
   - Choose model based on enterprise requirements
   - Consider security and compliance needs
   - Monitor quota and costs
   - Use appropriate model versions

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

- Enterprise Data Processing
- Secure Content Generation
- Document Analysis
- Compliance-Aware Processing
- Large-Scale Deployments

## Related Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Vertex AI's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Some features like partial streaming may not be available due to API limitations. Always check the latest documentation for feature availability.
