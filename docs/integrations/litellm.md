---
title: "Structured outputs with LiteLLM, a complete guide w/ instructor"
description: "Complete guide to using Instructor with LiteLLM's unified interface. Learn how to generate structured, type-safe outputs across multiple LLM providers."
---

# Structured outputs with LiteLLM, a complete guide w/ instructor

LiteLLM provides a unified interface for multiple LLM providers, making it easy to switch between different models and providers. This guide shows you how to use Instructor with LiteLLM for type-safe, validated responses across various LLM providers.

## Quick Start

Install Instructor with LiteLLM support:

=== "pip"
    ```bash
    pip install "instructor[litellm]"
    ```

=== "uv"
    ```bash
    uv pip install "instructor[litellm]"
    ```

## Simple User Example (Sync)

```python
from litellm import completion
import instructor
from pydantic import BaseModel

# Enable instructor patches
client = instructor.from_litellm(completion)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.completion(
    model="gpt-4-turbo-preview",  # Can use any supported model
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
client = instructor.from_litellm(acompletion)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.acompletion(
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

## Related Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with LiteLLM's latest releases. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Always verify provider-specific features and limitations in their respective documentation before implementation.
