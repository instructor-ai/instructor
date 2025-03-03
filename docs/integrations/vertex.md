---
title: "Structured outputs with Vertex AI, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Google Cloud's Vertex AI. Learn how to generate structured, type-safe outputs with enterprise-grade AI capabilities."
---

# Structured outputs with Vertex AI, a complete guide w/ instructor

Google Cloud's Vertex AI provides enterprise-grade AI capabilities with robust scaling and security features. This guide shows you how to use Instructor with Vertex AI for type-safe, validated responses.

## Quick Start

Install Instructor with Vertex AI support. You can do so by running the command below.

```bash
pip install "instructor[vertexai]"
```

## Simple User Example (Sync)

```python
import instructor
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel

vertexai.init()


class User(BaseModel):
    name: str
    age: int


client = instructor.from_vertexai(
    client=GenerativeModel("gemini-1.5-pro-preview-0409"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
)

# note that client.chat.completions.create will also work
resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

print(resp)
#> User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import asyncio
import instructor
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel

vertexai.init()


class User(BaseModel):
    name: str
    age: int


client = instructor.from_vertexai(
    client=GenerativeModel("gemini-1.5-pro-preview-0409"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
    _async=True,
)

async def extract_user():
    user = await client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=User,
    )
    return user


# Run async function
user = asyncio.run(extract_user())
print(user)  # User(name='Jason', age=25)
```

## Related Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Vertex AI's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Note: Some features like partial streaming may not be available due to API limitations. Always check the latest documentation for feature availability.
