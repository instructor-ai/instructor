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

## Streaming Support

Instructor now supports streaming capabilities with Vertex AI! You can use both `create_partial` for incremental model building and `create_iterable` for streaming collections.

### Streaming Partial Responses

```python
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
import instructor
from pydantic import BaseModel
from instructor.dsl.partial import Partial

vertexai.init()

class UserExtract(BaseModel):
    name: str
    age: int

client = instructor.from_vertexai(
    client=GenerativeModel("gemini-2.0-flash"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
)

# Stream partial responses
response_stream = client.chat.completions.create(
    response_model=Partial[UserExtract],
    stream=True,
    messages=[
        {"role": "user", "content": "Anibal is 23 years old"},
    ],
)

for partial_user in response_stream:
    print(f"Received update: {partial_user}")
# Output might show:
# Received update: UserExtract(name='Anibal', age=None)
# Received update: UserExtract(name='Anibal', age=23)
```

### Streaming Iterable Collections

```python
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
import instructor
from pydantic import BaseModel

vertexai.init()

class UserExtract(BaseModel):
    name: str
    age: int

client = instructor.from_vertexai(
    client=GenerativeModel("gemini-2.0-flash"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
)

# Stream iterable responses
response_stream = client.chat.completions.create_iterable(
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Make up two people"},
    ],
)

for user in response_stream:
    print(f"Generated user: {user}")
# Output:
# Generated user: UserExtract(name='Sarah Johnson', age=32)
# Generated user: UserExtract(name='David Chen', age=27)
```

### Async Streaming

You can also use async versions of both streaming approaches:

```python
import asyncio
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
import instructor
from pydantic import BaseModel
from instructor.dsl.partial import Partial

vertexai.init()

class UserExtract(BaseModel):
    name: str
    age: int

client = instructor.from_vertexai(
    client=GenerativeModel("gemini-2.0-flash"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
    _async=True,
)

async def stream_partial():
    response_stream = await client.chat.completions.create(
        response_model=Partial[UserExtract],
        stream=True,
        messages=[
            {"role": "user", "content": "Anibal is 23 years old"},
        ],
    )
    
    async for partial_user in response_stream:
        print(f"Received update: {partial_user}")

async def stream_iterable():
    response_stream = client.chat.completions.create_iterable(
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Make up two people"},
        ],
    )
    
    async for user in response_stream:
        print(f"Generated user: {user}")

# Run async functions
asyncio.run(stream_partial())
asyncio.run(stream_iterable())
```

## Related Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with Vertex AI's latest API versions. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.

Streaming support has been added for both partial responses and iterable collections, with both synchronous and asynchronous interfaces.
