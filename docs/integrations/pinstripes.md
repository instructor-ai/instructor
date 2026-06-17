---
title: Structured Outputs with Pinstripes and Pydantic
description: Learn how to use Pinstripes with Instructor for structured JSON outputs using Pydantic models. Create type-safe, validated responses from Pinstripes models with Python.
---

# Structured Outputs with Pinstripes

This guide demonstrates how to use [Pinstripes](https://pinstripes.io) with Instructor to generate structured outputs. Pinstripes is an OpenAI-compatible LLM inference API, so you can use it with any standard `openai.OpenAI` or `openai.AsyncOpenAI` client pointed at `https://pinstripes.io/v1`.

## Prerequisites

Sign up for a Pinstripes account to obtain an API key, then install the required packages:

```bash
export PINSTRIPES_API_KEY=<your-api-key-here>
pip install "instructor[pinstripes]"
```

### See Also

- [Getting Started](../getting-started.md) - Quick start guide
- [from_provider Guide](../concepts/from_provider.md) - Detailed client configuration
- [Provider Examples](../index.md#provider-examples) - Quick examples for all providers

## Sync Example

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Configure the OpenAI client to target Pinstripes
raw_client = OpenAI(
    api_key=os.environ["PINSTRIPES_API_KEY"],
    base_url="https://pinstripes.io/v1",
)

# Wrap with instructor
client = instructor.from_pinstripes(raw_client)


class User(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="ps/deepseek-v4-flash",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
)

print(user)
# > User(name='Jason', age=25)
```

## Async Example

```python
import asyncio
import os
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel

raw_client = AsyncOpenAI(
    api_key=os.environ["PINSTRIPES_API_KEY"],
    base_url="https://pinstripes.io/v1",
)

client = instructor.from_pinstripes(raw_client)


class User(BaseModel):
    name: str
    age: int


async def extract_user() -> User:
    return await client.chat.completions.create(
        model="ps/deepseek-v4-flash",
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        response_model=User,
    )


user = asyncio.run(extract_user())
print(user)
# > User(name='Jason', age=25)
```

## Nested Objects

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

client = instructor.from_pinstripes(
    OpenAI(
        api_key=os.environ["PINSTRIPES_API_KEY"],
        base_url="https://pinstripes.io/v1",
    )
)


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: list[Address]


user = client.chat.completions.create(
    model="ps/deepseek-v4-flash",
    messages=[
        {
            "role": "user",
            "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """,
        }
    ],
    response_model=User,
)

print(user)
#> User(
#>     name='Jason',
#>     age=25,
#>     addresses=[
#>         Address(street='123 Main St', city='New York', country='USA'),
#>         Address(street='456 Beach Rd', city='Miami', country='USA')
#>     ]
#> )
```

## Supported Models

Pinstripes provides access to the following models:

- `ps/deepseek-v4-flash` - DeepSeek V4 Flash
- `ps/glm-4.5-air` - GLM-4.5 Air
- `ps/qwen3-35b` - Qwen3 35B
- `ps/minimax-m2.7` - MiniMax M2.7

Check the [Pinstripes documentation](https://pinstripes.io) for the full and up-to-date list of available models.

## Supported Modes

Pinstripes supports all standard OpenAI-compatible modes:

| Mode | Description |
|------|-------------|
| `instructor.Mode.TOOLS` | Function / tool calling (recommended) |
| `instructor.Mode.JSON` | JSON mode |
| `instructor.Mode.JSON_SCHEMA` | JSON schema mode |
| `instructor.Mode.MD_JSON` | Markdown-wrapped JSON mode |
| `instructor.Mode.PARALLEL_TOOLS` | Parallel tool calls |

```python
client = instructor.from_pinstripes(raw_client, mode=instructor.Mode.TOOLS)
```

## Validation and Retries

Instructor automatically retries requests when Pydantic validation fails:

```python
from pydantic import BaseModel, field_validator
import instructor
import os
from openai import OpenAI

client = instructor.from_pinstripes(
    OpenAI(
        api_key=os.environ["PINSTRIPES_API_KEY"],
        base_url="https://pinstripes.io/v1",
    )
)


class User(BaseModel):
    name: str
    age: int

    @field_validator("age")
    def validate_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be a positive number")
        return v


user = client.chat.completions.create(
    model="ps/deepseek-v4-flash",
    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    response_model=User,
    max_retries=3,
)
```

## Additional Resources

- [Pinstripes Website](https://pinstripes.io)
- [Instructor API Reference](../api/index.md)
