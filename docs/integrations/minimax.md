---
title: MiniMax
description: Guide to using instructor with MiniMax
---

# Structured outputs with MiniMax, a complete guide w/ instructor

MiniMax offers the MiniMax-M2.5 and MiniMax-M2.5-highspeed models with a 204,800-token context window. MiniMax provides an OpenAI-compatible API, making integration straightforward with the standard OpenAI SDK.

## Quick Start

Install the required packages:

```bash
pip install instructor openai
```

Set up your API key:

```bash
export MINIMAX_API_KEY=your_api_key_here
```

## Basic Example

```python
import os

import openai
import instructor
from pydantic import BaseModel, Field


# Create an OpenAI client pointed at MiniMax's API
client = instructor.from_minimax(
    openai.OpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.io/v1",
    ),
    mode=instructor.Mode.MINIMAX_TOOLS,
)


class UserExtract(BaseModel):
    """Model for extracting user information from text."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")


user = client.chat.completions.create(
    model="MiniMax-M2.5",
    response_model=UserExtract,
    messages=[
        {
            "role": "user",
            "content": "Extract jason is 25 years old",
        },
    ],
    temperature=1.0,
)

print(user.model_dump_json(indent=2))
# {
#   "name": "Jason",
#   "age": 25
# }
```

## Async Example

```python
import os
import asyncio

import openai
import instructor
from pydantic import BaseModel, Field


client = instructor.from_minimax(
    openai.AsyncOpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.io/v1",
    ),
    mode=instructor.Mode.MINIMAX_TOOLS,
)


class UserExtract(BaseModel):
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")


async def extract_user(text: str) -> UserExtract:
    return await client.chat.completions.create(
        model="MiniMax-M2.5",
        response_model=UserExtract,
        messages=[{"role": "user", "content": text}],
        temperature=1.0,
    )


async def main():
    user = await extract_user("Extract jason is 25 years old")
    print(user.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

## Using `from_provider`

You can also use the `from_provider` helper:

```python
import instructor

client = instructor.from_provider("minimax/MiniMax-M2.5")
```

## Supported Modes

MiniMax supports the following instructor modes:

- `Mode.MINIMAX_TOOLS` - Uses OpenAI-compatible tool calling for structured extraction (recommended)
- `Mode.MINIMAX_JSON` - Uses system-prompt guidance for JSON output

## Models

| Model | Context Window | Description |
|-------|---------------|-------------|
| `MiniMax-M2.5` | 204,800 tokens | Peak performance, ultimate value |
| `MiniMax-M2.5-highspeed` | 204,800 tokens | Same performance, faster and more agile |

## API Documentation

- [OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api)
