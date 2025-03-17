---
title: SambaNova
description: Use Instructor with SambaNova's LLM API for structured outputs.
---

# SambaNova Integration

Instructor supports SambaNova's LLM API, allowing you to use structured outputs with their models.

## Installation

```bash
pip install "instructor[openai]"
```

## Basic Usage

```python
from openai import OpenAI
import instructor
import os
from pydantic import BaseModel

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.sambanova.ai/v1",
        api_key=os.environ["SAMBANOVA_API_KEY"]
    )
)

class User(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="Meta-Llama-3.1-405B-Instruct",
    messages=[
        {"role": "user", "content": "Ivan is 28"},
    ],
    response_model=User,
)

print(user)
# > User(name='Ivan', age=28)
```

## Async Usage

```python
from openai import AsyncOpenAI
import instructor
import os
from pydantic import BaseModel

client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://api.sambanova.ai/v1",
        api_key=os.environ["SAMBANOVA_API_KEY"]
    )
)

class User(BaseModel):
    name: str
    age: int

async def get_user():
    user = await client.chat.completions.create(
        model="Meta-Llama-3.1-405B-Instruct",
        messages=[
            {"role": "user", "content": "Ivan is 28"},
        ],
        response_model=User,
    )
    return user

# Run with asyncio
import asyncio
user = asyncio.run(get_user())
print(user)
# > User(name='Ivan', age=28)
```

## Available Models

Check the [SambaNova documentation](https://docs.sambanova.ai/cloud/docs/get-started/supported-models) for the latest model offerings and capabilities.
