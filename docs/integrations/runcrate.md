---
title: "Structured outputs with Runcrate"
description: "Use Instructor with Runcrate's OpenAI-compatible API to get typed outputs from open models such as DeepSeek, GLM, Qwen and gpt-oss."
---

# Structured outputs with Runcrate

[Runcrate](https://www.runcrate.ai/inference/free?utm_source=instructor&utm_medium=directory&utm_campaign=gpu_pricing) serves open models (DeepSeek, GLM, Qwen, gpt-oss, Kimi) through an OpenAI-compatible API at `https://api.runcrate.ai/v1`. Because the API speaks the OpenAI format, Instructor works with the standard OpenAI client pointed at Runcrate. Open models are free with your GitHub account.

## Installation

```bash
pip install "instructor[openai]"
```

Create an API key at [runcrate.ai/dashboard/api-keys](https://www.runcrate.ai/dashboard/api-keys) and export it:

```bash
export RUNCRATE_API_KEY=rc_live_...
```

## Basic usage

```python
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.runcrate.ai/v1", api_key=os.environ["RUNCRATE_API_KEY"]
    ),
    mode=instructor.Mode.TOOLS,
)


class User(BaseModel):
    name: str
    age: int


user = client.create(
    model="deepseek-ai/DeepSeek-V4.1-Flash",
    messages=[{"role": "user", "content": "Ivan is 28"}],
    response_model=User,
)
print(user)
#> name='Ivan' age=28
```

Model ids keep their organization prefix, for example `deepseek-ai/DeepSeek-V4.1-Flash`, `zai-org/GLM-5.3`, `Qwen/Qwen3.8-Flash` and `openai/gpt-oss-120b`. The full list is at `https://api.runcrate.ai/v1/models`.

## Async usage

```python
import asyncio
import os

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://api.runcrate.ai/v1", api_key=os.environ["RUNCRATE_API_KEY"]
    ),
    mode=instructor.Mode.TOOLS,
)


class User(BaseModel):
    name: str
    age: int


async def main() -> None:
    user = await client.create(
        model="zai-org/GLM-5.3-Flash",
        messages=[{"role": "user", "content": "Ivan is 28"}],
        response_model=User,
    )
    print(user)


asyncio.run(main())
```

## Modes

`Mode.TOOLS` (function calling) and `Mode.JSON_SCHEMA` (`response_format` with a JSON schema) both work on Runcrate's open models.
