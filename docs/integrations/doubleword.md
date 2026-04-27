---
title: "Structured outputs with Doubleword, a complete guide with instructor"
description: "Learn how to use Instructor with Doubleword's AI model gateway for type-safe, structured outputs with batch pricing up to 90% cheaper than realtime inference."
---

# Structured outputs with Doubleword, a complete guide with instructor

[Doubleword](https://doubleword.ai) is an AI model gateway providing unified routing, management, and security for inference across multiple model providers. It exposes an OpenAI-compatible API with batch pricing — up to 90% cost savings on non-latency-sensitive workloads like bulk extraction, evals, and data processing.

This guide shows you how to use Instructor with Doubleword for type-safe, validated responses, and how to use the batch API for significant cost savings.

## Quick Start

Instructor comes with support for the OpenAI client out of the box, so you don't need to install anything extra.

```bash
pip install "instructor"
```

Set your Doubleword API key:

```bash
export DOUBLEWORD_API_KEY='your-api-key-here'
```

## Simple User Example (Sync)

```python
import os
from openai import OpenAI
from pydantic import BaseModel
import instructor


client = instructor.from_openai(
    OpenAI(
        base_url="https://api.doubleword.ai/v1",
        api_key=os.environ["DOUBLEWORD_API_KEY"],
    )
)


class User(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
#> name='Jason' age=25
```

## Simple User Example (Async)

```python
import os
from openai import AsyncOpenAI
from pydantic import BaseModel
import instructor
import asyncio


client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://api.doubleword.ai/v1",
        api_key=os.environ["DOUBLEWORD_API_KEY"],
    )
)


class User(BaseModel):
    name: str
    age: int


async def extract_user():
    user = await client.chat.completions.create(
        model="Qwen/Qwen3.5-397B-A17B",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user


user = asyncio.run(extract_user())
print(user)
#> name='Jason' age=25
```

## Batch pricing with Autobatcher

For bulk extraction, evals, or any workload where latency is not critical, use the [`autobatcher`](https://pypi.org/project/autobatcher/) package to transparently route requests through Doubleword's Batch API at reduced cost.

`BatchOpenAI` is a drop-in subclass of `AsyncOpenAI` that collects concurrent requests into batch submissions automatically.

```bash
pip install "instructor" autobatcher
```

```python
import os
from autobatcher import BatchOpenAI
from pydantic import BaseModel
import instructor
import asyncio


client = instructor.from_openai(
    BatchOpenAI(
        base_url="https://api.doubleword.ai/v1",
        api_key=os.environ["DOUBLEWORD_API_KEY"],
    )
)


class User(BaseModel):
    name: str
    age: int


async def extract_users():
    sentences = [
        "Jason is 25 years old",
        "Sarah is 32 and lives in London",
        "Mike turned 41 last week",
    ]

    users = await asyncio.gather(*[
        client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B-FP8",
            messages=[
                {"role": "user", "content": f"Extract: {s}"},
            ],
            response_model=User,
        )
        for s in sentences
    ])
    return users


users = asyncio.run(extract_users())
for user in users:
    print(user)
#> name='Jason' age=25
#> name='Sarah' age=32
#> name='Mike' age=41
```

The concurrent `create` calls are collected into a single batch submission. Pricing follows Doubleword's 24-hour batch tier — up to 90% less than realtime. Your code stays the same; only the client changes.

## Async (1-hour flex) pricing with Autobatcher

When 24-hour batch turnaround is too slow but realtime cost is too high, use `autobatcher.AsyncOpenAI` instead. Same drop-in `AsyncOpenAI` machinery, but pinned to Doubleword's **1-hour flex tier** — sits between realtime and 24-hour batch on cost and latency.

```bash
pip install "instructor" autobatcher
```

```python
import os
from autobatcher import AsyncOpenAI
from pydantic import BaseModel
import instructor
import asyncio


client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://api.doubleword.ai/v1",
        api_key=os.environ["DOUBLEWORD_API_KEY"],
    )
)


class User(BaseModel):
    name: str
    age: int


async def extract_users():
    sentences = [
        "Jason is 25 years old",
        "Sarah is 32 and lives in London",
        "Mike turned 41 last week",
    ]

    users = await asyncio.gather(*[
        client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B-FP8",
            messages=[
                {"role": "user", "content": f"Extract: {s}"},
            ],
            response_model=User,
        )
        for s in sentences
    ])
    return users


users = asyncio.run(extract_users())
for user in users:
    print(user)
#> name='Jason' age=25
#> name='Sarah' age=32
#> name='Mike' age=41
```

`AsyncOpenAI` is a thin subclass of `BatchOpenAI` with a 1-hour completion window default — same fan-out semantics, results back within the hour rather than next-day.

## Available Models

| Model | Type | Best for |
|-------|------|----------|
| `Qwen/Qwen3.5-397B-A17B` | Chat + Vision + Reasoning | Maximum performance, complex extraction |
| `Qwen/Qwen3.5-35B-A3B-FP8` | Chat + Vision + Reasoning | Balanced cost/performance |
| `Qwen/Qwen3.5-9B` | Chat + Vision + Reasoning | High volume, cost-efficient |
| `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | Chat + Vision | Vision-heavy tasks |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | Chat + Vision | Production vision workloads |
| `Qwen/Qwen3-14B-FP8` | Chat (text-only) | Classification, extraction, summarization |
| `openai/gpt-oss-20b` | Chat + Reasoning | Low latency, specialized tasks |

For the full list, see the [Doubleword models documentation](https://docs.doubleword.ai/inference-api/models).

## Nested Models and Validation

```python
import os
from openai import OpenAI
from pydantic import BaseModel, field_validator
from typing import List
import instructor


client = instructor.from_openai(
    OpenAI(
        base_url="https://api.doubleword.ai/v1",
        api_key=os.environ["DOUBLEWORD_API_KEY"],
    )
)


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]

    @field_validator("age")
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Age must be between 0 and 150")
        return v


user = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B-FP8",
    messages=[
        {
            "role": "user",
            "content": "Extract: Jason is 25, lives at 123 Main St, NYC, USA and 456 Oak Ave, London, UK",
        },
    ],
    response_model=User,
    max_retries=3,
)

print(user)
#> name='Jason' age=25 addresses=[Address(street='123 Main St', city='NYC', country='USA'), Address(street='456 Oak Ave', city='London', country='UK')]
```

## Related Resources

- [Doubleword documentation](https://docs.doubleword.ai)
- [Available models](https://docs.doubleword.ai/inference-api/models)
- [Autobatcher package](https://pypi.org/project/autobatcher/)
