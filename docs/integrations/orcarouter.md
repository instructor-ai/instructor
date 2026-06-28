---
title: "Structured outputs with OrcaRouter, a complete guide with instructor"
description: "Learn how to use Instructor with OrcaRouter to access multiple LLM providers through a unified, adaptive routing API. Get type-safe, structured outputs from OpenAI, Anthropic, Google, DeepSeek, Qwen, and more."
---

# Structured outputs with OrcaRouter, a complete guide with instructor

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible meta-router that lets you call 150+ models from OpenAI, Anthropic, Google, DeepSeek, Qwen, MiniMax, xAI, and others through a single API key, with adaptive routing strategies that balance cost, latency, and quality. This guide shows you how to use Instructor with OrcaRouter for type-safe, validated responses across all of its upstream models.

## Quick Start

OrcaRouter uses an OpenAI-compatible API, so no extra dependency is needed beyond the base `instructor` package.

Set your API key:

```bash
export ORCAROUTER_API_KEY=<your-orcarouter-api-key>
```

Then call `instructor.from_provider` with the `orcarouter/<model>` prefix:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "orcarouter/openai/gpt-4o-mini",
    async_client=False,
)

resp = client.create(
    messages=[
        {"role": "user", "content": "Ivan is 28 years old."},
    ],
    response_model=User,
)
print(resp)
#> name='Ivan' age=28
```

## Async Example

```python
import asyncio
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider(
    "orcarouter/anthropic/claude-sonnet-4.6",
    async_client=True,
)


async def extract_user():
    return await client.create(
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )


user = asyncio.run(extract_user())
print(user)
#> name='Jason' age=25
```

## Adaptive Routing with `orcarouter/auto`

`orcarouter/auto` is a virtual router that picks the cheapest, fastest, or
highest-quality upstream per request based on a strategy you configure in the
[OrcaRouter console](https://www.orcarouter.ai/console/routing). You call it
exactly like any other model:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("orcarouter/auto")

resp = client.create(
    messages=[{"role": "user", "content": "Ivan is 27 and lives in Singapore"}],
    response_model=User,
)
print(resp)
```

> ⚠️ If you use agentic / tool-calling flows with `orcarouter/auto`, make sure
> the underlying router pool only contains models that support tool calling.
> Otherwise pin a specific tool-capable model such as
> `orcarouter/openai/gpt-4o` or `orcarouter/anthropic/claude-sonnet-4.6`.

## Supported Modes

OrcaRouter is OpenAI-compatible and supports the same modes as OpenAI:

- `instructor.Mode.TOOLS` (default, recommended)
- `instructor.Mode.JSON_SCHEMA` — structured outputs
- `instructor.Mode.MD_JSON`
- `instructor.Mode.PARALLEL_TOOLS`

```python
import instructor

client = instructor.from_provider(
    "orcarouter/openai/gpt-4o-mini",
    mode=instructor.Mode.JSON_SCHEMA,
)
```

## Routing Preferences via `extra_body`

OrcaRouter exposes routing controls (fallback chains, provider preferences) via
the standard OpenAI `extra_body` field:

```python
resp = client.create(
    messages=[{"role": "user", "content": "Ivan is 27"}],
    response_model=User,
    extra_body={
        "models": ["openai/gpt-4o-mini", "openai/gpt-4o"],
        "route": "fallback",
    },
)
```

## Reasoning Models

OrcaRouter passes reasoning controls through to the underlying upstream:

- OpenAI o-series / gpt-5: top-level `reasoning_effort="high|medium|low|minimal"`
- Anthropic Claude reasoning models: top-level `thinking={"type": "enabled", "budget_tokens": 2000}`
- DeepSeek reasoner: no extra field (the model reasons by default)

Note that some reasoning models do not accept `temperature` (the upstream API
will return a 400). Omit `temperature` when calling reasoning models.

## Manual Client Construction

If you'd rather build the underlying OpenAI client yourself:

```python
import os
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


openai_client = openai.OpenAI(
    api_key=os.environ["ORCAROUTER_API_KEY"],
    base_url="https://api.orcarouter.ai/v1",
)
client = instructor.from_orcarouter(openai_client, model="openai/gpt-4o-mini")

resp = client.create(
    messages=[{"role": "user", "content": "Ivan is 28"}],
    response_model=User,
)
```

## Supported Models

OrcaRouter routes to 150+ chat models across OpenAI, Anthropic, Google,
DeepSeek, Qwen, MiniMax, xAI, Kimi, and more. See the full catalog at
[orcarouter.ai/models](https://www.orcarouter.ai/models). API reference:
[docs.orcarouter.ai](https://docs.orcarouter.ai).
