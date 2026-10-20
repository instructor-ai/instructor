---
title: "Structured outputs with AI Stupid Level, a complete guide with instructor"
description: "Use Instructor with the AI Stupid Level router: one OpenAI-compatible endpoint that sends each request to the model currently scoring best on continuously re-run coding, reasoning and tool-use benchmarks."
---

# Structured outputs with AI Stupid Level

[AI Stupid Level](https://aistupidlevel.info/router) is an OpenAI-compatible router. Instead of naming a model, you name a routing mode such as `auto-coding`, and each request goes to the model that currently scores best on the matching benchmark. The benchmarks are re-run continuously, so the choice follows the measurements. Requests are made with your own provider keys (OpenAI, Anthropic, Google, DeepSeek and others), which you add to your AI Stupid Level account.

Instructor works with it through the OpenAI client, so nothing extra is needed beyond the base package.

## Setup

```bash
pip install instructor openai
export ASL_ROUTER_KEY=aism_...  # create a router key at https://aistupidlevel.info/router/keys
```

## Quick Start (Sync)

```python
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(
    OpenAI(
        base_url="https://aistupidlevel.info/v1",
        api_key=os.environ["ASL_ROUTER_KEY"],
    )
)

user = client.create(
    model="auto-coding",
    response_model=User,
    messages=[{"role": "user", "content": "Ivan is 28 years old"}],
)
print(user)
#> name='Ivan' age=28
```

`from_provider` works too: `instructor.from_provider("openai/auto-coding", base_url="https://aistupidlevel.info/v1", api_key=os.environ["ASL_ROUTER_KEY"])`.

## Async

```python
import asyncio
import os

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(
    AsyncOpenAI(
        base_url="https://aistupidlevel.info/v1",
        api_key=os.environ["ASL_ROUTER_KEY"],
    )
)


async def main():
    user = await client.create(
        model="auto-reasoning",
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
    )
    print(user)


asyncio.run(main())
```

## Routing modes

| Model | Sends the request to |
|---|---|
| `auto-coding` | the best model on the coding benchmark |
| `auto-reasoning` | the best model on the multi-turn reasoning benchmark |
| `auto-tooling` | the best model on the tool-use benchmark |
| `auto-best` | the best combined score |
| `auto-value` | the best score per dollar among near-top models |
| `auto-task` | picks one of the above from the request itself |

A specific model ID, such as `claude-opus-5-5` or `gpt-6-sol`, pins one model. `GET https://aistupidlevel.info/v1/models` lists every ID your key can use.

## Which model answered

`create_with_completion` returns the raw completion as well, and its `model` field is the model the router chose:

```python
user, completion = client.create_with_completion(
    model="auto-coding",
    response_model=User,
    messages=[{"role": "user", "content": "Ivan is 28 years old"}],
)
print(completion.model)
#> claude-opus-5-5
```

## Modes

The default `instructor.Mode.TOOLS` works, as does `instructor.Mode.JSON_SCHEMA`.

## Notes

- The router accepts text only, so image and audio inputs are rejected with a clear error.
- To try it without spending anything, send the header `X-ASL-Sandbox: true` (for example `OpenAI(..., default_headers={"X-ASL-Sandbox": "true"})`). The routing is real and the answer is a mock that matches your schema.

See the [router documentation](https://aistupidlevel.info/router/docs) for the full API.
