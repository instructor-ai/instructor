---
title: "Structured outputs with DaoXE via OpenAI-compatible API"
description: "Use Instructor with DaoXE multi-model multi-protocol API gateway through the OpenAI-compatible Chat Completions endpoint."
---

# Structured outputs with DaoXE

[DaoXE](https://daoxe.com) is a multi-model multi-protocol API gateway. Instructor can call it through the OpenAI-compatible Chat Completions path (`https://daoxe.com/v1`).

DaoXE also exposes OpenAI Responses and Anthropic Messages for other clients; this guide uses Chat Completions with Instructor's OpenAI provider interface.

## Quick start

```bash
pip install instructor openai pydantic
export DAOXE_API_KEY=your_key
export DAOXE_MODEL=your_exact_model_id   # from GET /v1/models for your account
```

Use an **exact** model ID from your DaoXE account catalog. Do not hardcode a public model price list. DaoXE is not available in mainland China.

### Simple example

```python
import os
from pydantic import BaseModel
import instructor
from openai import OpenAI


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(
    OpenAI(
        api_key=os.environ["DAOXE_API_KEY"],
        base_url="https://daoxe.com/v1",
    )
)

user = client.chat.completions.create(
    model=os.environ["DAOXE_MODEL"],
    messages=[{"role": "user", "content": "Ivan is 28 years old"}],
    response_model=User,
)

print(user)
```

### Using `from_provider` style (if supported in your Instructor version)

Some Instructor versions accept an OpenAI-compatible provider string with `base_url`:

```python
import os
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


# Prefer from_openai(OpenAI(base_url=...)) if from_provider routing differs by version
client = instructor.from_provider(
    f"openai/{os.environ['DAOXE_MODEL']}",
    base_url="https://daoxe.com/v1",
    api_key=os.environ["DAOXE_API_KEY"],
    async_client=False,
)
```

## Notes

- Prefer models that support tool calling / structured outputs for `response_model`.
- Smoke with a tiny request before agent loops.
- Examples: [DaoXE-AI](https://github.com/seven7763/DaoXE-AI)

## Disclosure

This integration page is contributed by a DaoXE maintainer.
