---
title: "Structured outputs with Sarvam AI"
description: "Use Instructor with Sarvam Indic LLMs for type-safe structured extraction across Indian languages."
---

# Structured outputs with Sarvam AI

[Sarvam AI](https://www.sarvam.ai/models) provides Indic-focused LLMs such as **Sarvam 30B** and **Sarvam 105B** through an [OpenAI-compatible chat API](https://docs.sarvam.ai/). Instructor now supports Sarvam as a first-class provider via `from_provider("sarvam/...")` and `from_sarvam()`.

## Install

```bash
pip install "instructor[openai]"
```

Set your API key:

```bash
export SARVAM_API_KEY='your-api-key-here'
```

## Quick start

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("sarvam/sarvam-30b")

user = client.chat.completions.create(
    model="sarvam-30b",
    response_model=User,
    messages=[{"role": "user", "content": "राहुल की उम्र 28 साल है।"}],
)

print(user)
# > User(name='Rahul', age=28)
```

## Supported models

Chat LLMs exposed via the Sarvam API (see [Sarvam models](https://www.sarvam.ai/models)):

| Model ID | Description |
|----------|-------------|
| `sarvam-30b` | Efficient MoE chat model for high-throughput Indic workloads |
| `sarvam-105b` | Flagship chat model for highest-quality Indic understanding |

Per [Sarvam's model guide](https://docs.sarvam.ai/api-reference-docs/getting-started/models), chat models support **11 languages** (10 Indian + English). `sarvam-m` is deprecated — use `sarvam-30b` or `sarvam-105b`.

Speech (Saaras), TTS (Bulbul), translation (Mayura, Sarvam Translate), and document intelligence (Sarvam Vision) use separate APIs and are not covered by this integration yet. See [Extending Sarvam support](#extending-sarvam-support).

## Modes

Sarvam works with Instructor's OpenAI-compatible modes:

| Mode | When to use |
|------|-------------|
| `Mode.MD_JSON` (default) | Most reliable across Indic scripts in our Jul 2026 benchmark |
| `Mode.TOOLS` | Fastest latency; best on `sarvam-105b`, but unreliable on `sarvam-30b` for extraction |
| `Mode.JSON_SCHEMA` | Native `response_format` JSON schema |

!!! warning "`Mode.TOOLS` on `sarvam-30b`"
    In the full 432-trial eval, `sarvam-30b + Mode.TOOLS` scored **29–46%** success on person extraction (vs **58–75%** for `Mode.MD_JSON`). Prefer `Mode.MD_JSON` for cost-sensitive extraction workloads.

```python
import instructor

client = instructor.from_provider(
    "sarvam/sarvam-105b",
    mode=instructor.Mode.JSON_SCHEMA,
)
```

Sarvam-specific API parameters such as `reasoning_effort` and `wiki_grounding` can be passed through `create()` kwargs.

## Async usage

```python
import asyncio
import instructor
from pydantic import BaseModel


class CityInfo(BaseModel):
    city: str
    state: str


async def main() -> None:
    client = instructor.from_provider("sarvam/sarvam-30b", async_client=True)
    result = await client.chat.completions.create(
        model="sarvam-30b",
        response_model=CityInfo,
        messages=[{"role": "user", "content": "Extract city info: Bengaluru is in Karnataka."}],
    )
    print(result)


asyncio.run(main())
```

## Extending Sarvam support

Future non-chat Sarvam products (Vision, Saaras STT, Bulbul TTS, Translate) would need dedicated provider handlers under `instructor/v2/providers/sarvam/` with separate API endpoints. The current integration is intentionally scoped to chat completions.

## Related resources

- [Sarvam models](https://www.sarvam.ai/models)
- [Sarvam API documentation](https://docs.sarvam.ai/)
- [Instructor modes comparison](../concepts/modes-comparison.md)
- [from_provider guide](../concepts/from_provider.md)
