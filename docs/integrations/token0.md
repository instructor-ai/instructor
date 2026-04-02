---
title: "Vision token optimization with Token0"
description: "Reduce vision LLM costs by 35-99% using Token0 as a pre-call hook in Instructor."
---

# Vision Token Optimization with Token0

[Token0](https://github.com/Pritom14/token0) is an open-source vision token optimizer that hooks into Instructor's `completion:kwargs` event to automatically compress images before every LLM call — reducing vision token costs by 35-99% with no changes to your existing code.

## Install

```bash
pip install token0
```

## Quick Start

```python
import instructor
import openai
from pydantic import BaseModel
from token0.instructor_hook import Token0Hook

client = instructor.from_openai(openai.OpenAI())
client.on("completion:kwargs", Token0Hook())


class Invoice(BaseModel):
    total: float
    vendor: str
    date: str


# Images are automatically optimized before the API call
result = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the invoice details"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
            ],
        }
    ],
    response_model=Invoice,
)
```

## What Token0 Does

Token0 applies up to 11 optimizations automatically:

| Optimization | Savings |
|---|---|
| Smart resize (downscale to provider max) | 0-44% |
| OCR routing (text images → text extraction) | 47-70% |
| PDF text layer extraction | ~94% |
| Prompt-aware detail mode | 13-92% |
| Tile-optimized resize (OpenAI) | 44% |
| JPEG recompression (PNG → JPEG) | 20-60% |
| Saliency cropping (crop to relevant region) | 20-60% |
| Model cascade (route simple tasks to cheaper models) | 5-20x cheaper |
| Semantic response cache | 100% on hits |
| QJL fuzzy cache | 62% additional on variations |
| Video frame deduplication | 13-83% |

## Options

```python
hook = Token0Hook(
    enable_cascade=True,    # auto-route simple tasks to cheaper models
    detail_override="low",  # force low/high detail mode (OpenAI only)
)
client.on("completion:kwargs", hook)
```

## Works With Any Provider

Token0 works with any Instructor-supported provider:

```python
import anthropic
import instructor

client = instructor.from_anthropic(anthropic.Anthropic())
client.on("completion:kwargs", Token0Hook())
```

## Remove the Hook

```python
hook = Token0Hook()
client.on("completion:kwargs", hook)

# Later — remove it
client.off("completion:kwargs", hook)
```

## Learn More

- [Token0 GitHub](https://github.com/Pritom14/token0)
- [Benchmark results](https://github.com/Pritom14/token0#benchmarks)
