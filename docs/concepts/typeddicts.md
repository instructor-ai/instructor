---
title: Using TypedDicts with OpenAI API
description: Learn how to utilize TypedDicts in Python with the OpenAI API for structured data responses.
---

# TypedDicts

We also support typed dicts.

```python
from typing_extensions import TypedDict
import instructor


class User(TypedDict):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4.1-mini")


response = client.chat.completions.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Timothy is a man from New York who is turning 32 this year",
        }
    ],
)
```