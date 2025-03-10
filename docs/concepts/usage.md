---
title: Handling Non-Streaming Requests in OpenAI with Usage Tracking
description: Learn how to manage non-streaming requests in OpenAI, track token usage, and handle exceptions with Python.
---

The easiest way to get usage for non streaming requests is to access the raw response.

```python
import instructor

from openai import OpenAI
from pydantic import BaseModel

client = instructor.from_openai(OpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


user, completion = client.chat.completions.create_with_completion(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(completion.usage)
"""
CompletionUsage(
    completion_tokens=10,
    prompt_tokens=82,
    total_tokens=92,
    completion_tokens_details=CompletionTokensDetails(
        audio_tokens=0, reasoning_tokens=0
    ),
    prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
)
"""
```

You can catch an IncompleteOutputException whenever the context length is exceeded and react accordingly, such as by trimming your prompt by the number of exceeding tokens.

```python
from instructor.exceptions import IncompleteOutputException
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


try:
    client.chat.completions.create_with_completion(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
except IncompleteOutputException as e:
    token_count = e.last_completion.usage.total_tokens  # type: ignore
    # your logic here
```
