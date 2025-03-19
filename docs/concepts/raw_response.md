---
title: Creating a Model with OpenAI Completions
description: Learn how to create a custom model using OpenAI's API to extract user data efficiently with Python.
---


# Creating a model with completions

In instructor>1.0.0 we have a custom client, if you wish to use the raw response you can do the following

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

print(user)
#> name='Jason' age=25

print(completion)
"""
ChatCompletion(
    id='chatcmpl-B7YgfMbbn3vOol0urrCAUUgCd7eej',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=None,
                refusal=None,
                role='assistant',
                audio=None,
                function_call=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='call_cHlDYOU8IV70YVHTqFCHpgGr',
                        function=Function(
                            arguments='{"name":"Jason","age":25}', name='UserExtract'
                        ),
                        type='function',
                    )
                ],
            ),
        )
    ],
    created=1741141333,
    model='gpt-3.5-turbo-0125',
    object='chat.completion',
    service_tier='default',
    system_fingerprint=None,
    usage=CompletionUsage(
        completion_tokens=10,
        prompt_tokens=82,
        total_tokens=92,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    ),
)
"""
```

## Anthropic Raw Response

You can also access the raw response from Anthropic models. This is useful for debugging or when you need to access additional information from the response.

```python
import instructor

from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())


user, completion = client.chat.completions.create_with_completion(
    model="claude-3-5-sonnet-latest",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

print(user)
#> name='Jason' age=25

print(completion)
"""