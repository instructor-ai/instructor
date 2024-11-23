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
    id='chatcmpl-AWl4kOf2XIrMZ2cBWC41gXCkFCpQs',
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
                        id='call_bGBFg2QrTqw30Y8zXCs9RYGY',
                        function=Function(
                            arguments='{"name":"Jason","age":25}', name='UserExtract'
                        ),
                        type='function',
                    )
                ],
            ),
        )
    ],
    created=1732370818,
    model='gpt-3.5-turbo-0125',
    object='chat.completion',
    service_tier=None,
    system_fingerprint=None,
    usage=CompletionUsage(
        completion_tokens=9,
        prompt_tokens=82,
        total_tokens=91,
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
        prompt_tokens_details=None,
        prompt_token_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
    ),
)
"""
```