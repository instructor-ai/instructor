---
title: Implementing Effective Retry Logic with Pydantic and Tenacity
description: Learn how to establish simple and advanced retry strategies in Python using Pydantic and Tenacity for robust application behavior.
---

# Retrying

One of the benefits of having Pydantic is the ease with which we can define validators. We cover this topic in many articles, like [Reasking Validation](./reask_validation.md) and in our blog post [Good LLM validation is just good validation](../blog/posts/validation-part1.md).

This post will mostly describe how to use simple and more complex retry and logic.

## Example of a Validator

Before we begin, we'll use a simple example of a validator. One that checks that the name is in all caps. While we could obviously prompt that we want the name in all caps, this serves as an example of how we can build in additional logic without changing our prompts.

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel


def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)]
    age: int


try:
    UserDetail(name="jason", age=12)
except Exception as e:
    print(e)
    """
    1 validation error for UserDetail
    name
      Value error, Name must be ALL CAPS [type=value_error, input_value='jason', input_type=str]
        For further information visit https://errors.pydantic.dev/2.9/v/value_error
    """
```

## Simple: Max Retries

The simplest way to set up retries is to assign an integer value to `max_retries`.

```python
import openai
import instructor
from pydantic import BaseModel


class UserDetail(BaseModel):
    name: str
    age: int


client = instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.TOOLS)

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract `jason is 12`"},
    ],
    max_retries=3,  # (1)!
)
print(response.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 12
}
"""
# (2)!
```

1. We set the maximum number of retries to 3. This means that if the model returns an error, we'll reask the model up to 3 times.
2. We assert that the name is in all caps.

## Catching Retry Exceptions

If you want to catch the retry exceptions, you can do so and access the `last_completion`, `n_attempts` and `messages` attributes.

```python
from pydantic import BaseModel, field_validator
import openai
import instructor
from instructor.exceptions import InstructorRetryException
from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt

# Patch the OpenAI client to enable response_model
client = instructor.from_openai(openai.OpenAI())


# Define a Pydantic model for the user details
class UserDetail(BaseModel):
    name: str
    age: int

    @field_validator("age")
    def validate_age(cls, v: int):
        raise ValueError(f"You will never succeed with {str(v)}")


retries = Retrying(
    retry=retry_if_not_exception_type(ZeroDivisionError), stop=stop_after_attempt(3)
)
# Use the client to create a user detail
try:
    user = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old"}],
        max_retries=retries,
    )
except InstructorRetryException as e:
    print(e.messages[-1]["content"])  # type: ignore
    """
    Validation Error found:
    1 validation error for UserDetail
    age
      Value error, You will never succeed with 25 [type=value_error, input_value=25, input_type=int]
        For further information visit https://errors.pydantic.dev/2.9/v/value_error
    Recall the function correctly, fix the errors
    """

    print(e.n_attempts)
    #> 3

    print(e.last_completion)
    """
    ChatCompletion(
        id='chatcmpl-B7YgHmfrWA8FxsSxvzUUvdSe2lo9h',
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
                            id='call_zvTyhnBKPIhDXrOCxNzlsgzN',
                            function=Function(
                                arguments='{"name":"Jason","age":25}', name='UserDetail'
                            ),
                            type='function',
                        )
                    ],
                ),
            )
        ],
        created=1741141309,
        model='gpt-3.5-turbo-0125',
        object='chat.completion',
        service_tier='default',
        system_fingerprint=None,
        usage=CompletionUsage(
            completion_tokens=30,
            prompt_tokens=522,
            total_tokens=552,
            completion_tokens_details=CompletionTokensDetails(
                audio_tokens=0, reasoning_tokens=0
            ),
            prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
        ),
    )
    """
```

## Advanced: Retry Logic

If you want more control over how we define retries such as back-offs and additional retry logic we can use a library called Tenacity. To learn more, check out the documentation on the [Tenacity](https://tenacity.readthedocs.io/en/latest/) website.

Rather than using the decorator `@retry`, we can use the `Retrying` and `AsyncRetrying` classes to define our own retry logic.

```python
import openai
import instructor
from pydantic import BaseModel
from tenacity import Retrying, stop_after_attempt, wait_fixed

client = instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.TOOLS)


class UserDetail(BaseModel):
    name: str
    age: int


response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract `jason is 12`"},
    ],
    max_retries=Retrying(
        stop=stop_after_attempt(2),  # (1)!
        wait=wait_fixed(1),  # (2)!
    ),  # (3)!
)
print(response.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 12
}
"""
```

1. We stop after 2 attempts
2. We wait 1 second between each attempt
3. We can now define our own retry logic

### asynchronous retries

If you're using asynchronous code, you can use `AsyncRetrying` instead.

```python
import openai
import instructor
from pydantic import BaseModel
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

client = instructor.from_openai(openai.AsyncOpenAI(), mode=instructor.Mode.TOOLS)


class UserDetail(BaseModel):
    name: str
    age: int


task = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract `jason is 12`"},
    ],
    max_retries=AsyncRetrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    ),
)

import asyncio

response = asyncio.run(task)
print(response.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 12
}
"""
```

## Other Features of Tenacity

Tenacity features a huge number of different retrying capabilities. A few of them are listed below.

- `Retrying(stop=stop_after_attempt(2))`: Stop after 2 attempts
- `Retrying(stop=stop_after_delay(10))`: Stop after 10 seconds
- `Retrying(wait=wait_fixed(1))`: Wait 1 second between each attempt
- `Retrying(wait=wait_random(0, 1))`: Wait a random amount of time between 0 and 1 seconds
- `Retrying(wait=wait_exponential(multiplier=1, min=4, max=10))`: Wait an exponential amount of time between 4 and 10 seconds
- `Retrying(wait=(stop_after_attempt(2) | stop_after_delay(10)))`: Stop after 2 attempts or 10 seconds
- `Retrying(wait=(wait_fixed(1) + wait_random(0.2)))`: Wait at least 1 second and add up to 0.2 seconds

Remember that for async clients you need to use `AsyncRetrying` instead of `Retrying`!

## Retry Callbacks

You can also define callbacks to be called before and after each attempt. This is useful for logging or debugging.

```python
from pydantic import BaseModel, field_validator
import instructor
import tenacity
import openai

client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def name_is_uppercase(cls, v: str):
        assert v.isupper(), "Name must be uppercase"
        return v


resp = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1024,
    max_retries=tenacity.Retrying(
        stop=tenacity.stop_after_attempt(3),
        before=lambda _: print("before:", _),
"""
before:
<RetryCallState 13531729680: attempt #1; slept for 0.0; last result: none yet>
"""
        after=lambda _: print("after:", _),
    ),  # type: ignore
    messages=[
        {
            "role": "user",
            "content": "Extract John is 18 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "JOHN"  # due to validation
assert resp.age == 18
print(resp)
#> name='JOHN' age=18
# Output: name='JOHN' age=18

# Sample output:
# before: <RetryCallState 4421908816: attempt #1; slept for 0.0; last result: none yet>
# after: <RetryCallState 4421908816: attempt #1; slept for 0.0; last result: failed (ValidationError 1 validation error for User
# name
#   Assertion failed, Name must be uppercase [type=assertion_error, input_value='John', input_type=str]
#     For further information visit https://errors.pydantic.dev/2.6/v/assertion_error)>
#
# before: <RetryCallState 4421908816: attempt #2; slept for 0.0; last result: none yet>
# name='JOHN' age=18
```
