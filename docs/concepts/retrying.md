# Retrying

One of the benefits of having Pydantic is the ease with which we can define validators. We cover this topic in many articles, like [Reasking Validation](./reask_validation.md) and in our blog post [Good LLM validation is just good validation](../blog/posts/validation-part1.md).

This post will mostly describe how to use simple and more complex retry and logic.

## Example of a Validator

Before we begin, we'll use a simple example of a validator. One that checks that the name is in all caps. While we could obviously prompt that we want the name in all caps, this serves as an example of how we can build in additional logic without changing our prompts.

To use simple retry, we just need to set `max_retries`` as an integer. In this example.

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
        For further information visit https://errors.pydantic.dev/2.6/v/value_error
    """
```

## Simple: Max Retries

The simplest way of defining a retry is just defining the maximum number of retries.

```python
import openai
import instructor
from pydantic import BaseModel


class UserDetail(BaseModel):
    name: str
    age: int


client = instructor.patch(openai.OpenAI(), mode=instructor.Mode.TOOLS)

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

## Advanced: Retry Logic

If you want more control over how we define retries such as back-offs and additional retry logic we can use a library called Tenacity. To learn more, check out the documentation on the [Tenacity](https://tenacity.readthedocs.io/en/latest/) website.

Rather than using the decorator `@retry`, we can use the `Retrying` and `AsyncRetrying` classes to define our own retry logic.

```python
import openai
import instructor
from pydantic import BaseModel
from tenacity import Retrying, stop_after_attempt, wait_fixed

client = instructor.patch(openai.OpenAI(), mode=instructor.Mode.TOOLS)


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

client = instructor.patch(openai.AsyncOpenAI(), mode=instructor.Mode.TOOLS)


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
