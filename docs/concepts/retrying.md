# Retrying

One of the benefits of having Pythantic is the ease with which we can define validators. We cover this topic in many articles, like [Reasking Validation](./reask_validation.md) and in our blog post [Good LLM validation is just good validation](../blog/posts/validation-part1.md).

This post will mostly describe how to use simple and more complex retry and logic.

## Example of a Validator

Before we begin, we'll use a simple example of a validator. One that checks that the name is an all cap. While we could obviously prompt that we want the name in all camps, this serves as an example of how we can build an additional logic without changing our prompts.

To use simple retry, we just need to set `max_retries`` as an integer. In this example.

```python
from typing import Annotated
import openai
from pydantic import AfterValidator, BaseModel
import instructor


def uppercase_validator(v):
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)]
    age: int
```

Now if we create a user detail with a lowercase name, we'll see an error.

```python
UserDetail(name="jason", age=12)
>>> 1 validation error for UserDetail
>>> name
>>>     Value error, Name must be ALL CAPS [type=value_error, input_value='jason', input_type=str]
```

## Simple: Max Retries

The simplest way of defining a retry is just defining the maximum number of retries.

```python
client = instructor.patch(
    openai.OpenAI(),
    mode=instructor.Mode.TOOLS
)

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract `jason is 12`"},
    ],
    max_retries=3, #(1)!
)
assert response.name == "JASON" #(2)!
```

1. We set the maximum number of retries to 3. Which means that if the model returns an error, we'll reask the model up to 3 times.
2. We assert that the name is in all caps

```json
{
  "name": "JASON",
  "age": 12
}
```

## Advanced: Retry Logic

If you want more control over how we define retries such as back-offs and additional retry logic We can use a library called Tenacity. To learn more, check out the documentation on the [Tenacity](https://tenacity.readthedocs.io/en/latest/) website.

Rather than using the decorator `@retry`, we can use the `Retrying` and `AsyncRetrying` classes to define our own retry logic.

```python
from tenacity import Retrying, stop_after_attempt, wait_fixed

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract `jason is 12`"},
    ],
    max_retries=Retrying(
        stop=stop_after_attempt(2), #(1)!
        wait=wait_fixed(1), #(2)!
    ) # (3)!
)
```

1. We stop after 2 attempts
2. We wait 1 second between each attempt
3. We can now define our own retry logic

### asynchronous retries

If you're using asynchronous code, you can use `AsyncRetrying` instead.

```python
from tenacity import AsyncRetrying, stop_after_attempt, wait_fixed

response = await client.chat.completions.create(
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
```

## Other Features of Tenacity

Tenacity features a huge number of different retrying capabilities. Here is a couple of them listed below.

- `Retrying(stop=stop_after_attempt(2))`: Stop after 2 attempts
- `Retrying(stop=stop_after_delay(10))`: Stop after 10 seconds
- `Retrying(wait=wait_fixed(1))`: Wait 1 second between each attempt
- `Retrying(wait=wait_random(0, 1))`: Wait a random amount of time between 0 and 1 seconds
- `Retrying(wait=wait_exponential(multiplier=1, min=4, max=10))`: Wait an exponential amount of time between 4 and 10 seconds
- `Retrying(wait=(stop_after_attempt(2) | stop_after_delay(10)))`: Stop after 2 attempts or 10 seconds
- `Retrying(wait=(wait_fixed(1) + wait_random(0.2)))`: Wait at least 1 second and add up to 0.2 seconds

Remember that for async clients you need to use `AsyncRetrying` instead of `Retrying`!
