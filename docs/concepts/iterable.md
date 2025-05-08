---
title: Extracting Structured Data with Iterable and Streaming in Python
description: Learn to use Iterable and streaming for structured data extraction with Pydantic and OpenAI in Python.
---

# Multi-Task and Streaming

Using an `Iterable` lets you extract multiple structured objects from a single LLM call, streaming them as they arrive. This is useful for entity extraction, multi-task outputs, and more.

**We recommend using the `create_iterable` method for most use cases.** It's simpler and less error-prone than manually specifying `Iterable[...]` and `stream=True`.

Here's a simple example showing how to extract multiple users from a single sentence. You can use either the recommended `create_iterable` method or the `create` method with `Iterable[User]`:

=== "Using `create_iterable` (recommended)"
    ```python
    import instructor
    import openai
    from pydantic import BaseModel

    client = instructor.from_openai(openai.OpenAI())

    class User(BaseModel):
        name: str
        age: int

    resp = client.chat.completions.create_iterable(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28, lives in Moscow and his friends are Alex, John and Mary who are 25, 30 and 27 respectively",
            }
        ],
        response_model=User,
    )

    for user in resp:
        print(user)
    ```
    _Recommended for most use cases. Handles streaming and iteration for you._

=== "Using `create` with `Iterable[User]`"
    ```python
    import instructor
    import openai
    from pydantic import BaseModel
    from typing import Iterable

    client = instructor.from_openai(openai.OpenAI())

    class User(BaseModel):
        name: str
        age: int

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28, lives in Moscow and his friends are Alex, John and Mary who are 25, 30 and 27 respectively",
            }
        ],
        response_model=Iterable[User],
    )

    for user in resp:
        print(user)
    ```
    _Use this if you need more manual control or compatibility with legacy code._

---


We also support more complex extraction patterns such as Unions as you'll see below out of the box. 

???+ warning 

    Unions don't work with Gemini because the AnyOf is not supported in the current response schema.

## Synchronous Usage

=== "Using `create`"

    ```python
    import instructor
    import openai
    from typing import Iterable, Union, Literal
    from pydantic import BaseModel

    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]

    class GoogleSearch(BaseModel):
        query: str

    client = instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.TOOLS)

    results = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {"role": "user", "content": "What is the weather in toronto and dallas and who won the super bowl?"},
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
        stream=True,
    )

    for item in results:
        print(item)
    ```

=== "Using `create_iterable` (recommended)"

    ```python

    import instructor
    import openai
    from typing import Union, Literal
    from pydantic import BaseModel

    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]

    class GoogleSearch(BaseModel):
        query: str

    client = instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.TOOLS)

    results = client.chat.completions.create_iterable(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {"role": "user", "content": "What is the weather in toronto and dallas and who won the super bowl?"},
        ],
        response_model=Union[Weather, GoogleSearch],
    )

    for item in results:
        print(item)
    ```

---

## Asynchronous Usage

=== "Using `create`"

    ```python
    import instructor
    import openai
    from typing import Iterable, Union, Literal
    from pydantic import BaseModel
    import asyncio

    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]

    class GoogleSearch(BaseModel):
        query: str

    aclient = instructor.from_openai(openai.AsyncOpenAI(), mode=instructor.Mode.TOOLS)

    async def main():
        results = await aclient.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You must always use tools"},
                {"role": "user", "content": "What is the weather in toronto and dallas and who won the super bowl?"},
            ],
            response_model=Iterable[Union[Weather, GoogleSearch]],
            stream=True,
        )
        async for item in results:
            print(item)

    asyncio.run(main())
    ```

=== "Using `create_iterable` (recommended)"

    ```python
    import instructor
    import openai
    from typing import Union, Literal
    from pydantic import BaseModel
    import asyncio

    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]

    class GoogleSearch(BaseModel):
        query: str

    aclient = instructor.from_openai(openai.AsyncOpenAI(), mode=instructor.Mode.TOOLS)

    async def main():
        results = await aclient.chat.completions.create_iterable(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You must always use tools"},
                {"role": "user", "content": "What is the weather in toronto and dallas and who won the super bowl?"},
            ],
            response_model=Union[Weather, GoogleSearch],
        )
        async for item in results:
            print(item)

    asyncio.run(main())
    ```
