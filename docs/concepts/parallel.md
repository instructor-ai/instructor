---
title: Understanding Parallel Function Calling in OpenAI
description: Learn about OpenAI's experimental parallel function calling to reduce latency and improve application performance.
---

# Parallel Tools

Parallel Tool Calling is a feature that allows you to call multiple functions in a single request. This makes it faster to get a response from the language model, especially if your tool calls are independent of each other.

!!! warning "Experimental Feature"

    Parallel tool calling is only supported by Gemini and OpenAI at the moment

## Understanding Parallel Function Calling

By using parallel function callings that allow you to call multiple functions in a single request, you can significantly reduce the latency of your application without having to use tricks with now one builds a schema.

=== "OpenAI"

    ```python hl_lines="19 31"
    from __future__ import annotations

    import openai
    import instructor

    from typing import Iterable, Literal
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_openai(
        openai.OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS
    )  # (1)!

    function_calls = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather | GoogleSearch],  # (2)!
    )

    for fc in function_calls:
        print(fc)
        #> location='Toronto' units='metric'
        #> location='Dallas' units='imperial'
        #> query='who won the super bowl'
    ```

=== "Vertex AI"

    ```python
    import instructor
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from typing import Iterable, Literal
    from pydantic import BaseModel

    vertexai.init()

    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_vertexai(
        GenerativeModel("gemini-1.5-pro-preview-0409"),
        mode=instructor.Mode.VERTEXAI_PARALLEL_TOOLS
    )  # (1)!

    function_calls = client.create(
        messages=[
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather | GoogleSearch],  # (2)!
    )

    for fc in function_calls:
        print(fc)
        #> location='Toronto' units='metric'
        #> location='Dallas' units='imperial'
        #> query='who won the super bowl'
    ```

1. Set the mode to `PARALLEL_TOOLS` to enable parallel function calling.
2. Set the response model to `Iterable[Weather | GoogleSearch]` to indicate that the response will be a list of `Weather` and `GoogleSearch` objects. This is necessary because the response will be a list of objects, and we need to specify the types of the objects in the list.

Noticed that the `response_model` Must be in the form `Iterable[Type1 | Type2 | ...]` or `Iterable[Type1]` where `Type1` and `Type2` are the types of the objects that will be returned in the response.
