---
title: Understanding Parallel Function Calling in OpenAI
description: Learn about OpenAI's experimental parallel function calling to reduce latency and improve application performance.
---

# Parallel Tools

Parallel Function Calling is a feature that allows you to call multiple functions in a single request.

!!! warning "Experimental Feature"

    Parallel Function calling is only supported by Google and OpenAI at the moment. Make sure to use the equivalent parallel tool `mode` for your client.

## Understanding Parallel Function Calling

Parallel Function Calling helps you to significantly reduce the latency of your application without having to build a parent schema as a wrapper around these tool calls.

=== "OpenAI"

    ```python hl_lines="20 32"
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


    client = instructor.from_openai(openai.OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

    function_calls = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather | GoogleSearch],
    )

    for fc in function_calls:
        print(fc)
        #> location='Toronto' units='metric'
        #> location='Dallas' units='metric'
        #> query='who won the super bowl 2023'
    ```

=== "Vertex AI"

    ```python hl_lines="20 30"
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
        mode=instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
    )

    function_calls = client.create(
        messages=[
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather | GoogleSearch],
    )

    for fc in function_calls:
        print(fc)
        #> location='Toronto' units='metric'
        #> location='Dallas' units='imperial'
        #> query='who won the super bowl'
    ```

We need to set the response model to `Iterable[Weather | GoogleSearch]` to indicate that the response will be a list of `Weather` and `GoogleSearch` objects.

This is necessary because the response will be a list of objects, and we need to specify the types of the objects in the list. This returns an iterable which you can then iterate over
