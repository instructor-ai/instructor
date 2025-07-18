---
title: Parallel Tools
description: Learn about parallel tools in OpenAI, Google, and Anthropic.
---

# Parallel Tools

Parallel Tool Calling is a feature that allows you to call multiple functions in a single request.

!!! warning "Experimental Feature"

    Parallel Tool Calling is supported by Google, OpenAI, and Anthropic. Make sure to use the equivalent parallel tool `mode` for your client.

## Understanding Parallel Tool Calling

Parallel Function Calling helps you to significantly reduce the latency of your application without having to build a parent schema as a wrapper around these tool calls.

=== "OpenAI"

    ```python hl_lines="20 32"
    from __future__ import annotations

    import instructor

    from typing import Iterable, Literal
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_provider(
        "openai/gpt-4.1-mini",
        mode=instructor.Mode.PARALLEL_TOOLS,
    )

    function_calls = client.chat.completions.create(
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
    from typing import Iterable, Literal
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_provider(
        "vertexai/gemini-2.5-flash",
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

=== "Anthropic"

    ```python hl_lines="20 32"
    import instructor
    from typing import Iterable, Literal
    from pydantic import BaseModel


    class Weather(BaseModel):
        location: str
        units: Literal["imperial", "metric"]


    class GoogleSearch(BaseModel):
        query: str


    client = instructor.from_provider(
        "anthropic/claude-3-7-sonnet-latest",
        mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
    )

    function_calls = client.chat.completions.create(
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

We need to set the response model to `Iterable[Weather | GoogleSearch]` to indicate that the response will be a list of `Weather` and `GoogleSearch` objects.

This is necessary because the response will be a list of objects, and we need to specify the types of the objects in the list. This returns an iterable which you can then iterate over
