# Parallel Tools

One of the latest capabilities that OpenAI has recently introduced is parallel function calling.
To learn more you can read up on [this](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling)

!!! warning "Experimental Feature"

    This feature is currently in preview and is subject to change. only supported by the `gpt-4-turbo-preview` model.

## Understanding Parallel Function Calling

By using parallel function callings that allow you to call multiple functions in a single request, you can significantly reduce the latency of your application without having to use tricks with now one builds a schema.

```python hl_lines="19 31"
import openai
import instructor

from typing import Iterable, Literal
from pydantic import BaseModel


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


client = instructor.patch(
    openai.OpenAI(),
    mode=instructor.Mode.PARALLEL_TOOLS #(1)!
)

function_calls = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You must always use tools"},
        {
            "role": "user",
            "content": "What is the weather in toronto and dallas and who won the super bowl?",
        },
    ],
    response_model=Iterable[Weather | GoogleSearch], #(2)!
)

for fc in function_calls:
    print(fc)
    """
```

1. Set the mode to `PARALLEL_TOOLS` to enable parallel function calling.
2. Set the response model to `Iterable[Weather | GoogleSearch]` to indicate that the response will be a list of `Weather` and `GoogleSearch` objects. This is necessary because the response will be a list of objects, and we need to specify the types of the objects in the list.

```python
Weather(location='toronto', units='imperial')
Weather(location='dallas', units='imperial')
GoogleSearch(query='who won the super bowl?')
```

Noticed that the `response_model` Must be in the form `Iterable[Type1 | Type2 | ...]` or `Iterable[Type1]` where `Type1` and `Type2` are the types of the objects that will be returned in the response.
