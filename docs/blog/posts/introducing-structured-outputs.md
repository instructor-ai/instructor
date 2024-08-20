---
draft: False
date: 2024-08-20
slug: is-instructor-dead
tags:
  - OpenAI
authors:
  - ivanleomk
---

# Is Instructor Dead?

## What's Open AI's Structured Output mode all about?

OpenAI's new Structured Output mode is a huge step change for developers building complex workflows. Given an arbitrary JSON Schema, Structured Output ensures that the response matches the schema exactly.

Here's a basic example.

```python
import openai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = openai.OpenAI()
resp = client.beta.chat.completions.parse(
    response_format=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the following user: Jason is 25 years old.",
        },
    ],
    model="gpt-4o-mini",
)

print(resp.choices[0].message.parsed)
#> name='Jason' age=25
```

With guaranteed schema adherence, outputs always conform to your defined Pydantic model, eliminating type mismatches and missing fields. However, while Structured Outputs solve many common issues, two key challenges emerge when building more sophisticated applications - that of Validation and Streaming.

<!-- more -->

### Limited Validation Feedback

Validation is crucial for allowing models to correct their mistakes and improve their responses. Let's see a simple example where we want to extract a user's name in all uppercase.

```python
import openai
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def ensure_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("All letters must be uppercase. Got: " + v)
        return v


client = openai.OpenAI()
try:
    resp = client.beta.chat.completions.parse(
        response_format=User,
        messages=[
            {
                "role": "user",
                "content": "Extract the following user: Jason is 25 years old.",
            },
        ],
        model="gpt-4o-mini",
    )
except Exception as e:
    print(e)
    """
    1 validation error for User
    name
      Value error, All letters must be uppercase. Got: Jason [type=value_error, input_value='Jason', input_type=str]
        For further information visit https://errors.pydantic.dev/2.8/v/value_error
    """
```

When validation fails, we lose the original completion, making it challenging to implement retry logic or provide specific prompts for correction. This limitation hinders our ability to offer detailed feedback to the model, ultimately impacting our capacity to improve its performance over time.

### Streaming

Streaming with Structured Outputs is supported but a challenging endeavour. There's no built-in partial validation and you need to manually parse the generated response while simultaneously having to now use a context manager to access the generated values.

In short, making it work well in practice requires a good amount of effort with their current `beta.chat.completions.stream` implementation.

```python
import openai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = openai.OpenAI()
with client.beta.chat.completions.stream(
    response_format=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the following user: Jason is 25 years old.",
        },
    ],
    model="gpt-4o-mini",
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.snapshot, flush=True, end="\n")
            #>
            #> {"
            #> {"name
            #> {"name":"
            #> {"name":"Jason
            #> {"name":"Jason","
            #> {"name":"Jason","age
            #> {"name":"Jason","age":
            #> {"name":"Jason","age":25
            #> {"name":"Jason","age":25}
```

## Should you be using Structured Output mode?

We performed some simple benchmarks on the new Structured Output model and obtained the following results. Note that for Structured Outputs, your schemas are cached and stored on the OpenAI servers. As a result, actual figures might differ slightly depending on your production usage.

??? "How did we perform the benchmarks?"

    We used the following snippet of code to perform the benchmarks

    ```python
    import instructor
    import openai
    from asyncio import run, Semaphore
    from tqdm.asyncio import tqdm_asyncio as asyncio
    from pydantic import BaseModel, field_validator
    import time
    import pandas as pd
    from typing import Union

    modes = [
        instructor.Mode.STRUCTURED_OUTPUTS,
        instructor.Mode.TOOLS,
    ]
    oai_client = openai.AsyncOpenAI()


    class User(BaseModel):
        name: str
        age: int

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError("Name must be uppercase")
            return v


    class Users(BaseModel):
        users: list[User]


    async def run_single_call(
        client: instructor.client.AsyncInstructor, semaphore: Semaphore
    ) -> float:
        start = time.time()
        async with semaphore:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=Users,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract Out users name and age from the following text: John is 25 years old. Sarah is 32. Mike, who is 41, loves sports. Emma, aged 19, is a student.",
                    },
                ],
            )
            return time.time() - start


    async def generate_responses(
        client: instructor.client.AsyncInstructor, max_concurrent_calls: int, n_samples: int
    ) -> list[float]:
        coros = [
            run_single_call(client, Semaphore(max_concurrent_calls))
            for _ in range(n_samples)
        ]
        return await asyncio.gather(*coros)


    results: list[dict[str, Union[float, str]]] = []

    for mode in modes:
        client = instructor.from_openai(oai_client, mode=mode)

        response = run(
            generate_responses(
                client,
                10,
                200,
            )
        )
        mean_response = sum(response) / len(response)
        min_response = min(response)
        max_response = max(response)
        results.append(
            {
                "mode": mode.name,
                "mean": mean_response,
                "min": min_response,
                "max": max_response,
            }
        )

    df = pd.DataFrame(results)
    ```

    All generated values were then rounded off to make them easier to compare

### Without Validators

| Sample Size | Mode               | Mean (s) | Min (s) | Max (s) |
| ----------- | ------------------ | -------- | ------- | ------- |
| 50          | STRUCTURED_OUTPUTS | 1.9      | 1.4     | 7.3     |
| 50          | TOOLS              | 1.5      | 1.2     | 3.4     |
| 200         | STRUCTURED_OUTPUTS | 3.2      | 2.0     | 6.4     |
| 200         | TOOLS              | 3.0      | 1.9     | 16      |

### With Validators

| Sample Size | Mode               | Mean (s) | Min (s) | Max (s) |
| ----------- | ------------------ | -------- | ------- | ------- |
| 50          | STRUCTURED_OUTPUTS | 4.2      | 2.4     | 17      |
| 50          | TOOLS              | 3.3      | 2.5     | 6.7     |
| 200         | STRUCTURED_OUTPUTS | 5.5      | 3.8     | 65.2    |
| 200         | TOOLS              | 6.4      | 4.7     | 18.2    |

## Why use `instructor`

In short, while OpenAI's Structured Output mode ensures schema adherence, developers still need to implement a good amount of functionality themselves.

`instructor` solves a lot of these issues with features such as automatically handling retries, streaming of validated inputs and full support for Pydantic validation among many others.

Let's see this in action below.

### Automatic Retries

With `instructor`, all it takes is a simple Pydantic Schema and a validator for you to get the extracted name as upper-case

```python
import instructor
import openai
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def ensure_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError("All letters must be uppercase. Got: " + v)
        return v


client = instructor.from_openai(
    openai.OpenAI(), mode=instructor.Mode.STRUCTURED_OUTPUTS
)

resp = client.chat.completions.create(
    response_model=User,
    messages=[
        {
            "role": "user",
            "content": "Extract the following user: Jason is 25 years old.",
        }
    ],
    model="gpt-4o-mini",
)

print(resp)
#> name='JASON' age=25
```

### Full Pydantic Support

You might also have some runtime information that you cannot encode in a Pydantic Schema. A great example would be citations from a paragraph that you pass in using a Validation Context.

Let's see an example below where we try to answer a question with citations.

```python
import instructor
import openai
from pydantic import BaseModel, field_validator, ValidationInfo


class Response(BaseModel):
    answer: str
    citation: str

    @field_validator("citation")
    def validate_citation(cls, v: str, info: ValidationInfo) -> str:
        paragraph = info.context.get("paragraph", "")
        sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
        if v not in sentences:
            raise ValueError(
                f"Extract out the exact sentence for your citation. {v} is not in the list of sentences  ( {sentences})."
            )
        return v


client = instructor.from_openai(
    openai.OpenAI(), mode=instructor.Mode.STRUCTURED_OUTPUTS
)

paragraph = "Jason is 25 years old. He enjoys playing basketball and reading science fiction novels."

resp = client.chat.completions.create(
    response_model=Response,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that extracts information from a paragraph.",
        },
        {
            "role": "user",
            "content": "What is Jason's age? Here is some information to refer to -: "
            + paragraph,
        },
    ],
    model="gpt-4o",
    validation_context={"paragraph": paragraph},
)

print(f"Answer: {resp.answer}")
#> Answer: 25 years old
print(f"Citation: {resp.citation}")
#> Citation: Jason is 25 years old
print(f"Paragraph: {paragraph}")
"""
Paragraph: Jason is 25 years old. He enjoys playing basketball and reading science fiction novels.
"""
```

### Streaming

A common use-case is to define a single Schema and extract multiple instances of it. With `instructor`, doing this is relatively straightforward by using our `create_iterable` method.

```python
import instructor
import openai
from pydantic import BaseModel

client = instructor.from_openai(
    openai.OpenAI(), mode=instructor.Mode.STRUCTURED_OUTPUTS
)


class User(BaseModel):
    name: str
    age: int


users = client.chat.completions.create_iterable(
    model="gpt-4o-mini",
    response_model=User,
    messages=[
        {
            "role": "system",
            "content": "You are a perfect entity extraction system",
        },
        {
            "role": "user",
            "content": (f"Extract `Jason is 10 and John is 10`"),
        },
    ],
)

for user in users:
    print(user)
```

Often times, we might also want to stream out information as it's dynamically generated into some sort of frontend component.

With `instructor`, you'll be able to do just that using the `create_partial` method.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from rich.console import Console

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.TOOLS)

text_block = """
In our recent online meeting, participants from various backgrounds joined to discuss the upcoming tech conference. The names and contact details of the participants were as follows:

- Name: John Doe, Email: johndoe@email.com, Twitter: @TechGuru44
- Name: Jane Smith, Email: janesmith@email.com, Twitter: @DigitalDiva88
- Name: Alex Johnson, Email: alexj@email.com, Twitter: @CodeMaster2023

During the meeting, we agreed on several key points. The conference will be held on March 15th, 2024, at the Grand Tech Arena located at 4521 Innovation Drive. Dr. Emily Johnson, a renowned AI researcher, will be our keynote speaker.

The budget for the event is set at $50,000, covering venue costs, speaker fees, and promotional activities. Each participant is expected to contribute an article to the conference blog by February 20th.

A follow-up meetingis scheduled for January 25th at 3 PM GMT to finalize the agenda and confirm the list of speakers.
"""


class User(BaseModel):
    name: str
    email: str
    twitter: str


class MeetingInfo(BaseModel):
    users: list[User]
    date: str
    location: str
    budget: int
    deadline: str


extraction_stream = client.chat.completions.create_partial(
    model="gpt-4",
    response_model=MeetingInfo,
    messages=[
        {
            "role": "user",
            "content": f"Get the information about the meeting and the users {text_block}",
        },
    ],
    stream=True,
)


console = Console()

for extraction in extraction_stream:
    obj = extraction.model_dump()
    console.clear()
    console.print(obj)
```

This will output the following

![Structured Output Extraction](./img/Structured_Output_Extraction.gif)

# Conclusion

In short, OpenAI's Structured Output format mode is promising in helping to ensure more reliable and consistent generations. especially when you pair it with `instructor`.

Give Instructor a try and see how much easier it makes getting valid outputs from LLMs!
