---
draft: False
date: 2023-12-11
slug: pydantic
tags:
  - pydantic
authors:
  - jxnl
---

# Steering Large Language Models with Pydantic

In the past year, significant progress has been made in utilizing large language models. Prompt engineering, in particular, has gained attention, and new prompting techniques are being developed to guide language models toward specific tasks. While many are building chat bots, an even more exciting application is the generation of structured outputs, whether its extracting structured data, augmenting your RAG application, or even generating

??? question "What is Prompt Engineering?"

    Prompt Engineering, also known as In-Context Prompting, is a method used to guide the behavior of LLMs without updating the model. It involves techniques to enhance the quality of outputs, formatting, reasoning, and factuality. You can learn more about it in [this post](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/).

While some have resorted to [threatening human life](https://twitter.com/goodside/status/1657396491676164096?s=20) to generate structured data, we have found that Pydantic even more effective.

In this post, we will explore how we can easily validate structured outputs from language models using Pydantic and OpenAI, to write code that we can trust, then we will introduce a new library called "instructor" that simplifies this process even further while adding additional features as well.

## Pydantic

Unlike libraries like `dataclasses` or `attrs`, `Pydantic` goes a step further and allows you to define a schema for your dataclass. This schema can be used to validate data, but also to generate documentation and even to generate a JSON schema, which is perfect for our use case of generating structured data with language models!

By Just prompting the model with the following prompt, we can generate a JSON schema for a `PythonPackage` dataclass.

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()


class Package(BaseModel):
    name: str
    author: str

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Return the `name`, and `author` of pydantic, in a json object."
        },
    ]
)

Package.model_validate_json(resp.choices[0].message.content)
```

If all is well, we might get `{"name": "pydantic", "author": "Samuel Colvin"}` as a response. But if something is wrong, we might get a whole bunch of invalid json that contains prose or markdown code blocks that we'd have to parse ourselves:

````text
```json
{
  "name": "pydantic",
  "author": "Samuel Colvin"
}
```
```

```text
Ok heres the authors of pydantic: Samuel Colvin, and the name this library

{
  "name": "pydantic",
  "author": "Samuel Colvin"
}
````

All of which is invalid JSON, but could contain useful information that we would have to handle ourselves. Luckily, `OpenAI` has given us a few options to handle this.

## Introducing Tools Calling

While tool calling was originally designed to make calls to external APIs using JSON schema, its real value lies in allowing us to specify the desired output. Fortunately, `Pydantic` provides utilities for generating a JSON schema and supports nested structures, which would be difficult to describe in plain text.

In this example, instead of describing the desired output in plain text, we can simply provide the JSON schema for the `Packages` class, which includes a list of `Package` objects.

As an exercise, try prompting the model to generate this prompt without using Pydantic!
??? note "Example without Pydantic"

    Heres the same example as below without using pydantic's schema generation

    ```python
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Pydantic and FastAPI?",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Requirements",
                    "description": "A list of packages and their first authors.",
                    "parameters": {
                        "$defs": {
                            "Package": {
                                "properties": {
                                    "name": {"title": "Name", "type": "string"},
                                    "author": {"title": "Author", "type": "string"},
                                },
                                "required": ["name", "author"],
                                "title": "Package",
                                "type": "object",
                            }
                        },
                        "properties": {
                            "packages": {
                                "items": {"$ref": "#/$defs/Package"},
                                "title": "Packages",
                                "type": "array",
                            }
                        },
                        "required": ["packages"],
                        "title": "Packages",
                        "type": "object",
                    },
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "Requirements"},
        },
    )

    resp = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    ```

Now, notice in this example that the prompts we use contain purely the data we want, where the tools and tools choice now capture the schemas we want to output. This speration of concerns makes it much easier organize the 'data' and the 'description' of the data that we want back out.

```python
from typing import List
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()


class Package(BaseModel):
    name: str
    author: str


class Packages(BaseModel):
    packages: List[Package]


resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Pydantic and FastAPI?",
        },
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "Requirements",
                "description": "A list of packages and their first authors.",
                "parameters": Packages.model_json_schema(),
            },
        }
    ],
    tool_choice={
        "type": "function",
        "function": {"name": "Requirements"},
    },
)

Packages.model_validate_json(
    resp.choices[0].message.tool_calls[0].function.arguments
)
```

```json
{
  "packages": [
    {
      "name": "pydantic",
      "author": "Samuel Colvin"
    },
    {
      "name": "fastapi",
      "author": "Sebastián Ramírez"
    }
  ]
}
```

## Instructor

Although this example may seem contrived, it demonstrates how Pydantic can be used to generate structured data from language models. To simplify this pattern, I have developed a small library called `instructor` that patches the `OpenAI` client. This library offers convenient features such as JSON mode, function calling, tool usage, and open source models.

To achieve the same result using `instructor`, simply follow these steps:

```python
from typing import List
from pydantic import BaseModel
from openai import OpenAI

import instructor

client = instructor.patch(OpenAI())


class Package(BaseModel):
    name: str
    author: str


class Packages(BaseModel):
    """A list of packages and their first authors"""
    packages: List[Package]


packages = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Pydantic and FastAPI?",
        },
    ],
    response_model=Packages,
)
```

Now, by using the `response_model` argument, inspired by `FastAPI`, you can specify the desired output, and `instructor` will take care of the rest!

In our upcoming posts, we will provide more practical examples and explore how we can leverage `Pydantic`'s validation features to ensure that the data we receive is not only valid JSON, but also valid data.
