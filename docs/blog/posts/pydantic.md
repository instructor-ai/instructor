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

In the last year, there's been a big leap in how we use advanced AI programs, especially in how we communicate with them to get specific tasks done. People are not just making chatbots; they're also using these AIs to sort information, improve their apps, and create synthetic data to train smaller task specific models.

!!! question "What is Prompt Engineering?"

    Prompt Engineering is a technique to direct large language models (LLMs) like ChatGPT. It doesn't change the AI itself but tweaks how we ask questions or give instructions. This method improves the AI's responses, making them more accurate and helpful. It's like finding the best way to ask something to get the answer you need. There's a detailed article about it [here](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/).

While some have resorted to [threatening human life](https://twitter.com/goodside/status/1657396491676164096?s=20) to generate structured data, we have found that Pydantic is even more effective.

In this post, we will discuss validating structured outputs from language models using Pydantic and OpenAI. We'll show you how to write reliable code. Additionally, we'll introduce a new library called "instructor" that simplifies this process and offers extra features to leverage validation to improve the quality of your outputs.

## Pydantic

Unlike libraries like `dataclasses`, `Pydantic` goes a step further and allows you to define a schema for your dataclass. This schema can be used to validate data, but also to generate documentation and even to generate a JSON schema, which is perfect for our use case of generating structured data with language models!

??? note "Understanding Validation"

    A simple example of validation involves ensuring that a value has the correct type. For instance, let's consider a `Person` dataclass with a `name` field of type `str`. We can validate that the value is indeed a string.

    ```python
    from dataclasses import dataclass

    @dataclass
    class Person:
        name: str
        age: int

    Person(name="Sam", age="10")
    >>> Person(name="Sam", age="10")
    ```

    By using the `dataclass` decorator, we can pass in the values as strings without any complaints from the dataclass. This would mean that we could run into issues later on if we try to use the `age` field as an `int`.

    ```python
    Person(name="Sam", age="10").age + 1
    >>> TypeError: can only concatenate str (not "int") to str
    ```

    However, if we use `Pydantic`, we will obtain the correct type!

    ```python
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    Person(name="Sam", age="10")
    >>> Person(name='Sam', age=10)

    Person(name="Sam", age="10").age + 1
    >>> 11
    ```

    The `age` field has been updated to an `int` from a `str` in the demonstration. Pydantic validates and coerces the type, ensuring the correct type is obtained. If we provide data that cannot be converted to an `int`, an error will be returned.

    ```python
    Person(name="Sam", age="13.4")
    >>> ValidationError: 1 validation error for Person
    >>> age
    >>> Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='13.4', input_type=str]
    >>>        For further information visit https://errors.pydantic.dev/2.5/v/int_parsing
    ```

    This behavior is great when we may not have trusted inputs, but is even more critical when inputs are coming from a language model!

    To learn more about validation, check out the section [validation is a misnomer](https://docs.pydantic.dev/latest/concepts/models/#tldr)

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

If all is well, we might get something that looks like `json.loads({"name": "pydantic", "author": "Samuel Colvin"})` which is correctly deserialized. But if something is wrong, we might get a whole bunch of text that contains prose or markdown code blocks that we'd have to reasonable

**LLM responses with markdown code blocks**

````python
json.loads("""
```json
{
  "name": "pydantic",
  "author": "Samuel Colvin"
}
```
""")
>>> JSONDecodeError: Expecting value: line 1 column 1 (char 0
````

**LLM responses with prose**

```python
json.loads("""
Ok heres the authors of pydantic: Samuel Colvin, and the name this library

{
  "name": "pydantic",
  "author": "Samuel Colvin"
}
""")
>>> JSONDecodeError: Expecting value: line 1 column 1 (char 0
```

The content may contain valid JSON, but it isn't considered valid JSON without understanding the language model's behavior. However, it could still provide useful information that we need to handle independently. Fortunately, `OpenAI` offers several options to address this situation.

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

## Using `pip install instructor`

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
