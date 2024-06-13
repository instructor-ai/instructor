---
draft: False
date: 2024-05-30
slug: vertexai
tags:
  - patching
authors:
  - ajac-zero
---

# Structured Outputs with Vertex AI

Vertex AI is the recommended way to deploy the Gemini family of models in production. These models support up to 1 million tokens in their context window and boast native multimodality with files, video, and audio. The Vertex AI SDK offers a preview of tool calling that we can use to obtain structured outputs.

By the end of this blog post, you will learn how to effectively utilize Instructor with the Gemini family of models.

<!-- more -->

## Patching

Instructor's patch enhances the gemini api with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Vertex AI Client

The Vertex AI client employs a different client than OpenAI, making the patching process slightly different than other examples

!!! note "Getting access"

    If you want to try this out for yourself check out the [Vertex AI](https://cloud.google.com/vertex-ai?hl=en) console. You can get started [here](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform).

```python
import instructor

from pydantic import BaseModel
import vertexai.generative_models as gm
import vertexai

vertexai.init()

client = gm.GenerativeModel("gemini-1.5-pro-preview-0409")

# enables `response_model` in chat call
client = instructor.from_vertexai(client)


if __name__ == "__main__":

    class UserDetails(BaseModel):
        name: str
        age: int

    resp = client.create(
        response_model=UserDetails,
        messages=[
            {
                "role": "user",
                "content": f'Extract the following entities: "Jason is 20"',
            },
        ],
    )
    print(resp)
    #> name='Jason' age=20
```

## Limitations

Currently, Vertex AI offers does not support the following attributes from the OpenAPI schema: `optional`, `maximum`, `anyOf`. This means that not all pydantic models will be supported. Below, I'll share some models that could trigger this error and some work-arounds.

### optional / anyOf

Using a pydantic model with an `Optional` field raise an exception, because the Optional type is translated to `"anyOf": [integer , null]` which is not yet supported.

```python
from typing import Optional

class User(BaseModel):
    name: str
    age: Optional[int]

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is 23 years old.",
        }
    ],
    response_model=User,
)

print(resp)
# ValueError: Protocol message Schema has no "anyOf" field.
```

A workaround if to set a certain default value that Gemini can fall back on if the information is not present:

```python
from pydantic import Field

class User(BaseModel):
    name: str
    age: int = Field(default=0) # or just age: int = 0

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is _ years old.",
        }
    ],
    response_model=User,
)

print(resp)
# name='Anibal' age=0
```

This workaround can also work with default_factories:

```python
class User(BaseModel):
    name: str
    age: int
    siblings: list[str] = Field(default_factory=lambda: [])

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is 23 years old.",
        }
    ],
    response_model=User,
)

print(resp)
# name='Anibal' age=23 siblings=[]
```

### maximum

Using the `lt`(less than) or `gt`(greater than) paramateres in a pydantic field will raise exceptions:


```python
class User(BaseModel):
    name: str
    age: int = Field(gt=0)

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is 23 years old.",
        }
    ],
    response_model=User,
)

print(resp)
# ValueError: Protocol message Schema has no "exclusiveMinimum" field.

class User(BaseModel):
    name: str
    age: int = Field(lt=100)

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is _ years old.",
        }
    ],
    response_model=User,
)

print(resp)
# ValueError: Protocol message Schema has no "exclusiveMaximum" field
```

A workaround for this is to use pydantic validadors to change these values post creation

```python
from pydantic import field_validator

class User(BaseModel):
    name: str
    age: int

    @field_validator("age")
    def age_range_limit(cls, age: int) -> int:
        if age > 100:
            age = 100
        elif age < 0:
            age = 0
        return age

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is 1023 years old.",
        }
    ],
    response_model=User,
)

print(resp)
# name='Anibal' age=100

resp = client.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Anibal is -12 years old.",
        }
    ],
    response_model=User,
)

print(resp)
# name='Anibal' age=0
```

So by relying on pydantic, we can mitigate some of the current limitations with the Gemini models 😊.
