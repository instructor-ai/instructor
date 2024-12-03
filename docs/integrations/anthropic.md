---
title: "Structured outputs with Anthropic, a complete guide w/ instructor"
description: Learn how to combine Anthropic and Instructor clients to create user models with complex properties in Python.
---

# Structured outputs with Anthropic, a complete guide w/ instructor

Now that we have a [Anthropic](https://www.anthropic.com/) client, we can use it with the `instructor` client to make requests.

Let's first install the instructor client with anthropic support

```
pip install "instructor[anthropic]"
```

Once we've done so, getting started is as simple as using our `from_anthropic` method to patch the client up.

```python
from pydantic import BaseModel
from typing import List
import anthropic
import instructor

# Patching the Anthropics client with the instructor for enhanced capabilities
client = instructor.from_anthropic(
    anthropic.Anthropic(),
)


class Properties(BaseModel):
    name: str
    value: str


class User(BaseModel):
    name: str
    age: int
    properties: List[Properties]


# client.messages.create will also work due to the instructor client
user_response = client.chat.completions.create(
    model="claude-3-haiku-20240307",
    max_tokens=1024,
    max_retries=0,
    messages=[
        {
            "role": "user",
            "content": "Create a user for a model with a name, age, and properties.",
        }
    ],
    response_model=User,
)  # type: ignore

print(user_response.model_dump_json(indent=2))
"""
{
  "name": "John Doe",
  "age": 35,
  "properties": [
    {
      "name": "City",
      "value": "New York"
    },
    {
      "name": "Occupation",
      "value": "Software Engineer"
    }
  ]
}
"""
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

You can use our `create_partial` method to stream a single object. Note that validators should not be declared in the response model when streaming objects because it will break the streaming process.

```python
from instructor import from_anthropic
import anthropic
from pydantic import BaseModel

client = from_anthropic(anthropic.Anthropic())


class User(BaseModel):
    name: str
    age: int
    bio: str


# Stream partial objects as they're generated
for partial_user in client.chat.completions.create_partial(
    model="claude-3-5-haiku-20241022",
    messages=[
        {"role": "user", "content": "Create a user profile for Jason, age 25"},
    ],
    response_model=User,
    max_tokens=4096,
):
    print(f"Current state: {partial_user}")
    # > Current state: name='Jason' age=None bio=None
    # > Current state: name='Jason' age=25 bio='Jason is a 25-year-old with an adventurous spirit and a love for technology. He is'
    # > Current state: name='Jason' age=25 bio='Jason is a 25-year-old with an adventurous spirit and a love for technology. He is always on the lookout for new challenges and opportunities to grow both personally and professionally.'

```

### Iterable Example

You can also use our `create_iterable` method to stream a list of objects. This is helpful when you'd like to extract multiple instances of the same response model from a single prompt.

```python
from instructor import from_anthropic
import anthropic
from pydantic import BaseModel

client = from_anthropic(anthropic.Anthropic())


class User(BaseModel):
    name: str
    age: int


users = client.chat.completions.create_iterable(
    model="claude-3-5-haiku-20241022",
    messages=[
        {
            "role": "user",
            "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """,
        },
    ],
    max_tokens=4096,
    response_model=User,
)

for user in users:
    print(user)
    #> name='Jason' age=25
    #> name='Sarah' age=30
    #> name='Mike' age=28
```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that Anthropic supports

1. `instructor.Mode.ANTHROPIC_JSON` : This uses the text completion API from the Anthropic API and then extracts out the desired response model from the text completion model
2. `instructor.Mode.ANTHROPIC_TOOLS` : This uses Anthropic's [tools calling API](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) to return structured outputs to the client

In general, we recommend using `Mode.ANTHROPIC_TOOLS` because it's the best way to ensure you have the desired response schema that you want.

## Caching

If you'd like to use caching with the Anthropic Client, we also support it for images and text input.

### Caching Text Input

Here's how you can implement caching for text input ( assuming you have a giant `book.txt` file that you read in).

We've written a comprehensive walkthrough of how to use caching to implement Anthropic's new Contextual Retrieval method that gives a significant bump to retrieval accuracy.

```python
from instructor import Instructor, Mode, patch
from anthropic import Anthropic
from pydantic import BaseModel

# Set up the client with prompt caching
client = instructor.from_anthropic(Anthropic())

# Define your Pydantic model
class Character(BaseModel):
    name: str
    description: str

# Load your large context
with open("./book.txt", "r") as f:
    book = f.read()

# Make multiple calls using the cached context
for _ in range(2):
    resp, completion = client.chat.completions.create_with_completion(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<book>" + book + "</book>",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "Extract a character from the text given above",
                    },
                ],
            },
        ],
        response_model=Character,
        max_tokens=1000,
    )
```

### Caching Images

We also support caching for images. This helps significantly, especially if you're using images repeatedly to save on costs. Read more about it [here](../concepts/caching.md)

```python
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic(), enable_prompt_caching=True)

cache_control = {"type": "ephemeral"}
response = client.chat.completions.create(
    model="claude-3-haiku-20240307",
    response_model=ImageAnalyzer,  # This can be set to `None` to return an Anthropic prompt caching message
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this two images?",
                {"type": "image", "source": "https://example.com/image.jpg", "cache_control": cache_control},
                {"type": "image", "source": "path/to/image.jpg", "cache_control": cache_control},
            ]
        }
    ],
    autodetect_images=True
)
```

## Beta support
The Anthropic beta API has [multiple features](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support) other than those mentioned above. You can access the beta api by setting the beta flag to `True` when creating your client. The example below shows extraction from a pdf.

```python
import instructor
import requests
import base64
from pydantic import BaseModel
from anthropic import Anthropic

class Character(BaseModel):
    name: str
    description: str

client = instructor.from_anthropic(anthropic.AsyncAnthropic(), beta=True)

pdf_data=base64.b64encode(requests.get("https://freekidsbooks.org/wp-content/uploads/2019/12/FKB-Kids-Stories-Peter-Rabbit.pdf").content).decode()

extracted_content = await self.instructor_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25"],
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "Extract a character from the pdf and describe them from any pictures ine the document"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                ],
            },
        ],
        response_model=Character,
    )
```
