---
draft: False
date: 2024-10-15
slug: introducing-cerebras-support
categories:
  - LLM
  - Cerebras
authors:
  - ivanleomk
---

# Introducing Cerebras Support

## What's Cerebras?

Cerebras provides a new AI chip that is purpose-built for large language models. It is a custom chip that is designed to be more efficient and powerful than existing chips. With Cerebras inference, you can get up to 550 tokens/seconds over their API.

We're happy to announce that we've added support for Cerebras inference in Instructor using the `from_cerebras` method.

### Basic Usage

To use Cerebras inference, you can use the `from_cerebras` method to create a new Instructor client, define a Pydantic model to pass into the `response_model` parameter and get back a validated response exactly as you would expect.

<!-- more -->

You'll also need to install the Cerebras SDK to use the client. You can install it with the command below.

```bash
pip install "instructor[cerebras_cloud_sdk]"
```

This ensures that you have the necessary dependencies to use the Gemini or VertexAI SDKs with instructor.

### Getting Started

Before running the following code, you'll need to make sure that you have your Cerebras API Key set in your shell under the alias `CEREBRAS_API_KEY`.

```python
import instructor
from cerebras.cloud.sdk import Cerebras
from pydantic import BaseModel

client = instructor.from_cerebras(Cerebras())


class Person(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create(
    model="llama3.1-70b",
    messages=[
        {
            "role": "user",
            "content": "Extract the name and age of the person in this sentence: John Smith is 29 years old.",
        }
    ],
    response_model=Person,
)

print(resp)
#> Person(name='John Smith', age=29)
```

We support both the `AsyncCerebras` and `Cerebras` clients.

### Streaming

We also support streaming with the Cerebras client with `CEREBRAS_JSON` mode so that you can take advantage of the speed of Cerebras and process the response as it comes in.

```python
import instructor
from cerebras.cloud.sdk import Cerebras, AsyncCerebras
from pydantic import BaseModel
from typing import Iterable

client = instructor.from_cerebras(Cerebras(), mode=instructor.Mode.CEREBRAS_JSON)


class Person(BaseModel):
    name: str
    age: int


resp = client.chat.completions.create(
    model="llama3.1-70b",
    messages=[
        {
            "role": "user",
            "content": "Extract all users from this sentence : Chris is 27 and lives in San Francisco, John is 30 and lives in New York while their college roomate Jessica is 26 and lives in London",
        }
    ],
    response_model=Iterable[Person],
    stream=True,
)

for person in resp:
    print(person)
    # > Person(name='Chris', age=27)
    # > Person(name='John', age=30)
    # > Person(name='Jessica', age=26)
```

We're excited to see what you build with Instructor and Cerebras!
