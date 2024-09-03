---
draft: False
date: 2024-09-03
authors:
  - ivanleomk
---

# Announcing Gemini and VertexAI Tool Calling Support

We've excited to announce that Instructor now supports tool calling for both the Gemini SDK and the VertexAI SDk.

A special shoutout to [Sonal](https://x.com/sonalsaldanha) for his contributions to the Gemini Tool Calling support.

Let's walk through a simple example of how to use these new features

## Installation

To get started, install the latest version of Instructor. Depending on whether you're using Gemini or VertexAI, you should install the following:

=== "Gemini"

    ```bash
    uv pip install "instructor[gemini]"
    ```

=== "VertexAI"

    ```bash
    uv pip install "instructor[vertexai]"
    ```

This ensures that you have the necessary dependencies to use the Gemini or VertexAI SDKs wtih instructor.

## Getting Started

With our provider agnostric API, you can use the same interface to interact with the gemini API, the only thing that changes here is how we initialise the client itself.

Before running the following code, you'll need to make sure that you have your Gemini API Key set in your shell under the alias `GOOGLE_API_KEY`.

```python
import instructor
import google.generativeai as genai
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest", # (1)!
    ),
    mode=instructor.Mode.GEMINI_TOOLS, # (2)!
)

resp = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

print(resp)
#> name='Jason' age=25
```

1. Current Gemini models that support tool calling are `gemini-1.5-flash-latest` and `gemini-1.5-pro-latest`.

2. Make sure to set the `mode` to `instructor.Mode.GEMINI_TOOLS` in order to use Gemini Tool Calling

We can achieve a similar thing with the VertexAI SDk. For this to work, you'll need to authenticate to VertexAI.

There are some instructions [here](https://cloud.google.com/vertex-ai/docs/authentication) but the easiest way I found was to simply download the GCloud cli and run `gcloud auth application-default login`.

```python
iimport instructor
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel

vertexai.init()


class User(BaseModel):
    name: str
    age: int


client = instructor.from_vertexai(
    client=GenerativeModel("gemini-1.5-pro-preview-0409"), # (1)!
    mode=instructor.Mode.VERTEXAI_TOOLS,# (2)!
)


resp = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

print(resp)
```

1. Current Gemini models that support tool calling are `gemini-1.5-flash-latest` and `gemini-1.5-pro-latest`.

2. Make sure to set the `mode` to `instructor.Mode.GEMINI_TOOLS` in order to use Gemini Tool Calling
