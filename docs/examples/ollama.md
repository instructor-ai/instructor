# Structured Outputs with Ollama

Open-source LLMS are gaining popularity, and with the release of Ollama's OpenAI compatibility layer, it has become possible to obtain structured outputs using JSON schema.

By the end of this blog post, you will learn how to effectively utilize instructor with Ollama. But before we proceed, let's first explore the concept of patching.

## Patching

Instructor's patch enhances an openai api with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Ollama

Start by downloading [Ollama](https://ollama.ai/download), and then pull a model such as Llama 3 or Mistral.

!!! tip "Make sure you update your `ollama` to the latest version!"

```
ollama pull llama3
```

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

import instructor


class Character(BaseModel):
    name: str
    age: int
    fact: List[str] = Field(..., description="A list of facts about the character")


# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

resp = client.chat.completions.create(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": "Tell me about the Harry Potter",
        }
    ],
    response_model=Character,
)
print(resp.model_dump_json(indent=2))
"""
{
  "name": "Harry James Potter",
  "age": 37,
  "fact": [
    "He is the chosen one.",
    "He has a lightning-shaped scar on his forehead.",
    "He is the son of James and Lily Potter.",
    "He attended Hogwarts School of Witchcraft and Wizardry.",
    "He is a skilled wizard and sorcerer.",
    "He fought against Lord Voldemort and his followers.",
    "He has a pet owl named Snowy."
  ]
}
"""
```
