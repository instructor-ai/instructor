---
draft: False
date: 2024-02-26
slug: mistral
tags:
  - patching
authors:
  - shanktt
---

# Structured Outputs with Mistral Large

If you want to try this example using `instructor hub`, you can pull it by running

```bash
instructor hub pull --slug mistral --py > mistral_example.py
```

Mistral Large is the flagship model from Mistral AI, supporting 32k context windows and functional calling abilities. Mistral Large's addition of [function calling](https://docs.mistral.ai/guides/function-calling/) makes it possible to obtain structured outputs using JSON schema.

By the end of this blog post, you will learn how to effectively utilize Instructor with Mistral Large.

<!-- more -->

## Patching

Instructor's patch enhances the mistral api with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Mistral Client

The Mistral client employs a different client than OpenAI, making the patching process slightly different than other examples

!!! note "Getting access"

    If you want to try this out for yourself check out the [Mistral AI](https://mistral.ai/) website. You can get started [here](https://docs.mistral.ai/).

```python
import instructor

from pydantic import BaseModel
from mistralai.client import MistralClient

# enables `response_model` in chat call
client = MistralClient()

patched_chat = instructor.from_openai(create=client.chat, mode=instructor.Mode.MISTRAL_TOOLS)

if __name__ == "__main__":

    class UserDetails(BaseModel):
        name: str
        age: int

    resp = patched_chat(
        model="mistral-large-latest",
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
