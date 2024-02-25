---
draft: False
date: 2024-01-27
slug: together
tags:
  - patching
  - open source
authors:
  - jxnl
---

# Structured Outputs with Together AI

If you want to try this example using `instructor hub`, you can pull it by running

```bash
instructor hub pull --slug together --py > together_example.py
```

Open-source LLMS are gaining popularity, and with the release of Together's Function calling models, its been easier than ever to get structured outputs.

By the end of this blog post, you will learn how to effectively utilize instructor with Together AI. But before we proceed, let's first explore the concept of patching.

!!! note "Other Languages"

    This blog post is written in Python, but the concepts are applicable to other languages as well, as we currently have support for [Javascript](
        https://instructor-ai.github.io/instructor-js) and [Elixir](https://hexdocs.pm/instructor/Instructor.html)

<!-- more -->

## Patching

Instructor's patch enhances the openai api it with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more, please refer to the [docs](../index.md). To understand the benefits of using Pydantic with Instructor, visit the tips and tricks section of the [why use Pydantic](../why.md) page.

## Together AI

The good news is that Together employs the same OpenAI client, and its models support some of these output modes too!

!!! note "Getting access"

    If you want to try this out for yourself check out the [Together AI](https://www.together.ai/) website. You can get started [here](http://api.together.ai/).

```python
import os
import openai
from pydantic import BaseModel
import instructor

client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
)


# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.patch(client, mode=instructor.Mode.TOOLS)


# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
class UserExtract(BaseModel):
    name: str
    age: int


user: UserExtract = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(user, UserExtract), "Should be instance of UserExtract"
assert user.name.lower() == "jason"
assert user.age == 25

print(user.model_dump_json(indent=2))
"""
{
  "name": "jason",
  "age": 25
}
"""
{
    "name": "Jason",
    "age": 25,
}
```

You can find more information about Together's function calling support [here](https://docs.together.ai/docs/function-calling).
