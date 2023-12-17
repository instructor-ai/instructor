---
draft: False
date: 2023-12-15
slug: patching
tags:
  - patching
  - open source
authors:
  - anmol
  - jxnl
---

# Structured Outputs with Anyscale

Open-source LLMS are gaining popularity, and the release of Anyscale's Mistral model has made it possible to obtain structured outputs using JSON schema at any scale. Instead of relying on a model's default output mode, you can utilize JSON schema to obtain structured outputs. This approach is a time-saving alternative to extensive prompt engineering.

By the end of this blog post, you will learn how to effectively utilize the instructor at any scale. But before we proceed, let's first explore the concept of patching.

## Patching

Instructor's patch enhances a openai api it with the following features:

- `response_model` in `create` calls that returns a pydantic model
- `max_retries` in `create` calls that retries the call if it fails by using a backoff strategy

!!! note "Learn More"

    To learn more check out the [docs](../../index.md)

## Anyscale

The good news is that Anyscale employs the same OpenAI client, and its models support some of these output modes too!

Let's explore one of the models available in Anyscale's extensive collection!

```python
from openai import OpenAI
from pydantic import BaseModel

import instructor


class UserDetails(BaseModel):
    name: str
    age: int

# enables `response_model` in create call
client = instructor.patch(
    OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key="<YOUR_ANYSCALE_API_KEY>
    ),
    # This uses Anyscale's json schema output mode
    mode=instructor.Mode.JSON_SCHEMA
)

resp = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    messages=[
        {
            "role": "system",
            "content": "You are a world class extractor"
        },
        {
            "role": "user",
            "content": 'Extract the following entities: "Jason is 20"'
        },
    ],
    response_model=UserDetails,
)
print(resp)
>>> name='Jason' age=20
```

You can find more information about Anyscale's output mode support [here](https://docs.endpoints.anyscale.com/).
