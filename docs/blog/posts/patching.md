---
draft: False
date: 2023-12-15
slug: patching
tags:
  - patching
  - open source
authors:
  - anmol
---

# Introduction

Assuming you have _some_ familiarity with instructor, this blog will go over how we can use instructor with various third-party models other than OpenAI.

# Models models everywhere

With the explosion in open source models, users have seen a corresponding increase in model choice.
Different models make different tradeoffs across cost, feature-set & latency (among other things).

Instructor's core functionality -- implemented as a patch -- is meant to support these different models.

# Patching

At its core, Instructor acts as a patch on an open ai model client. This patch adds the following on top of the client:

- It makes it easier to get the structured outputs you want - you can model expected outputs as pydantic dataclasses v/s spending hours on prompt engineering
- It lets you define custom validators to validate model responses

The patch works across different open ai output modes (function calling, json, tools).

But what if you want to use a model _outside_ of OpenAI (should you be so daring)?

## Anyscale

The good news is that Anyscale uses the same OpenAI client & the models support some of these output modes!

Let's use a model from Anyscale's repertoire of many models!

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel


class UserDetails(BaseModel):
    name: str
    age: int

client = instructor.patch(
    OpenAI(
        base_url = "https://api.endpoints.anyscale.com/v1",
        api_key="<YOUR_ANYSCALE_API_KEY>
    ),
    mode=instructor.Mode.JSON_SCHEMA
)

resp = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_model=UserDetails,
    messages=[
        {"role": "system", "content": "You are a world class extractor"},
        {"role": "user", "content": 'Extract the following entities: "Jason is 20"'},
    ],
)
print(resp)
>>> name='Jason' age=20
```

You can read more Anyscale's output mode support [here](https://docs.endpoints.anyscale.com/)
