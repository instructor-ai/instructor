---
draft: False
date: 2024-02-26
title: "Structured outputs with Mistral, a complete guide w/ instructor"
description: "Complete guide to using Instructor with Mistral. Learn how to generate structured, type-safe outputs with Mistral."
slug: mistral
tags:
  - patching
authors:
  - shanktt
---

# Structured outputs with Mistral, a complete guide w/ instructor

This guide demonstrates how to use Mistral with Instructor to generate structured outputs. You'll learn how to use function calling with Mistral Large to create type-safe responses.

Mistral Large is the flagship model from Mistral AI, supporting 32k context windows and functional calling abilities. Mistral Large's addition of [function calling](https://docs.mistral.ai/guides/function-calling/) makes it possible to obtain structured outputs using JSON schema.

By the end of this blog post, you will learn how to effectively utilize Instructor with Mistral Large.

```python
import os
from pydantic import BaseModel
from mistralai import Mistral
from instructor import from_mistral, Mode


class UserDetails(BaseModel):
    name: str
    age: int


# enables `response_model` in chat call
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

instructor_client = from_mistral(
    client=client,
    model="mistral-large-latest",
    mode=Mode.MISTRAL_TOOLS,
    max_tokens=1000,
)

resp = instructor_client.messages.create(
    response_model=UserDetails,
    messages=[{"role": "user", "content": "Jason is 10"}],
    temperature=0,
)

print(resp)

```
