---
title: Structured outputs with Cohere, a complete guide w/ instructor
description: Learn how to leverage Cohere's command models with Python's instructor library for structured data outputs.
---

# Structured outputs with Cohere, a complete guide w/ instructor

This guide demonstrates how to use Cohere with Instructor to generate structured outputs. You'll learn how to use Cohere's command models to create type-safe responses.

You can now use any of the Cohere's [command models](https://docs.cohere.com/docs/models) with the `instructor` library to get structured outputs.

You'll need a cohere API key which can be obtained by signing up [here](https://dashboard.cohere.com/) and gives you [free](https://cohere.com/pricing), rate-limited usage for learning and prototyping.

## Setup

```
pip install "instructor[cohere]"

```

Export your key:

```
export CO_API_KEY=<YOUR_COHERE_API_KEY>
```

## Example

```python
from pydantic import BaseModel, Field
from typing import List
import cohere
import instructor


# Patching the Cohere client with the instructor for enhanced capabilities
client = instructor.from_cohere(
    cohere.Client(),
    max_tokens=1000,
    model="command-r-plus",
)


class Person(BaseModel):
    name: str = Field(description="name of the person")
    country_of_origin: str = Field(description="country of origin of the person")


class Group(BaseModel):
    group_name: str = Field(description="name of the group")
    members: List[Person] = Field(description="list of members in the group")


task = """\
Given the following text, create a Group object for 'The Beatles' band

Text:
The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time. The group were integral to the development of 1960s counterculture and popular music's recognition as an art form.
"""
group = client.messages.create(
    response_model=Group,
    messages=[{"role": "user", "content": task}],
    temperature=0,
)

print(group.model_dump_json(indent=2))
"""
{
  "group_name": "The Beatles",
  "members": [
    {
      "name": "John Lennon",
      "country_of_origin": "England"
    },
    {
      "name": "Paul McCartney",
      "country_of_origin": "England"
    },
    {
      "name": "George Harrison",
      "country_of_origin": "England"
    },
    {
      "name": "Ringo Starr",
      "country_of_origin": "England"
    }
  ]
}
"""
```
