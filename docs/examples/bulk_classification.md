---
title: User-Provided Tag Classification Tutorial
description: Learn to classify user-provided tags effectively using async functions and FastAPI for parallel processing.
---

# Bulk Classification from User-Provided Tags.

This tutorial shows how to do classification from user provided tags. This is valuable when you want to provide services that allow users to do some kind of classification.

!!! tips "Motivation"

    Imagine allowing the user to upload documents as part of a RAG application. Oftentimes, we might want to allow the user to specify an existing set of tags, give descriptions, and do the classification for them.

## Defining the Structures

One of the easy things to do is to allow users to define a set of tags in some kind of schema and save that in a database. Here's an example of a schema that we might use:

| tag_id | name     | instructions         |
| ------ | -------- | -------------------- |
| 0      | personal | Personal information |
| 1      | phone    | Phone number         |
| 2      | email    | Email address        |
| 3      | address  | Address              |
| 4      | Other    | Other information    |

1. **tag_id** — The unique identifier for the tag.
2. **name** — The name of the tag.
3. **instructions** — A description of the tag, which can be used as a prompt to describe the tag.

## Implementing the Classification

In order to do this we'll do a couple of things:

0. We'll use the `instructor` library to patch the `openai` library to use the `AsyncOpenAI` client.
1. Implement a `Tag` model that will be used to validate the tags from the context. (This will allow us to avoid hallucinating tags that are not in the context.)
2. Helper models for the request and response.
3. An async function to do the classification.
4. A main function to run the classification using the `asyncio.gather` function to run the classification in parallel.

If you want to learn more about how to do bad computations, check out our post on AsyncIO [here](../blog/posts/learn-async.md).

```python
import openai
import instructor

client = instructor.from_openai(
    openai.AsyncOpenAI(),
)
```

First, we'll need to import all of our Pydantic and instructor code and use the AsyncOpenAI client. Then, we'll define the tag model along with the tag instructions to provide input and output.

This is very helpful because once we use something like FastAPI to create endpoints, the Pydantic functions will serve as multiple tools:

1. A description for the developer
2. Type hints for the IDE
3. OpenAPI documentation for the FastAPI endpoint
4. Schema and Response Model for the language model.

```python
from typing import List
from pydantic import BaseModel, field_validator, ConfigDict

class Tag(BaseModel):
    id: int
    name: str
    model_config = ConfigDict(validate_default=True)

    @field_validator("id", "name")
    @classmethod
    def validate_ids(cls, value, info):
        context = info.context
        if context:
            tags: List[Tag] = context.get("tags")
            if info.field_name == "id":
                assert value in {tag.id for tag in tags}, f"Tag ID {value} not found in context"
            else:  # name
                assert value in {tag.name for tag in tags}, f"Tag name {value} not found in context"
        return value

class TagWithInstructions(Tag):
    instructions: str

class TagRequest(BaseModel):
    texts: List[str]
    tags: List[TagWithInstructions]

class TagResponse(BaseModel):
    texts: List[str]
    predictions: List[Tag]
```

Let's delve deeper into what the `validate_ids` function does. Notice that its purpose is to extract tags from the context and ensure that each ID and name exists in the set of tags. This approach helps minimize hallucinations. If we mistakenly identify either the ID or the tag, an error will be thrown, and the instructor will prompt the language model to retry until the correct item is successfully extracted.

```python
from pydantic import field_validator, ConfigDict

@field_validator("id", "name")
@classmethod
def validate_ids(cls, value, info):
    context = info.context
    if context:
        tags: List[Tag] = context.get("tags")
        if info.field_name == "id":
            assert value in {tag.id for tag in tags}, f"Tag ID {value} not found in context"
        else:  # name
            assert value in {tag.name for tag in tags}, f"Tag name {value} not found in context"
    return value

{ unchanged content from line 116 to line 362 }

from typing import List
from pydantic import BaseModel, field_validator, ConfigDict, Field

class TagWithConfidence(Tag):
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="The confidence of the prediction, 0 is low, 1 is high",
    )

{ unchanged multiclass classification example and remaining content }
```
