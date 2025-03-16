---
title: Integrating Instructor with SQLModel in Python
description: Learn how to integrate Instructor with SQLModel for seamless database interactions and API development in Python.
---

# Integrating Instructor with SQLModel

[SQLModel](https://sqlmodel.tiangolo.com/) is a library designed for interacting with SQL databases from Python code using Python objects. 

`SQLModel` is based on `Pydantic` and `SQLAlchemy` and was created by [tiangolo](https://twitter.com/tiangolo) who also developed `FastAPI`. 

So you can expect seamless integration across all these libraries, reducing code duplicating and improving your developer experience. 

# Example: Adding responses from Instructor directly to your DB

## Defining the Models

First we'll define a model that will serve as a table for our database and the structure of our outputs from `Instructor`

!!! tips "Model Definition"

    You'll need to subclass your models with both `SQLModel` and `instructor.OpenAISchema` for them to work with SQLModel

```python
from typing import Optional
from uuid import UUID, uuid4
from pydantic.json_schema import SkipJsonSchema
from sqlmodel import Field, SQLModel
import instructor


class Hero(SQLModel, instructor.OpenAISchema, table=True):
    id: SkipJsonSchema[UUID] = Field(default_factory=lambda: uuid4(), primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None
```

### The Importance of Using SkipJsonSchema

Notice the use of `SkipJsonSchema` for the `id` field. 

This prunes the field from the JSON schema sent to the LLM, so it won't try to generate a UUID.

When Instructor unpacks the response and loads it into the Hero model, it will automatically generate a UUID using the default_factory.
    
This approach saves tokens during LLM generation and more importantly protects against errors that might occur if the LLM generates an incorrect UUID format. 

The resulting JSON schema sent to the LLM will look like:

```json
{
    "properties": {
        "name": {
            "title": "Name",
            "type": "string"
        },
        "secret_name": {
            "title": "Secret Name",
            "type": "string"
        },
        "age": {
            "anyOf": [
                {
                    "type": "integer"
                },
                {
                    "type": "null"
                }
            ],
            "default": null,
            "title": "Age"
        }
    },
    "required": [
        "name",
        "secret_name"
    ],
    "title": "Hero",
    "type": "object"
}
```


## Generating a record

The `create_hero` function will query `OpenAI` for a `Hero` record

```python
import instructor
from openai import OpenAI

# <%hide%>
from typing import Optional
from uuid import UUID, uuid4
from pydantic.json_schema import SkipJsonSchema
from sqlmodel import Field, SQLModel


class Hero(SQLModel, instructor.OpenAISchema, table=True):
    __table_args__ = {'extend_existing': True}
    id: SkipJsonSchema[UUID] = Field(default_factory=lambda: uuid4(), primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None


# <%hide%>

client = instructor.from_openai(OpenAI())


def create_hero() -> Hero:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Hero,
        messages=[
            {"role": "user", "content": "Make a new superhero"},
        ],
    )
```

## Inserting the response into the DB

```python
# <%hide%>
import instructor
from openai import OpenAI
from typing import Optional
from uuid import UUID, uuid4
from pydantic.json_schema import SkipJsonSchema
from sqlmodel import Field, SQLModel, create_engine, Session


class Hero(SQLModel, instructor.OpenAISchema, table=True):
    __table_args__ = {'extend_existing': True}
    id: SkipJsonSchema[UUID] = Field(default_factory=lambda: uuid4(), primary_key=True)
    name: str
    secret_name: str
    age: Optional[int] = None


client = instructor.from_openai(OpenAI())


def create_hero() -> Hero:
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Hero,
        messages=[
            {"role": "user", "content": "Make a new superhero"},
        ],
    )


# <%hide%>
engine = create_engine("sqlite:///database.db")
SQLModel.metadata.create_all(engine)

hero = create_hero()

# The Raw Response from the LLM will not have an id due to the SkipJsonSchema
print(hero._raw_response.choices[0].message.content)
#> {'name': 'Superman', 'secret_name': 'Clark Kent', 'age': 30}

# The model_dump() method will include the generated id as it has been loaded as a Hero object
print(hero.model_dump())
#> {'name': 'Superman', 'secret_name': 'Clark Kent', 'age': 30, 'id': UUID('1234-5678-...')}

with Session(engine) as session:
    session.add(hero)
    session.commit()
```

![Image of hero record in the database](db.png)

And there you have it! You can now use the same models for your database and `Instructor` enabling them work seamlessly! Also checkout the [FastAPI](../concepts/fastapi.md) guide to see how you can use these models in an API as well. 