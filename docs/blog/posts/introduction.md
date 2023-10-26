---
draft: False 
date: 2023-09-11
tags:
  - Introduction
---

# Bridging Language Models with Python using Instructor, Pydantic, and OpenAI's Function Calls

Language models have seen significant growth. Using them effectively often requires complex frameworks. This post discusses how Instructor simplifies this process using Pydantic.

## The Problem with Existing LLM Frameworks

Current frameworks for Language Learning Models (LLMs) have complex setups. Developers find it hard to control interactions with language models. Some frameworks require complex JSON Schema setups.

## The OpenAI Function Calling Game-Changer

OpenAI's Function Calling feature provides a constrained interaction model. However, it has its own complexities, mostly around JSON Schema.

## Why Pydantic?

Instructor uses Pydantic to simplify the interaction between the programmer and the language model.

- **Widespread Adoption**: Pydantic is a popular tool among Python developers.
- **Simplicity**: Pydantic allows model definition in Python.
- **Framework Compatibility**: Many Python frameworks already use Pydantic.

```python
import pydantic
import instructor
import openai

# Enables the response_model
instructor.patch()

class UserDetail(pydantic.BaseModel):
    name: str
    age: int
    
    def introduce(self):
        return f"Hello I'm {self.name} and I'm {self.age} years old"

user: UserDetail = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)
```

## Simplifying Validation Flow with Pydantic

Pydantic validators simplify features like re-asking or self-critique. This makes these tasks less complex compared to other frameworks.

```python
from typing_extensions import Annotated
from pydantic import BaseModel, BeforeValidator
from instructor import llm_validator, patch

import openai

class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things")
        ),
    ]
```

## The Modular Approach

Pydantic allows for modular output schemas. This leads to more organized code.

### Composition of Schemas
```python
class UserDetails(BaseModel):
    name: str
    age: int

class UserWithAddress(UserDetails):
    address: str
```

### Defining Relationships
```python
class UserDetail(BaseModel):
    id: int
    age: int
    name: str
    friends: List[int]

class UserRelationships(BaseModel):
    users: List[UserDetail]
```

### Using Enums
```python
from enum import Enum, auto

class Role(Enum):
    PRINCIPAL = auto()
    TEACHER = auto()
    STUDENT = auto()
    OTHER = auto()

class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

### Flexible Schemas
```python
from typing import List

class Property(BaseModel):
    key: str
    value: str

class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property]
```

### Chain of Thought
```python
class TimeRange(BaseModel):
    chain_of_thought: str
    start_time: int
    end_time: int

class UserDetail(BaseModel):
    id: int
    age: int
    name: str
    work_time: TimeRange
    leisure_time: TimeRange
```

## Language Models as Microservices

The architecture resembles FastAPI. Most code can be written as Python functions that use Pydantic objects. This eliminates the need for prompt chains.

### FastAPI Stub

```python
app = FastAPI()

@app.get("/user/{user_id}", response_model=UserDetails)
async def get_user(user_id: int) -> UserDetails:
    return UserDetails(...)
```

### Using Instructor as a Function

```python
def extract_user(str) -> UserDetails:
    return openai.ChatCompletion(
           response_model=UserDetails,
           messages=[...]
    )
```

### Response Modeling
```python
class MaybeUser(BaseModel):
    result: Optional[UserDetail]
    error: bool
    message: Optional[str]
```

## Conclusion

Instructor, with Pydantic, simplifies interaction with language models. It is usable for both experienced and new developers.

If you enjoy the content or want to try out `instructor` please check out the [github](https://github.com/jxnl/instructor) and give us a star!