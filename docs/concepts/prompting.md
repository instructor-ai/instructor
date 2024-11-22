---
title: Effective Prompt Engineering with Pydantic and Instructor
description: Discover best practices for prompt engineering using Pydantic and Instructor to enhance modularity, flexibility, and data integrity.
---

# General Tips for Prompt Engineering

The overarching theme of using Instructor and Pydantic for function calling is to make the models as self-descriptive, modular, and flexible as possible, while maintaining data integrity and ease of use.

- **Modularity**: Design self-contained components for reuse.
- **Self-Description**: Use Pydantic's `Field` for clear field descriptions.
- **Optionality**: Use Python's `Optional` type for nullable fields and set sensible defaults.
- **Standardization**: Employ enumerations for fields with a fixed set of values; include a fallback option.
- **Dynamic Data**: Use key-value pairs for arbitrary properties and limit list lengths.
- **Entity Relationships**: Define explicit identifiers and relationship fields.
- **Contextual Logic**: Optionally add a "chain of thought" field in reusable components for extra context.

## Modular Chain of Thought {#chain-of-thought}

This approach to "chain of thought" improves data quality but can have modular components rather than global CoT.

```python hl_lines="4 5"
from pydantic import BaseModel, Field


class Role(BaseModel):
    chain_of_thought: str = Field(
        ..., description="Think step by step to determine the correct title"
    )
    title: str


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

## Utilize Optional Attributes

Use Python's Optional type and set a default value to prevent undesired defaults like empty strings.

```python hl_lines="6"
from typing import Optional
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)
```

## Handling Errors Within Function Calls

You can create a wrapper class to hold either the result of an operation or an error message. This allows you to remain within a function call even if an error occurs, facilitating better error handling without breaking the code flow.

```python
from pydantic import BaseModel, Field
from typing import Optional


class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)


class MaybeUser(BaseModel):
    result: Optional[UserDetail] = Field(default=None)
    error: bool = Field(default=False)
    message: Optional[str]

    def __bool__(self):
        return self.result is not None
```

With the `MaybeUser` class, you can either receive a `UserDetail` object in result or get an error message in message.

### Simplification with the Maybe Pattern

You can further simplify this using instructor to create the `Maybe` pattern dynamically from any `BaseModel`.

```python
import instructor
from pydantic import BaseModel


class UserDetail(BaseModel):
    age: int
    name: str


MaybeUser = instructor.Maybe(UserDetail)
```

This allows you to quickly create a Maybe type for any class, streamlining the process.

## Tips for Enumerations

To prevent data misalignment, use Enums for standardized fields. Always include an "Other" option as a fallback so the model can signal uncertainty.

```python hl_lines="7 12"
from enum import Enum, auto
from pydantic import BaseModel, Field


class Role(Enum):
    PRINCIPAL = auto()
    TEACHER = auto()
    STUDENT = auto()
    OTHER = auto()


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role = Field(
        description="Correctly assign one of the predefined roles to the user."
    )
```

## Literals {#literals}

If you're having a hard time with `Enum` an alternative is to use `Literal`

```python hl_lines="4"
from typing import Literal
from pydantic import BaseModel


class UserDetail(BaseModel):
    age: int
    name: str
    role: Literal["PRINCIPAL", "TEACHER", "STUDENT", "OTHER"]
```

If you'd like to improve performance more you can reiterate the requirements in the field descriptions or in the docstrings.

## Reiterate Long Instructions

For complex attributes, it helps to reiterate the instructions in the field's description.

```python hl_lines="5 11"
from pydantic import BaseModel, Field


class Role(BaseModel):
    """
    Extract the role based on the following rules ...
    """

    instructions: str = Field(
        ...,
        description="Restate the instructions and rules to correctly determine the title.",
    )
    title: str


class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

## Handle Arbitrary Properties

When you need to extract undefined attributes, use a list of key-value pairs.

```python hl_lines="10"
from typing import List
from pydantic import BaseModel, Field


class Property(BaseModel):
    key: str
    value: str


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property] = Field(
        ..., description="Extract any other properties that might be relevant."
    )
```

## Limiting the Length of Lists

When dealing with lists of attributes, especially arbitrary properties, it's crucial to manage the length. You can use prompting and enumeration to limit the list length, ensuring a manageable set of properties.

```python hl_lines="2 9"
from typing import List
from pydantic import BaseModel, Field


class Property(BaseModel):
    index: str = Field(..., description="Monotonically increasing ID")
    key: str
    value: str


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property] = Field(
        ...,
        description="Numbered list of arbitrary extracted properties, should be less than 6",
    )
```

**Using Tuples for Simple Types**

For simple types, tuples can be a more compact alternative to custom classes, especially when the properties don't require additional descriptions.

```python hl_lines="4"
from typing import List, Tuple
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Tuple[int, str]] = Field(
        ...,
        description="Numbered list of arbitrary extracted properties, should be less than 6",
    )
```

## Advanced Arbitrary Properties

For multiple users, aim to use consistent key names when extracting properties.

```python
from typing import List
from pydantic import BaseModel


class UserDetail(BaseModel):
    id: int
    age: int
    name: str


class UserDetails(BaseModel):
    """
    Extract information for multiple users.
    Use consistent key names for properties across users.
    """

    users: List[UserDetail]
```

This refined guide should offer a cleaner and more organized approach to structure engineering in Python.

## Defining Relationships Between Entities

In cases where relationships exist between entities, it's vital to define them explicitly in the model. The following example demonstrates how to define relationships between users by incorporating an id and a friends field:

```python hl_lines="2 5 8"
from typing import List
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier for each user.")
    age: int
    name: str
    friends: List[int] = Field(
        ...,
        description="Correct and complete list of friend IDs, representing relationships between users.",
    )


class UserRelationships(BaseModel):
    users: List[UserDetail] = Field(
        ...,
        description="Collection of users, correctly capturing the relationships among them.",
    )
```

## Reusing Components with Different Contexts

You can reuse the same component for different contexts within a model. In this example, the TimeRange component is used for both work_time and leisure_time.

```python hl_lines="9 10"
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    start_time: int = Field(..., description="The start time in hours.")
    end_time: int = Field(..., description="The end time in hours.")


class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier for each user.")
    age: int
    name: str
    work_time: TimeRange = Field(
        ..., description="Time range during which the user is working."
    )
    leisure_time: TimeRange = Field(
        ..., description="Time range reserved for leisure activities."
    )
```

Sometimes, a component like TimeRange may require some context or additional logic to be used effectively. Employing a "chain of thought" field within the component can help in understanding or optimizing the time range allocations.

```python hl_lines="2"
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    chain_of_thought: str = Field(
        ..., description="Step by step reasoning to get the correct time range"
    )
    start_time: int = Field(..., description="The start time in hours.")
    end_time: int = Field(..., description="The end time in hours.")
```
