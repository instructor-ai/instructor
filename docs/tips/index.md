# Tips for Structure Engineering in Python

## Modular Chain of Thought

This approach to "chain of thought" improves data quality but can have modular components rather than global CoT.

```python hl_lines="4 5"
from pydantic import BaseModel, Field

class Role(BaseModel):
    chain_of_thought: str = Field(..., 
        description="Think step by step to determine the correct title")
    title: str

class UserDetail(BaseModel):
    age: int
    name: str
    role: Role
```

## Utilize Optional Attributes

For optional fields, use Python's **`Optional`** type and set a default value to avoid the model filling in undesired defaults like `''` for strings that are missing

```python hl_lines="6"
from typing import Optional

class UserDetail(BaseModel):
    age: int
    name: str
    role: Optional[str] = Field(default=None)

```

## Tips for Enumerations

To prevent data misalignment, use Enums for standardized fields. Always include an "Other" option as a fallback so the model can signal uncertainty.

```python hl_lines="7 12"
from enum import Enum, auto

class Role(Enum):
    PRINCIPAL = auto()
    TEACHER = auto()
    STUDENT = auto()
    OTHER = auto()

class UserDetail(BaseModel):
    age: int
    name: str
    role: Role = Field(description="Correctly assign one of the predefined roles to the user.")

```

## Reiterate Long Instructions

For complex attributes, it helps to reiterate the instructions in the field's description.

```python hl_lines="5 11"
class Role(BaseModel):
    """
    Extract the role based on the following rules ...
    """
    instructions: str = Field(..., description="Restate the instructions and rules to correctly determine the title.")
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

class Property(BaseModel):
    key: str
    value: str

class UserDetail(BaseModel):
    age: int
    name: str
    properties: List[Property] = Field(..., description="Extract any other properties that might be relevant.")

```

## Advanced Arbitrary Properties

For multiple users, aim to use consistent key names when extracting properties.

```python
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
class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier for each user.")
    age: int
    name: str
    friends: List[int] = Field(..., description="Correctl and complete list of friend IDs, representing relationships between users.")

class UserRelationships(BaseModel):
    users: List[UserDetail] = Field(..., description="Collection of users, correctly capturing the relationships among them.")
```

## Reusing Components with Different Contexts

You can reuse the same component for different contexts within a model. In this example, the TimeRange component is used for both work_time and leisure_time.

```python hl_lines="9 10"
class TimeRange(BaseModel):
    start_time: int = Field(..., description="The start time in hours.")
    end_time: int = Field(..., description="The end time in hours.")

class UserDetail(BaseModel):
    id: int = Field(..., description="Unique identifier for each user.")
    age: int
    name: str
    work_time: TimeRange = Field(..., description="Time range during which the user is working.")
    leisure_time: TimeRange = Field(..., description="Time range reserved for leisure activities.")
```

Sometimes, a component like TimeRange may require some context or additional logic to be used effectively. Employing a "chain of thought" field within the component can help in understanding or optimizing the time range allocations.

```python hl_lines="2"
class TimeRange(BaseModel):
    chain_of_thought: str = Field(..., description="Step by step reasoning to get the correct time range")
    start_time: int = Field(..., description="The start time in hours.")
    end_time: int = Field(..., description="The end time in hours.")
```