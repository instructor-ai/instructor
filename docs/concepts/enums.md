To prevent data misalignment, we can use Enums for standardized fields. Always include an "Other" option as a fallback so the model can signal uncertainty.

```python hl_lines="7 12"
from enum import Enum, auto

class Role(Enum):
    PRINCIPAL = "PRINCIPAL"
    TEACHER = "TEACHER"
    STUDENT = "STUDENT"
    OTHER = "OTHER""

class UserDetail(BaseModel):
    age: int
    name: str
    role: Role = Field(description="Correctly assign one of the predefined roles to the user.")
```

If you're having a hard time with `Enum` and alternative is to use `Literal` instead.

```python hl_lines="4"
class UserDetail(BaseModel):
    age: int
    name: str
    role: Literal["PRINCIPAL", "TEACHER", "STUDENT", "OTHER"]
```
