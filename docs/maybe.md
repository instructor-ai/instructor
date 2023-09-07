# Handling Errors Within Function Calls

You can create a wrapper class to hold either the result of an operation or an error message. This allows you to remain within a function call even if an error occurs, facilitating better error handling without breaking the code flow.

```python
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

## Simplification with the Maybe Pattern

You can further simplify this using instructor to create the `Maybe` pattern.

```python
import instructor

MaybeUser = instructor.Maybe(UserDetail)
```

This allows you to quickly create a Maybe type for any class, streamlining the process.