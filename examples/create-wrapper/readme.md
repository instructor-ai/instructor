# Chat Completion Wrapper

## Overview

A generic method to create a wrapped reusable `chat.completion.create` function. It could very well be more interesting than useful but it's functional and doesn't break types in 3.10.

## Dependencies

- python >= 3.10.13 # lowest tested
- openai
- Pydantic

## Usage

```python
import openai
from typing import Any, Coroutine, Callable, TypeVar, ParamSpec, Concatenate, Type

T = TypeVar('T')
M = TypeVar('M')
P = ParamSpec('P')

def patched_create(**kwargs: Any):
    _client = openai.AsyncClient(**kwargs)
    client: AsyncOpenAI = instructor.patch(_client)
    func = client.chat.completions.create
    def async_create_wrapper(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[Concatenate[Type[M], P], Coroutine[Any, Any, M]]:
        async def wrapper(val: Type[M] | None = None, *args: P.args, **kwargs: P.kwargs) -> M:
            if response_model := kwargs.pop("response_model", val):
                val = cast(Type[M], response_model)
            return await cast(Callable[Concatenate[Type[M] | None, P], Coroutine[Any, Any, M]], func)(val, *args, **kwargs)
        return wrapper
    return async_create_wrapper(func)

class Staff(BaseModel):
    """Correctly determine employee information."""
    name: str = Field(..., description="Name of the staff member.")

create = patched_create()
staff = await create(
    Staff,
    messages=[{"role": "user", "content": ...}],
    model="gpt-3.5-turbo-1106",
)

assert isinstance(staff, Staff)

```

## Examples

- ***./run.py*** - standard multi-model typed async example via `asyncio.run`
- ***./gather.py*** - multi-model typed concurrent async example using [structural pattern matching](https://jxnl.github.io/instructor/concepts/models/#structural-pattern-matching) via `asyncio.gather`
- ***./as_completed.py*** - multi-model typed concurrent async example using [structural pattern matching](https://jxnl.github.io/instructor/concepts/models/#structural-pattern-matching) via `asyncio.as_completed`.


> Continuation idea: cast `Awaitable` return object compatible with `coroutine` parameters accepted by `asyncio.create_task`.
