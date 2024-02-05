import asyncio
import openai
import instructor
import datetime

from pydantic import BaseModel, Field
from typing import Any, Coroutine, Callable, TypeVar, ParamSpec, Concatenate, Type, cast

from openai import AsyncOpenAI

models="gpt-4-0125-preview", "gpt-3.5-turbo-1106"
model = models[0]

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
    chain_of_thought: str = Field(..., description="Step by step reasoning to get the correct time range")
    name: str = Field(..., description="Name of the staff member.")
    position: str | None = Field(None, description="Position of the staff member.")
    start_date: datetime.date | None = Field(None, description="Start date for the staff member.")

class Recruits(BaseModel):
    """Correctly determine newly recruited staff details."""
    staff: list[Staff] = Field(..., description="List of new staff recruits.")

system_message_text=f"Date: {datetime.date.today().isoformat()}\nPerform accurate extractions."
system_message = {"role": "system", "content": system_message_text}
query1="We just hired Felicity Quill, she starts on Monday."
message1 = {"role": "user", "content": query1}
query2="Archibald Fiddlesticks begins in 2 weeks from Monday."
message2 = {"role": "user", "content": query2}


async def main():
    create = patched_create()
    staff1 = await create(
        Staff,
        messages=[
            {"role": "system", "content": system_message_text},
            {"role": "user", "content": query1},
        ],
        model=model,
    )
    staff2 = await create(
        Staff,
        messages=[
            {"role": "system", "content": system_message_text},
            {"role": "user", "content": query1},
        ],
        model=model,
    )
    recruits = await create(
        Recruits,
        messages=[
            {"role": "system", "content": system_message_text},
            {"role": "user", "content": query1},
        ],
        model=model,
    )
    assert isinstance(staff1, Staff)
    print(staff1.model_dump_json())
    assert isinstance(staff2, Staff)
    print(staff2.model_dump_json())
    assert isinstance(recruits, Recruits)
    print(recruits.model_dump_json())

if __name__ == "__main__":
    asyncio.run(main())

_="""
[
    {
        "chain_of_thought": "To determine Felicity's start date, we must identify the current date and then find the upcoming Monday. The current date is 2024-02-04, and the upcoming Monday from this date is 2024-02-05, which is the start date for Felicity Quill.",
        "name": "Felicity Quill",
        "position": null,
        "start_date": "2024-02-05"
    },
    {
        "chain_of_thought": "To find the start date for Archibald Fiddlesticks, we first identify today's date, which is February 4, 2024. Knowing that the context is set from a Monday, we calculate two weeks from the next Monday following today. The next Monday from February 4 is February 5. Two weeks from February 5 is February 19, 2024, which marks the start date for Archibald Fiddlesticks.",
        "name": "Archibald Fiddlesticks",
        "position": null,
        "start_date": "2024-02-19"
    },
    {
        "staff": [
            {
                "chain_of_thought": "Today is Saturday, February 3, 2024. Monday refers to February 5, 2024, which is the start date for Felicity Quill.",
                "name": "Felicity Quill",
                "position": null,
                "start_date": "2024-02-05"
            },
            {
                "chain_of_thought": "Today is Saturday, February 3, 2024. Two weeks from Monday refers to the Monday after next, which is February 19, 2024, the start date for Archibald Fiddlesticks.",
                "name": "Archibald Fiddlesticks",
                "position": null,
                "start_date": "2024-02-19"
            }
        ]
    }
]
"""

