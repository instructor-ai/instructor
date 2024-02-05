"""Example creating and scheduling tasks `asyncio.create_task`, processing them concurrently via `as_completed`. As they complete, results are parsed using [structure pattern matching](https://jxnl.github.io/instructor/concepts/models/#structural-pattern-matching)."""

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
    tasks: list[asyncio.Task[Staff | Recruits]] = []
    task1 = asyncio.create_task(create(
        Staff,
        messages=[
            {"role": "system", "content": system_message_text},
            {"role": "user", "content": query1},
        ],
        model=model,
    ))
    tasks.append(task1)

    task2 = asyncio.create_task(create(
        Staff,
        messages=[
            {"role": "system", "content": system_message_text},
            {"role": "user", "content": query2},
        ],
        model=model,
    ))
    tasks.append(task2)
    task3 = asyncio.create_task(create(
        Recruits,
        messages=[
            {"role": "system", "content": system_message_text},
            {"role": "user", "content": query1},
            {"role": "user", "content": query2},
        ],
        model=model,
    ))
    tasks.append(task3)
    length = []
    for task in asyncio.as_completed(tasks):
        result = await task
        match result:
            case Recruits(staff=recruits):
                assert isinstance(result, Recruits)
                assert isinstance(recruits, list)
                data_dump = result.model_dump_json()
                length.append(len(data_dump))
            case _:
                assert isinstance(result, Staff)
                data_dump = result.model_dump_json()
                length.append(len(data_dump))
        print(data_dump)
    print(f"{length}\t= {sum(length)}")

if __name__ == "__main__":
    asyncio.run(main())

_="""
[
    {
        "chain_of_thought": "To determine the start date for Felicity Quill, we need to know what day today is. Generally speaking, when someone says 'starts on Monday,' without specifying a date, it implies the closest Monday from today. Given the current date is 2024-02-04, the closest Monday is actually today, which means Felicity Quill's start date is 2024-02-05.",
        "name": "Felicity Quill",
        "position": null,
        "start_date": "2024-02-05"
    }
{
        "chain_of_thought": "To determine Archibald Fiddlesticks' start date, we take today's date, 2024-02-04, and calculate 2 weeks from the next Monday. Considering today is Sunday, the next Monday is 2024-02-05. Adding 2 weeks to this date gives us 2024-02-19.",
        "name": "Archibald Fiddlesticks",
        "position": null,
        "start_date": "2024-02-19"
    }
{
        "staff": [
            {
                "chain_of_thought": "Today is Sunday, February 4, 2024, so Monday would be February 5, 2024. Hence, Felicity Quill's start date is February 5, 2024.",
                "name": "Felicity Quill",
                "position": null,
                "start_date": "2024-02-05"
            },
            {
                "chain_of_thought": "Starting from Monday, which is February 5, 2024, adding 2 weeks leads to February 19, 2024, making it Archibald Fiddlesticks' start date.",
                "name": "Archibald Fiddlesticks",
                "position": null,
                "start_date": "2024-02-19"
            }
        ]
    }
]
"""

