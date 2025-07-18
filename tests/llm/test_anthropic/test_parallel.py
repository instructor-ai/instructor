import instructor
import pytest
from pydantic import BaseModel
from typing import Union, Literal
from collections.abc import Iterable


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


def test_sync_parallel_tools_or(client):
    client = instructor.from_anthropic(
        client, mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS
    )
    resp = client.chat.completions.create(
        model="claude-3-5-haiku-latest",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    assert len(list(resp)) == 3


@pytest.mark.asyncio
async def test_async_parallel_tools_or(aclient):
    client = instructor.from_anthropic(
        aclient, mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS
    )
    resp = await client.chat.completions.create(
        model="claude-3-5-haiku-latest",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    assert len(list(resp)) == 3
