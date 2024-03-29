from typing import Iterable, Literal
from pydantic import BaseModel

import pytest
import instructor


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


def test_sync_parallel_tools__error(client):
    client = instructor.patch(client, mode=instructor.Mode.PARALLEL_TOOLS)

    with pytest.raises(TypeError):
        resp = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You must always use tools"},
                {
                    "role": "user",
                    "content": "What is the weather in toronto and dallas and who won the super bowl?",
                },
            ],
            response_model=Weather,
        )


def test_sync_parallel_tools_or(client):
    client = instructor.patch(client, mode=instructor.Mode.PARALLEL_TOOLS)
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather | GoogleSearch],
    )
    assert len(list(resp)) == 3


@pytest.mark.asyncio
async def test_async_parallel_tools_or(aclient):
    client = instructor.patch(aclient, mode=instructor.Mode.PARALLEL_TOOLS)
    resp = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Iterable[Weather | GoogleSearch],
    )
    assert len(list(resp)) == 3


def test_sync_parallel_tools_one(client):
    client = instructor.patch(client, mode=instructor.Mode.PARALLEL_TOOLS)
    resp = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas?",
            },
        ],
        response_model=Iterable[Weather],
    )
    assert len(list(resp)) == 2


@pytest.mark.asyncio
async def test_async_parallel_tools_one(aclient):
    client = instructor.patch(aclient, mode=instructor.Mode.PARALLEL_TOOLS)
    resp = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas?",
            },
        ],
        response_model=Iterable[Weather],
    )
    assert len(list(resp)) == 2
