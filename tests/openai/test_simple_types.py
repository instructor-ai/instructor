import pytest
import instructor
import enum

from typing import Annotated, Literal, Union
from pydantic import Field


@pytest.mark.asyncio
async def test_response_simple_types(aclient):
    client = instructor.patch(aclient, mode=instructor.Mode.TOOLS)

    for response_model in [int, bool, str]:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=response_model,
            messages=[
                {
                    "role": "user",
                    "content": "Produce a Random but correct response given the desired output",
                },
            ],
        )
        assert type(response) == response_model


@pytest.mark.asyncio
async def test_annotate(aclient):
    client = instructor.patch(aclient, mode=instructor.Mode.TOOLS)

    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Annotated[int, Field(description="test")],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == int


def test_literal(client):
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Literal["1231", "212", "331"],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in ["1231", "212", "331"]


def test_union(client):
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Union[int, str],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) in [int, str]


def test_enum(client):
    class Options(enum.Enum):
        A = "A"
        B = "B"
        C = "C"

    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Options,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in [Options.A, Options.B, Options.C]


def test_bool(client):
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == bool
