import pytest
import instructor
import enum
from typing import Annotated, Literal, Union, Any
from pydantic import Field
from openai import AsyncClient, Client
from .util import models


@pytest.mark.asyncio
async def test_response_simple_types(aclient: AsyncClient) -> None:
    client = instructor.patch(aclient, mode=instructor.Mode.TOOLS)

    for response_model in [int, bool, str]:
        response = await client.chat.completions.create(
            model=models[0],
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
async def test_annotate(aclient: AsyncClient) -> None:
    client = instructor.patch(aclient, mode=instructor.Mode.TOOLS)

    response = await client.chat.completions.create(
        model=models[0],
        response_model=Annotated[int, Field(description="test")],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == int


def test_literal(client: Client) -> None:
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model=models[0],
        response_model=Literal["1231", "212", "331"],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in ["1231", "212", "331"]


def test_union(client: Client) -> None:
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model=models[0],
        response_model=Union[int, str],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) in [int, str]


def test_enum(client: Client) -> None:
    class Options(enum.Enum):
        A = "A"
        B = "B"
        C = "C"

    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model=models[0],
        response_model=Options,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in [Options.A, Options.B, Options.C]


def test_bool(client: Client) -> None:
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    response = client.chat.completions.create(
        model=models[0],
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == bool
