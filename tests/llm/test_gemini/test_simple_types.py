import instructor
import pytest

from typing import Literal


@pytest.mark.asyncio
async def test_literal():
    client = instructor.from_provider("google/gemini-2.5-flash", async_client=True)

    response = await client.chat.completions.create(
        response_model=Literal["1231", "212", "331"],
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert response in ["1231", "212", "331"]


@pytest.mark.asyncio
async def test_bool():
    client = instructor.from_provider("google/gemini-2.5-flash", async_client=True)

    response = await client.chat.completions.create(
        response_model=bool,
        messages=[
            {
                "role": "user",
                "content": "Produce a Random but correct response given the desired output",
            },
        ],
    )
    assert type(response) == bool
