from typing import Iterable
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import pytest

import instructor
from instructor.function_calls import Mode


class User(BaseModel):
    name: str
    age: int


Users = Iterable[User]


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
def test_multi_user(mode):
    client = instructor.patch(OpenAI(), mode=mode)

    def stream_extract(input: str) -> Iterable[User]:
        return client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "system",
                    "content": "You are a perfect entity extraction system",
                },
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = [user for user in stream_extract(input="Jason is 20, Sarah is 30")]
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
async def test_multi_user_tools_mode_async(mode):
    client = instructor.patch(AsyncOpenAI(), mode=mode)

    async def stream_extract(input: str) -> Iterable[User]:
        return await client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            stream=True,
            response_model=Users,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Consider the data below:\n{input}"
                        "Correctly segment it into entitites"
                        "Make sure the JSON is correct"
                    ),
                },
            ],
            max_tokens=1000,
        )

    resp = []
    async for user in await stream_extract(input="Jason is 20, Sarah is 30"):
        resp.append(user)
    print(resp)
    assert len(resp) == 2
    assert resp[0].name == "Jason"
    assert resp[0].age == 20
    assert resp[1].name == "Sarah"
    assert resp[1].age == 30
