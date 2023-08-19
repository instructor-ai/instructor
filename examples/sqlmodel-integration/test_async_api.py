import openai
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from instructor import OpenAISchema
from patch_sql import instrument_with_sqlalchemy
import pytest

async_engine = create_async_engine("sqlite+aiosqlite:///chat.db", echo=True)

instrument_with_sqlalchemy(async_engine)


@pytest.mark.asyncio
async def test_normal():
    resp = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a world class adder",
            },
            {
                "role": "user",
                "content": "1+1",
            },
        ],
    )
    assert "2" in resp.choices[0].message.content


@pytest.mark.asyncio
async def test_schema():
    class Add(OpenAISchema):
        a: int
        b: int

    resp = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo-0613",
        functions=[Add.openai_schema],
        function_call={"name": "Add"},
        messages=[
            {
                "role": "system",
                "content": "You are a world class adder",
            },
            {
                "role": "user",
                "content": "1+1",
            },
        ],
    )
    add = Add.from_response(resp)
    assert add.a == 1
    assert add.b == 1


@pytest.mark.asyncio
async def test_response_model():
    from instructor import patch

    patch()

    class Add(BaseModel):
        a: int
        b: int

    add: Add = await openai.ChatCompletion.acreate(  # type: ignore
        response_model=Add,
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": "1+1",
            }
        ],
    )
    assert add.a == 1
    assert add.b == 1
