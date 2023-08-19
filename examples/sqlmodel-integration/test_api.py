import openai
from pydantic import BaseModel
from sqlalchemy import create_engine
from instructor import OpenAISchema
from patch_sql import instrument_with_sqlalchemy

engine = create_engine("sqlite:///chat.db", echo=True)

instrument_with_sqlalchemy(engine)


def test_normal():
    resp = openai.ChatCompletion.create(
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


def test_schema():
    class Add(OpenAISchema):
        a: int
        b: int

    resp = openai.ChatCompletion.create(
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


def test_response_model():
    from instructor import patch
    patch()

    class Add(BaseModel):
        a: int
        b: int

    add: Add = openai.ChatCompletion.create(
        response_model=Add,
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": "1+1",
            }
        ],
    ) # type: ignore
    assert add.a == 1
    assert add.b == 1
