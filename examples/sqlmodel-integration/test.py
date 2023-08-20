import openai
from pydantic import BaseModel
from sqlalchemy import create_engine
from patch_sql import instrument_with_sqlalchemy
from instructor import patch

engine = create_engine("sqlite:///chat.db", echo=True)
instrument_with_sqlalchemy(engine)
patch()


class Add(BaseModel):
    a: int
    b: int


resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    response_model=Add,
    messages=[
        {
            "role": "user",
            "content": "1+1",
        }
    ],
)

assert resp.a == 1
assert resp.b == 1
