import instructor
from instructor import from_writer
from writerai import Writer, AsyncWriter
from pydantic import BaseModel
from .util import models, modes


class User(BaseModel):
    first_name: str
    age: int


import pytest
from itertools import product


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_format_sync(model: str, mode: instructor.Mode):
    client = from_writer(
        client=Writer(),
        mode=mode,
    )

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract {{name}} is {{age}} years old.",
            }
        ],
        response_model=User,
        context={"name": "Jason", "age": 25},
    )

    assert isinstance(resp, User)
    assert resp.first_name == "Jason"
    assert resp.age == 25


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_writer_format_async(mode: instructor.Mode, model: str):
    client = instructor.from_writer(
        client=AsyncWriter(),
        mode=mode,
    )

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : {{name}} is {{age}} and lives in Berlin",
            },
        ],
        context={
            "name": "Yan",
            "age": 27,
        },
        response_model=User,
    )

    assert resp.first_name == "Yan"
    assert resp.age == 27