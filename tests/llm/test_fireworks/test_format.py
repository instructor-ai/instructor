import instructor
from fireworks.client import Fireworks, AsyncFireworks
from pydantic import BaseModel
import pytest
from .util import modes


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_sync(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : {{ name }} is {{ age }} and lives in Singapore",
            },
        ],
        context={
            "name": "Ivan",
            "age": 27,
        },
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio
async def test_fireworks_async(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(AsyncFireworks(), mode=mode)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : {{ name }} is {{ age }} and lives in Singapore",
            },
        ],
        context={
            "name": "Ivan",
            "age": 27,
        },
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27
