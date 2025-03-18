import pytest
from pydantic import BaseModel
import instructor
from .util import models, modes


class User(BaseModel):
    name: str
    age: int


class Users(BaseModel):
    users: list[User]


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_simple_extraction(client, model, mode):
    client = instructor.from_genai(client, mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28 years old",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28


@pytest.mark.asyncio
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
async def test_simple_extraction_async(aclient, model, mode):
    aclient = instructor.from_genai(aclient, mode=mode, use_async=True)
    response = await aclient.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Ivan is 28 years old",
            },
        ],
        response_model=Users,
    )
    assert isinstance(response, Users)
    assert len(response.users) > 0
    assert response.users[0].name == "Ivan"
    assert response.users[0].age == 28
