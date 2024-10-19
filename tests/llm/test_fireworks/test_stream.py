import pytest
from pydantic import BaseModel
from .util import modes
import instructor
from fireworks.client import Fireworks, AsyncFireworks


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_iterables(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create_iterable(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that extracts users from a sentence. Make sure to adhere to the desired JSON output format of {User.model_json_schema()} and make sure that your output can be parsed by a JSON parser.",
            },
            {
                "role": "user",
                "content": "Extract all users from this sentence : Ivan is 27 and lives in Singapore. Darren is around the same age and lives in the same city. ",
            },
        ],
        response_model=User,
        stream=True,
    )

    users = [user for user in resp]

    assert len(users) == 2
    assert {user.name for user in users} == {"Ivan", "Darren"}
    assert {user.age for user in users} == {27}


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio(scope="session")
async def test_fireworks_iterables_async(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(AsyncFireworks(), mode=mode)

    resp = client.chat.completions.create_iterable(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that extracts users from a sentence. Make sure to adhere to the desired JSON output format of {User.model_json_schema()} and make sure that your output can be parsed by a JSON parser.",
            },
            {
                "role": "user",
                "content": "Extract all users from this sentence : Ivan is 27 and lives in Singapore. Darren is around the same age and lives in the same city.",
            },
        ],
        response_model=User,
    )

    users = []
    async for m in resp:
        assert isinstance(m, User)
        users.append(m)

    assert len(users) == 2
    assert {user.name for user in users} == {"Ivan", "Darren"}
    assert {user.age for user in users} == {27}


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_partial(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create_partial(
        model=model,
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that extracts users from a sentence. Make sure to adhere to the desired JSON output format of {User.model_json_schema()} and make sure that your output can be parsed by a JSON parser.",
            },
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
    )

    user = None
    for r in resp:
        user = r

    assert user is not None
    assert user.name.lower() == "ivan"
    assert user.age == 27


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio
async def test_fireworks_partial_async(mode: instructor.Mode, model: str):
    class User(BaseModel):
        name: str
        age: int

    client = instructor.from_fireworks(AsyncFireworks(), mode=mode)

    resp = client.chat.completions.create_partial(
        model=model,
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that extracts users from a sentence. Make sure to adhere to the desired JSON output format of {User.model_json_schema()} and make sure that your output can be parsed by a JSON parser.",
            },
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
    )

    user = None
    async for r in resp:
        user = r

    assert user is not None
    assert user.name.lower() == "ivan"
    assert user.age == 27
