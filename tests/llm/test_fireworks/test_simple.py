import instructor
from fireworks.client import Fireworks, AsyncFireworks
from pydantic import BaseModel, field_validator
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
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
def test_fireworks_sync_validated(mode: instructor.Mode, model: str):
    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_validator(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All letters in the name must be uppercase (Eg. JOHN, SMITH) - {v} is not a valid example."
                )
            return v

    client = instructor.from_fireworks(Fireworks(), mode=mode)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        max_retries=5,
        response_model=ValidatedUser,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio(scope="session")
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
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("mode, model", modes)
@pytest.mark.asyncio(scope="session")
async def test_fireworks_async_validated(mode: instructor.Mode, model: str):
    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_validator(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"Make sure to uppercase all letters in the name field. Examples include: JOHN, SMITH, etc. {v} is not a valid example of a name that has all its letters uppercased"
                )
            return v

    client = instructor.from_fireworks(AsyncFireworks(), mode=mode)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence : Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=ValidatedUser,
        max_retries=5,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27
