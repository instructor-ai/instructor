import instructor
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, field_validator
import pytest
from itertools import product
from .util import models, modes


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_pinstripes_sync(model: str, mode: instructor.Mode, client: OpenAI):
    """Test basic sync structured extraction with Pinstripes."""
    instructor_client = instructor.from_pinstripes(client, mode=mode)

    resp = instructor_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_pinstripes_sync_validated(model: str, mode: instructor.Mode, client: OpenAI):
    """Test sync structured extraction with validation retries."""

    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_must_be_uppercase(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"All letters in the name must be uppercase (e.g. JOHN, SMITH). "
                    f"'{v}' is not a valid example."
                )
            return v

    instructor_client = instructor.from_pinstripes(client, mode=mode)

    resp = instructor_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        max_retries=5,
        response_model=ValidatedUser,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio(scope="session")
async def test_pinstripes_async(
    model: str, mode: instructor.Mode, aclient: AsyncOpenAI
):
    """Test basic async structured extraction with Pinstripes."""
    instructor_client = instructor.from_pinstripes(aclient, mode=mode)

    resp = await instructor_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=User,
    )

    assert resp.name.lower() == "ivan"
    assert resp.age == 27


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio(scope="session")
async def test_pinstripes_async_validated(
    model: str, mode: instructor.Mode, aclient: AsyncOpenAI
):
    """Test async structured extraction with validation retries."""

    class ValidatedUser(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_must_be_uppercase(cls, v: str) -> str:
            if not v.isupper():
                raise ValueError(
                    f"Make sure to uppercase all letters in the name field. "
                    f"Examples: JOHN, SMITH. '{v}' is not valid."
                )
            return v

    instructor_client = instructor.from_pinstripes(aclient, mode=mode)

    resp = await instructor_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract a user from this sentence: Ivan is 27 and lives in Singapore",
            },
        ],
        response_model=ValidatedUser,
        max_retries=5,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27
