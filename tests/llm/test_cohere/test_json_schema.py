import pytest
from pydantic import BaseModel, Field, field_validator
import instructor
from instructor.mode import Mode


modes = [Mode.COHERE_JSON_SCHEMA, Mode.COHERE_TOOLS]


class User(BaseModel):
    name: str = Field(description="User's first name")
    age: int


@pytest.mark.parametrize("mode", modes)
def test_parse_user_sync(client_v2, mode):
    client = instructor.from_cohere(client_v2, mode=mode)

    resp = client.chat.completions.create(
        response_model=User,
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
    )

    assert resp.name == "Ivan"
    assert resp.age == 27


@pytest.mark.parametrize("mode", modes)
def test_parse_user_sync_jinja(client_v1, mode):
    client = instructor.from_cohere(client_v1, mode=mode, model="command-a-03-2025")

    resp = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - {{ name }} is a {{ age }} year old developer from Singapore",
            }
        ],
        context={"name": "Ivan", "age": 27},
    )

    assert resp.name == "Ivan"
    assert resp.age == 27


class ValidatedUser(BaseModel):
    name: str = Field(description="User's first name")
    age: int

    @field_validator("name")
    @classmethod
    def ensure_uppercase(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError(
                f"{v} should have all its characters uppercased (Eg. TOM, JACK, JEFFREY)"
            )
        return v


@pytest.mark.parametrize("mode", modes)
def test_parse_validated_user_sync(client_v1, mode):
    client = instructor.from_cohere(client_v1, mode=mode)

    resp = client.chat.completions.create(
        response_model=ValidatedUser,
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
        max_retries=3,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
async def test_parse_user_async(aclient_v2, mode):
    client = instructor.from_cohere(aclient_v2, mode=mode)

    resp = await client.chat.completions.create(
        response_model=ValidatedUser,
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - Ivan is a 27 year old developer from Singapore",
            }
        ],
        max_retries=4,
    )

    assert resp.name == "IVAN"
    assert resp.age == 27


@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
async def test_parse_user_async_jinja(aclient_v2, mode):
    client = instructor.from_cohere(aclient_v2, mode=mode)

    resp = await client.chat.completions.create(
        response_model=ValidatedUser,
        model="command-a-03-2025",
        messages=[
            {
                "role": "user",
                "content": "Extract user data from this sentence - {{ name }} is a {{ age }} year old developer from Singapore",
            }
        ],
        max_retries=4,
        context={"name": "Ivan", "age": 27},
    )  # type: ignore

    assert resp.name == "IVAN"
    assert resp.age == 27
