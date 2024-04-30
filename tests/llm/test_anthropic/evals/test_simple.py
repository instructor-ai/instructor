from enum import Enum
from typing import Literal

import anthropic
import pytest
from pydantic import BaseModel, field_validator

import instructor
from instructor.retry import InstructorRetryException

client = instructor.from_anthropic(
    anthropic.Anthropic(), mode=instructor.Mode.ANTHROPIC_TOOLS
)


def test_simple():
    class User(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_is_uppercase(cls, v: str):
            assert v.isupper(), "Name must be uppercase, please fix"
            return v

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old.",
            }
        ],
        response_model=User,
    )  # type: ignore

    assert isinstance(resp, User)
    assert resp.name == "JOHN"  # due to validation
    assert resp.age == 18


def test_nested_type():
    class Address(BaseModel):
        house_number: int
        street_name: str

    class User(BaseModel):
        name: str
        age: int
        address: Address

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Extract John is 18 years old and lives at 123 First Avenue.",
            }
        ],
        response_model=User,
    )  # type: ignore

    assert isinstance(resp, User)
    assert resp.name == "John"
    assert resp.age == 18

    assert isinstance(resp.address, Address)
    assert resp.address.house_number == 123
    assert resp.address.street_name == "First Avenue"


def test_list_str():
    class User(BaseModel):
        name: str
        age: int
        family: list[str]

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and family members.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert isinstance(resp.family, list)
    for member in resp.family:
        assert isinstance(member, str)


@pytest.mark.skip("Just use Literal!")
def test_enum():
    class Role(str, Enum):
        ADMIN = "admin"
        USER = "user"

    class User(BaseModel):
        name: str
        role: Role

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=1,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name and role of admin.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.role == Role.ADMIN


def test_literal():
    class User(BaseModel):
        name: str
        role: Literal["admin", "user"]

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": "Create a admin user for a model with a name and role.",
            }
        ],
        response_model=User,
    )  # type: ignore

    assert isinstance(resp, User)
    assert resp.role == "admin"


def test_nested_list():
    class Properties(BaseModel):
        key: str
        value: str

    class User(BaseModel):
        name: str
        age: int
        properties: list[Properties]

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            }
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    for property in resp.properties:
        assert isinstance(property, Properties)


def test_system_messages_allcaps():
    class User(BaseModel):
        name: str
        age: int

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        max_retries=0,
        messages=[
            {"role": "system", "content": "EVERYTHING MUST BE IN ALL CAPS"},
            {
                "role": "user",
                "content": "Create a user for a model with a name and age.",
            },
        ],
        response_model=User,
    )

    assert isinstance(resp, User)
    assert resp.name.isupper()


def test_retry_error():
    class User(BaseModel):
        name: str

        @field_validator("name")
        def validate_name(cls, _):
            raise ValueError("Never succeed")

    try:
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": "Extract John is 18 years old",
                },
            ],
            response_model=User,
        )
    except InstructorRetryException as e:
        assert e.total_usage.input_tokens > 0 and e.total_usage.output_tokens > 0


@pytest.mark.asyncio
async def test_async_retry_error():
    client = instructor.from_anthropic(anthropic.AsyncAnthropic())

    class User(BaseModel):
        name: str

        @field_validator("name")
        def validate_name(cls, _):
            raise ValueError("Never succeed")

    try:
        await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            max_retries=2,
            messages=[
                {
                    "role": "user",
                    "content": "Extract John is 18 years old",
                },
            ],
            response_model=User,
        )
    except InstructorRetryException as e:
        assert e.total_usage.input_tokens > 0 and e.total_usage.output_tokens > 0
