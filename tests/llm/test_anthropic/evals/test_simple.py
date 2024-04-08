import anthropic
import instructor
from pydantic import BaseModel, field_validator
from typing import List, Literal
from enum import Enum

client = instructor.from_anthropic(
    anthropic.Anthropic(), mode=instructor.Mode.ANTHROPIC_TOOLS
)


def test_simple():
    class User(BaseModel):
        name: str
        age: int

        @field_validator("name")
        def name_is_uppercase(cls, v: str):
            assert v.isupper(), "Name must be uppercase"
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
        family: List[str]

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
    assert isinstance(resp.family, List)
    for member in resp.family:
        assert isinstance(member, str)


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
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": "Create a user for a model with a name and role of admin.",
            }
        ],
        response_model=User,
    )  # type: ignore

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
        properties: List[Properties]

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
    )  # type: ignore

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
    )  # type: ignore

    assert isinstance(resp, User)
    assert resp.name.isupper()
