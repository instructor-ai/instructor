"""Integration tests for MD_YAML mode with OpenAI."""

import pytest
import instructor
from openai import OpenAI
from pydantic import BaseModel


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str
    age: int


class NestedData(BaseModel):
    """Nested data model for testing."""

    title: str
    items: list[str]
    metadata: dict[str, str]


@pytest.fixture
def client():
    """Create an OpenAI client with MD_YAML mode."""
    return instructor.from_openai(OpenAI(), mode=instructor.Mode.MD_YAML)


def test_simple_extraction(client):
    """Test simple data extraction with MD_YAML mode."""
    user = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Create a user named Alice who is 25 years old"}
        ],
        response_model=SimpleUser,
    )

    assert isinstance(user, SimpleUser)
    assert user.name == "Alice"
    assert user.age == 25


def test_nested_extraction(client):
    """Test nested data extraction with MD_YAML mode."""
    data = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Create project data titled 'ML Pipeline' with items: training, testing, deployment. Metadata: author=John, version=1.0",
            }
        ],
        response_model=NestedData,
    )

    assert isinstance(data, NestedData)
    assert data.title == "ML Pipeline"
    assert len(data.items) == 3
    assert "training" in data.items
    assert "author" in data.metadata
    assert "version" in data.metadata


def test_multiple_requests(client):
    """Test multiple sequential requests to ensure consistency."""
    users = []
    for i, name in enumerate(["Bob", "Carol", "Dave"], start=30):
        user = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Create a user named {name} who is {i} years old",
                }
            ],
            response_model=SimpleUser,
        )
        users.append(user)

    assert len(users) == 3
    assert users[0].name == "Bob"
    assert users[0].age == 30
    assert users[2].name == "Dave"
    assert users[2].age == 32
