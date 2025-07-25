import pytest
import os
import instructor
from pydantic import BaseModel, field_validator
from itertools import product

from .util import models, modes


class User(BaseModel):
    name: str
    age: int


class UserValidated(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError(
                "Name should have all letters in uppercase. Make sure to use the `uppercase` form of the name"
            )
        return v


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_raw_response_sync(model, mode):
    """Test that _raw_response is attached to sync XAI responses"""
    client = instructor.from_provider(f"xai/{model}", mode=mode)

    user = client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old. Make sure the name is in UPPERCASE format.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name.lower() == "jason"
    assert user.age == 25
    assert hasattr(user, "_raw_response"), (
        "The raw response should be available from XAI"
    )
    assert user._raw_response is not None


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_raw_response_async(model, mode):
    """Test that _raw_response is attached to async XAI responses"""
    client = instructor.from_provider(f"xai/{model}", mode=mode, async_client=True)

    user = await client.chat.completions.create(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old. Make sure the name is in UPPERCASE format.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name.lower() == "jason"
    assert user.age == 25
    assert hasattr(user, "_raw_response"), (
        "The raw response should be available from XAI"
    )
    assert user._raw_response is not None


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_raw_response_with_validator_sync(model, mode):
    """Test that _raw_response works with validated models in sync mode"""
    client = instructor.from_provider(f"xai/{model}", mode=mode)

    user = client.chat.completions.create(
        response_model=UserValidated,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old. Make sure the name is in UPPERCASE format.",
            },
        ],
    )

    assert isinstance(user, UserValidated)
    assert user.name == "JASON"
    assert user.age == 25
    assert hasattr(user, "_raw_response"), (
        "The raw response should be available from XAI"
    )
    assert user._raw_response is not None


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_raw_response_with_validator_async(model, mode):
    """Test that _raw_response works with validated models in async mode"""
    client = instructor.from_provider(f"xai/{model}", mode=mode, async_client=True)

    user = await client.chat.completions.create(
        response_model=UserValidated,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old. Make sure the name is in UPPERCASE format.",
            },
        ],
    )

    assert isinstance(user, UserValidated)
    assert user.name == "JASON"
    assert user.age == 25
    assert hasattr(user, "_raw_response"), (
        "The raw response should be available from XAI"
    )
    assert user._raw_response is not None


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_xai_create_with_completion():
    """Test that create_with_completion works with XAI provider"""
    client = instructor.from_provider("xai/grok-3-mini", mode=instructor.Mode.XAI_JSON)

    user, raw_response = client.chat.completions.create_with_completion(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old. Make sure the name is in UPPERCASE format.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name.lower() == "jason"
    assert user.age == 25
    assert hasattr(user, "_raw_response"), (
        "The raw response should be available from XAI"
    )
    assert raw_response is not None
    assert user._raw_response == raw_response


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_xai_create_with_completion_async():
    """Test that create_with_completion works with XAI provider in async mode"""
    client = instructor.from_provider(
        "xai/grok-3-mini", mode=instructor.Mode.XAI_JSON, async_client=True
    )

    user, raw_response = await client.chat.completions.create_with_completion(
        response_model=User,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information.",
            },
            {
                "role": "user",
                "content": "Extract: Jason is 25 years old. Make sure the name is in UPPERCASE format.",
            },
        ],
    )

    assert isinstance(user, User)
    assert user.name.lower() == "jason"
    assert user.age == 25
    assert hasattr(user, "_raw_response"), (
        "The raw response should be available from XAI"
    )
    assert raw_response is not None
    assert user._raw_response == raw_response
