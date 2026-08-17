"""Tests that async validator decorators actually run inside the retry loop.

Regression tests for issue #2528, where `@async_field_validator` and
`@async_model_validator` attached metadata to a model but nothing ever awaited
the validators.
"""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel, ValidationInfo

import instructor
from instructor.core.exceptions import InstructorRetryException
from instructor.mode import Mode
from instructor.validation import (
    AsyncValidationError,
    async_field_validator,
    async_model_validator,
)


def test_decorators_are_exported_from_the_package_root():
    assert instructor.async_field_validator is async_field_validator
    assert instructor.async_model_validator is async_model_validator


def _make_response(content: str) -> Mock:
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = content
    response.choices[0].finish_reason = "stop"
    response.usage = None
    return response


def _make_async_client(content: str):
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_response(content)
    )
    return instructor.patch(mock_client, mode=Mode.JSON)


def _make_sync_client(content: str):
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(return_value=_make_response(content))
    return instructor.patch(mock_client, mode=Mode.JSON)


@pytest.mark.asyncio
async def test_async_field_validator_runs_on_parsed_model():
    validated: list[str] = []

    class User(BaseModel):
        name: str

        @async_field_validator("name")
        async def uppercase_name(self, value: str) -> str:
            validated.append(value)
            return value

    client = _make_async_client('{"name": "JASON"}')

    user = await client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
        model="gpt-4o-mini",
        response_model=User,
        messages=[{"role": "user", "content": "test"}],
        max_retries=0,
    )

    assert user.name == "JASON"
    assert validated == ["JASON"]


@pytest.mark.asyncio
async def test_async_model_validator_failure_is_retried_and_reported():
    class User(BaseModel):
        name: str

        @async_model_validator()
        async def name_must_be_uppercase(self):
            if not self.name.isupper():
                raise ValueError("name must be uppercase")

    client = _make_async_client('{"name": "jason"}')

    with pytest.raises(InstructorRetryException) as exc_info:
        await client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=1,
        )

    exception = exc_info.value
    assert exception.n_attempts == 2
    assert exception.failed_attempts is not None
    assert len(exception.failed_attempts) == 2
    for attempt in exception.failed_attempts:
        assert isinstance(attempt.exception, AsyncValidationError)
        assert "name must be uppercase" in str(attempt.exception)


@pytest.mark.asyncio
async def test_async_validators_run_on_nested_models():
    validated: list[str] = []

    class Address(BaseModel):
        city: str

        @async_model_validator()
        async def city_must_be_known(self):
            validated.append(self.city)
            if self.city == "Atlantis":
                raise ValueError("unknown city")

    class User(BaseModel):
        name: str
        addresses: list[Address]

    client = _make_async_client(
        '{"name": "Jason", "addresses": [{"city": "Singapore"}, {"city": "Atlantis"}]}'
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        await client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=0,
        )

    assert validated == ["Singapore", "Atlantis"]
    assert "addresses.1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_validators_receive_validation_context():
    class User(BaseModel):
        name: str

        @async_field_validator("name")
        async def name_is_allowed(self, value: str, info: ValidationInfo) -> str:
            context = info.context or {}
            if value in context.get("forbidden_names", []):
                raise ValueError(f"{value} is forbidden")
            return value

    client = _make_async_client('{"name": "Jason"}')

    with pytest.raises(InstructorRetryException) as exc_info:
        await client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            context={"forbidden_names": ["Jason"]},
            max_retries=0,
        )

    assert "Jason is forbidden" in str(exc_info.value)


def test_sync_client_warns_that_async_validators_are_skipped(
    caplog: pytest.LogCaptureFixture,
):
    class User(BaseModel):
        name: str

        @async_model_validator()
        async def name_must_be_uppercase(self):
            raise ValueError("name must be uppercase")

    client = _make_sync_client('{"name": "jason"}')

    with caplog.at_level(logging.WARNING, logger="instructor.v2.retry"):
        user = client.chat.completions.create(  # ty: ignore[no-matching-overload] - runtime-patched API
            model="gpt-4o-mini",
            response_model=User,
            messages=[{"role": "user", "content": "test"}],
            max_retries=0,
        )

    assert user.name == "jason"
    assert any(
        "async validators" in record.message.lower() for record in caplog.records
    )
