"""Async-validator response models must fail closed before provider execution."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import retry_async_v2, retry_sync_v2
from instructor.v2.validation import async_field_validator, async_model_validator


class Email(BaseModel):
    address: str

    @async_field_validator("address")
    async def must_contain_at(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("Invalid email address")
        return value.lower()


class Account(BaseModel):
    email: Email

    @async_model_validator()
    async def normalize(self) -> Account:
        return self


@pytest.mark.parametrize("response_model", [Email, Account, list[Email]])
def test_sync_retry_rejects_async_validators_before_provider(
    response_model: Any,
) -> None:
    def unexpected_request(**_kwargs: Any) -> None:
        raise AssertionError("Provider must not be called for unsupported validators")

    with pytest.raises(ValueError, match="async validators are not supported"):
        retry_sync_v2(
            func=unexpected_request,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={},
            strict=True,
            hooks=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("response_model", [Email, Account, list[Email]])
async def test_async_retry_rejects_async_validators_before_provider(
    response_model: Any,
) -> None:
    async def unexpected_request(**_kwargs: Any) -> None:
        raise AssertionError("Provider must not be called for unsupported validators")

    with pytest.raises(ValueError, match="async validators are not supported"):
        await retry_async_v2(
            func=unexpected_request,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={},
            strict=True,
            hooks=None,
        )
