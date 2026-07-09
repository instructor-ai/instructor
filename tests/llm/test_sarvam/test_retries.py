"""Sarvam retry and validation tests."""

from __future__ import annotations

from itertools import product

import instructor
import pytest
from pydantic import BaseModel, Field, field_validator

from util import models, modes

pytestmark = pytest.mark.sarvam


class ValidatedUser(BaseModel):
    name: str
    age: int = Field(ge=0, le=120)

    @field_validator("name")
    @classmethod
    def name_must_have_content(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Name must not be empty")
        return value.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name, mode", product(models, modes))
async def test_sarvam_max_retries(
    model_name: str,
    mode: instructor.Mode,
    sarvam_api_key: str,
) -> None:
    client = instructor.from_provider(
        f"sarvam/{model_name}",
        mode=mode,
        api_key=sarvam_api_key,
        async_client=True,
    )

    user = await client.chat.completions.create(
        model=model_name,
        response_model=ValidatedUser,
        messages=[{"role": "user", "content": "Extract: Priya is 33 years old."}],
        max_retries=3,
        reasoning_effort="medium",
    )

    assert isinstance(user, ValidatedUser)
    assert user.age == 33
