"""Sarvam streaming tests."""

from __future__ import annotations

from itertools import product

import instructor
import pytest
from pydantic import BaseModel

from instructor.dsl.partial import Partial
from util import models, modes

pytestmark = pytest.mark.sarvam


class User(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name, mode", product(models, modes))
async def test_sarvam_partial_streaming(
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

    updates = []
    async for partial_user in await client.chat.completions.create(
        model=model_name,
        response_model=Partial[User],
        messages=[{"role": "user", "content": "Rahul is 30 years old"}],
        stream=True,
        reasoning_effort="low",
    ):
        updates.append(partial_user)

    assert len(updates) >= 1
    final = updates[-1]
    assert final.age == 30
