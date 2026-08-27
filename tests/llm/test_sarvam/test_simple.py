"""Basic Sarvam extraction tests."""

from __future__ import annotations

from itertools import product

import instructor
import pytest
from pydantic import BaseModel

from util import models, modes, name_matches

pytestmark = pytest.mark.sarvam


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("model_name, mode", product(models, modes))
def test_sarvam_simple_extraction(
    model_name: str,
    mode: instructor.Mode,
    sarvam_api_key: str,
) -> None:
    client = instructor.from_provider(
        f"sarvam/{model_name}",
        mode=mode,
        api_key=sarvam_api_key,
    )

    user = client.chat.completions.create(
        model=model_name,
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "निकालें: अमित की उम्र 32 साल है।",
            }
        ],
        reasoning_effort="medium",
        max_retries=3,
    )

    assert user.age == 32
    assert name_matches(user.name, ("Amit", "अमित")), (
        f"Expected Amit, got {user.name!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name, mode", product(models, modes))
async def test_sarvam_async_extraction(
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
        response_model=User,
        messages=[
            {
                "role": "user",
                "content": "निकालें: अमित की उम्र 32 साल है।",
            }
        ],
        reasoning_effort="medium",
    )

    assert user.age == 32
    assert user.name  # romanized or native script
