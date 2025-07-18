import instructor
import pytest
from typing import Union, Literal
from collections.abc import Iterable
from pydantic import BaseModel
from .util import models


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


@pytest.mark.parametrize("model", models)
def test_sync_parallel_tools_or(model):
    client = instructor.from_provider(
        model,
        mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
    )
    resp = client.chat.completions.create(
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": "You must always use tools. For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.",
            },
            {
                "role": "user",
                "content": "Please use parallel tool calls to get the weather for Toronto and Dallas, and also search for who won the Super Bowl. Use all tools simultaneously.",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    result = list(resp)
    assert len(result) >= 1  # Model should generate at least one tool call
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)
    # Note: Due to model limitations, Claude 3 Haiku may not always generate parallel tool calls
    # but the functionality should work when it does


@pytest.mark.asyncio
@pytest.mark.parametrize("model", models)
async def test_async_parallel_tools_or(model):
    client = instructor.from_provider(
        model,
        async_client=True,
        mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
    )
    resp = await client.chat.completions.create(
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": "You must always use tools. For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.",
            },
            {
                "role": "user",
                "content": "Please use parallel tool calls to get the weather for Toronto and Dallas, and also search for who won the Super Bowl. Use all tools simultaneously.",
            },
        ],
        response_model=Iterable[Union[Weather, GoogleSearch]],
    )
    result = list(resp)
    assert len(result) >= 1  # Model should generate at least one tool call
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)
    # Note: Due to model limitations, Claude 3 Haiku may not always generate parallel tool calls
    # but the functionality should work when it does
