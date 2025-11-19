"""
LLM tests for v2 Anthropic provider.

These tests make actual API calls and require ANTHROPIC_API_KEY.
"""

import pytest
import instructor
from typing import Literal
from collections.abc import Iterable
from pydantic import BaseModel

# All tests in this module require API key
pytestmark = pytest.mark.requires_api_key


class Answer(BaseModel):
    """Simple answer model."""

    answer: float


class Weather(BaseModel):
    """Weather tool."""

    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    """Search tool."""

    query: str


# Test parallel tools mode (auto-detected from Iterable[Union[...]])
# This tests the auto-detection feature when using Mode.TOOLS with Iterable[Union[...]]
def test_v2_parallel_tools_auto_detection():
    """Test v2 TOOLS mode auto-detecting parallel from Iterable[Union[...]]."""
    client = instructor.from_provider(
        "anthropic/claude-3-5-haiku-latest",
        mode=instructor.Mode.TOOLS,
    )
    response = client.chat.completions.create(
        response_model=Iterable[Weather | GoogleSearch],
        messages=[
            {
                "role": "system",
                "content": "You must always use tools. Use them simultaneously when appropriate.",
            },
            {
                "role": "user",
                "content": "Get weather for Toronto and search for current events.",
            },
        ],
        max_tokens=1000,
    )

    result = list(response)
    assert len(result) >= 1
    assert all(isinstance(r, (Weather, GoogleSearch)) for r in result)


def test_v2_json_schema_mode_live():
    """Test JSON_SCHEMA mode using Claude Sonnet structured outputs."""
    client = instructor.from_provider(
        "anthropic/claude-3-7-sonnet-20250219",
        mode=instructor.Mode.JSON_SCHEMA,
    )
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "What is 9 + 1? Reply with a number.",
            },
        ],
        max_tokens=1000,
    )

    assert isinstance(response, Answer)
    assert response.answer == 10.0
