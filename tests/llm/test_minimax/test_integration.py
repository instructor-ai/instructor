"""Integration tests for MiniMax provider.

These tests require a valid MINIMAX_API_KEY environment variable.
"""

from __future__ import annotations

import os

import pytest
import openai
from pydantic import BaseModel, Field

import instructor
from instructor.providers.minimax.client import from_minimax

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")


class UserExtract(BaseModel):
    """Model for extracting user information from text."""

    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")


class Sentiment(BaseModel):
    """Model for sentiment analysis."""

    label: str = Field(description="Sentiment label: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMiniMaxToolsIntegration:
    """Integration tests for MiniMax TOOLS mode."""

    def _make_client(self) -> instructor.Instructor:
        return from_minimax(
            openai.OpenAI(
                api_key=MINIMAX_API_KEY,
                base_url="https://api.minimax.io/v1",
            ),
            mode=instructor.Mode.MINIMAX_TOOLS,
        )

    def test_basic_extraction(self):
        client = self._make_client()
        user = client.chat.completions.create(
            model="MiniMax-M2.5",
            response_model=UserExtract,
            messages=[
                {"role": "user", "content": "Extract: Jason is 25 years old"},
            ],
            temperature=1.0,
        )
        assert isinstance(user, UserExtract)
        assert user.name.lower() == "jason"
        assert user.age == 25

    def test_sentiment_analysis(self):
        client = self._make_client()
        result = client.chat.completions.create(
            model="MiniMax-M2.5",
            response_model=Sentiment,
            messages=[
                {"role": "user", "content": "Analyze sentiment: I love this product!"},
            ],
            temperature=1.0,
        )
        assert isinstance(result, Sentiment)
        assert result.label.lower() == "positive"
        assert 0 <= result.confidence <= 1


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMiniMaxJSONIntegration:
    """Integration tests for MiniMax JSON mode."""

    def _make_client(self) -> instructor.Instructor:
        return from_minimax(
            openai.OpenAI(
                api_key=MINIMAX_API_KEY,
                base_url="https://api.minimax.io/v1",
            ),
            mode=instructor.Mode.MINIMAX_JSON,
        )

    def test_basic_extraction(self):
        client = self._make_client()
        user = client.chat.completions.create(
            model="MiniMax-M2.5",
            response_model=UserExtract,
            messages=[
                {"role": "user", "content": "Extract: Alice is 30 years old"},
            ],
            temperature=1.0,
        )
        assert isinstance(user, UserExtract)
        assert user.name.lower() == "alice"
        assert user.age == 30
