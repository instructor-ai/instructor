from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from instructor.core.client import two_step, two_step_async


class ReasoningModel(BaseModel):
    reasoning: str


class Answer(BaseModel):
    final_answer: str


def test_two_step_returns_correct_types():
    mock_client = MagicMock()
    mock_client.create.side_effect = [
        ReasoningModel(reasoning="Rayleigh scattering causes it."),
        Answer(final_answer="The sky is blue due to Rayleigh scattering."),
    ]

    answer, reasoning = two_step(
        client=mock_client,
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        think_model=ReasoningModel,
        final_model=Answer,
        think_max_tokens=256,
        final_max_tokens=128,
    )

    assert isinstance(answer, Answer)
    assert isinstance(reasoning, ReasoningModel)
    assert mock_client.create.call_count == 2


def test_two_step_rejects_system_messages():
    mock_client = MagicMock()

    with pytest.raises(ValueError, match="System messages"):
        two_step(
            client=mock_client,
            messages=[
                {"role": "system", "content": "You are a physics expert."},
                {"role": "user", "content": "Why is the sky blue?"},
            ],
            think_model=ReasoningModel,
            final_model=Answer,
            think_max_tokens=256,
            final_max_tokens=128,
        )

    assert mock_client.create.call_count == 0


@pytest.mark.asyncio
async def test_two_step_async_returns_correct_types():
    mock_client = MagicMock()
    mock_client.create = AsyncMock(
        side_effect=[
            ReasoningModel(reasoning="Rayleigh scattering causes it."),
            Answer(final_answer="The sky is blue due to Rayleigh scattering."),
        ]
    )

    answer, reasoning = await two_step_async(
        client=mock_client,
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        think_model=ReasoningModel,
        final_model=Answer,
        think_max_tokens=256,
        final_max_tokens=128,
    )

    assert isinstance(answer, Answer)
    assert isinstance(reasoning, ReasoningModel)
    assert mock_client.create.call_count == 2


@pytest.mark.asyncio
async def test_two_step_async_rejects_system_messages():
    mock_client = MagicMock()
    mock_client.create = AsyncMock()

    with pytest.raises(ValueError, match="System messages"):
        await two_step_async(
            client=mock_client,
            messages=[
                {"role": "system", "content": "You are a physics expert."},
                {"role": "user", "content": "Why is the sky blue?"},
            ],
            think_model=ReasoningModel,
            final_model=Answer,
            think_max_tokens=256,
            final_max_tokens=128,
        )

    assert mock_client.create.call_count == 0
