import anthropic
import pytest
import instructor
from pydantic import BaseModel
from itertools import product
from .util import models, modes
from anthropic.types.message import Message


class Answer(BaseModel):
    answer: float


def test_reasoning():
    anthropic_client = anthropic.Anthropic()
    client = instructor.from_anthropic(
        anthropic_client, mode=instructor.Mode.ANTHROPIC_JSON
    )
    response = client.chat.completions.create(
        model="claude-3-7-sonnet-latest",
        response_model=Answer,
        messages=[
            {
                "role": "user",
                "content": "Which is larger, 9.11 or 9.8",
            },
        ],
        temperature=1,
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    # Assertions to validate the response
    assert isinstance(response, Answer)
    assert response.answer == 9.8
