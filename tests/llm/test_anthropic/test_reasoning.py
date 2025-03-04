import anthropic
import pytest
import instructor
from pydantic import BaseModel


class Answer(BaseModel):
    answer: float


modes = [
    instructor.Mode.ANTHROPIC_REASONING_TOOLS,
    instructor.Mode.ANTHROPIC_JSON,
]


@pytest.mark.parametrize("mode", modes)
def test_reasoning(mode):
    anthropic_client = anthropic.Anthropic()
    client = instructor.from_anthropic(anthropic_client, mode=mode)
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
