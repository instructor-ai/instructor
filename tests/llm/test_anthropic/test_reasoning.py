import instructor
from pydantic import BaseModel


class Answer(BaseModel):
    answer: float


def test_reasoning():
    client = instructor.from_provider(
        "anthropic/claude-3-7-sonnet-latest",
        mode=instructor.Mode.ANTHROPIC_REASONING_TOOLS,
    )
    response = client.chat.completions.create(
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
