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
                "content": "Which is larger, 9.11 or 9.8? Think carefully about decimal places.",
            },
        ],
        temperature=0,  # Use temperature=0 for deterministic results
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        max_retries=3,  # Retry if the model gets it wrong
    )

    # Assertions to validate the response
    assert isinstance(response, Answer)
    assert response.answer == 9.8
