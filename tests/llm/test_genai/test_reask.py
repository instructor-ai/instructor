import os
import pytest
from pydantic import BaseModel, field_validator
import instructor


@pytest.mark.parametrize("mode", [instructor.Mode.GENAI_TOOLS])
def test_genai_tools_validation_retry(client, mode):
    """Test that validation retries work correctly with GENAI_TOOLS mode.

    This tests the fix for thought_signature preservation in reask_genai_tools.
    """
    model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.0-flash")

    class PositiveNumber(BaseModel):
        value: int

        @field_validator("value")
        @classmethod
        def must_be_positive(cls, v: int) -> int:
            if v <= 0:
                raise ValueError("Value must be positive")
            return v

    client = instructor.from_provider(f"google/{model}", mode=mode)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Return the number 42",
            }
        ],
        response_model=PositiveNumber,
        max_retries=2,
    )

    assert isinstance(response, PositiveNumber)
    assert response.value > 0
