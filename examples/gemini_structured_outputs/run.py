"""
Example demonstrating Gemini's enhanced structured outputs with JSON Schema support.

This example showcases the new features announced by Google AI Studio:
- Expanded JSON Schema support (anyOf for unions, $ref for recursive schemas)
- Implicit property ordering preservation
- Support for Pydantic models out-of-the-box

Based on: https://x.com/googleaistudio/status/1986127034800914543
"""

import instructor
from pydantic import BaseModel, Field
from typing import Union, Literal


class SpamDetails(BaseModel):
    """Details for content classified as spam."""

    reason: str = Field(description="The reason why the content is considered spam.")
    spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"] = Field(
        description="The type of spam."
    )


class NotSpamDetails(BaseModel):
    """Details for content classified as not spam."""

    summary: str = Field(description="A brief summary of the content.")
    is_safe: bool = Field(description="Whether the content is safe for all audiences.")


class ModerationResult(BaseModel):
    """The result of content moderation."""

    decision: Union[SpamDetails, NotSpamDetails]


def main():
    client = instructor.from_provider(
        "google/gemini-2.0-flash-exp",
        mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
    )

    prompt = """
Please moderate the following content and provide a decision.
Content: 'Congratulations! You've won a free cruise to the Bahamas. Click here to claim your prize: www.definitely-not-a-scam.com'
"""

    result = client.chat.completions.create(
        model="gemini-2.0-flash-exp",
        response_model=ModerationResult,
        messages=[{"role": "user", "content": prompt}],
    )

    print("=== Moderation Result ===")
    print(f"Decision type: {type(result.decision).__name__}")

    if isinstance(result.decision, SpamDetails):
        print(f"Classification: SPAM")
        print(f"Reason: {result.decision.reason}")
        print(f"Spam Type: {result.decision.spam_type}")
    else:
        print(f"Classification: NOT SPAM")
        print(f"Summary: {result.decision.summary}")
        print(f"Is Safe: {result.decision.is_safe}")


if __name__ == "__main__":
    main()
