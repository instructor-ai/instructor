"""
Example: Using Instructor with Moonshot AI's kimi-k2-thinking model

This example demonstrates how to use instructor with Moonshot AI's kimi-k2-thinking model,
which doesn't support tool calling. Instructor automatically falls back to MD_JSON mode.

Issue #1925: https://github.com/instructor-ai/instructor/issues/1925
"""

import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


class UserDetail(BaseModel):
    """User details extracted from text."""

    name: str = Field(description="The name of the user.")
    age: int = Field(description="The age of the user.")


def main():
    """
    Extract user details using Moonshot AI's kimi-k2-thinking model.

    Note: This example requires a Moonshot API key set in MOONSHOT_API_KEY environment variable.
    """
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("⚠️  MOONSHOT_API_KEY not set. This example requires a valid API key.")
        print("   Set it with: export MOONSHOT_API_KEY='your-api-key'")
        return

    # Option 1: Let instructor auto-detect and fallback to JSON mode
    print("Option 1: Auto-fallback (Mode.TOOLS → Mode.MD_JSON)")
    print("-" * 60)

    client = instructor.patch(
        OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        ),
        mode=instructor.Mode.TOOLS,  # Will auto-fallback to MD_JSON with warning
    )

    try:
        user = client.chat.completions.create(
            model="kimi-k2-thinking",
            response_model=UserDetail,
            messages=[
                {
                    "role": "user",
                    "content": "Extract user details from this sentence: Jason is 25 years old.",
                }
            ],
        )
        print(f"✅ Successfully extracted user:")
        print(f"   Name: {user.name}")
        print(f"   Age: {user.age}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print()

    # Option 2: Explicitly use JSON mode (recommended, no warning)
    print("Option 2: Explicit MD_JSON mode (recommended)")
    print("-" * 60)

    client_json = instructor.patch(
        OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        ),
        mode=instructor.Mode.MD_JSON,  # Explicit mode, no warning
    )

    try:
        user = client_json.chat.completions.create(
            model="kimi-k2-thinking",
            response_model=UserDetail,
            messages=[
                {
                    "role": "user",
                    "content": "Extract user details from this sentence: Sarah is 30 years old.",
                }
            ],
        )
        print(f"✅ Successfully extracted user:")
        print(f"   Name: {user.name}")
        print(f"   Age: {user.age}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
