# Gemini Enhanced Structured Outputs

This example demonstrates Gemini's enhanced structured outputs with expanded JSON Schema support, announced by Google AI Studio in November 2025.

## New Features

Google AI Studio announced several enhancements to structured outputs in the Gemini API:

1. **Expanded JSON Schema Support**: Full support for JSON Schema keywords including:
   - `anyOf` for conditional structures (Union types)
   - `$ref` for recursive schemas
   - `minimum` and `maximum` for numeric constraints
   - `additionalProperties` and `type: 'null'`
   - `prefixItems` for tuple-like arrays

2. **Implicit Property Ordering**: The API now preserves the same order as the ordering of keys in the schema (supported for Gemini 2.5+ models)

3. **Pydantic/Zod Integration**: Libraries like Pydantic (Python) and Zod (JavaScript/TypeScript) now work out-of-the-box with the Gemini API

## Example: Content Moderation with Union Types

This example demonstrates using Union types (which generate `anyOf` in JSON Schema) for content moderation:

```python
import instructor
from pydantic import BaseModel, Field
from typing import Union, Literal

class SpamDetails(BaseModel):
    """Details for content classified as spam."""
    reason: str = Field(description="The reason why the content is considered spam.")
    spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"]

class NotSpamDetails(BaseModel):
    """Details for content classified as not spam."""
    summary: str = Field(description="A brief summary of the content.")
    is_safe: bool = Field(description="Whether the content is safe for all audiences.")

class ModerationResult(BaseModel):
    """The result of content moderation."""
    decision: Union[SpamDetails, NotSpamDetails]

# Create client with GENAI_STRUCTURED_OUTPUTS mode
client = instructor.from_provider(
    "google/gemini-2.0-flash-exp",
    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
)

result = client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    response_model=ModerationResult,
    messages=[{"role": "user", "content": "Moderate this content..."}],
)
```

## Running the Example

1. Set your Google API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

2. Install instructor with Google GenAI support:
```bash
pip install "instructor[google-genai]"
```

3. Run the example:
```bash
python run.py
```

## How It Works

Instructor automatically:
1. Converts your Pydantic model to JSON Schema (with `anyOf`, `$ref`, etc.)
2. Passes the schema to Gemini using the `response_json_schema` parameter
3. Validates the response against your Pydantic model
4. Returns a fully typed, validated Python object

## Reference

- [Google AI Studio Announcement](https://x.com/googleaistudio/status/1986127034800914543)
- [Gemini API Structured Outputs Documentation](https://ai.google.dev/gemini-api/docs/structured-output)
