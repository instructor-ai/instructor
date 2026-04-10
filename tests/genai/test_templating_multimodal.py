"""Regression test for #2253: templating crashes on genai image parts.

When a genai Content message contains non-text Parts (images, URIs, bytes),
the `text` attribute exists but is None. process_message must skip
templating for these parts instead of passing None to Jinja2.
"""

from google.genai import types

from instructor.mode import Mode
from instructor.templating import handle_templating, process_message


def test_process_message_skips_non_text_genai_parts():
    """Image/URI parts with text=None must pass through untemplated."""
    msg = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="Describe {{ subject }}"),
            types.Part.from_uri(
                file_uri="gs://bucket/image.jpg", mime_type="image/jpeg"
            ),
        ],
    )

    result = process_message(msg, {"subject": "this image"}, Mode.GENAI_TOOLS)

    assert result.parts[0].text == "Describe this image"
    # The URI part should pass through unchanged
    assert result.parts[1].file_data is not None
    assert result.parts[1].file_data.file_uri == "gs://bucket/image.jpg"


def test_process_message_handles_bytes_part():
    """Parts created from raw bytes also have text=None."""
    msg = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="Analyze this"),
            types.Part.from_bytes(data=b"\x89PNG", mime_type="image/png"),
        ],
    )

    result = process_message(msg, {"key": "val"}, Mode.GENAI_STRUCTURED_OUTPUTS)

    assert result.parts[0].text == "Analyze this"
    assert result.parts[1].inline_data is not None


def test_handle_templating_with_multimodal_genai_contents():
    """End-to-end: handle_templating must not crash on multimodal genai messages."""
    kwargs = {
        "messages": [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="What is {{ item }}?"),
                    types.Part.from_uri(
                        file_uri="gs://b/img.jpg", mime_type="image/jpeg"
                    ),
                ],
            )
        ]
    }

    result = handle_templating(kwargs, Mode.GENAI_TOOLS, context={"item": "this"})
    processed = result["messages"][0]
    assert processed.parts[0].text == "What is this?"


def test_handle_templating_no_context_passthrough():
    """Without context, messages should pass through unchanged."""
    msg = types.Content(
        role="user",
        parts=[types.Part.from_text(text="{{ not_a_template }}")],
    )
    kwargs = {"messages": [msg]}

    result = handle_templating(kwargs, Mode.GENAI_TOOLS, context=None)
    assert result["messages"][0].parts[0].text == "{{ not_a_template }}"
