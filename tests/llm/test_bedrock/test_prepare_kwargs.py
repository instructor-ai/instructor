from __future__ import annotations
from instructor.providers.bedrock.utils import _prepare_bedrock_converse_kwargs_internal


def test_prepare_bedrock_kwargs_openai_text_plus_image(image_url: str):
    call_kwargs = {
        "model": "anthropic.claude-3-5-sonnet",
        "temperature": 0.3,
        "max_tokens": 256,
        "top_p": 0.9,
        "stop": ["<END>"],
        "system": [{"text": "You are helpful."}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }

    out = _prepare_bedrock_converse_kwargs_internal(call_kwargs)

    assert out["modelId"] == "anthropic.claude-3-5-sonnet"
    inf = out["inferenceConfig"]
    assert inf["temperature"] == 0.3
    assert inf["maxTokens"] == 256
    assert inf["topP"] == 0.9
    assert inf["stopSequences"] == ["<END>"]
    assert out["system"][0]["text"] == "You are helpful."

    parts = out["messages"][0]["content"]
    assert parts[0] == {"text": "hi"}
    assert parts[1]["image"]["format"] in {"jpeg", "png", "gif", "webp"}
    assert isinstance(parts[1]["image"]["source"]["bytes"], (bytes, bytearray))
    assert len(parts[1]["image"]["source"]["bytes"]) > 0
