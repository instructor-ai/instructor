from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.core.registry import mode_registry

genai = pytest.importorskip("google.genai")


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("mode", [Mode.TOOLS, Mode.JSON])
@pytest.mark.parametrize("config_object", [False, True])
@pytest.mark.parametrize("cached_content", ["cachedContents/example", None])
def test_cached_content_omits_cache_owned_config(
    mode: Mode, config_object: bool, cached_content: str | None
) -> None:
    config: Any = {"cached_content": cached_content, "labels": {"test": "cache"}}
    if config_object:
        config = genai.types.GenerateContentConfig(**config)
    kwargs: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "Extract the answer."},
            {"role": "user", "content": "The answer is 42."},
        ],
        "config": config,
        "temperature": 0.25,
    }
    original = deepcopy(kwargs)

    prepare = mode_registry.get_handlers(Provider.GENAI, mode).request_handler
    prepared_model, prepared = prepare(Answer, kwargs)
    prepared_config = prepared["config"]

    assert kwargs == original
    assert prepared_model is not None
    assert prepared_config.cached_content == cached_content
    assert prepared_config.temperature == 0.25
    assert prepared_config.labels == {"test": "cache"}
    assert prepared["contents"][0].parts[0].text == "The answer is 42."
    if cached_content is not None:
        assert prepared_config.system_instruction is None
        assert prepared_config.tools is None
        assert prepared_config.tool_config is None
    else:
        assert prepared_config.system_instruction == "Extract the answer.\n\n"
        if mode == Mode.TOOLS:
            assert prepared_config.tools[0].function_declarations[0].name == "Answer"
            assert (
                prepared_config.tool_config.function_calling_config.allowed_function_names
                == ["Answer"]
            )
    if mode == Mode.JSON:
        assert prepared_config.response_mime_type == "application/json"
        assert prepared_config.response_schema is prepared_model
