from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor import Mode
from instructor.v2.providers.genai.handlers import (
    GenAIToolsHandler,
    GenAIStructuredOutputsHandler,
    reask_genai_structured_outputs,
    reask_genai_tools,
)
from tests.v2._fake_genai import FakeContent, FakePart, install_fake_genai


class FakeModelContent(FakeContent):
    def __init__(self, parts: list[FakePart], role: str = "model") -> None:
        super().__init__(role=role, parts=parts)


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class CachedResponse(BaseModel):
    value: str


def _install_fake_genai_types(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_genai(
        monkeypatch,
        extra_types={
            "ModelContent": FakeModelContent,
            "GenerateContentConfig": FakeGenerateContentConfig,
            "FunctionDeclaration": SimpleNamespace,
            "Tool": SimpleNamespace,
            "FunctionCallingConfigMode": SimpleNamespace(ANY="ANY"),
            "FunctionCallingConfig": SimpleNamespace,
            "ToolConfig": SimpleNamespace,
        },
    )


def test_reask_genai_tools_without_function_call_appends_user_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    existing_content = FakeContent(role="model", parts=[FakePart(text="hi")])
    response = SimpleNamespace(candidates=[SimpleNamespace(content=existing_content)])

    result = reask_genai_tools({"contents": []}, response, ValueError("bad schema"))

    assert result["contents"][0] is existing_content
    assert result["contents"][1].role == "user"
    assert "Validation Error found" in result["contents"][1].parts[0].text


def test_reask_genai_tools_with_function_call_appends_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    function_call = SimpleNamespace(name="Answer")
    content = FakeContent(role="model", parts=[FakePart(function_call=function_call)])
    response = SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    result = reask_genai_tools({"contents": []}, response, ValueError("bad schema"))

    assert result["contents"][0] is content
    assert result["contents"][1].role == "tool"
    function_response = result["contents"][1].parts[0].function_response
    assert function_response["name"] == "Answer"
    assert "Validation Error found" in function_response["response"]["error"]


def test_reask_genai_structured_outputs_appends_model_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    kwargs = {"contents": []}
    response = SimpleNamespace(text='{"bad": true}')

    result = reask_genai_structured_outputs(kwargs, response, ValueError("bad json"))

    assert isinstance(result["contents"][-1], FakeModelContent)
    assert "bad json" in result["contents"][-1].parts[0].text
    assert '{"bad": true}' in result["contents"][-1].parts[0].text


def test_tools_handler_prepare_request_without_response_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai_types(monkeypatch)
    monkeypatch.setattr(
        "instructor.v2.providers.gemini.utils.convert_to_genai_messages",
        lambda messages: ["converted", *messages],
    )
    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.extract_genai_multimodal_content",
        lambda contents, autodetect_images: [*contents, autodetect_images],
    )

    handler = GenAIToolsHandler(mode=Mode.TOOLS)
    model, kwargs = handler.prepare_request(
        None,
        {
            "messages": [{"role": "user", "content": "hello"}],
            "system": "system note",
            "autodetect_images": True,
            "temperature": 0.2,
        },
    )

    assert model is None
    assert kwargs["contents"] == [
        "converted",
        {"role": "user", "content": "hello"},
        True,
    ]
    assert kwargs["config"].kwargs["system_instruction"] == "system note"
    assert "temperature" not in kwargs


@pytest.mark.parametrize(
    ("handler", "forbidden_fields"),
    [
        (
            GenAIToolsHandler(mode=Mode.TOOLS),
            {"system_instruction", "tools", "tool_config"},
        ),
        (
            GenAIStructuredOutputsHandler(mode=Mode.JSON),
            {"system_instruction"},
        ),
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        {"cached_content": "cachedContents/test"},
        SimpleNamespace(cached_content="cachedContents/test"),
    ],
    ids=["dict-config", "object-config"],
)
def test_genai_handlers_omit_cached_content_request_fields(
    monkeypatch: pytest.MonkeyPatch,
    handler: GenAIToolsHandler | GenAIStructuredOutputsHandler,
    forbidden_fields: set[str],
    config: dict[str, str] | SimpleNamespace,
) -> None:
    _install_fake_genai_types(monkeypatch)

    def merge_cached_content(
        kwargs: dict[str, Any], base_config: dict[str, Any]
    ) -> dict[str, Any]:
        user_config = kwargs["config"]
        cached_content = (
            user_config["cached_content"]
            if isinstance(user_config, dict)
            else user_config.cached_content
        )
        return {**base_config, "cached_content": cached_content}

    monkeypatch.setattr(
        "instructor.v2.providers.gemini.utils.map_to_genai_schema",
        lambda schema: schema,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.gemini.utils.map_to_gemini_function_schema",
        lambda schema: schema,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.gemini.utils.update_genai_kwargs",
        merge_cached_content,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.gemini.utils.convert_to_genai_messages",
        lambda messages: messages,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.extract_genai_multimodal_content",
        lambda contents, _autodetect_images: contents,
    )

    prepared_model, prepared = handler.prepare_request(
        CachedResponse,
        {
            "messages": [
                {"role": "system", "content": "cached instruction"},
                {"role": "user", "content": "hello"},
            ],
            "config": config,
        },
    )

    prepared_config = prepared["config"].kwargs
    assert prepared_config["cached_content"] == "cachedContents/test"
    assert forbidden_fields.isdisjoint(prepared_config)
    if handler.mode == Mode.JSON:
        assert prepared_config["response_mime_type"] == "application/json"
        assert prepared_config["response_schema"] is prepared_model
    else:
        assert "response_mime_type" not in prepared_config
        assert "response_schema" not in prepared_config


def test_structured_outputs_parse_response_unwraps_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = GenAIStructuredOutputsHandler(mode=Mode.JSON)

    class FakeAdapterBase:
        pass

    class FakeAdapter(FakeAdapterBase):
        def __init__(self, content: str) -> None:
            self.content = content

    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.AdapterBase",
        FakeAdapterBase,
    )
    monkeypatch.setattr(
        "instructor.v2.providers.genai.handlers.parse_genai_structured_outputs",
        lambda *_args, **_kwargs: FakeAdapter("done"),
    )

    class FakeResponseModel(BaseModel):
        pass

    result = handler.parse_response(
        response=SimpleNamespace(),
        response_model=FakeResponseModel,
        stream=False,
    )

    assert result == "done"
