from __future__ import annotations

from types import ModuleType, SimpleNamespace
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


class FakePart:
    def __init__(
        self,
        *,
        text: str | None = None,
        function_call: Any = None,
        function_response: Any = None,
    ) -> None:
        self.text = text
        self.function_call = function_call
        self.function_response = function_response

    @classmethod
    def from_text(cls, text: str) -> FakePart:
        return cls(text=text)

    @classmethod
    def from_function_response(cls, name: str, response: dict[str, Any]) -> FakePart:
        return cls(function_response={"name": name, "response": response})


class FakeContent:
    def __init__(self, role: str, parts: list[FakePart]) -> None:
        self.role = role
        self.parts = parts


class FakeModelContent(FakeContent):
    def __init__(self, parts: list[FakePart], role: str = "model") -> None:
        super().__init__(role=role, parts=parts)


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_genai_types(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_types = SimpleNamespace(
        Content=FakeContent,
        Part=FakePart,
        ModelContent=FakeModelContent,
        GenerateContentConfig=FakeGenerateContentConfig,
    )
    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.types = fake_types  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(__import__("sys").modules, "google", google_module)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", genai_module)


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
        "instructor.v2.providers.genai.handlers.extract_multimodal_content",
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
