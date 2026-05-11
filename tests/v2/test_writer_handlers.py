"""Writer-specific handler behavior not covered by shared compatibility tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    answer: float


class User(BaseModel):
    name: str
    age: int


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    function: MockFunction

    @classmethod
    def build(cls, model_name: str, arguments: dict[str, Any]) -> MockToolCall:
        return cls(MockFunction(model_name, json.dumps(arguments)))


@dataclass
class MockMessage:
    content: str | None = None
    tool_calls: list[MockToolCall] | None = None


@dataclass
class MockChoice:
    message: MockMessage
    finish_reason: str = "stop"


@dataclass
class MockResponse:
    choices: list[MockChoice]

    @classmethod
    def from_content(
        cls,
        content: str | None = None,
        *,
        tool_calls: list[MockToolCall] | None = None,
    ) -> MockResponse:
        return cls([MockChoice(MockMessage(content, tool_calls))])


def test_tools_request_uses_auto_tool_choice() -> None:
    _, kwargs = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS).request_handler(
        Answer,
        {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    )

    assert kwargs["tool_choice"] == "auto"


def test_tools_parse_user_model() -> None:
    response = MockResponse.from_content(
        tool_calls=[MockToolCall.build("User", {"name": "Alice", "age": 30})]
    )

    result = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS).response_parser(
        response,
        User,
    )

    assert result == User(name="Alice", age=30)


def test_md_json_parses_nested_codeblock() -> None:
    response = MockResponse.from_content('```json\n{"name": "Bob", "age": 25}\n```')

    result = mode_registry.get_handlers(Provider.WRITER, Mode.MD_JSON).response_parser(
        response,
        User,
    )

    assert result == User(name="Bob", age=25)
