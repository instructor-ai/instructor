"""Tests for Anthropic handlers routed through process_response."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Union

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.dsl.iterable import IterableBase
from instructor.processing.response import process_response
from instructor.v2.providers.anthropic.handlers import (
    AnthropicJSONHandler,
    AnthropicStructuredOutputsHandler,
    AnthropicToolsHandler,
)


class StreamItem(BaseModel):
    value: int


class SimpleIterable(IterableBase):
    """Minimal iterable model for testing streaming integration."""

    task_type = StreamItem

    @classmethod
    def from_streaming_response(
        cls, completion: Generator[dict[str, int], None, None], _mode: Mode, **_: object
    ) -> Generator[StreamItem, None, None]:
        for payload in completion:
            yield StreamItem.model_validate(payload)


@dataclass
class FakeToolContent:
    """Simplified tool_use block for parallel parsing tests."""

    type: str
    name: str
    input: dict[str, object]


@dataclass
class FakeMessage:
    """Minimal Anthropic message stand-in for tests."""

    content: list[FakeToolContent]
    stop_reason: str | None = None


class ToolA(BaseModel):
    foo: int


class ToolB(BaseModel):
    bar: str


def test_process_response_streaming_iterable_uses_registry():
    """Streaming iterable models should be parsed via Anthropic handler."""

    response = ({"value": 1}, {"value": 2})

    result = process_response(
        response=response,
        response_model=SimpleIterable,
        stream=True,
        validation_context=None,
        strict=True,
        mode=Mode.ANTHROPIC_TOOLS,
    )

    assert [item.value for item in result] == [1, 2]


def test_process_response_parallel_tools_matches_previous_behavior():
    """Parallel tool responses should yield validated models via handler."""

    response = FakeMessage(
        content=[
            FakeToolContent(type="tool_use", name="ToolA", input={"foo": 5}),
            FakeToolContent(type="tool_use", name="ToolB", input={"bar": "x"}),
        ]
    )

    # Use typehint instead of AnthropicParallelBase instance
    ParallelTools = Iterable[Union[ToolA, ToolB]]

    parsed = process_response(
        response=response,
        response_model=ParallelTools,
        stream=False,
        validation_context=None,
        strict=True,
        mode=Mode.ANTHROPIC_PARALLEL_TOOLS,
    )

    assert [model.model_dump() for model in parsed] == [
        {"foo": 5},
        {"bar": "x"},
    ]


@dataclass
class FakeToolUseBlock:
    """Content block representing a tool call."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "type": self.type,
        }


@dataclass
class FakeTextBlock:
    """Content block representing text output."""

    text: str
    type: str = "text"

    def model_dump(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


def _make_message(*content_blocks: Any) -> Any:
    """Helper to construct a fake message with arbitrary content."""

    return SimpleNamespace(content=list(content_blocks))


def test_tools_handler_handle_reask_with_tool_result():
    handler = AnthropicToolsHandler()
    kwargs = {"messages": [{"role": "user", "content": "initial"}]}
    response = _make_message(
        FakeToolUseBlock(id="tool_1", name="ToolA", input={"foo": 1})
    )

    new_kwargs = handler.handle_reask(kwargs, response, ValueError("boom"))

    assert len(new_kwargs["messages"]) == 3
    assistant_msg = new_kwargs["messages"][-2]
    assert assistant_msg["role"] == "assistant"
    tool_msg = new_kwargs["messages"][-1]["content"][0]
    assert tool_msg["tool_use_id"] == "tool_1"
    assert tool_msg["is_error"] is True
    assert "Validation Error found" in tool_msg["content"]


def test_tools_handler_handle_reask_without_tool_use():
    handler = AnthropicToolsHandler()
    kwargs = {"messages": [{"role": "user", "content": "initial"}]}
    response = _make_message(FakeTextBlock("no tool used"))

    new_kwargs = handler.handle_reask(kwargs, response, ValueError("boom"))

    assert len(new_kwargs["messages"]) == 3
    final_msg = new_kwargs["messages"][-1]
    assert final_msg["role"] == "user"
    assert "no tool invocation" in final_msg["content"]


def test_json_handler_handle_reask_includes_last_text():
    handler = AnthropicJSONHandler()
    kwargs = {"messages": [{"role": "user", "content": "initial"}]}
    response = _make_message(FakeTextBlock("previous attempt"))

    new_kwargs = handler.handle_reask(kwargs, response, ValueError("json boom"))

    assert len(new_kwargs["messages"]) == 2
    reask_msg = new_kwargs["messages"][-1]
    assert "previous attempt" in reask_msg["content"]


def test_structured_outputs_handle_reask_includes_last_text():
    handler = AnthropicStructuredOutputsHandler()
    kwargs = {"messages": [{"role": "user", "content": "initial"}]}
    response = _make_message(FakeTextBlock("structured output"))

    new_kwargs = handler.handle_reask(kwargs, response, ValueError("schema boom"))

    assert len(new_kwargs["messages"]) == 2
    reask_msg = new_kwargs["messages"][-1]
    assert "structured output" in reask_msg["content"]
