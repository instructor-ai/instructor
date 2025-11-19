"""Tests for Anthropic handlers routed through process_response."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.dsl.iterable import IterableBase
from instructor.dsl.parallel import AnthropicParallelBase
from instructor.processing.response import process_response


class StreamItem(BaseModel):
    value: int


class SimpleIterable(IterableBase):
    """Minimal iterable model for testing streaming integration."""

    task_type = StreamItem

    @classmethod
    def from_streaming_response(
        cls, completion: Generator[dict[str, int], None, None], mode: Mode, **_: object
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

    parallel_model = AnthropicParallelBase(ToolA, ToolB)

    parsed = process_response(
        response=response,
        response_model=parallel_model,
        stream=False,
        validation_context=None,
        strict=True,
        mode=Mode.ANTHROPIC_PARALLEL_TOOLS,
    )

    assert [model.model_dump() for model in parsed] == [
        {"foo": 5},
        {"bar": "x"},
    ]
