"""Tests for ParallelBase response processing.

Ensures that ``process_response`` and ``process_response_async`` correctly
handle the generator returned by ``ParallelBase.from_response()`` without
attempting to set attributes on the generator object (fixes #2049).
"""

from __future__ import annotations

from types import GeneratorType

import pytest
from pydantic import BaseModel

from instructor.dsl.parallel import ParallelBase
from instructor.mode import Mode
from instructor.processing.response import process_response, process_response_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class UserInfo(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, function: _FakeFunction):
        self.function = function


class _FakeMessage:
    def __init__(self, tool_calls: list[_FakeToolCall]):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeResponse:
    """Minimal stand-in for an OpenAI-like completion with parallel tool calls."""

    def __init__(self, tool_calls: list[_FakeToolCall]):
        self.choices = [_FakeChoice(_FakeMessage(tool_calls))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_response(*tool_calls: tuple[str, str]) -> _FakeResponse:
    return _FakeResponse(
        [_FakeToolCall(_FakeFunction(name, args)) for name, args in tool_calls]
    )


class TestParallelBaseProcessResponse:
    """process_response with ParallelBase should not raise on generator."""

    def test_returns_generator_without_attribute_error(self):
        """Regression test for #2049: model._raw_response on generator."""
        response_model = ParallelBase(UserInfo, Address)
        raw = _build_response(
            ("UserInfo", '{"name": "Alice", "age": 30}'),
            ("Address", '{"street": "123 Main St", "city": "Springfield"}'),
        )

        result = process_response(
            raw,
            response_model=response_model,
            stream=False,
            mode=Mode.PARALLEL_TOOLS,
        )

        # Result should be a generator yielding parsed models
        assert isinstance(result, GeneratorType)
        items = list(result)
        assert len(items) == 2
        assert isinstance(items[0], UserInfo)
        assert items[0].name == "Alice"
        assert isinstance(items[1], Address)
        assert items[1].city == "Springfield"

    def test_raw_response_attached_to_response_model(self):
        """_raw_response should be set on the ParallelBase instance."""
        response_model = ParallelBase(UserInfo)
        raw = _build_response(("UserInfo", '{"name": "Bob", "age": 25}'))

        process_response(
            raw,
            response_model=response_model,
            stream=False,
            mode=Mode.PARALLEL_TOOLS,
        )

        assert hasattr(response_model, "_raw_response")
        assert response_model._raw_response is raw

    def test_single_tool_call(self):
        response_model = ParallelBase(UserInfo)
        raw = _build_response(("UserInfo", '{"name": "Carol", "age": 40}'))

        result = process_response(
            raw,
            response_model=response_model,
            stream=False,
            mode=Mode.PARALLEL_TOOLS,
        )

        items = list(result)
        assert len(items) == 1
        assert items[0].name == "Carol"


class TestParallelBaseProcessResponseAsync:
    """process_response_async with ParallelBase should not raise on generator."""

    @pytest.mark.asyncio
    async def test_returns_generator_without_attribute_error(self):
        """Regression test for #2049 (async path)."""
        response_model = ParallelBase(UserInfo, Address)
        raw = _build_response(
            ("UserInfo", '{"name": "Dave", "age": 35}'),
            ("Address", '{"street": "456 Oak Ave", "city": "Shelbyville"}'),
        )

        result = await process_response_async(
            raw,
            response_model=response_model,
            stream=False,
            mode=Mode.PARALLEL_TOOLS,
        )

        assert isinstance(result, GeneratorType)
        items = list(result)
        assert len(items) == 2
        assert isinstance(items[0], UserInfo)
        assert items[0].name == "Dave"
        assert isinstance(items[1], Address)
        assert items[1].city == "Shelbyville"

    @pytest.mark.asyncio
    async def test_raw_response_attached_to_response_model(self):
        response_model = ParallelBase(UserInfo)
        raw = _build_response(("UserInfo", '{"name": "Eve", "age": 28}'))

        await process_response_async(
            raw,
            response_model=response_model,
            stream=False,
            mode=Mode.PARALLEL_TOOLS,
        )

        assert hasattr(response_model, "_raw_response")
        assert response_model._raw_response is raw
