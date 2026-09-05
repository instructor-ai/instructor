from __future__ import annotations

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

import instructor
from instructor.core.exceptions import ConfigurationError

from instructor.v2.validation.async_validators import (
    async_field_validator,
    run_async_validators,
    model_declares_async_validators,
)


class Ordered(BaseModel):
    value: str

    @async_field_validator("value")
    async def prefix(cls, value: str) -> str:
        return "prefix:" + value

    @async_field_validator("value")
    async def suffix(cls, value: str) -> str:
        return value + ":suffix"


@pytest.mark.asyncio
async def test_async_field_validators_chain_results() -> None:
    result = await run_async_validators(Ordered(value="x"), context=None)
    assert result.value == "prefix:x:suffix"


@pytest.mark.asyncio
async def test_undecorated_override_suppresses_inherited_validator() -> None:
    class Child(Ordered):
        async def prefix(cls, value: str) -> str:
            return value

    result = await run_async_validators(Child(value="x"), context=None)
    assert result.value == "x:suffix"


def test_nested_container_validators_are_detected() -> None:
    class Parent(BaseModel):
        children: dict[str, list[Ordered]]

    assert model_declares_async_validators(Parent)
    assert model_declares_async_validators(list[Parent])


def test_recursive_model_without_validators_terminates() -> None:
    class Node(BaseModel):
        children: list[Node] = []

    Node.model_rebuild()
    assert not model_declares_async_validators(Node)


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["stream", "list", "partial", "parallel"])
async def test_unsupported_async_workflows_fail_before_request(workflow: str) -> None:
    async with AsyncOpenAI(api_key="unused", base_url="http://localhost:1") as sdk:
        mode = (
            instructor.Mode.PARALLEL_TOOLS
            if workflow == "parallel"
            else instructor.Mode.TOOLS
        )
        client = instructor.from_openai(sdk, mode=mode)
        response_model = (
            list[Ordered]
            if workflow == "list"
            else instructor.Partial[Ordered]
            if workflow == "partial"
            else Ordered
        )
        with pytest.raises(ConfigurationError, match="non-streaming single response"):
            await client.create(
                model="unused",
                response_model=response_model,
                messages=[{"role": "user", "content": "unused"}],
                stream=workflow == "stream",
            )
