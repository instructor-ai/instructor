from __future__ import annotations  # type: ignore

from typing import Any, Literal, overload

import boto3
from botocore.client import BaseClient

import instructor
from instructor.client import AsyncInstructor, Instructor


@overload  # type: ignore
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    _async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload  # type: ignore
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    _async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def handle_bedrock_json(
    response_model: Any,
    new_kwargs: Any,
) -> tuple[Any, Any]:
    print(f"handle_bedrock_json: response_model {response_model}")
    print(f"handle_bedrock_json: new_kwargs {new_kwargs}")
    return response_model, new_kwargs


def from_bedrock(
    client: BaseClient,
    mode: instructor.Mode = instructor.Mode.BEDROCK_JSON,
    _async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    assert mode in {
        instructor.Mode.BEDROCK_TOOLS,
        instructor.Mode.BEDROCK_JSON,
    }, (
        "Mode must be one of {instructor.Mode.BEDROCK_TOOLS, instructor.Mode.BEDROCK_JSON}"
    )
    assert isinstance(
        client,
        BaseClient,
    ), "Client must be an instance of boto3.client"

    async def async_wrapper(**kwargs: Any):
        return client.converse(**kwargs)

    create = client.converse

    if _async:
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.BEDROCK,
            mode=mode,
            **kwargs,
        )
    else:
        return Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.BEDROCK,
            mode=mode,
            **kwargs,
        )
