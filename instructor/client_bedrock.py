from __future__ import annotations  # type: ignore

from typing import Any, Literal, overload
import warnings

import boto3
from botocore.client import BaseClient

import instructor
from instructor.client import AsyncInstructor, Instructor


@overload  # type: ignore
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    async_client: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload  # type: ignore
def from_bedrock(
    client: boto3.client,
    mode: instructor.Mode = instructor.Mode.BEDROCK_TOOLS,
    async_client: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def handle_bedrock_json(
    response_model: Any,
    new_kwargs: Any,
) -> tuple[Any, Any]:
    """
    This function is deprecated and no longer used.
    Bedrock JSON handling is now done in process_response.py via handle_bedrock_json().
    """
    return response_model, new_kwargs


def from_bedrock(
    client: BaseClient,
    mode: instructor.Mode = instructor.Mode.BEDROCK_JSON,
    async_client: bool = False,
    _async: bool | None = None,  # Deprecated, use async_client
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """
    Accepts both 'async_client' (preferred) and '_async' (deprecated) for async mode.
    """
    valid_modes = {
        instructor.Mode.BEDROCK_TOOLS,
        instructor.Mode.BEDROCK_JSON,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Bedrock",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, BaseClient):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of boto3.client (BaseClient). "
            f"Got: {type(client).__name__}"
        )

    # Deprecation warning for _async usage
    if _async is not None and not async_client:
        warnings.warn(
            "The '_async' argument to from_bedrock is deprecated. Use 'async_client' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Prefer async_client, fallback to _async for backward compatibility
    use_async = async_client or (_async is not None and _async is True)

    async def async_wrapper(**kwargs: Any):
        return client.converse(**kwargs)

    create = client.converse

    if use_async:
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
