from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, TypeVar, cast, overload

import cohere
import instructor
from pydantic import BaseModel
from typing_extensions import ParamSpec


T_Model = TypeVar("T_Model", bound=BaseModel)
T_ParamSpec = ParamSpec("T_ParamSpec")


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_cohere(
    client: cohere.ClientV2,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClient,
    mode: instructor.Mode = instructor.Mode.COHERE_JSON_SCHEMA,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_cohere(
    client: cohere.AsyncClientV2,
    mode: instructor.Mode = instructor.Mode.COHERE_JSON_SCHEMA,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_cohere(
    client: cohere.Client | cohere.AsyncClient | cohere.ClientV2 | cohere.AsyncClientV2,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs: Any,
):
    valid_modes = {
        instructor.Mode.COHERE_TOOLS,
        instructor.Mode.COHERE_JSON_SCHEMA,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Cohere", valid_modes=[str(m) for m in valid_modes]
        )

    # Determine if we're dealing with an async client
    is_async = isinstance(client, (cohere.AsyncClient, cohere.AsyncClientV2))

    if isinstance(client, (cohere.ClientV2, cohere.AsyncClientV2)):
        client_version = "v2"
    elif isinstance(client, (cohere.Client, cohere.AsyncClient)):
        client_version = "v1"
    else:
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of cohere.Client or cohere.AsyncClient or cohere.ClientV2 or cohere.AsyncClientV2. "
            f"Got: {type(client).__name__}"
        )
    kwargs["_cohere_client_version"] = client_version

    if is_async:

        async def async_wrapper(*args: Any, **call_kwargs: Any):
            if call_kwargs.pop("stream", False):
                return client.chat_stream(*args, **call_kwargs)
            result = client.chat(*args, **call_kwargs)
            if inspect.isawaitable(result):
                return await cast(Awaitable[Any], result)
            return result

        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )
    else:

        def sync_wrapper(*args: Any, **call_kwargs: Any):
            if call_kwargs.pop("stream", False):
                return client.chat_stream(*args, **call_kwargs)
            return client.chat(*args, **call_kwargs)

        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=sync_wrapper, mode=mode),
            provider=instructor.Provider.COHERE,
            mode=mode,
            **kwargs,
        )
