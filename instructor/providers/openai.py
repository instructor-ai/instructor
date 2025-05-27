"""
OpenAI provider implementation for Instructor.
"""
from __future__ import annotations

import openai
import instructor
from functools import partial
from typing import Any, Callable, Awaitable, overload

from instructor.utils import Provider, get_provider


@overload
def from_openai(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor:
    pass


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor:
    pass


def map_chat_completion_to_response(messages, client, *args, **kwargs) -> Any:
    return client.responses.create(
        *args,
        input=messages,
        **kwargs,
    )


async def async_map_chat_completion_to_response(
    messages, client, *args, **kwargs
) -> Any:
    return await client.responses.create(
        *args,
        input=messages,
        **kwargs,
    )


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    """Create an Instructor instance from an OpenAI client.

    Args:
        client: An instance of OpenAI client (sync or async)
        mode: The mode to use for the client
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        AssertionError: If mode is not compatible with the provider
    """
    if hasattr(client, "base_url"):
        provider = get_provider(str(client.base_url))
    else:
        provider = Provider.OPENAI

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        import warnings

        warnings.warn(
            "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. Unexpected behavior may occur with other client types.",
            stacklevel=2,
        )

    if provider in {Provider.OPENROUTER}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS,
            instructor.Mode.JSON,
        }

    if provider in {Provider.ANYSCALE, Provider.TOGETHER}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.JSON_SCHEMA,
            instructor.Mode.MD_JSON,
        }

    if provider in {Provider.OPENAI, Provider.DATABRICKS}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.FUNCTIONS,
            instructor.Mode.PARALLEL_TOOLS,
            instructor.Mode.MD_JSON,
            instructor.Mode.TOOLS_STRICT,
            instructor.Mode.JSON_O1,
            instructor.Mode.RESPONSES_TOOLS,
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }

    if isinstance(client, openai.OpenAI):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(
                create=client.chat.completions.create
                if mode
                not in {
                    instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                    instructor.Mode.RESPONSES_TOOLS,
                }
                else partial(map_chat_completion_to_response, client=client),
                mode=mode,
            ),
            mode=mode,
            provider=provider,
            **kwargs,
        )

    if isinstance(client, openai.AsyncOpenAI):
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(
                create=client.chat.completions.create
                if mode
                not in {
                    instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                    instructor.Mode.RESPONSES_TOOLS,
                }
                else partial(async_map_chat_completion_to_response, client=client),
                mode=mode,
            ),
            mode=mode,
            provider=provider,
            **kwargs,
        )
    
    raise ValueError(f"Unsupported client type: {type(client)}")
