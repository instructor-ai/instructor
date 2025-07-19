# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations


from mistralai import Mistral
import instructor
from typing import overload, Any, Literal


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_mistral(
    client: Mistral,
    mode: instructor.Mode = instructor.Mode.MISTRAL_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.MISTRAL_TOOLS,
        instructor.Mode.MISTRAL_STRUCTURED_OUTPUTS,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Mistral",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, Mistral):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of mistralai.Mistral. "
            f"Got: {type(client).__name__}"
        )

    if use_async:

        async def async_wrapper(
            *args: Any, **kwargs: dict[str, Any]
        ):  # Handler for async streaming
            if kwargs.pop("stream", False):
                return await client.chat.stream_async(*args, **kwargs)
            return await client.chat.complete_async(*args, **kwargs)

        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=async_wrapper, mode=mode),
            provider=instructor.Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )

    def sync_wrapper(
        *args: Any, **kwargs: dict[str, Any]
    ):  # Handler for sync streaming
        if kwargs.pop("stream", False):
            return client.chat.stream(*args, **kwargs)
        return client.chat.complete(*args, **kwargs)

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=sync_wrapper, mode=mode),
        provider=instructor.Provider.MISTRAL,
        mode=mode,
        **kwargs,
    )
