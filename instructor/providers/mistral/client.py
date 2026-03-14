# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations


import instructor
from typing import overload, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mistralai import Mistral  # type: ignore[import-untyped]


def _import_mistral_class() -> type:
    """Import the Mistral client class, supporting both v1.x and v2.x.

    mistralai 2.0.0 moved the Mistral class from ``mistralai.Mistral``
    to ``mistralai.client.Mistral``.  We try the v1 path first (more
    common in existing installs) and fall back to the v2 path.
    """
    try:
        from mistralai import Mistral as _Mistral  # v1.x
    except ImportError:
        from mistralai.client import Mistral as _Mistral  # v2.x  # type: ignore[no-redef]
    return _Mistral


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

    MistralClass = _import_mistral_class()
    if not isinstance(client, MistralClass):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of mistralai.Mistral. "
            f"Got: {type(client).__name__}"
        )

    if use_async:

        async def async_wrapper(
            *args: Any, **kwargs: Any
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

    def sync_wrapper(*args: Any, **kwargs: Any):  # Handler for sync streaming
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
