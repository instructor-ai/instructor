# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations


import instructor
from writerai import AsyncWriter, Writer
from typing import overload, Any


@overload
def from_writer(
    client: Writer,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_writer(
    client: AsyncWriter,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_writer(
    client: Writer | AsyncWriter,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {instructor.Mode.WRITER_TOOLS, instructor.Mode.WRITER_JSON}

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Writer", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (Writer, AsyncWriter)):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of Writer or AsyncWriter. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, Writer):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat.chat, mode=mode),
            provider=instructor.Provider.WRITER,
            mode=mode,
            **kwargs,
        )

    return instructor.AsyncInstructor(
        client=client,
        create=instructor.patch(create=client.chat.chat, mode=mode),
        provider=instructor.Provider.WRITER,
        mode=mode,
        **kwargs,
    )
