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
) -> instructor.client.Instructor | instructor.client.AsyncInstructor:
    assert mode in {
        instructor.Mode.WRITER_TOOLS,
    }, "Mode must be instructor.Mode.WRITER_TOOLS"

    assert isinstance(client, (Writer, AsyncWriter)), (
        "Client must be an instance of Writer or AsyncWriter"
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
