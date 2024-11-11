# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

import instructor
from writerai import AsyncWriter, Writer
from typing import overload, Any, Literal

@overload
def from_writer(
    client: Writer,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_writer(
    client: AsyncWriter,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_writer(
    client: Writer,
    mode: instructor.Mode = instructor.Mode.WRITER_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    assert mode in {
        instructor.Mode.WRITER_TOOLS,
        instructor.Mode.WRITER_JSON,
    }, "Mode must be instructor.Mode.WRITER_TOOLS or instructor.Mode.WRITER_JSON"

    assert isinstance(
        client, (Writer, AsyncWriter)
    ), "Client must be an instance of Writer or AsyncWriter"

    args = {
        "client": client,
        "create": instructor.patch(create=client.chat.chat, mode=mode),
        "provider": instructor.Provider.WRITER,
        "mode": mode,
        **kwargs,
    }

    if not use_async: return instructor.Instructor(**args)
    else: return instructor.AsyncInstructor(**args)