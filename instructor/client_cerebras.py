from __future__ import annotations

from typing import Any, Literal, overload

import instructor
from instructor.client import AsyncInstructor, Instructor


from cerebras.cloud.sdk import Cerebras, AsyncCerebras


@overload
def from_cerebras(
    client: Cerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cerebras(
    client: AsyncCerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cerebras(
    client: Cerebras | AsyncCerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    assert (
        mode
        in {
            instructor.Mode.CEREBRAS_TOOLS,
            instructor.Mode.CEREBRAS_JSON,
        }
    ), "Mode must be one of {instructor.Mode.CEREBRAS_TOOLS, instructor.Mode.CEREBRAS_JSON}"

    assert isinstance(
        client, (Cerebras, AsyncCerebras)
    ), "Client must be an instance of Cerebras or AsyncCerebras"

    if use_async:
        create = client.chat.completions.create
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.CEREBRAS,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.CEREBRAS,
        mode=mode,
        **kwargs,
    )
