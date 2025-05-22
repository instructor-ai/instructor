from __future__ import annotations

from typing import overload, Any

import groq
import instructor


@overload
def from_groq(
    client: groq.Groq,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_groq(
    client: groq.AsyncGroq,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_groq(
    client: groq.Groq | groq.AsyncGroq,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    valid_modes = {
        instructor.Mode.JSON,
        instructor.Mode.TOOLS,
    }

    if mode not in valid_modes:
        from instructor.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Groq", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (groq.Groq, groq.AsyncGroq)):
        from instructor.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of groq.Groq or groq.AsyncGroq. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, groq.Groq):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            provider=instructor.Provider.GROQ,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create, mode=mode),
            provider=instructor.Provider.GROQ,
            mode=mode,
            **kwargs,
        )
