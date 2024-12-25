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
    assert mode in {
        instructor.Mode.MISTRAL_TOOLS,
    }, "Mode be one of {instructor.Mode.MISTRAL_TOOLS}"

    

    assert isinstance(
        client, Mistral
    ), "Client must be an instance of mistralai.Mistral"

    if not use_async:
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat.complete, mode=mode),
            provider=instructor.Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.complete_async, mode=mode),
            provider=instructor.Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )
