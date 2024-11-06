# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model as Watsonx
import instructor
from typing import overload, Any, Literal
from instructor.client import AsyncInstructor, Instructor


@overload
def from_watsonx(
    client: Watsonx,
    mode: instructor.Mode = instructor.Mode.WATSONX_TOOLS,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...

@overload
def from_watsonx(
    client: Watsonx,
    mode: instructor.Mode = instructor.Mode.WATSONX_TOOLS,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...

def from_watsonx(
    client: Watsonx,
    mode: instructor.Mode = instructor.Mode.WATSONX_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert mode in {
        instructor.Mode.WATSONX_TOOLS,
        instructor.Mode.WATSONX_MD_JSON,
    }, "Mode be one of {instructor.Mode.WATSONX_TOOLS, WATSONX_MD_JSON}"

    assert isinstance(
        client, Watsonx
    ), "Client must be an instance of ibm_watsonx_ai.foundation_models.Model"

    if "stream" in kwargs and kwargs["stream"] is True:
        raise ValueError("Unsupported stream functionality")

    if not use_async:
        
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=client.chat, mode=mode),
            provider=instructor.Provider.WATSONX,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError("Unsupported async functionality")
