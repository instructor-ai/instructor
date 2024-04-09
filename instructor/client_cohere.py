import cohere
import instructor

from typing import overload


@overload
def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
) -> instructor.Instructor: ...


def from_cohere(
    client: cohere.Client,
    mode: instructor.Mode = instructor.Mode.COHERE_TOOLS,
    **kwargs,
) -> instructor.Instructor:
    assert (
        mode
        in {
            instructor.Mode.COHERE_TOOLS,
        }
    ), "Mode be one of {instructor.Mode.COHERE_TOOLS}"

    assert isinstance(
        client, (cohere.Client)
    ), "Client must be an instance of cohere.Cohere"

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=client.chat, mode=mode),
        provider=instructor.Provider.COHERE,
        mode=mode,
        **kwargs,
    )
