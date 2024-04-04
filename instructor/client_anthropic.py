import anthropic
import instructor

from typing import overload


@overload
def from_anthropic(
    client: anthropic.Anthropic,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> instructor.Instructor: ...


@overload
def from_anthropic(
    client: anthropic.AsyncAnthropic,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> instructor.Instructor: ...


def from_anthropic(
    client: anthropic.Anthropic | anthropic.AsyncAnthropic,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert (
        mode
        in {
            instructor.Mode.ANTHROPIC_JSON,
            instructor.Mode.ANTHROPIC_TOOLS,
        }
    ), "Mode be one of {instructor.Mode.ANTHROPIC_JSON, instructor.Mode.ANTHROPIC_TOOLS}"

    assert isinstance(
        client, (anthropic.Anthropic, anthropic.AsyncAnthropic)
    ), "Client must be an instance of anthropic.Anthropic or anthropic.AsyncAnthropic"

    if mode == instructor.Mode.ANTHROPIC_TOOLS:
        create = client.beta.tools.messages.create
    else:
        create = client.messages.create

    if isinstance(client, anthropic.Anthropic):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.ANTHROPIC,
            mode=mode,
            **kwargs,
        )
