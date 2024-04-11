import anthropic
import instructor

from typing import overload


@overload
def from_anthropic(
    client: anthropic.Anthropic | anthropic.AnthropicBedrock,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> instructor.Instructor: ...


@overload
def from_anthropic(
    client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> instructor.Instructor: ...


def from_anthropic(
    client: (
        anthropic.Anthropic
        | anthropic.AsyncAnthropic
        | anthropic.AnthropicBedrock
        | anthropic.AsyncAnthropicBedrock
    ),
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert mode in {
        instructor.Mode.ANTHROPIC_JSON,
        instructor.Mode.ANTHROPIC_TOOLS,
    }, "Mode be one of {instructor.Mode.ANTHROPIC_JSON, instructor.Mode.ANTHROPIC_TOOLS}"

    assert isinstance(
        client,
        (
            anthropic.Anthropic,
            anthropic.AsyncAnthropic,
            anthropic.AnthropicBedrock,
            anthropic.AsyncAnthropicBedrock,
        ),
    ), "Client must be an instance of {anthropic.Anthropic, anthropic.AsyncAnthropic, anthropic.AnthropicBedrock, anthropic.AsyncAnthropicBedrock}"

    if mode == instructor.Mode.ANTHROPIC_TOOLS:
        create = client.beta.tools.messages.create
    else:
        create = client.messages.create

    if isinstance(client, (anthropic.Anthropic, anthropic.AnthropicBedrock)):
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
