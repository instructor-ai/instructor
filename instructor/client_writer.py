# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

import json
import re

import instructor
from writerai import AsyncWriter, Writer
from writerai.types.chat import ChoiceMessageToolCall, ChoiceMessageToolCallFunction
from openai.types.chat import ChatCompletion
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
    }, "Mode must be instructor.Mode.WRITER_TOOLS"

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

def parse_multiple_writer_tools(completion: ChatCompletion) -> ChatCompletion:
    content = completion.choices[0].message.content
    if content:
        tool_calls = re.findall(r"\[.*\]", content)
        if len(tool_calls or []) != 2:
            raise ValueError(f"Wrong message content, it should be "
                             f"'[TOOL_CALLS] [info about calls]' got {content} instead.")

        tool_calls_dict = json.loads(tool_calls[-1])
        tool_calls_dict.update({"arguments": json.dumps(tool_calls_dict["arguments"])})

        function = ChoiceMessageToolCallFunction(**tool_calls_dict)

        completion.choices[0].finish_reason = "tool_calls"
        completion.choices[0].message.content = None
        completion.choices[0].message.tool_calls=[ChoiceMessageToolCall(id="id", function=function, type="function")]
        return completion
    else:
        raise ValueError("Can't handle empty message content.")