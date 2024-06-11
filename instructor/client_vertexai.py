from __future__ import annotations

from typing import Any

from vertexai.preview.generative_models import ToolConfig  # type: ignore[reportMissingTypeStubs]
import vertexai.generative_models as gm  # type: ignore[reportMissingTypeStubs]
from pydantic import BaseModel
import instructor
import jsonref  # type: ignore[reportMissingTypeStubs]


def _create_vertexai_tool(model: BaseModel) -> gm.Tool:
    schema: dict[Any, Any] = jsonref.replace_refs(model.model_json_schema())  # type: ignore[reportMissingTypeStubs]

    parameters: dict[Any, Any] = {
        "type": schema["type"],
        "properties": schema["properties"],
        "required": schema["required"],
    }

    declaration = gm.FunctionDeclaration(
        name=model.__name__, description=model.__doc__, parameters=parameters
    )

    tool = gm.Tool(function_declarations=[declaration])

    return tool


def _vertexai_message_parser(message: dict[str, str]) -> gm.Content:
    return gm.Content(
        role=message["role"], parts=[gm.Part.from_text(message["content"])]
    )


def vertexai_function_response_parser(
    response: gm.GenerationResponse, exception: Exception
) -> gm.Content:
    return gm.Content(
        parts=[
            gm.Part.from_function_response(
                name=response.candidates[0].content.parts[0].function_call.name,
                response={
                    "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                },
            )
        ]
    )


def vertexai_process_response(_kwargs: dict[str, Any], model: BaseModel):
    messages = _kwargs.pop("messages")
    contents = [
        _vertexai_message_parser(message)  # type: ignore[reportUnkownArgumentType]
        if isinstance(message, dict)
        else message
        for message in messages
    ]
    tool = _create_vertexai_tool(model=model)
    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def from_vertexai(
    client: gm.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.VERTEXAI_TOOLS,
    _async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    assert (
        mode == instructor.Mode.VERTEXAI_TOOLS
    ), "Mode must be instructor.Mode.VERTEXAI_TOOLS"

    assert isinstance(
        client, gm.GenerativeModel
    ), "Client must be an instance of vertexai.generative_models.GenerativeModel"

    create = client.generate_content_async if _async else client.generate_content

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.VERTEXAI,
        mode=mode,
        **kwargs,
    )
