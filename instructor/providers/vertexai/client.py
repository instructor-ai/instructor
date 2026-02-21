from __future__ import annotations

from typing import Any, Union, get_origin

from vertexai.preview.generative_models import ToolConfig  # type: ignore[import-not-found]
import vertexai.generative_models as gm  # type: ignore[import-not-found]
from pydantic import BaseModel
import instructor
from ...dsl.parallel import get_types_array
import jsonref


def _create_gemini_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    # Add type check to ensure we have a concrete model class
    if get_origin(model) is not None:
        raise TypeError(f"Expected concrete model class, got type hint {model}")

    schema = model.model_json_schema()
    schema_without_refs: dict[str, Any] = jsonref.replace_refs(schema)  # type: ignore[assignment]
    gemini_schema: dict[Any, Any] = {
        "type": schema_without_refs["type"],
        "properties": schema_without_refs["properties"],
        "required": (
            schema_without_refs["required"] if "required" in schema_without_refs else []
        ),  # TODO: Temporary Fix for Iterables which throw an error when their tasks field is specified in the required field
    }
    return gemini_schema


def _create_vertexai_tool(
    models: type[BaseModel] | list[type[BaseModel]] | Any,
) -> gm.Tool:  # noqa: UP007
    """Creates a tool with function declarations for single model or list of models"""
    # Handle Iterable case first
    if get_origin(models) is not None:
        model_list = list(get_types_array(models))
    else:
        # Handle both single model and list of models
        model_list = models if isinstance(models, list) else [models]

    declarations = []
    for model in model_list:
        parameters = _create_gemini_json_schema(model)
        declaration = gm.FunctionDeclaration(
            name=model.__name__,
            description=model.__doc__,
            parameters=parameters,
        )
        declarations.append(declaration)

    return gm.Tool(function_declarations=declarations)


def vertexai_message_parser(
    message: dict[str, str | gm.Part | list[str | gm.Part]],
) -> gm.Content:
    if isinstance(message["content"], str):
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=[gm.Part.from_text(message["content"])],
        )
    elif isinstance(message["content"], list):
        parts: list[gm.Part] = []
        for item in message["content"]:
            if isinstance(item, str):
                parts.append(gm.Part.from_text(item))
            elif isinstance(item, gm.Part):
                parts.append(item)
            else:
                raise ValueError(f"Unsupported content type in list: {type(item)}")
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=parts,
        )
    else:
        raise ValueError("Unsupported message content type")


def _vertexai_message_list_parser(
    messages: list[dict[str, str | gm.Part | list[str | gm.Part]]],
) -> list[gm.Content]:
    contents = [
        vertexai_message_parser(message) if isinstance(message, dict) else message
        for message in messages
    ]
    return contents


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


def vertexai_process_response(
    _kwargs: dict[str, Any],
    model: Union[type[BaseModel], list[type[BaseModel]], Any],  # noqa: UP007
):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)  # type: ignore[arg-type]

    tool = _create_vertexai_tool(models=model)

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def vertexai_process_json_response(_kwargs: dict[str, Any], model: type[BaseModel]):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)  # type: ignore[arg-type]

    config: dict[str, Any] | None = _kwargs.pop("generation_config", None)

    response_schema = _create_gemini_json_schema(model)

    generation_config = gm.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        **(config if config else {}),
    )

    return contents, generation_config


def from_vertexai(
    client: gm.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.VERTEXAI_TOOLS,
    _async: bool = False,
    use_async: bool | None = None,
    **kwargs: Any,
) -> instructor.Instructor:
    import warnings

    warnings.warn(
        "from_vertexai is deprecated and will be removed in a future version. "
        "Please use from_genai with vertexai=True or from_provider instead. "
        "Install google-genai with: pip install google-genai\n"
        "Example migration:\n"
        "  # Old way\n"
        "  from instructor import from_vertexai\n"
        "  import vertexai.generative_models as gm\n"
        "  client = from_vertexai(gm.GenerativeModel('gemini-3-flash'))\n\n"
        "  # New way\n"
        "  from instructor import from_genai\n"
        "  from google import genai\n"
        "  client = from_genai(genai.Client(vertexai=True, project='your-project', location='us-central1'))\n"
        "  # OR use from_provider\n"
        "  client = instructor.from_provider('vertexai/gemini-3-flash')",
        DeprecationWarning,
        stacklevel=2,
    )

    valid_modes = {
        instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
        instructor.Mode.VERTEXAI_TOOLS,
        instructor.Mode.VERTEXAI_JSON,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="VertexAI",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, gm.GenerativeModel):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of vertexai.generative_models.GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    if use_async is not None and _async != False:
        from ...core.exceptions import ConfigurationError

        raise ConfigurationError(
            "Cannot provide both '_async' and 'use_async'. Use 'use_async' instead."
        )

    if _async and use_async is None:
        import warnings

        warnings.warn(
            "'_async' is deprecated. Use 'use_async' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        use_async = _async

    is_async = use_async if use_async is not None else _async

    create = client.generate_content_async if is_async else client.generate_content

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.VERTEXAI,
        mode=mode,
        **kwargs,
    )
