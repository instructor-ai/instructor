"""OpenAI-specific utilities.

This module contains utilities specific to the OpenAI provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from openai import pydantic_function_tool

from ...dsl.parallel import ParallelModel, handle_parallel_model
from ...core.exceptions import ConfigurationError
from ...mode import Mode
from ...utils.core import dump_message, merge_consecutive_messages
from ...processing.schema import generate_openai_schema


def reask_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for OpenAI tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (tool response messages indicating validation errors)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",  # type: ignore
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_responses_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for OpenAI responses tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user messages with validation errors)
    """
    kwargs = kwargs.copy()

    reask_messages = []
    for tool_call in response.output:
        reask_messages.append(
            {
                "role": "user",  # type: ignore
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors with {tool_call.arguments}"
                ),
            }
        )

    kwargs["messages"].extend(reask_messages)
    return kwargs


def reask_md_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for OpenAI JSON modes when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_default(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for OpenAI default mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting function correction)
    """
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Recall the function correctly, fix the errors, exceptions found\n{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


# Response handlers
def handle_parallel_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle OpenAI parallel tools mode for concurrent function calls.

    This mode enables making multiple independent function calls in a single request,
    useful for batch processing or when you need to extract multiple structured outputs
    simultaneously. The response_model should be a list/iterable type or use the
    ParallelModel wrapper.

    Example usage:
        # Define models for parallel extraction
        class PersonInfo(BaseModel):
            name: str
            age: int

        class EventInfo(BaseModel):
            date: str
            location: str

        # Use with PARALLEL_TOOLS mode
        result = client.chat.completions.create(
            model="gpt-4",
            response_model=[PersonInfo, EventInfo],
            mode=instructor.Mode.PARALLEL_TOOLS,
            messages=[{"role": "user", "content": "Extract person and event info..."}]
        )

    Kwargs modifications:
    - Adds: "tools" (multiple function schemas from parallel model)
    - Adds: "tool_choice" ("auto" to allow model to choose which tools to call)
    - Validates: stream=False (streaming not supported in parallel mode)
    """
    if new_kwargs.get("stream", False):
        raise ConfigurationError(
            "stream=True is not supported when using PARALLEL_TOOLS mode"
        )
    new_kwargs["tools"] = handle_parallel_model(response_model)
    new_kwargs["tool_choice"] = "auto"
    return ParallelModel(typehint=response_model), new_kwargs


def handle_functions(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI functions mode (deprecated).

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "functions" (list with function schema)
      - Adds: "function_call" (forced function call)
    """
    Mode.warn_mode_functions_deprecation()

    if response_model is None:
        return None, new_kwargs

    new_kwargs["functions"] = [generate_openai_schema(response_model)]
    new_kwargs["function_call"] = {
        "name": generate_openai_schema(response_model)["name"]
    }
    return response_model, new_kwargs


def handle_tools_strict(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI strict tools mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "tools" (list with strict function schema)
      - Adds: "tool_choice" (forced function call)
    """
    if response_model is None:
        return None, new_kwargs

    response_model_schema = pydantic_function_tool(response_model)
    response_model_schema["function"]["strict"] = True
    new_kwargs["tools"] = [response_model_schema]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model_schema["function"]["name"]},
    }
    return response_model, new_kwargs


def handle_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI tools mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "tools" (list with function schema)
      - Adds: "tool_choice" (forced function call)
    """
    if response_model is None:
        return None, new_kwargs

    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": generate_openai_schema(response_model),
        }
    ]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": generate_openai_schema(response_model)["name"]},
    }
    return response_model, new_kwargs


def handle_responses_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI responses tools mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "tools" (list with function schema)
      - Adds: "tool_choice" (forced function call)
      - Adds: "max_output_tokens" (converted from max_tokens)
    """
    # Handle max_tokens to max_output_tokens conversion for RESPONSES_TOOLS modes
    if new_kwargs.get("max_tokens") is not None:
        new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

    # If response_model is None, just return without setting up tools
    if response_model is None:
        return None, new_kwargs

    schema = pydantic_function_tool(response_model)
    del schema["function"]["strict"]

    tool_definition = {
        "type": "function",
        "name": schema["function"]["name"],
        "parameters": schema["function"]["parameters"],
    }

    if "description" in schema["function"]:
        tool_definition["description"] = schema["function"]["description"]
    else:
        tool_definition["description"] = (
            f"Correctly extracted `{response_model.__name__}` with all "
            f"the required parameters with correct types"
        )

    new_kwargs["tools"] = [
        {
            "type": "function",
            "name": schema["function"]["name"],
            "parameters": schema["function"]["parameters"],
        }
    ]

    new_kwargs["tool_choice"] = {
        "type": "function",
        "name": generate_openai_schema(response_model)["name"],
    }

    return response_model, new_kwargs


def handle_responses_tools_with_inbuilt_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI responses tools with inbuilt tools mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Adds: "tools" (list with function schema)
      - Adds: "tool_choice" (forced function call)
      - Adds: "max_output_tokens" (converted from max_tokens)
    """
    # Handle max_tokens to max_output_tokens conversion for RESPONSES_TOOLS modes
    if new_kwargs.get("max_tokens") is not None:
        new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

    # If response_model is None, just return without setting up tools
    if response_model is None:
        return None, new_kwargs

    schema = pydantic_function_tool(response_model)
    del schema["function"]["strict"]

    tool_definition = {
        "type": "function",
        "name": schema["function"]["name"],
        "parameters": schema["function"]["parameters"],
    }

    if "description" in schema["function"]:
        tool_definition["description"] = schema["function"]["description"]
    else:
        tool_definition["description"] = (
            f"Correctly extracted `{response_model.__name__}` with all "
            f"the required parameters with correct types"
        )

    if not new_kwargs.get("tools"):
        new_kwargs["tools"] = [tool_definition]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "name": generate_openai_schema(response_model)["name"],
        }
    else:
        new_kwargs["tools"].append(tool_definition)

    return response_model, new_kwargs


def handle_json_o1(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI o1 JSON mode.

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Modifies: "messages" (appends user message with JSON schema)
      - Validates: No system messages allowed for O1 models
    """
    roles = [message["role"] for message in new_kwargs.get("messages", [])]
    if "system" in roles:
        raise ValueError("System messages are not supported For the O1 models")

    if response_model is None:
        return None, new_kwargs

    message = dedent(
        f"""
        Understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    new_kwargs["messages"].append(
        {
            "role": "user",
            "content": message,
        },
    )
    return response_model, new_kwargs


def handle_json_modes(
    response_model: type[Any] | None, new_kwargs: dict[str, Any], mode: Mode
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle OpenAI JSON modes (JSON, MD_JSON, JSON_SCHEMA).

    Kwargs modifications:
    - When response_model is None: No modifications
    - When response_model is provided:
      - Mode.JSON_SCHEMA: Adds "response_format" with json_schema
      - Mode.JSON: Adds "response_format" with type="json_object", modifies system message
      - Mode.MD_JSON: Appends user message for markdown JSON response
    """
    if response_model is None:
        return None, new_kwargs

    message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        """
    )

    if mode == Mode.JSON:
        new_kwargs["response_format"] = {"type": "json_object"}
    elif mode == Mode.JSON_SCHEMA:
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
            },
        }
    elif mode == Mode.MD_JSON:
        new_kwargs["messages"].append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        )
        new_kwargs["messages"] = merge_consecutive_messages(new_kwargs["messages"])

    if new_kwargs["messages"][0]["role"] != "system":
        new_kwargs["messages"].insert(
            0,
            {
                "role": "system",
                "content": message,
            },
        )
    elif isinstance(new_kwargs["messages"][0]["content"], str):
        new_kwargs["messages"][0]["content"] += f"\n\n{message}"
    elif isinstance(new_kwargs["messages"][0]["content"], list):
        new_kwargs["messages"][0]["content"][0]["text"] += f"\n\n{message}"
    else:
        raise ValueError(
            "Invalid message format, must be a string or a list of messages"
        )

    return response_model, new_kwargs


def handle_openrouter_structured_outputs(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle OpenRouter structured outputs mode.

    Kwargs modifications:
    - Adds: "response_format" (json_schema with strict mode enabled)
    """
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": schema,
            "strict": True,
        },
    }
    return response_model, new_kwargs


# Handler registry for OpenAI
OPENAI_HANDLERS = {
    Mode.TOOLS: {
        "reask": reask_tools,
        "response": handle_tools,
    },
    Mode.TOOLS_STRICT: {
        "reask": reask_tools,
        "response": handle_tools_strict,
    },
    Mode.FUNCTIONS: {
        "reask": reask_default,
        "response": handle_functions,
    },
    Mode.JSON: {
        "reask": reask_md_json,
        "response": lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON),
    },
    Mode.MD_JSON: {
        "reask": reask_md_json,
        "response": lambda rm, nk: handle_json_modes(rm, nk, Mode.MD_JSON),
    },
    Mode.JSON_SCHEMA: {
        "reask": reask_md_json,
        "response": lambda rm, nk: handle_json_modes(rm, nk, Mode.JSON_SCHEMA),
    },
    Mode.JSON_O1: {
        "reask": reask_md_json,
        "response": handle_json_o1,
    },
    Mode.PARALLEL_TOOLS: {
        "reask": reask_tools,
        "response": handle_parallel_tools,
    },
    Mode.RESPONSES_TOOLS: {
        "reask": reask_responses_tools,
        "response": handle_responses_tools,
    },
    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: {
        "reask": reask_responses_tools,
        "response": handle_responses_tools_with_inbuilt_tools,
    },
    Mode.OPENROUTER_STRUCTURED_OUTPUTS: {
        "reask": reask_md_json,
        "response": handle_openrouter_structured_outputs,
    },
}
