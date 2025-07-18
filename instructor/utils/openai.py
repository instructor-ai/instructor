"""OpenAI-specific utilities.

This module contains utilities specific to the OpenAI provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from openai import pydantic_function_tool

from instructor.dsl.parallel import ParallelModel, handle_parallel_model
from instructor.exceptions import ConfigurationError
from instructor.mode import Mode
from instructor.utils.core import dump_message, merge_consecutive_messages
from instructor.schema_utils import generate_openai_schema


def reask_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
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
    if new_kwargs.get("stream", False):
        raise ConfigurationError(
            "stream=True is not supported when using PARALLEL_TOOLS mode"
        )
    new_kwargs["tools"] = handle_parallel_model(response_model)
    new_kwargs["tool_choice"] = "auto"
    return ParallelModel(typehint=response_model), new_kwargs


def handle_functions(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    Mode.warn_mode_functions_deprecation()
    new_kwargs["functions"] = [generate_openai_schema(response_model)]
    new_kwargs["function_call"] = {
        "name": generate_openai_schema(response_model)["name"]
    }
    return response_model, new_kwargs


def handle_tools_strict(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    response_model_schema = pydantic_function_tool(response_model)
    response_model_schema["function"]["strict"] = True
    new_kwargs["tools"] = [response_model_schema]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": response_model_schema["function"]["name"]},
    }
    return response_model, new_kwargs


def handle_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
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
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
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
    if new_kwargs.get("max_tokens") is not None:
        new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

    return response_model, new_kwargs


def handle_responses_tools_with_inbuilt_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
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

    if new_kwargs.get("max_tokens") is not None:
        new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

    return response_model, new_kwargs


def handle_json_o1(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    roles = [message["role"] for message in new_kwargs.get("messages", [])]
    if "system" in roles:
        raise ValueError("System messages are not supported For the O1 models")

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
    response_model: type[Any], new_kwargs: dict[str, Any], mode: Mode
) -> tuple[type[Any], dict[str, Any]]:
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
