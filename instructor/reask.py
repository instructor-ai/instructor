from __future__ import annotations

import logging
from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.utils import dump_message
from pydantic import BaseModel, ValidationError
from typing_extensions import ParamSpec

logger = logging.getLogger("instructor")

T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def reask_anthropic_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    from anthropic.types import Message

    assert isinstance(response, Message), "Response must be a Anthropic Message"

    assistant_content = []
    tool_use_id = None
    for content in response.content:
        assistant_content.append(content.model_dump())  # type: ignore
        if (
            content.type == "tool_use"
            and isinstance(exception, ValidationError)
            and content.name == exception.title
        ):
            tool_use_id = content.id

    reask_msgs = [{"role": "assistant", "content": assistant_content}]  # type: ignore
    if tool_use_id is not None:
        reask_msgs.append(  # type: ignore
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
                        "is_error": True,
                    }
                ],
            }
        )
    else:
        reask_msgs.append(  # type: ignore
            {
                "role": "user",
                "content": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors",
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_anthropic_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    from anthropic.types import Message

    assert isinstance(response, Message), "Response must be a Anthropic Message"

    reask_msg = {
        "role": "user",
        "content": f"""Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{response.content[0].text}""",  # type: ignore
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def reask_cohere_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Cohere
    exception: Exception,
):
    # Get message outside the function
    message = kwargs.get("message", "")

    # Fetch or initialize chat_history in one operation
    if "chat_history" in kwargs:
        # Only modify chat_history if it exists
        kwargs["chat_history"].append({"role": "user", "message": message})
    else:
        # Create a new chat_history if it doesn't exist
        kwargs["chat_history"] = [{"role": "user", "message": message}]

    # Set the message directly without string concatenation with f-strings
    kwargs["message"] = (
        "Correct the following JSON response, based on the errors given below:\n\n"
        f"JSON:\n{response.text}\n\nExceptions:\n{exception}"
    )
    return kwargs


def reask_gemini_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Gemini
    exception: Exception,
):
    from google.ai import generativelanguage as glm  # type: ignore

    reask_msgs = [
        {
            "role": "model",
            "parts": [
                glm.FunctionCall(
                    name=response.parts[0].function_call.name,
                    args=response.parts[0].function_call.args,
                )
            ],
        },
        {
            "role": "function",
            "parts": [
                glm.Part(
                    function_response=glm.FunctionResponse(
                        name=response.parts[0].function_call.name,
                        response={"error": f"Validation Error(s) found:\n{exception}"},
                    )
                ),
            ],
        },
        {
            "role": "user",
            "parts": ["Recall the function arguments correctly and fix the errors"],
        },
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_gemini_json(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Gemini
    exception: Exception,
):
    kwargs["contents"].append(
        {
            "role": "user",
            "parts": [
                f"Correct the following JSON response, based on the errors given below:\n\n"
                f"JSON:\n{response.text}\n\nExceptions:\n{exception}"
            ],
        }
    )
    return kwargs


def reask_vertexai_tools(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Vertex AI
    exception: Exception,
):
    from .client_vertexai import vertexai_function_response_parser

    kwargs = kwargs.copy()
    reask_msgs = [
        response.candidates[0].content,
        vertexai_function_response_parser(response, exception),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


def reask_vertexai_json(
    kwargs: dict[str, Any],
    response: Any,  # Replace with actual response type for Vertex AI
    exception: Exception,
):
    from .client_vertexai import vertexai_message_parser

    kwargs = kwargs.copy()

    reask_msgs = [
        response.candidates[0].content,
        vertexai_message_parser(
            {
                "role": "user",
                "content": (
                    f"Validation Errors found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors found in the following attempt:\n{response.text}"
                ),
            }
        ),
    ]
    kwargs["contents"].extend(reask_msgs)
    return kwargs


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


def reask_cerebras_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, "
                    f"fix the errors and call the tool {tool_call.function.name} again, "
                    f"taking into account the problems with {tool_call.function.arguments} that was previously generated."
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
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


def reask_bedrock_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [response["output"]["message"]]
    reask_msgs.append(
        {
            "role": "user",
            "content": [
                {
                    "text": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}"
                },
            ],
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


def reask_fireworks_tools(kwargs: dict[str, Any], response: Any, exception: Exception):
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


def reask_fireworks_json(
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


def reask_writer_tools(
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
                f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_perplexity_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reasking for Perplexity JSON mode."""
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

  
def reask_genai_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    from google.genai import types

    kwargs = kwargs.copy()
    function_call = response.candidates[0].content.parts[0].function_call
    kwargs["contents"].append(
        types.ModelContent(
            parts=[
                types.Part.from_function_call(
                    name=function_call.name,
                    args=function_call.args,
                ),
                types.Part.from_text(
                    text=f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            ]
        ),
    )
    return kwargs


def reask_genai_structured_outputs(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    from google.genai import types

    kwargs = kwargs.copy()
    kwargs["contents"].append(
        types.ModelContent(
            parts=[
                types.Part.from_text(
                    text=f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors in the following attempt:\n{response.text}"
                ),
            ]
        ),
    )
    return kwargs  

def reask_mistral_structured_outputs(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    kwargs = kwargs.copy()
    reask_msgs = [
        {
            "role": "assistant",
            "content": response.choices[0].message.content,
        }
    ]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_mistral_tools(
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



def handle_reask_kwargs(
    kwargs: dict[str, Any],
    mode: Mode,
    response: Any,  # Replace with actual response type based on the mode
    exception: Exception,
):
    # Create a shallow copy of kwargs to avoid modifying the original
    kwargs_copy = kwargs.copy()

    # Use a more efficient mapping approach with mode groupings to reduce lookup time
    # Group similar modes that use the same reask function
    if mode in {Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS}:
        return reask_anthropic_tools(kwargs_copy, response, exception)
    elif mode == Mode.ANTHROPIC_JSON:
        return reask_anthropic_json(kwargs_copy, response, exception)
    elif mode in {Mode.COHERE_TOOLS, Mode.COHERE_JSON_SCHEMA}:
        return reask_cohere_tools(kwargs_copy, response, exception)
    elif mode == Mode.GEMINI_TOOLS:
        return reask_gemini_tools(kwargs_copy, response, exception)
    elif mode == Mode.GEMINI_JSON:
        return reask_gemini_json(kwargs_copy, response, exception)
    elif mode == Mode.VERTEXAI_TOOLS:
        return reask_vertexai_tools(kwargs_copy, response, exception)
    elif mode == Mode.VERTEXAI_JSON:
        return reask_vertexai_json(kwargs_copy, response, exception)
    elif mode in {Mode.TOOLS, Mode.TOOLS_STRICT}:
        return reask_tools(kwargs_copy, response, exception)
    elif mode == Mode.CEREBRAS_TOOLS:
        return reask_cerebras_tools(kwargs_copy, response, exception)
    elif mode == Mode.MD_JSON:
        return reask_md_json(kwargs_copy, response, exception)
    elif mode == Mode.FIREWORKS_TOOLS:
        return reask_fireworks_tools(kwargs_copy, response, exception)
    elif mode == Mode.FIREWORKS_JSON:
        return reask_fireworks_json(kwargs_copy, response, exception)
    elif mode == Mode.WRITER_TOOLS:
        return reask_writer_tools(kwargs_copy, response, exception)
    elif mode == Mode.BEDROCK_JSON:
        return reask_bedrock_json(kwargs_copy, response, exception)
    elif mode == Mode.PERPLEXITY_JSON:
        return reask_perplexity_json(kwargs_copy, response, exception)
    elif mode == Mode.GENAI_TOOLS:
        return reask_genai_tools(kwargs_copy, response, exception)
    elif mode == Mode.GENAI_STRUCTURED_OUTPUTS:
        return reask_genai_structured_outputs(kwargs_copy, response, exception)
    elif mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
        return reask_mistral_structured_outputs(kwargs_copy, response, exception)
    elif mode == Mode.MISTRAL_TOOLS:
        return reask_mistral_tools(kwargs_copy, response, exception)
    else:
        return reask_default(kwargs_copy, response, exception)
