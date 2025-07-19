"""AWS Bedrock-specific utilities.

This module contains utilities specific to the AWS Bedrock provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from ...mode import Mode


def generate_bedrock_schema(response_model: type[Any]) -> dict[str, Any]:
    """
    Generate Bedrock tool schema from a Pydantic model.

    Bedrock Converse API expects tools in this format:
    {
        "toolSpec": {
            "name": "tool_name",
            "description": "tool description",
            "inputSchema": {
                "json": { JSON Schema }
            }
        }
    }
    """
    schema = response_model.model_json_schema()

    return {
        "toolSpec": {
            "name": response_model.__name__,
            "description": response_model.__doc__
            or f"Correctly extracted `{response_model.__name__}` with all the required parameters with correct types",
            "inputSchema": {"json": schema},
        }
    }


def reask_bedrock_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Bedrock JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
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


def reask_bedrock_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Bedrock tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (assistant message with tool use, then user message with tool result error)
    """
    kwargs = kwargs.copy()

    # Add the assistant's response message
    assistant_message = response["output"]["message"]
    reask_msgs = [assistant_message]

    # Find the tool use ID from the assistant's response to reference in the error
    tool_use_id = None
    if "content" in assistant_message:
        for content_block in assistant_message["content"]:
            if "toolUse" in content_block:
                tool_use_id = content_block["toolUse"]["toolUseId"]
                break

    # Add a user message with tool result indicating validation error
    if tool_use_id:
        reask_msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [
                                {
                                    "text": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                                }
                            ],
                            "status": "error",
                        }
                    }
                ],
            }
        )
    else:
        # Fallback if no tool use ID found
        reask_msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors"
                    }
                ],
            }
        )

    kwargs["messages"].extend(reask_msgs)
    return kwargs


def _prepare_bedrock_converse_kwargs_internal(
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Prepare kwargs for the Bedrock Converse API.

    Kwargs modifications:
    - Moves: system list to messages as a system role
    - Renames: "model" -> "modelId"
    - Collects: temperature, max_tokens, top_p, stop into inferenceConfig
    - Converts: messages content to Bedrock format
    """
    # Handle Bedrock-native system parameter format: system=[{'text': '...'}]
    # Convert to OpenAI format by adding to messages as system role
    if "system" in call_kwargs and isinstance(call_kwargs["system"], list):
        system_content = call_kwargs.pop("system")
        if (
            system_content
            and isinstance(system_content[0], dict)
            and "text" in system_content[0]
        ):
            # Convert system=[{'text': '...'}] to OpenAI format
            system_text = system_content[0]["text"]
            if "messages" not in call_kwargs:
                call_kwargs["messages"] = []
            # Insert system message at beginning
            call_kwargs["messages"].insert(
                0, {"role": "system", "content": system_text}
            )

    # Bedrock expects 'modelId' over 'model'
    if "model" in call_kwargs and "modelId" not in call_kwargs:
        call_kwargs["modelId"] = call_kwargs.pop("model")

    # Prepare inferenceConfig for parameters like temperature, maxTokens, etc.
    inference_config_params = {}

    # Temperature
    if "temperature" in call_kwargs:
        inference_config_params["temperature"] = call_kwargs.pop("temperature")

    # Max Tokens (OpenAI uses max_tokens)
    if "max_tokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("max_tokens")
    elif "maxTokens" in call_kwargs:  # If Bedrock-style maxTokens is already top-level
        inference_config_params["maxTokens"] = call_kwargs.pop("maxTokens")

    # Top P (OpenAI uses top_p)
    if "top_p" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("top_p")
    elif "topP" in call_kwargs:  # If Bedrock-style topP is already top-level
        inference_config_params["topP"] = call_kwargs.pop("topP")

    # Stop Sequences (OpenAI uses 'stop')
    # Bedrock 'Converse' API expects 'stopSequences'
    if "stop" in call_kwargs:
        stop_val = call_kwargs.pop("stop")
        if isinstance(stop_val, str):
            inference_config_params["stopSequences"] = [stop_val]
        elif isinstance(stop_val, list):
            inference_config_params["stopSequences"] = stop_val
    elif "stop_sequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stop_sequences")
    elif (
        "stopSequences" in call_kwargs
    ):  # If Bedrock-style stopSequences is already top-level
        inference_config_params["stopSequences"] = call_kwargs.pop("stopSequences")

    # If any inference parameters were collected, add them to inferenceConfig
    # Merge with existing inferenceConfig if user provided one.
    # User-provided inferenceConfig keys take precedence over top-level params if conflicts.
    if inference_config_params:
        if "inferenceConfig" in call_kwargs:
            # Merge, giving precedence to what's already in call_kwargs["inferenceConfig"]
            # This could be more sophisticated, but for now, if inferenceConfig is set, assume it's intentional.
            existing_inference_config = call_kwargs["inferenceConfig"]
            for key, value in inference_config_params.items():
                if key not in existing_inference_config:
                    existing_inference_config[key] = value
        else:
            call_kwargs["inferenceConfig"] = inference_config_params

    # Process messages for Bedrock: separate system prompts and format text content.
    if "messages" in call_kwargs and isinstance(call_kwargs["messages"], list):
        original_input_messages = call_kwargs.pop("messages")

        bedrock_system_list: list[dict[str, Any]] = []
        bedrock_user_assistant_messages_list: list[dict[str, Any]] = []

        for msg_dict in original_input_messages:
            if not isinstance(msg_dict, dict):
                # If an item in the messages list is not a dictionary,
                # pass it through to the user/assistant messages list as is.
                # This allows non-standard message items to be handled by subsequent Boto3 validation
                # or if they represent something other than standard role/content messages.
                bedrock_user_assistant_messages_list.append(msg_dict)
                continue

            # Make a copy to avoid modifying the original dict if it's part of a larger structure
            # or if the original list/dicts are expected to remain unchanged by the caller.
            current_message_for_api = msg_dict.copy()
            role = current_message_for_api.get("role")
            content = current_message_for_api.get(
                "content"
            )  # content can be None or other types

            if role == "system":
                if isinstance(content, str):
                    bedrock_system_list.append({"text": content})
                else:  # System message content is not a string (could be None, list, int, etc.)
                    raise ValueError(
                        "System message content must be a string for Bedrock processing by this handler. "
                        f"Found type: {type(content)}."
                    )
            else:  # For user, assistant, or other roles that go into Bedrock's 'messages' list
                if "content" in current_message_for_api:
                    if isinstance(content, str):
                        current_message_for_api["content"] = [{"text": content}]
                    elif (
                        isinstance(content, list)
                        and content
                        and isinstance(content[0], dict)
                        and "text" in content[0]
                    ):
                        # Handle Bedrock-native content format: [{'text': "..."}]
                        current_message_for_api["content"] = content
                    else:  # Content is not a string or supported list format (e.g., None, int, unsupported list).
                        # This matches the original behavior which raised for any non-string content.
                        raise NotImplementedError(
                            "Non-text prompts are not currently supported in the Bedrock provider."
                        )
                # If 'content' key is not in current_message_for_api, message is added as is (e.g. for tool calls without content)
                bedrock_user_assistant_messages_list.append(current_message_for_api)

        if bedrock_system_list:
            call_kwargs["system"] = bedrock_system_list

        # Always re-assign the 'messages' key with the processed list.
        # If original_input_messages was empty or only contained system messages that were extracted,
        # bedrock_user_assistant_messages_list will be empty, correctly resulting in `messages: []`.
        call_kwargs["messages"] = bedrock_user_assistant_messages_list
    return call_kwargs


def handle_bedrock_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Bedrock JSON mode.

    Kwargs modifications:
    - Adds: "response_format" with json_schema
    - Adds/Modifies: "system" (prepends JSON instructions)
    - Applies: _prepare_bedrock_converse_kwargs_internal transformations
    """
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)
    json_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        and don't include any other text in the response apart from the json
        """
    )
    system_message = new_kwargs.pop("system", None)
    if not system_message:
        new_kwargs["system"] = [{"text": json_message}]
    else:
        if not isinstance(system_message, list):
            raise ValueError(
                """system must be a list of SystemMessage, refer to:
                https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
                """
            )
        system_message.append({"text": json_message})
        new_kwargs["system"] = system_message

    return response_model, new_kwargs


def handle_bedrock_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """
    Handle Bedrock tools mode.

    Kwargs modifications:
    - When response_model is None: Only applies _prepare_bedrock_converse_kwargs_internal transformations
    - When response_model is provided:
      - Adds: "toolConfig" with tools list and toolChoice configuration
      - Applies: _prepare_bedrock_converse_kwargs_internal transformations
    """
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)

    if response_model is None:
        return None, new_kwargs

    # Generate Bedrock tool schema
    tool_schema = generate_bedrock_schema(response_model)

    # Set up tools configuration for Bedrock Converse API
    new_kwargs["toolConfig"] = {
        "tools": [tool_schema],
        "toolChoice": {"tool": {"name": response_model.__name__}},
    }

    return response_model, new_kwargs


# Handler registry for Bedrock
BEDROCK_HANDLERS = {
    Mode.BEDROCK_JSON: {
        "reask": reask_bedrock_json,
        "response": handle_bedrock_json,
    },
    Mode.BEDROCK_TOOLS: {
        "reask": reask_bedrock_tools,
        "response": handle_bedrock_tools,
    },
}
