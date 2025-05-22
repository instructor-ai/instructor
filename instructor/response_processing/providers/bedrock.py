"""AWS Bedrock-specific response handlers."""

import json
from textwrap import dedent
from typing import Any, TypeVar

from instructor.mode import Mode
from instructor.response_processing.base import BaseHandler
from instructor.response_processing.messages import MessageHandler
from instructor.response_processing.registry import handler_registry

T = TypeVar("T")


def _prepare_bedrock_converse_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Minimal processing to support `converse` parameters for the Bedrock client."""
    # Bedrock expects 'modelId' over 'model'
    if "model" in call_kwargs and "modelId" not in call_kwargs:
        call_kwargs["modelId"] = call_kwargs.pop("model")

    # Prepare inferenceConfig for parameters like temperature, maxTokens, etc.
    inference_config_params = {}

    # Temperature
    if "temperature" in call_kwargs:
        inference_config_params["temperature"] = call_kwargs.pop("temperature")

    # Max Tokens
    if "max_tokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("max_tokens")
    elif "maxTokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("maxTokens")

    # Top P
    if "top_p" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("top_p")
    elif "topP" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("topP")

    # Stop Sequences
    if "stop" in call_kwargs:
        stop_val = call_kwargs.pop("stop")
        if isinstance(stop_val, str):
            inference_config_params["stopSequences"] = [stop_val]
        elif isinstance(stop_val, list):
            inference_config_params["stopSequences"] = stop_val
    elif "stop_sequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stop_sequences")
    elif "stopSequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stopSequences")

    # Add inference config if any parameters were collected
    if inference_config_params:
        if "inferenceConfig" in call_kwargs:
            existing_inference_config = call_kwargs["inferenceConfig"]
            for key, value in inference_config_params.items():
                if key not in existing_inference_config:
                    existing_inference_config[key] = value
        else:
            call_kwargs["inferenceConfig"] = inference_config_params

    # Process messages for Bedrock
    if "messages" in call_kwargs and isinstance(call_kwargs["messages"], list):
        bedrock_system_list, bedrock_user_assistant_messages_list = (
            MessageHandler.prepare_bedrock_messages(call_kwargs.pop("messages"))
        )

        if bedrock_system_list:
            call_kwargs["system"] = bedrock_system_list

        call_kwargs["messages"] = bedrock_user_assistant_messages_list

    return call_kwargs


class BedrockJSONHandler(BaseHandler):
    """Handler for Bedrock JSON mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Bedrock JSON mode."""
        kwargs = _prepare_bedrock_converse_kwargs(kwargs)
        json_message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n\n

            {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            and don't include any other text in the response apart from the json
            """
        )
        system_message = kwargs.pop("system", None)
        if not system_message:
            kwargs["system"] = [{"text": json_message}]
        else:
            if not isinstance(system_message, list):
                raise ValueError(
                    """system must be a list of SystemMessage, refer to:
                    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
                    """
                )
            system_message.append({"text": json_message})
            kwargs["system"] = system_message

        return response_model, kwargs


class BedrockToolsHandler(BaseHandler):
    """Handler for Bedrock tools mode."""

    def handle(
        self, response_model: type[T], kwargs: dict[str, Any]
    ) -> tuple[type[T], dict[str, Any]]:
        """Handle Bedrock tools mode."""
        kwargs = _prepare_bedrock_converse_kwargs(kwargs)
        return response_model, kwargs


def register_bedrock_handlers() -> None:
    """Register all Bedrock handlers."""
    handler_registry.register(Mode.BEDROCK_JSON, BedrockJSONHandler())
    handler_registry.register(Mode.BEDROCK_TOOLS, BedrockToolsHandler())
