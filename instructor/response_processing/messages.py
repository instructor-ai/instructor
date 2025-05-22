"""Message processing utilities for different providers."""

from typing import Any

from instructor.mode import Mode
from instructor.multimodal import convert_messages
from instructor.utils import (
    combine_system_messages,
    extract_system_messages,
)


class MessageHandler:
    """Handles message processing for different providers."""

    @staticmethod
    def process_messages(
        kwargs: dict[str, Any],
        mode: Mode,
        autodetect_images: bool = False,
    ) -> dict[str, Any]:
        """Process messages based on the mode.

        Args:
            kwargs: The kwargs containing messages
            mode: The processing mode
            autodetect_images: Whether to auto-detect images

        Returns:
            Updated kwargs with processed messages
        """
        if "messages" not in kwargs:
            return kwargs

        messages = convert_messages(
            kwargs["messages"],
            mode,
            autodetect_images=autodetect_images,
        )

        if mode in {Mode.ANTHROPIC_JSON, Mode.ANTHROPIC_TOOLS}:
            # Handle Anthropic style messages
            kwargs["messages"] = [m for m in messages if m["role"] != "system"]
            if "system" not in kwargs:
                system_message = extract_system_messages(messages)
                if system_message:
                    kwargs["system"] = system_message
        else:
            if mode in {
                Mode.RESPONSES_TOOLS,
                Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            } and kwargs.get("max_tokens"):
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

            kwargs["messages"] = messages

        return kwargs

    @staticmethod
    def extract_and_combine_system_messages(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract system messages from the messages list and combine with existing system.

        Args:
            kwargs: The kwargs to process

        Returns:
            Updated kwargs
        """
        system_messages = extract_system_messages(kwargs.get("messages", []))

        if system_messages:
            kwargs["system"] = combine_system_messages(
                kwargs.get("system"), system_messages
            )

        kwargs["messages"] = [
            m for m in kwargs.get("messages", []) if m["role"] != "system"
        ]

        return kwargs

    @staticmethod
    def prepare_bedrock_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Prepare messages for Bedrock format.

        Args:
            messages: Original messages

        Returns:
            Tuple of (system_messages, user_assistant_messages)
        """
        bedrock_system_list: list[dict[str, Any]] = []
        bedrock_user_assistant_messages_list: list[dict[str, Any]] = []

        for msg_dict in messages:
            if not isinstance(msg_dict, dict):
                bedrock_user_assistant_messages_list.append(msg_dict)
                continue

            current_message_for_api = msg_dict.copy()
            role = current_message_for_api.get("role")
            content = current_message_for_api.get("content")

            if role == "system":
                if isinstance(content, str):
                    bedrock_system_list.append({"text": content})
                else:
                    raise ValueError(
                        "System message content must be a string for Bedrock processing. "
                        f"Found type: {type(content)}."
                    )
            else:
                if "content" in current_message_for_api:
                    if isinstance(content, str):
                        current_message_for_api["content"] = [{"text": content}]
                    else:
                        raise NotImplementedError(
                            "Non-text prompts are not currently supported in the Bedrock provider."
                        )
                bedrock_user_assistant_messages_list.append(current_message_for_api)

        return bedrock_system_list, bedrock_user_assistant_messages_list

    @staticmethod
    def prepare_cohere_messages(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert messages to Cohere format.

        Args:
            kwargs: The kwargs containing messages

        Returns:
            Updated kwargs in Cohere format
        """
        messages = kwargs.pop("messages", [])
        chat_history = []

        for message in messages[:-1]:
            chat_history.append(
                {
                    "role": message["role"],
                    "message": message["content"],
                }
            )

        kwargs["message"] = messages[-1]["content"]
        kwargs["chat_history"] = chat_history

        if "model_name" in kwargs and "model" not in kwargs:
            kwargs["model"] = kwargs.pop("model_name")

        kwargs.pop("strict", None)
        return kwargs
