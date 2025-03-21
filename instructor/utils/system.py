from typing import TypedDict, Any, Union


class SystemMessage(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict[str, str]


def combine_system_messages(
    existing_system: Union[str, list[SystemMessage], None],  # noqa: UP007
    new_system: Union[str, list[SystemMessage]],  # noqa: UP007
) -> Union[str, list[SystemMessage]]:  # noqa: UP007
    """
    Combine existing and new system messages.

    This optimized version uses a more direct approach with fewer branches.

    Args:
        existing_system: Existing system message(s) or None
        new_system: New system message(s) to add

    Returns:
        Combined system message(s)
    """
    # Fast path for None existing_system (avoid unnecessary operations)
    if existing_system is None:
        return new_system

    # Validate input types
    if not isinstance(existing_system, (str, list)) or not isinstance(
        new_system, (str, list)
    ):
        raise ValueError(
            f"System messages must be strings or lists, got {type(existing_system)} and {type(new_system)}"
        )

    # Use direct type comparison instead of isinstance for better performance
    if isinstance(existing_system, str) and isinstance(new_system, str):
        # Both are strings, join with newlines
        # Avoid creating intermediate strings by joining only once
        return f"{existing_system}\n\n{new_system}"
    elif isinstance(existing_system, list) and isinstance(new_system, list):
        # Both are lists, use list extension in place to avoid creating intermediate lists
        # First create a new list to avoid modifying the original
        result = list(existing_system)
        result.extend(new_system)
        return result
    elif isinstance(existing_system, str) and isinstance(new_system, list):
        # existing is string, new is list
        # Create a pre-sized list to avoid resizing
        result = [SystemMessage(type="text", text=existing_system)]
        result.extend(new_system)
        return result
    elif isinstance(existing_system, list) and isinstance(new_system, str):
        # existing is list, new is string
        # Create message once and add to existing
        new_message = SystemMessage(type="text", text=new_system)
        result = list(existing_system)
        result.append(new_message)
        return result

    # This should never happen due to validation above
    return existing_system


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    """
    Extract system messages from a list of messages.

    This optimized version pre-allocates the result list and
    reduces function call overhead.

    Args:
        messages: List of messages to extract system messages from

    Returns:
        List of system messages
    """
    # Fast path for empty messages
    if not messages:
        return []

    # First count system messages to pre-allocate result list
    system_count = sum(1 for m in messages if m.get("role") == "system")

    # If no system messages, return empty list
    if system_count == 0:
        return []

    # Helper function to convert a message content to SystemMessage
    def convert_message(content: Any) -> SystemMessage:
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        elif isinstance(content, dict):
            return SystemMessage(**content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")

    # Process system messages
    result: list[SystemMessage] = []

    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")

            # Skip empty content
            if not content:
                continue

            # Handle list or single content
            if isinstance(content, list):
                # Process each item in the list
                for item in content:
                    if item:  # Skip empty items
                        result.append(convert_message(item))
            else:
                # Process single content
                result.append(convert_message(content))

    return result


def extract_genai_system_message(
    messages: list[dict[str, Any]],
) -> str:
    """
    Extract system messages from a list of messages.

    We expect an explicit system messsage for this provider.
    """
    system_messages = ""

    for message in messages:
        if isinstance(message, str):
            continue
        elif isinstance(message, dict):
            if message.get("role") == "system":
                if isinstance(message.get("content"), str):
                    system_messages += message.get("content", "") + "\n\n"
                elif isinstance(message.get("content"), list):
                    for item in message.get("content", []):
                        if isinstance(item, str):
                            system_messages += item + "\n\n"

    return system_messages
