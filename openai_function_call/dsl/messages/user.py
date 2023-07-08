from typing import List

from .base import Message, MessageRole


def TipsMessage(
    tips: List[str], header: str = "Here are some tips to help you complete the task"
) -> Message:
    """
    Create a system message that gives the user tips to help them complete the task.

    Parameters:
        tips (List[str]): A list of tips to help the user complete the task.
        header (str): The header of the message.

    Returns:
        message (Message): A user message that gives the user tips to help them complete the
    """
    tips_str = "\n* ".join(tips)
    return Message(
        content=f"{header}:\n\n* {tips_str}",
        role=MessageRole.USER,
    )


def UserMessage(content: str) -> Message:
    """
    Create a user message.

    Parameters:
        content (str): The content of the message.

    Returns:
        message (Message): A user message.
    """
    return Message(content=content, role=MessageRole.USER)


def TaggedMessage(
    content: str, tag: str = "data", header: str = "Consider the following data:"
) -> Message:
    """
    Create a user message.

    Parameters:
        content (str): The content of the message.
        tag (str): The tag to use, will show up as <tag>content</tag>.
        header (str): The header to reference the data

    Returns:
        message (Message): A user message with the data tagged.
    """
    content = f"{header}\n\n<{tag}>{content}</{tag}>"
    return Message(content=content, role=MessageRole.USER)
