from typing import List

from .base import Message, MessageRole
from pydantic.dataclasses import dataclass


def SystemIdentity(identity: str) -> Message:
    """
    Create a system message that tells the user what their identity is.

    Parameters:
        identity (str): The identity of the user.

    Returns:
        message (Message): A system message that tells the user what their identity is.
    """
    return Message(content=f"You are a {identity.lower()}.", role=MessageRole.SYSTEM)


def SystemTask(task: str) -> Message:
    """
    Create a system message that tells the user what task they are doing, uses language to
    push the system to behave as a world class algorithm.

    Parameters:
        task (str): The task the user is doing.

    Returns:
        message (Message): A system message that tells the user what task they are doing.
    """
    return Message(
        content=f"You are a world class state of the art algorithm capable of correctly completing the following task: `{task}`.",
        role=MessageRole.SYSTEM,
    )


def SystemStyle(style: str) -> Message:
    """
    Create a system message that tells the user what style they are responding in.

    Parameters:
        style (str): The style the user is responding in.

    Returns:
        message (Message): A system message that tells the user what style they are responding in.
    """
    return Message(
        content=f"You must respond with in following style: {style.lower()}.",
        role=MessageRole.SYSTEM,
    )


def SystemMessage(content: str) -> Message:
    """
    Create a system message.

    Parameters:
        content (str): The content of the message.

    Returns:
        message (Message): A system message."""
    return Message(content=content, role=MessageRole.SYSTEM)


def SystemGuidelines(guidelines: List[str]) -> Message:
    """
    Create a system message that tells the user what guidelines they must follow when responding.

    Parameters:
        guidelines (List[str]): The guidelines the user must follow when responding.

    Returns:
        message (Message): A system message that tells the user what guidelines they must follow when responding.
    """
    guideline_str = "\n* ".join(guidelines)
    return Message(
        content=f"Here are the guidelines you must to follow when responding:\n\n* {guideline_str}",
        role=MessageRole.SYSTEM,
    )


def SystemTips(tips: List[str]) -> Message:
    """
    Create a system message that gives the user some tips before responding.

    Parameters:
        tips (List[str]): The tips the user should follow when responding.

    Returns:
        message (Message): A system message that gives the user some tips before responding.
    """
    tips_str = "\n* ".join(tips)
    return Message(
        content=f"Here are some tips before responding:\n\n* {tips_str}",
        role=MessageRole.SYSTEM,
    )


@dataclass
class ChainOfThought(Message):
    """
    Special message type to correctly leverage chain of thought reasoning
    for the task. This is automatically set as the last message.
    """

    def __post_init__(self):
        self.content = "Lets think step by step to get the correct answer:"
        self.role = MessageRole.ASSISTANT
