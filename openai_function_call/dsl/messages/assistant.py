from pydantic.dataclasses import dataclass

from .base import Message, MessageRole


@dataclass
class AssistantMessage(Message):
    role: MessageRole = MessageRole.ASSISTANT


@dataclass
class ChainOfThought(AssistantMessage):
    def __post_init__(self):
        self.content = "Lets think step by step to get the correct answer:"
