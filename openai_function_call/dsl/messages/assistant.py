from pydantic.dataclasses import dataclass

from .base import Message, MessageRole


@dataclass
class AssistantMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.ASSISTANT


@dataclass
class ChainOfThought(Message):
    def __post_init__(self):
        self.role = MessageRole.ASSISTANT
        self.content = "Lets think step by step to get the correct answer:"
