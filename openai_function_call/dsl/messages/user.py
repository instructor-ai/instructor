from typing import List

from .base import Message, MessageRole
from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class TipsMessage(Message):
    tips: List[str] = Field(default_factory=list)
    header: str = "Here are some tips to help you complete the task"

    def __post_init__(self):
        self.role = MessageRole.USER
        tips = "\n* ".join(self.tips)
        self.content = f"{self.header}:\n\n* {tips}"


@dataclass
class UserMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.USER


@dataclass
class TaggedMessage(Message):
    tag: str = Field(default="data", repr=True)

    def __post_init__(self):
        self.role = MessageRole.USER
        self.content = f"<{self.tag}>{self.content}</{self.tag}>"
