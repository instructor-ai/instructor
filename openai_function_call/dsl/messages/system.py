from typing import List

from pydantic import Field
from pydantic.dataclasses import dataclass

from .base import Message, MessageRole


@dataclass
class SystemMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.SYSTEM


@dataclass
class Identity(Message):
    identity: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        if self.identity:
            self.content = f"You are a {self.identity.lower()}."
        else:
            self.content = "You are a world class, state of the art agent."


@dataclass
class Task(Message):
    task: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        if self.task:
            self.content = (
                f"Your purpose is to correctly complete this task: `{self.task}`."
            )


@dataclass
class Style(Message):
    style: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        self.content = f"Your style when answering is {self.style.lower()}"


@dataclass
class Guidelines(Message):
    guidelines: List[str] = Field(default_factory=list)
    header: str = (
        "These are the guidelines you need to follow when answering user queries:"
    )

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        guidelines = "\n* ".join(self.guidelines)
        self.content = f"{self.header}:\n\n* {guidelines}"


@dataclass
class Tips(Message):
    tips: List[str] = Field(default_factory=list)
    header: str = "Here are some tips to help you complete the task:"

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        tips = "\n* ".join(self.tips)
        self.content = f"{self.header}:\n\n* {tips}"
