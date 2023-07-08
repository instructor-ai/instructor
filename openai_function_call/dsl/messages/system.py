from typing import List

from pydantic import Field
from pydantic.dataclasses import dataclass

from .base import Message, MessageRole


@dataclass
class SystemMessage(Message):
    role: MessageRole = MessageRole.SYSTEM


@dataclass
class SystemIdentity(SystemMessage):
    identity: str = Field(default=None, repr=True)

    def __post_init__(self):
        if self.identity:
            self.content = f"You are a {self.identity.lower()}."
        else:
            self.content = "You are a world class, state of the art agent."


@dataclass
class SystemTask(SystemMessage):
    task: str = Field(default=None, repr=True)

    def __post_init__(self):
        assert self.task is not None
        self.content = f"You are a world class state of the art algorithm capable of correctly completing the following task: `{self.task}`."


@dataclass
class SystemStyle(SystemMessage):
    style: str = Field(default=None, repr=True)

    def __post_init__(self):
        assert self.style is not None
        self.content = f"You must respond with in following style: {self.style.lower()}"


@dataclass
class SystemGuidelines(SystemMessage):
    guidelines: List[str] = Field(default_factory=list)
    header: str = "Here are the guidelines you must to follow when responding"

    def __post_init__(self):
        guidelines = "\n* ".join(self.guidelines)
        self.content = f"{self.header}:\n\n* {guidelines}"


@dataclass
class SystemTips(SystemMessage):
    tips: List[str] = Field(default_factory=list)
    header: str = "Here are some tips before responding"

    def __post_init__(self):
        tips = "\n* ".join(self.tips)
        self.content = f"{self.header}:\n\n* {tips}"
