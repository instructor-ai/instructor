from enum import Enum, auto
from typing import List, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass


class MessageRole(Enum):
    USER = auto()
    SYSTEM = auto()
    ASSISTANT = auto()


@dataclass
class Message:
    content: str = Field(default=None, repr=True)
    role: MessageRole = Field(default=MessageRole.USER, repr=False)
    name: Optional[str] = Field(default=None)

    def dict(self):
        assert self.content is not None, "Content must be set!"
        obj = {
            "role": self.role.name.lower(),
            "content": self.content,
        }
        if self.name and self.role == MessageRole.USER:
            obj["name"] = self.name
        return obj


@dataclass
class SystemMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.SYSTEM


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


@dataclass
class AssistantMessage(Message):
    def __post_init__(self):
        self.role = MessageRole.ASSISTANT


@dataclass
class SystemIdentity(Message):
    identity: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        if self.identity:
            self.content = f"You are a {self.identity.lower()}."
        else:
            self.content = "You are a world class, state of the art agent."


@dataclass
class SystemTask(Message):
    task: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        if self.task:
            self.content = (
                f"Your purpose is to correctly complete this task: `{self.task}`."
            )


@dataclass
class SystemStyle(Message):
    style: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        self.content = f"Your style when answering is {self.style.lower()}"


@dataclass
class SystemGuidelines(Message):
    guidelines: List[str] = Field(default_factory=list)
    header: str = (
        "These are the guidelines you need to follow when answering user queries:"
    )

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        guidelines = "\n* ".join(self.guidelines)
        self.content = f"{self.header}:\n\n* {guidelines}"


@dataclass
class SystemTips(Message):
    tips: List[str] = Field(default_factory=list)
    header: str = "Here are some tips to help you complete the task:"

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        tips = "\n* ".join(self.tips)
        self.content = f"{self.header}:\n\n* {tips}"


@dataclass
class TipsMessage(Message):
    tips: List[str] = Field(default_factory=list)
    header: str = "Here are some tips to help you complete the task"

    def __post_init__(self):
        self.role = MessageRole.USER
        tips = "\n* ".join(self.tips)
        self.content = f"{self.header}:\n\n* {tips}"


@dataclass
class ChainOfThought(Message):
    def __post_init__(self):
        self.role = MessageRole.ASSISTANT
        self.content = "Lets think step by step to get the correct answer:"
