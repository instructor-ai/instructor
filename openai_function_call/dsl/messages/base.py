from enum import Enum, auto
from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass


class MessageRole(Enum):
    """
    An enum that represents the role of a message.

    Attributes:
        USER: A message from the user.
        SYSTEM: A message from the system.
        ASSISTANT: A message from the assistant.
    """

    USER = auto()
    SYSTEM = auto()
    ASSISTANT = auto()


@dataclass
class Message:
    """
    A message class that helps build messages for the chat interface.

    Attributes:
        content (str): The content of the message.
        role (MessageRole): The role of the message.
        name (Optional[str]): The name of the user, only used if the role is USER.

    Tips:
        If you want to make custom messages simple make a function that returns the `Message` class and use that as part of your pipes. For example if you want to add additional context:

        ```python
        def GetUserData(user_id) -> Message:
            data = ...
            return Message(
                content="This is some more user data: {data} for {user_id}
                role=MessageRole.USER
            )
        ```
    """

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
