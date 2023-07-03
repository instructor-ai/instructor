from typing import List, Optional, Type, Union
from pydantic import BaseModel, Field, create_model
import openai

from openai_function_call import OpenAISchema

from .messages import (
    Message,
    SystemMessage,
    ChainOfThought,
    ExpertSystem,
    TaggedMessage,
    TipsMessage,
)


class ChatCompletion(BaseModel):
    name: str
    model: str = Field(default="gpt3.5-turbo-0613")
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.1)
    stream: bool = Field(default=False)

    messages: List[Message] = Field(default_factory=list, repr=False)
    system_message: SystemMessage = Field(default=None, repr=False)
    cot_message: ChainOfThought = Field(default=None, repr=False)
    function: OpenAISchema = Field(default=None, repr=False)

    def __post_init__(self):
        assert self.stream == False, "Stream is not supported yet"

    def __or__(self, other: Union[Message, OpenAISchema]) -> "ChatCompletion":
        if isinstance(other, Message):
            if isinstance(other, SystemMessage):
                if self.system_message:
                    self.system_message.content += "\n\n" + other.content
                self.system_message = other

            if isinstance(other, ChainOfThought):
                if self.cot_message:
                    raise ValueError(
                        "Only one chain of thought message can be used per completion"
                    )
                self.cot_message = other
            self.messages.append(other)
        else:
            if self.function:
                raise ValueError(
                    "Only one function can be used per completion, wrap your tools into a single toolkit schema"
                )
            self.function = other

            assert self.model not in {
                "gpt3.5-turbo",
                "gpt4",
            }, "Only *-0613 models can currently use functions"
        return self

    @property
    def kwargs(self) -> dict:
        kwargs = {}

        messages = []

        if self.system_message:
            messages.append(self.system_message.dict())

        if self.messages:
            special_types = {
                SystemMessage,
                ChainOfThought,
            }
            messages += [
                message.dict()
                for message in self.messages
                if type(message) not in special_types
            ]

        if self.cot_message:
            messages.append(self.cot_message.dict())

        kwargs["messages"] = messages

        if self.function:
            kwargs["functions"] = [self.function.openai_schema]
            kwargs["function_call"] = {"name": self.function.openai_schema["name"]}

        kwargs["max_tokens"] = self.max_tokens
        kwargs["temperature"] = self.temperature
        kwargs["model"] = self.model
        return kwargs

    def create(self):
        kwargs = self.kwargs
        completion = openai.ChatCompletion.create(**kwargs)
        if self.function:
            return self.function.from_response(completion)

    async def acreate(self):
        kwargs = self.kwargs
        completion = openai.ChatCompletion.acreate(**kwargs)
        if self.function:
            return self.function.from_response(await completion)
        return await completion

