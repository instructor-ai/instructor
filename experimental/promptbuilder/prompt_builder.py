from typing import List, Optional, Type, Union
from enum import Enum, auto
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field, create_model
import openai

from openai_function_call import OpenAISchema


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
class ExpertSystem(Message):
    task: str = Field(default=None, repr=True)

    def __post_init__(self):
        self.role = MessageRole.SYSTEM
        self.content = f"You are a world class, state of the art agent capable of correctly completing the task: `{self.task}`"


@dataclass
class ChainOfThought(Message):
    def __post_init__(self):
        self.role = MessageRole.ASSISTANT
        self.content = "Lets think step by step to get the correct answer:"


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
                    raise ValueError(
                        "Only one system message can be used per completion"
                    )
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
            messages += [message.dict() for message in self.messages]

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
        completion = openai.Completion.create(**kwargs)
        if self.function:
            return self.function.from_response(completion)

    async def acreate(self):
        kwargs = self.kwargs
        completion = openai.Completion.acreate(**kwargs)
        if self.function:
            # TODO async from_response
            return self.function.from_response(await completion)
        return await completion


def MultiTask(
    subtask_class: Type[OpenAISchema],
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    Dynamically create a MultiTask OpenAISchema that can be used to segment multiple
    tasks given a base class. This creates class that can be used to create a toolkit
    for a specific task, names and descriptions are automatically generated. However
    they can be overridden.

    :param subtask_class: The base class to use for the MultiTask
    :param name: The name of the MultiTask
    :param description: The description of the MultiTask

    :return: new schema class called `Multi{subtask_class.name}`
    """
    task_name = subtask_class.__name__ if name is None else name

    name = f"Multi{task_name}"

    list_tasks = (
        List[subtask_class],
        Field(
            default_factory=list,
            repr=False,
            description=f"Correctly segmented list of `{task_name}` tasks",
        ),
    )

    new_cls = create_model(name, tasks=list_tasks, __base__=(OpenAISchema,))

    new_cls.__doc__ = (
        f"Correct segmentation of `{task_name}` tasks"
        if description is None
        else description
    )

    return new_cls


if __name__ == "__main__":
    from pprint import pprint

    class Search(OpenAISchema):
        id: int
        query: str

    task = (
        ChatCompletion(name="Acme Inc Email Segmentation", model="gpt3.5-turbo-0613")
        | ExpertSystem(task="Segment emails into search queries")
        | MultiTask(subtask_class=Search)
        | ChainOfThought()
        | TaggedMessage(tag="email", content="Segment emails into search queries")
    )
    assert isinstance(task, ChatCompletion)

    import json

    print(json.dumps(task.kwargs, indent=4))
    """
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a world class, state of the art agent capable of correctly completing the task: `Segment emails into search queries`"
            },
            {
                "role": "assistant",
                "content": "Lets think step by step to get the correct answer:"
            },
            {
                "role": "user",
                "content": "<email>Segment emails into search queries</email>"
            },
            {
                "role": "assistant",
                "content": "Lets think step by step to get the correct answer:"
            }
        ],
        "functions": [
            {
                "name": "MultiSearch",
                "description": "Correct segmentation of `Search` tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "description": "Correctly segmented list of `Search` tasks",
                            "type": "array",
                            "items": {
                                "$ref": "#/definitions/Search"
                            }
                        }
                    },
                    "definitions": {
                        "Search": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "integer"
                                },
                                "query": {
                                    "type": "string"
                                }
                            },
                            "required": [
                                "id",
                                "query"
                            ]
                        }
                    },
                    "required": [
                        "tasks"
                    ]
                }
            }
        ],
        "function_call": {
            "name": "MultiSearch"
        },
        "max_tokens": 1000,
        "temperature": 0.1,
    "model": "gpt3.5-turbo-0613"
    }
    """
