import openai
from typing import List, Union
from pydantic import BaseModel, Field
from openai_function_call import OpenAISchema
from .messages import ChainOfThought, Message, MessageRole, SystemMessage


class ChatCompletion(BaseModel):
    """
    A chat completion is a collection of messages and configration options that can be used to
    generate a chat response from the OpenAI API.


    Usage:
        In order to generate a chat response from the OpenAI API, you need to create a chat completion and then pipe it to a message and a `OpenAISchema`. Then when `create` or `acreate` is called we'll return the response from the API as an instance of `OpenAISchema`.


    Example:
        ```python
        class Sum(OpenAISchema):
            a: int
            b: int

        completion = (
            ChatCompletion("example")
            | TaggedMessage(content="What is 1 + 1?", tag="question")
            | Schema
        )

        print(completion.create())
        # Sum(a=1, b=1)
        ```


    Tips:
        * You can use the `|` operator to chain multiple messages and functions together
        * There should be exactly one function call class (OpenAISchema) per chat completion
        * System messages will be concatenated together
        * Only one chain of thought message can be used per completion


    Attributes:
        name (str): The name of the chat completion
        model (str): The model to use for the chat completion (default: "gpt-3.5-turbo-0613")
        max_tokens (int): The maximum number of tokens to generate (default: 1000)
        temperature (float): The temperature to use for the chat completion (default: 0.1)
        stream (bool): Whether to stream the response from the API (default: False)

    Warning:
        Currently we do not support streaming the response from the API, so the stream parameter is not supported yet.
    """

    name: str
    model: str = Field(default="gpt-3.5-turbo-0613")
    max_tokens: int = Field(default=1000)
    temperature: float = Field(default=0.1)
    stream: bool = Field(default=False)

    messages: List[Message] = Field(default_factory=list, repr=False)
    system_message: Message = Field(default=None, repr=False)
    cot_message: ChainOfThought = Field(default=None, repr=False)
    function: OpenAISchema = Field(default=None, repr=False)

    def __post_init__(self):
        assert self.stream == False, "Stream is not supported yet"

    def __or__(self, other: Union[Message, OpenAISchema]) -> "ChatCompletion":
        """
        Add a message or function to the chat completion, this can be used to chain multiple messages and functions together. It should contain some set of user or system messages along with a function call class (OpenAISchema)

        """

        if isinstance(other, Message):
            if other.role == MessageRole.SYSTEM:
                if not self.system_message:
                    self.system_message = other  # type: ignore
                else:
                    self.system_message.content += "\n\n" + other.content
            else:
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
                "gpt-3.5-turbo",
                "gpt-4",
            }, "Only *-0613 models can currently use functions"
        return self

    @property
    def kwargs(self) -> dict:
        """
        Construct the kwargs for the OpenAI API call

        Example:
            ```python
            result = openai.ChatCompletion.create(**self.kwargs)
            ```
        """
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
        """
        Create a chat response from the OpenAI API

        Returns:
            response (OpenAISchema): The response from the OpenAI API
        """
        kwargs = self.kwargs
        completion = openai.ChatCompletion.create(**kwargs)
        if self.function:
            return self.function.from_response(completion)
        return completion

    async def acreate(self):
        """
        Create a chat response from the OpenAI API asynchronously

        Returns:
            response (OpenAISchema): The response from the OpenAI API
        """
        kwargs = self.kwargs
        completion = openai.ChatCompletion.acreate(**kwargs)
        if self.function:
            return self.function.from_response(await completion)
        return await completion
