from typing import Literal, Any, Union
from pydantic import BaseModel, Field
from instructor.process_response import handle_response_model
import uuid

openai_models = Literal[
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4-turbo-preview",
    "gpt-4-vision-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
]


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: Union[openai_models, str]
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=1000)
    tools: list[Tool]


class BatchModel(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions"]
    body: RequestBody


class BatchJob:
    @classmethod
    def create_from_messages(
        cls,
        messages_batch: list[list[dict[str, Any]]],
        model: Union[openai_models, str],
        response_model: type[BaseModel],
        max_tokens: int = 1000,
    ):
        _, tools = handle_response_model(response_model=response_model)
        return [
            BatchModel(
                custom_id=str(uuid.uuid4()),
                method="POST",
                url="/v1/chat/completions",
                body=RequestBody(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    **tools,
                ),
            ).model_dump(mode="json")
            for messages in messages_batch
        ]
