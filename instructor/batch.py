from typing import Literal, Any, Union, TypeVar, Optional
from collections.abc import Iterable
from pydantic import BaseModel, Field
from instructor.process_response import handle_response_model
import uuid
import json

T = TypeVar("T", bound=BaseModel)

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
    tool_choice: dict[str, Any]


class BatchModel(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions"]
    body: RequestBody


class EmbedRequestBody(BaseModel):
    model: Literal[
        "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    ]
    encoding_format: Literal["float", "base64"]
    input: list[str]
    dimensions: Optional[int] = Field(default=None)


class EmbedBatchModel(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/embeddings"]
    body: EmbedRequestBody


class BatchJob:
    @classmethod
    def parse_from_file(
        cls, file_path: str, response_model: type[T]
    ) -> tuple[list[T], list[dict[Any, Any]]]:
        with open(file_path) as file:
            res: list[T] = []
            error_objs: list[dict[Any, Any]] = []
            for line in file:
                data = json.loads(line)
                try:
                    res.append(
                        response_model(
                            **json.loads(
                                data["response"]["body"]["choices"][0]["message"][
                                    "tool_calls"
                                ][0]["function"]["arguments"]
                            )
                        )
                    )
                except Exception:
                    error_objs.append(data)

            return res, error_objs

    @classmethod
    def parse_embeddings_from_file(cls, file_path: str) -> list[float]:
        with open(file_path) as file:
            return [
                json_obj["response"]["body"]["data"][0]["embedding"]
                for line in file
                for json_obj in [json.loads(line)]
            ]

    @classmethod
    def embed_list(
        cls,
        embedding_input: list[str],
        model: Literal[
            "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
        ],
        file_path: str,
        encoding_format: Literal["float", "base64"] = "float",
        dimensions: Optional[int] = None,
    ):
        assert (
            len(embedding_input) < 50000
        ), "Batch Job has a maximum size of 50,000 items per batch job for now"
        if dimensions:
            assert model != "text-embedding-ada-002"

        with open(file_path, "w") as file:
            for input_string in embedding_input:
                assert (
                    len(input_string) > 0
                ), "OpenAI Embedding API does not support empty strings"
                file.write(
                    EmbedBatchModel(
                        custom_id=str(uuid.uuid4()),
                        method="POST",
                        url="/v1/embeddings",
                        body=EmbedRequestBody(
                            model=model,
                            encoding_format=encoding_format,
                            input=[input_string],
                            dimensions=dimensions,
                        ),
                    ).model_dump_json(exclude_none=True)
                    + "\n"
                )

    @classmethod
    def create_from_messages(
        cls,
        messages_batch: Union[
            list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]]
        ],
        model: Union[openai_models, str],
        response_model: type[BaseModel],
        file_path: str,
        max_tokens: int = 1000,
    ):
        _, kwargs = handle_response_model(response_model=response_model)

        with open(file_path, "w") as file:
            for messages in messages_batch:
                file.write(
                    BatchModel(
                        custom_id=str(uuid.uuid4()),
                        method="POST",
                        url="/v1/chat/completions",
                        body=RequestBody(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            **kwargs,
                        ),
                    ).model_dump_json()
                    + "\n"
                )
