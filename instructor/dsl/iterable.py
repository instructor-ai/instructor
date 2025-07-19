from collections.abc import AsyncGenerator, Generator, Iterable
from typing import (
    Any,
    ClassVar,
    Optional,
    cast,
    get_origin,
    get_args,
    Union,
    TYPE_CHECKING,
)
import json
from pydantic import BaseModel, Field, create_model
from ..mode import Mode
from ..utils import extract_json_from_stream, extract_json_from_stream_async

if TYPE_CHECKING:
    pass


class IterableBase:
    task_type: ClassVar[Optional[type[BaseModel]]] = None

    @classmethod
    def from_streaming_response(
        cls, completion: Iterable[Any], mode: Mode, **kwargs: Any
    ) -> Generator[BaseModel, None, None]:  # noqa: ARG003
        json_chunks = cls.extract_json(completion, mode)

        if mode in {Mode.MD_JSON, Mode.GEMINI_TOOLS}:
            json_chunks = extract_json_from_stream(json_chunks)

        if mode in {Mode.VERTEXAI_TOOLS, Mode.MISTRAL_TOOLS}:
            response = next(json_chunks)
            if not response:
                return

            json_response = json.loads(response)
            if not json_response["tasks"]:
                return

            for item in json_response["tasks"]:
                yield cls.extract_cls_task_type(json.dumps(item), **kwargs)

        yield from cls.tasks_from_chunks(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls, completion: AsyncGenerator[Any, None], mode: Mode, **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        json_chunks = cls.extract_json_async(completion, mode)

        if mode == Mode.MD_JSON:
            json_chunks = extract_json_from_stream_async(json_chunks)

        if mode in {Mode.MISTRAL_TOOLS, Mode.VERTEXAI_TOOLS}:
            return cls.tasks_from_mistral_chunks(json_chunks, **kwargs)

        return cls.tasks_from_chunks_async(json_chunks, **kwargs)

    @classmethod
    async def tasks_from_mistral_chunks(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        """Process streaming chunks from Mistral and VertexAI.

        Handles the specific JSON format used by these providers when streaming."""

        async for chunk in json_chunks:
            if not chunk:
                continue
            json_response = json.loads(chunk)
            if not json_response["tasks"]:
                continue

            for item in json_response["tasks"]:
                obj = cls.extract_cls_task_type(json.dumps(item), **kwargs)
                yield obj

    @classmethod
    def tasks_from_chunks(
        cls, json_chunks: Iterable[str], **kwargs: Any
    ) -> Generator[BaseModel, None, None]:
        started = False
        potential_object = ""
        for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]

            while True:
                task_json, potential_object = cls.get_object(potential_object, 0)
                if task_json:
                    assert cls.task_type is not None
                    obj = cls.extract_cls_task_type(task_json, **kwargs)
                    yield obj
                else:
                    break

    @classmethod
    async def tasks_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        started = False
        potential_object = ""
        async for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]

            while True:
                task_json, potential_object = cls.get_object(potential_object, 0)
                if task_json:
                    assert cls.task_type is not None
                    obj = cls.extract_cls_task_type(task_json, **kwargs)
                    yield obj
                else:
                    break

    @classmethod
    def extract_cls_task_type(
        cls,
        task_json: str,
        **kwargs: Any,
    ):
        assert cls.task_type is not None
        if get_origin(cls.task_type) is Union:
            union_members = get_args(cls.task_type)
            for member in union_members:
                try:
                    obj = member.model_validate_json(task_json, **kwargs)
                    return obj
                except Exception:
                    pass
        else:
            return cls.task_type.model_validate_json(task_json, **kwargs)
        raise ValueError(
            f"Failed to extract task type with {task_json} for {cls.task_type}"
        )

    @staticmethod
    def extract_json(
        completion: Iterable[Any], mode: Mode
    ) -> Generator[str, None, None]:
        for chunk in completion:
            try:
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.GEMINI_JSON:
                    yield chunk.text
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
                    yield chunk.data.choices[0].delta.content
                if mode == Mode.MISTRAL_TOOLS:
                    if not chunk.data.choices[0].delta.tool_calls:
                        continue
                    yield chunk.data.choices[0].delta.tool_calls[0].function.arguments

                if mode in {Mode.GENAI_TOOLS}:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                if mode in {Mode.GENAI_STRUCTURED_OUTPUTS}:
                    yield chunk.candidates[0].content.parts[0].text

                if mode in {Mode.GEMINI_TOOLS}:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type:ignore

                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])

                if mode in {
                    Mode.RESPONSES_TOOLS,
                    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                }:
                    from openai.types.responses import (
                        ResponseFunctionCallArgumentsDeltaEvent,
                    )

                    if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                        yield chunk.delta
                elif chunk.choices:
                    if mode == Mode.FUNCTIONS:
                        Mode.warn_mode_functions_deprecation()
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif mode in {
                        Mode.JSON,
                        Mode.MD_JSON,
                        Mode.JSON_SCHEMA,
                        Mode.CEREBRAS_JSON,
                        Mode.FIREWORKS_JSON,
                        Mode.PERPLEXITY_JSON,
                        Mode.WRITER_JSON,
                    }:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif mode in {
                        Mode.TOOLS,
                        Mode.TOOLS_STRICT,
                        Mode.FIREWORKS_TOOLS,
                        Mode.WRITER_TOOLS,
                    }:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            if json_chunk[0].function.arguments is not None:
                                yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass

    @staticmethod
    async def extract_json_async(
        completion: AsyncGenerator[Any, None], mode: Mode
    ) -> AsyncGenerator[str, None]:
        async for chunk in completion:
            try:
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
                    yield chunk.data.choices[0].delta.content
                if mode == Mode.MISTRAL_TOOLS:
                    if not chunk.data.choices[0].delta.tool_calls:
                        continue
                    yield chunk.data.choices[0].delta.tool_calls[0].function.arguments
                if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
                    yield chunk.text
                if mode in {Mode.GENAI_TOOLS}:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                if mode in {
                    Mode.RESPONSES_TOOLS,
                    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                }:
                    from openai.types.responses import (
                        ResponseFunctionCallArgumentsDeltaEvent,
                    )

                    if isinstance(chunk, ResponseFunctionCallArgumentsDeltaEvent):
                        yield chunk.delta
                elif chunk.choices:
                    if mode == Mode.FUNCTIONS:
                        Mode.warn_mode_functions_deprecation()
                        if json_chunk := chunk.choices[0].delta.function_call.arguments:
                            yield json_chunk
                    elif mode in {
                        Mode.JSON,
                        Mode.MD_JSON,
                        Mode.JSON_SCHEMA,
                        Mode.CEREBRAS_JSON,
                        Mode.FIREWORKS_JSON,
                        Mode.PERPLEXITY_JSON,
                        Mode.WRITER_JSON,
                    }:
                        if json_chunk := chunk.choices[0].delta.content:
                            yield json_chunk
                    elif mode in {
                        Mode.TOOLS,
                        Mode.TOOLS_STRICT,
                        Mode.FIREWORKS_TOOLS,
                        Mode.WRITER_TOOLS,
                    }:
                        if json_chunk := chunk.choices[0].delta.tool_calls:
                            if json_chunk[0].function.arguments is not None:
                                yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass

    @staticmethod
    def get_object(s: str, stack: int) -> tuple[Optional[str], str]:
        start_index = s.find("{")
        for i, c in enumerate(s):
            if c == "{":
                stack += 1
            if c == "}":
                stack -= 1
                if stack == 0:
                    return s[start_index : i + 1], s[i + 2 :]
        return None, s


def IterableModel(
    subtask_class: type[BaseModel],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> type[BaseModel]:
    # Import at runtime to avoid circular import
    from ..processing.function_calls import OpenAISchema

    """
    Dynamically create a IterableModel OpenAISchema that can be used to segment multiple
    tasks given a base class. This creates class that can be used to create a toolkit
    for a specific task, names and descriptions are automatically generated. However
    they can be overridden.

    ## Usage

    ```python
    from pydantic import BaseModel, Field
    from instructor import IterableModel

    class User(BaseModel):
        name: str = Field(description="The name of the person")
        age: int = Field(description="The age of the person")
        role: str = Field(description="The role of the person")

    MultiUser = IterableModel(User)
    ```

    ## Result

    ```python
    class MultiUser(OpenAISchema, MultiTaskBase):
        tasks: List[User] = Field(
            default_factory=list,
            repr=False,
            description="Correctly segmented list of `User` tasks",
        )

        @classmethod
        def from_streaming_response(cls, completion) -> Generator[User]:
            '''
            Parse the streaming response from OpenAI and yield a `User` object
            for each task in the response
            '''
            json_chunks = cls.extract_json(completion)
            yield from cls.tasks_from_chunks(json_chunks)
    ```

    Parameters:
        subtask_class (Type[OpenAISchema]): The base class to use for the MultiTask
        name (Optional[str]): The name of the MultiTask class, if None then the name
            of the subtask class is used as `Multi{subtask_class.__name__}`
        description (Optional[str]): The description of the MultiTask class, if None
            then the description is set to `Correct segmentation of `{subtask_class.__name__}` tasks`

    Returns:
        schema (OpenAISchema): A new class that can be used to segment multiple tasks
    """
    task_name = subtask_class.__name__ if name is None else name

    name = f"Iterable{task_name}"

    list_tasks = (
        list[subtask_class],
        Field(
            default_factory=list,
            repr=False,
            description=f"Correctly segmented list of `{task_name}` tasks",
        ),
    )

    base_models = cast(tuple[type[BaseModel], ...], (OpenAISchema, IterableBase))
    new_cls = create_model(
        name,
        tasks=list_tasks,
        __base__=base_models,
    )
    new_cls = cast(type[IterableBase], new_cls)

    # set the class constructor BaseModel
    new_cls.task_type = subtask_class

    new_cls.__doc__ = (
        f"Correct segmentation of `{task_name}` tasks"
        if description is None
        else description
    )
    assert issubclass(new_cls, OpenAISchema), (
        "The new class should be a subclass of OpenAISchema"
    )
    return new_cls
