from collections.abc import AsyncGenerator, Callable, Generator, Iterable
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

if TYPE_CHECKING:
    pass


class IterableBase:
    task_type: ClassVar[Optional[type[BaseModel]]] = None

    @classmethod
    def from_streaming_response(
        cls,
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]],
        task_parser: Callable[..., Generator[BaseModel, None, None]] | None = None,
        **kwargs: Any,
    ) -> Generator[BaseModel, None, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        json_chunks = stream_extractor(completion)
        parser = task_parser or cls.tasks_from_chunks
        yield from parser(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls,
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ],
        task_parser: Callable[..., AsyncGenerator[BaseModel, None]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseModel, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        json_chunks = stream_extractor(completion)
        parser = task_parser or cls.tasks_from_chunks_async
        async for item in parser(json_chunks, **kwargs):
            yield item

    @classmethod
    async def tasks_from_task_list_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[BaseModel, None]:
        """Process streaming chunks that contain a full tasks list."""

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
    def tasks_from_task_list_chunks(
        cls, json_chunks: Iterable[str], **kwargs: Any
    ) -> Generator[BaseModel, None, None]:
        """Process streaming chunks that contain a full tasks list."""
        for chunk in json_chunks:
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
        depth = 0
        scanned = 0
        for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]
                    depth = 0
                    scanned = 0

            while True:
                task_json, potential_object, depth, scanned = cls._scan_for_object(
                    potential_object, depth, scanned
                )
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
        depth = 0
        scanned = 0
        async for chunk in json_chunks:
            potential_object += chunk
            if not started:
                if "[" in chunk:
                    started = True
                    potential_object = chunk[chunk.find("[") + 1 :]
                    depth = 0
                    scanned = 0

            while True:
                task_json, potential_object, depth, scanned = cls._scan_for_object(
                    potential_object, depth, scanned
                )
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
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]],
    ) -> Generator[str, None, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        yield from stream_extractor(completion)

    @staticmethod
    async def extract_json_async(
        completion: AsyncGenerator[Any, None],
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]
        ],
    ) -> AsyncGenerator[str, None]:
        if stream_extractor is None:
            raise ValueError("stream_extractor is required for streaming responses")
        async for chunk in stream_extractor(completion):
            yield chunk

    @staticmethod
    def get_object(s: str, stack: int) -> tuple[Optional[str], str]:
        # Public (reachable as IterableBase.get_object) and the reference oracle
        # that _scan_for_object is differential-tested against; keep its brace
        # accounting byte-for-byte stable when refactoring.
        start_index = s.find("{")
        for i, c in enumerate(s):
            if c == "{":
                stack += 1
            if c == "}":
                stack -= 1
                if stack == 0:
                    return s[start_index : i + 1], s[i + 2 :]
        return None, s

    @staticmethod
    def _scan_for_object(
        buffer: str, depth: int, scanned: int
    ) -> tuple[Optional[str], str, int, int]:
        """Incrementally scan ``buffer`` for the next complete top-level object.

        Resumes from ``scanned`` (the index up to which ``buffer`` has already
        been brace-counted) carrying the running brace ``depth``, so each
        appended chunk is examined only once instead of re-scanning the whole
        growing buffer. This is behaviourally identical to calling
        :meth:`get_object` from index 0 on every chunk (the buffer is
        append-only while an object is incomplete, so a carried running depth
        equals the depth recomputed from scratch) but runs in O(total chars)
        rather than O(chars^2) over a streamed object.

        Returns ``(object_str_or_None, remainder, depth, scanned)``. On a
        completed object, ``remainder`` is the buffer past the separator that
        follows the closing brace and ``depth``/``scanned`` reset to ``0`` for
        the fresh remainder; when still incomplete, ``buffer`` is returned
        unchanged with the carried ``depth`` and ``scanned == len(buffer)``.
        """
        start_index = buffer.find("{")
        n = len(buffer)
        i = scanned
        while i < n:
            c = buffer[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return buffer[start_index : i + 1], buffer[i + 2 :], 0, 0
            i += 1
        return None, buffer, depth, n


def IterableModel(
    subtask_class: type[BaseModel],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> type[BaseModel]:
    # Import at runtime to avoid circular import
    from instructor.v2.core.function_calls import ResponseSchema

    """
    Dynamically create a IterableModel ResponseSchema that can be used to segment multiple
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
    class MultiUser(ResponseSchema, MultiTaskBase):
        tasks: List[User] = Field(
            default_factory=list,
            repr=False,
            description="Correctly segmented list of `User` tasks",
        )

        @classmethod
        def from_streaming_response(cls, completion) -> Generator[User]:
            '''
            Parse the streaming response and yield a `User` object
            for each task in the response.
            '''
            json_chunks = cls.extract_json(completion, stream_extractor)
            yield from cls.tasks_from_chunks(json_chunks)
    ```

    Parameters:
        subtask_class (Type[ResponseSchema]): The base class to use for the MultiTask
        name (Optional[str]): The name of the MultiTask class, if None then the name
            of the subtask class is used as `Multi{subtask_class.__name__}`
        description (Optional[str]): The description of the MultiTask class, if None
            then the description is set to `Correct segmentation of `{subtask_class.__name__}` tasks`

    Returns:
        schema (ResponseSchema): A new class that can be used to segment multiple tasks
    """
    if name is not None:
        task_name = name
    else:
        # Handle `Union[A, B]` / `A | B` task types.
        # `types.UnionType` does not have `__name__`, so fall back to a stable name.
        task_name = getattr(subtask_class, "__name__", None)
        if task_name is None and get_origin(subtask_class) is Union:
            members = get_args(subtask_class)
            task_name = "Or".join(getattr(m, "__name__", str(m)) for m in members)
        if task_name is None:
            task_name = str(subtask_class)

    name = f"Iterable{task_name}"

    list_tasks = (
        list[subtask_class],  # type: ignore
        Field(
            default_factory=list,
            repr=False,
            description=f"Correctly segmented list of `{task_name}` tasks",
        ),
    )

    base_models = cast(tuple[type[BaseModel], ...], (ResponseSchema, IterableBase))
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
    assert issubclass(new_cls, ResponseSchema), (
        "The new class should be a subclass of ResponseSchema"
    )
    return new_cls
