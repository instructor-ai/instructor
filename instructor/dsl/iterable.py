from typing import Any, Generic, Optional, TypeVar, cast, ClassVar
from collections.abc import AsyncGenerator, Generator, Iterable
from typing import Any, ClassVar, Optional, cast
import json
from pydantic import BaseModel, Field, create_model

from instructor.function_calls import OpenAISchema
from instructor.mode import Mode
from instructor.utils import extract_json_from_stream, extract_json_from_stream_async


T = TypeVar("T", bound=BaseModel)


class IterableBase(Generic[T]):
    task_type: ClassVar[Optional[type[BaseModel]]] = None

    @classmethod
    def from_streaming_response(
        cls, completion: Iterable[Any], mode: Mode, **kwargs: Any
    ) -> Generator[T, None, None]:
        """Process streaming responses and yield task objects.

        Args:
            completion: The streaming response from the model
            mode: The response mode being used
            kwargs: Additional arguments to pass to model_validate

        Returns:
            Generator yielding task objects
        """
        json_chunks = cls.extract_json(completion, mode)

        # Handle markdown JSON or Gemini tools format
        if mode in {Mode.MD_JSON, Mode.GEMINI_TOOLS}:
            json_chunks = extract_json_from_stream(json_chunks)

        # Handle Vertexai and Mistral special cases
        if mode in {Mode.VERTEXAI_TOOLS, Mode.MISTRAL_TOOLS}:
            import json
            response = next(json_chunks, None)
            if not response:
                return

            try:
                json_response = json.loads(response)
                if not json_response.get("tasks"):
                    return

                # Check if task_type is None before using it
                if cls.task_type is None:
                    raise ValueError("task_type is not defined for this class")

                for item in json_response["tasks"]:
                    yield cast(T, cls.task_type.model_validate(item))
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                # Better error handling for malformed responses
                raise ValueError(f"Error processing response: {e}") from e

        yield from cls.tasks_from_chunks(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls, completion: AsyncGenerator[Any, None], mode: Mode, **kwargs: Any
    ) -> AsyncGenerator[T, None]:
        """Process async streaming responses and yield task objects.

        Args:
            completion: The async streaming response from the model
            mode: The response mode being used
            kwargs: Additional arguments to pass to model_validate

        Returns:
            AsyncGenerator yielding task objects
        """
        json_chunks = cls.extract_json_async(completion, mode)

        if mode == Mode.MD_JSON:
            json_chunks = extract_json_from_stream_async(json_chunks)

        if mode in {Mode.MISTRAL_TOOLS, Mode.VERTEXAI_TOOLS}:
            return cls.tasks_from_mistral_chunks(json_chunks, **kwargs)

        return cls.tasks_from_chunks_async(json_chunks, **kwargs)

    @classmethod
    async def tasks_from_mistral_chunks(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T, None]:
        """Process streaming chunks from Mistral and VertexAI.

        Handles the specific JSON format used by these providers when streaming.

        Args:
            json_chunks: The JSON chunks from the model
            kwargs: Additional arguments to pass to model_validate

        Returns:
            AsyncGenerator yielding task objects
        """
        import json

        async for chunk in json_chunks:
            if not chunk:
                continue

            try:
                json_response = json.loads(chunk)
                if not json_response.get("tasks"):
                    continue

                # Check if task_type is None before using it
                if cls.task_type is None:
                    raise ValueError("task_type is not defined for this class")

                for item in json_response["tasks"]:
                    yield cast(T, cls.task_type.model_validate(item, **kwargs))
            except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
                # Skip malformed chunks instead of crashing
                continue

    @classmethod
    def tasks_from_chunks(
        cls, json_chunks: Iterable[str], **kwargs: Any
    ) -> Generator[T, None, None]:
        """Extract task objects from JSON chunks.

        Args:
            json_chunks: The JSON chunks from the model
            kwargs: Additional arguments to pass to model_validate

        Returns:
            Generator yielding task objects
        """
        started = False
        potential_object = ""

        for chunk in json_chunks:
            potential_object += chunk
            if not started and "[" in chunk:
                started = True
                potential_object = chunk[chunk.find("[") + 1 :]
                continue

            task_json, potential_object = cls.get_object(potential_object, 0)
            if task_json:
                # The assert already checks for None, but let's be explicit for the type checker
                if cls.task_type is None:
                    continue

                try:
                    obj = cast(
                        T, cls.task_type.model_validate_json(task_json, **kwargs)
                    )
                    yield obj
                except Exception:
                    # Skip invalid objects instead of failing
                    continue

    @classmethod
    async def tasks_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T, None]:
        """Extract task objects from async JSON chunks.

        Args:
            json_chunks: The async JSON chunks from the model
            kwargs: Additional arguments to pass to model_validate

        Returns:
            AsyncGenerator yielding task objects
        """
        started = False
        potential_object = ""

        async for chunk in json_chunks:
            potential_object += chunk
            if not started and "[" in chunk:
                started = True
                potential_object = chunk[chunk.find("[") + 1 :]
                continue

            task_json, potential_object = cls.get_object(potential_object, 0)
            if task_json:
                # The assert already checks for None, but let's be explicit for the type checker
                if cls.task_type is None:
                    continue

                try:
                    obj = cast(
                        T, cls.task_type.model_validate_json(task_json, **kwargs)
                    )
                    yield obj
                except Exception:
                    # Skip invalid objects instead of failing
                    continue

    @staticmethod
    def _process_common_modes(chunk: Any, mode: Mode) -> Optional[str]:
        """Process common modes shared between sync and async extraction.

        Args:
            chunk: The response chunk
            mode: The response mode being used

        Returns:
            Extracted JSON string or None
        """
        # Common mode handling for both sync and async
        if (
            mode == Mode.ANTHROPIC_JSON
            and hasattr(chunk, "delta")
            and hasattr(chunk.delta, "text")
        ):
            return chunk.delta.text

        if mode == Mode.ANTHROPIC_TOOLS and hasattr(chunk, "delta"):
            return chunk.delta.partial_json

        if mode == Mode.GEMINI_JSON and hasattr(chunk, "text"):
            return chunk.text

        if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
            try:
                return chunk.data.choices[0].delta.content
            except (AttributeError, IndexError):
                return None

        if mode == Mode.MISTRAL_TOOLS:
            try:
                if not chunk.data.choices[0].delta.tool_calls:
                    return None
                return chunk.data.choices[0].delta.tool_calls[0].function.arguments
            except (AttributeError, IndexError):
                return None

        # Common OpenAI-compatible modes
        if hasattr(chunk, "choices") and chunk.choices:
            if mode == Mode.FUNCTIONS:
                Mode.warn_mode_functions_deprecation()
                try:
                    return chunk.choices[0].delta.function_call.arguments
                except AttributeError:
                    return None

            elif mode in {
                Mode.JSON,
                Mode.MD_JSON,
                Mode.JSON_SCHEMA,
                Mode.CEREBRAS_JSON,
                Mode.FIREWORKS_JSON,
                Mode.PERPLEXITY_JSON,
            }:
                try:
                    return chunk.choices[0].delta.content
                except AttributeError:
                    return None

            elif mode in {
                Mode.TOOLS,
                Mode.TOOLS_STRICT,
                Mode.FIREWORKS_TOOLS,
                Mode.WRITER_TOOLS,
            }:
                try:
                    if json_chunk := chunk.choices[0].delta.tool_calls:
                        if json_chunk[0].function.arguments is not None:
                            return json_chunk[0].function.arguments
                except (AttributeError, IndexError):
                    return None

        return None

    @classmethod
    def extract_json(
        cls, completion: Iterable[Any], mode: Mode
    ) -> Generator[str, None, None]:
        """Extract JSON content from streaming completions.

        Args:
            completion: The streaming response from the model
            mode: The response mode being used

        Returns:
            Generator of JSON strings
        """
        for chunk in completion:
            try:
                # Handle common modes
                if json_chunk := cls._process_common_modes(chunk, mode):
                    yield json_chunk
                    continue

                # Special case handling
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                    continue

                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                    continue

                if mode == Mode.GEMINI_TOOLS:
                    import json
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
                    continue

            except (AttributeError, IndexError):
                # Skip chunks with missing attributes
                continue

            # Handle unsupported modes only once we've tried all supported modes
            if mode not in {
                Mode.ANTHROPIC_JSON,
                Mode.ANTHROPIC_TOOLS,
                Mode.GEMINI_JSON,
                Mode.GEMINI_TOOLS,
                Mode.VERTEXAI_JSON,
                Mode.VERTEXAI_TOOLS,
                Mode.MISTRAL_STRUCTURED_OUTPUTS,
                Mode.MISTRAL_TOOLS,
                Mode.FUNCTIONS,
                Mode.JSON,
                Mode.MD_JSON,
                Mode.JSON_SCHEMA,
                Mode.CEREBRAS_JSON,
                Mode.FIREWORKS_JSON,
                Mode.PERPLEXITY_JSON,
                Mode.TOOLS,
                Mode.TOOLS_STRICT,
                Mode.FIREWORKS_TOOLS,
                Mode.WRITER_TOOLS,
            }:
                raise NotImplementedError(
                    f"Mode {mode} is not supported for MultiTask streaming"
                )

    @classmethod
    async def extract_json_async(
        cls, completion: AsyncGenerator[Any, None], mode: Mode
    ) -> AsyncGenerator[str, None]:
        """Extract JSON content from async streaming completions.

        Args:
            completion: The async streaming response from the model
            mode: The response mode being used

        Returns:
            AsyncGenerator of JSON strings
        """
        async for chunk in completion:
            try:
                # Handle common modes
                if json_chunk := cls._process_common_modes(chunk, mode):
                    yield json_chunk
                    continue

                # Special case handling
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                    continue

                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                    continue

            except (AttributeError, IndexError):
                # Skip chunks with missing attributes
                continue

            # Handle unsupported modes only once we've tried all supported modes
            if mode not in {
                Mode.ANTHROPIC_JSON,
                Mode.ANTHROPIC_TOOLS,
                Mode.GEMINI_JSON,
                Mode.GEMINI_TOOLS,
                Mode.VERTEXAI_JSON,
                Mode.VERTEXAI_TOOLS,
                Mode.MISTRAL_STRUCTURED_OUTPUTS,
                Mode.MISTRAL_TOOLS,
                Mode.FUNCTIONS,
                Mode.JSON,
                Mode.MD_JSON,
                Mode.JSON_SCHEMA,
                Mode.CEREBRAS_JSON,
                Mode.FIREWORKS_JSON,
                Mode.PERPLEXITY_JSON,
                Mode.TOOLS,
                Mode.TOOLS_STRICT,
                Mode.FIREWORKS_TOOLS,
                Mode.WRITER_TOOLS,
            }:
                raise NotImplementedError(
                    f"Mode {mode} is not supported for MultiTask streaming"
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
        """Extract a complete JSON object from a string.

        Args:
            s: The string containing JSON
            stack: The current bracket stack depth

        Returns:
            Tuple of (extracted object, remaining string)
        """
        start_index = s.find("{")
        if start_index == -1:
            return None, s

        for i, c in enumerate(s):
            if c == "{":
                stack += 1
            if c == "}":
                stack -= 1
                if stack == 0:
                    return s[start_index : i + 1], s[i + 2 :]
        return None, s


def IterableModel(
    subtask_class: type[T],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> type[IterableBase[T]]:
    """
    Dynamically create a IterableModel OpenAISchema that can be used to segment multiple
    tasks given a base class. This creates a class that can be used to create a toolkit
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
        subtask_class (Type[BaseModel]): The base class to use for the MultiTask
        name (Optional[str]): The name of the MultiTask class, if None then the name
            of the subtask class is used as `Multi{subtask_class.__name__}`
        description (Optional[str]): The description of the MultiTask class, if None
            then the description is set to `Correct segmentation of `{subtask_class.__name__}` tasks`

    Returns:
        schema (OpenAISchema): A new class that can be used to segment multiple tasks
    """
    task_name = subtask_class.__name__ if name is None else name
    class_name = f"Iterable{task_name}"

    list_tasks = (
        list[subtask_class],
        Field(
            default_factory=list,
            repr=False,
            description=f"Correctly segmented list of `{task_name}` tasks",
        ),
    )

    # Create the new class with proper generic typing
    base_models = cast(tuple[type[BaseModel], ...], (OpenAISchema, IterableBase))
    new_cls = create_model(
        class_name,
        tasks=list_tasks,
        __base__=base_models,
    )
    new_cls = cast(type[IterableBase[T]], new_cls)

    # Set the class constructor BaseModel
    new_cls.task_type = subtask_class

    # Set docstring
    new_cls.__doc__ = (
        f"Correct segmentation of `{task_name}` tasks"
        if description is None
        else description
    )

    assert issubclass(new_cls, OpenAISchema), (
        "The new class should be a subclass of OpenAISchema"
    )
    return new_cls
