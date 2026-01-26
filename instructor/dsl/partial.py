# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import sys
import types
import warnings
from collections.abc import AsyncGenerator, Generator, Iterable
from copy import deepcopy
from functools import cache
from typing import (  # noqa: UP035
    Any,
    Generic,
    List,  # needed for runtime check against typing.List annotations from user code
    NoReturn,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from jiter import from_json
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from instructor.mode import Mode
from instructor.utils import extract_json_from_stream, extract_json_from_stream_async
from instructor.dsl.json_tracker import JsonCompleteness, is_json_complete

T_Model = TypeVar("T_Model", bound=BaseModel)

if sys.version_info >= (3, 10):
    # types.UnionType is only available in Python 3.10 and above
    UNION_ORIGINS = (Union, types.UnionType)
else:
    UNION_ORIGINS = (Union,)

# Track models currently being processed to prevent infinite recursion
# with self-referential models (e.g., TreeNode with children: List["TreeNode"])
_processing_models: set[type] = set()


class MakeFieldsOptional:
    pass


class PartialLiteralMixin:
    """DEPRECATED: This mixin is no longer necessary.

    With completeness-based validation, Literal and Enum types are handled
    automatically during streaming:
    - Incomplete JSON: no validation runs, partial values are stored as-is
    - Complete JSON: full validation against original model

    You can safely remove this mixin from your models.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        warnings.warn(
            "PartialLiteralMixin is deprecated and no longer necessary. "
            "Completeness-based validation now handles Literal and Enum types "
            "automatically during streaming. You can safely remove this mixin.",
            DeprecationWarning,
            stacklevel=2,
        )


def remove_control_chars(s):
    return re.sub(r"[\x00-\x1F\x7F-\x9F]", "", s)


def process_potential_object(potential_object, partial_mode, partial_model, **kwargs):
    """Process a potential JSON object using completeness-based validation.

    - If JSON is complete (closed braces/brackets): validate against original model
    - If JSON is incomplete: build partial object using model_construct (no validation)

    Note: Pydantic v2.10+ has `experimental_allow_partial` but it doesn't support
    BaseModel constraints during partial validation (only TypedDict). If Pydantic
    adds BaseModel support in the future, this could potentially be simplified.
    See: https://docs.pydantic.dev/latest/concepts/partial_validation/
    """
    json_str = potential_object.strip() or "{}"
    parsed = from_json(json_str.encode(), partial_mode=partial_mode)

    tracker = JsonCompleteness()
    tracker.analyze(json_str)

    # Get original model for validation
    original_model = getattr(partial_model, "_original_model", None)

    # Check if root is complete AND has actual data (not just empty {})
    root_complete = tracker.is_root_complete()
    has_data = bool(parsed) if isinstance(parsed, dict) else True

    if root_complete and has_data and original_model is not None:
        # Root object is complete with data - validate against original model
        return original_model.model_validate(parsed, **kwargs)
    else:
        # Object is incomplete or empty - build instance using model_construct (no validation)
        model_for_construct = (
            original_model if original_model is not None else partial_model
        )
        return _build_partial_object(parsed, model_for_construct, tracker, "", **kwargs)


def _build_partial_object(
    data: Any,
    model: type[BaseModel],
    tracker: JsonCompleteness,
    path: str,
    **kwargs: Any,
) -> Any:
    """Build a partial object using model_construct() to skip validation.

    For each field:
    - If the field's JSON is complete AND it's a nested BaseModel: validate it
    - Otherwise: store without validation
    """
    if data is None:
        return None

    if not isinstance(data, dict):
        return data

    result = {}

    for field_name in data:
        field_value = data[field_name]
        field_path = f"{path}.{field_name}" if path else field_name

        if field_value is None:
            result[field_name] = None
            continue

        field_complete = tracker.is_path_complete(field_path)
        field_info = model.model_fields.get(field_name)
        field_type = field_info.annotation if field_info else None

        if field_complete and field_type is not None:
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                result[field_name] = field_type.model_validate(field_value, **kwargs)
                continue

        if isinstance(field_value, dict):
            nested_model = None
            if field_type is not None and isinstance(field_type, type):
                if issubclass(field_type, BaseModel):
                    nested_model = field_type

            if nested_model:
                result[field_name] = _build_partial_object(
                    field_value, nested_model, tracker, field_path, **kwargs
                )
            else:
                result[field_name] = field_value
        elif isinstance(field_value, list):
            result[field_name] = _build_partial_list(
                field_value, model, field_name, tracker, field_path, **kwargs
            )
        else:
            result[field_name] = field_value

    # Set missing fields to None or empty nested models
    for field_name, field_info in model.model_fields.items():
        if field_name not in result:
            field_type = field_info.annotation
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                result[field_name] = _build_partial_object(
                    {}, field_type, tracker, "", **kwargs
                )
            else:
                result[field_name] = None

    return model.model_construct(**result)


def _build_partial_list(
    items: list,
    original_model: type[BaseModel] | None,
    field_name: str,
    tracker: JsonCompleteness,
    path: str,
    **kwargs: Any,
) -> list:
    """Build a partial list, validating complete items."""
    result = []

    item_type = None
    if original_model:
        field_info = original_model.model_fields.get(field_name)
        if field_info:
            field_type = field_info.annotation
            if get_origin(field_type) in (list, List):  # noqa: UP006
                args = get_args(field_type)
                if args:
                    item_type = args[0]

    for i, item in enumerate(items):
        item_path = f"{path}[{i}]"
        item_complete = tracker.is_path_complete(item_path)

        if item_complete and item_type and isinstance(item_type, type):
            if issubclass(item_type, BaseModel) and isinstance(item, dict):
                result.append(item_type.model_validate(item, **kwargs))
                continue

        result.append(item)

    return result


def _process_generic_arg(
    arg: Any,
    make_fields_optional: bool = False,
) -> Any:
    arg_origin = get_origin(arg)

    if arg_origin is not None:
        # Handle any nested generic type (Union, List, Dict, etc.)
        nested_args = get_args(arg)
        modified_nested_args = tuple(
            _process_generic_arg(
                t,
                make_fields_optional=make_fields_optional,
            )
            for t in nested_args
        )
        # Special handling for Union types (types.UnionType isn't subscriptable)
        if arg_origin in UNION_ORIGINS:
            return Union[modified_nested_args]  # type: ignore

        return arg_origin[modified_nested_args]
    else:
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            # Prevent infinite recursion for self-referential models
            if arg in _processing_models:
                return arg  # Already processing this model, return unwrapped
            _processing_models.add(arg)
            try:
                return (
                    Partial[arg, MakeFieldsOptional]  # type: ignore[valid-type]
                    if make_fields_optional
                    else Partial[arg]
                )
            finally:
                _processing_models.discard(arg)
        else:
            return arg


def _make_field_optional(
    field: FieldInfo,
) -> tuple[Any, FieldInfo]:
    tmp_field = deepcopy(field)

    annotation = field.annotation

    # Handle generics (like List, Dict, Union, Literal, etc.)
    if get_origin(annotation) is not None:
        # Get the generic base (like List, Dict) and its arguments (like User in List[User])
        generic_base = get_origin(annotation)
        generic_args = get_args(annotation)

        modified_args = tuple(
            _process_generic_arg(arg, make_fields_optional=True) for arg in generic_args
        )

        # Reconstruct the generic type with modified arguments
        tmp_field.annotation = (
            Optional[generic_base[modified_args]] if generic_base else None
        )
        tmp_field.default = None
        tmp_field.default_factory = None
    # If the field is a BaseModel, then recursively convert it's
    # attributes to optionals.
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        tmp_field.annotation = Optional[Partial[annotation, MakeFieldsOptional]]  # type: ignore[assignment, valid-type]
        tmp_field.default = {}
        tmp_field.default_factory = None
    else:
        tmp_field.annotation = Optional[field.annotation]  # type:ignore
        tmp_field.default = None
        tmp_field.default_factory = None

    return tmp_field.annotation, tmp_field  # type: ignore


class PartialBase(Generic[T_Model]):
    @classmethod
    @cache
    def get_partial_model(cls) -> type[T_Model]:
        """Return a partial model for holding incomplete streaming data.

        With completeness-based validation, we use model_construct() to build
        partial objects without validation. This method creates a model with
        all fields optional and stores a reference to the original model
        for validation when JSON is complete.
        """
        assert issubclass(cls, BaseModel), (
            f"{cls.__name__} must be a subclass of BaseModel"
        )

        model_name = (
            cls.__name__
            if cls.__name__.startswith("Partial")
            else f"Partial{cls.__name__}"
        )

        # Create partial model with optional fields
        partial_model = create_model(
            model_name,
            __base__=cls,
            __module__=cls.__module__,
            **{
                field_name: _make_field_optional(field_info)
                for field_name, field_info in cls.model_fields.items()
            },  # type: ignore[all]
        )

        # Store reference to original model for validation of complete objects
        original = getattr(cls, "_original_model", cls)
        partial_model._original_model = original  # type: ignore[attr-defined]

        return partial_model

    @classmethod
    def from_streaming_response(
        cls, completion: Iterable[Any], mode: Mode, **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        json_chunks = cls.extract_json(completion, mode)

        if mode in {Mode.MD_JSON, Mode.GEMINI_TOOLS}:
            json_chunks = extract_json_from_stream(json_chunks)

        if mode == Mode.WRITER_TOOLS:
            yield from cls.writer_model_from_chunks(json_chunks, **kwargs)
        else:
            yield from cls.model_from_chunks(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls, completion: AsyncGenerator[Any, None], mode: Mode, **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        json_chunks = cls.extract_json_async(completion, mode)

        if mode == Mode.MD_JSON:
            json_chunks = extract_json_from_stream_async(json_chunks)

        if mode == Mode.WRITER_TOOLS:
            async for item in cls.writer_model_from_chunks_async(json_chunks, **kwargs):
                yield item
        else:
            async for item in cls.model_from_chunks_async(json_chunks, **kwargs):
                yield item

    @classmethod
    def writer_model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        # Always use trailing-strings mode to preserve incomplete data during streaming
        # PartialLiteralMixin is deprecated - completeness-based validation handles Literals
        partial_mode = "trailing-strings"
        final_obj = None
        for chunk in json_chunks:
            # Writer mode special handling: chunk might be complete JSON replacing accumulated
            if (
                len(chunk) > len(potential_object)
                and chunk.startswith("{")
                and chunk.endswith("}")
            ):
                potential_object = chunk
            else:
                potential_object += chunk
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            final_obj = obj
            yield obj

        # Final validation: only validate if the JSON is structurally complete
        # If JSON is incomplete (stream ended mid-object), skip validation
        if final_obj is not None:
            original_model = getattr(cls, "_original_model", None)
            if original_model is not None:
                if is_json_complete(potential_object.strip() or "{}"):
                    original_model.model_validate(
                        final_obj.model_dump(exclude_none=True), **kwargs
                    )

    @classmethod
    async def writer_model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        # Always use trailing-strings mode to preserve incomplete data during streaming
        # PartialLiteralMixin is deprecated - completeness-based validation handles Literals
        partial_mode = "trailing-strings"
        final_obj = None
        async for chunk in json_chunks:
            # Writer mode special handling: chunk might be complete JSON replacing accumulated
            if (
                len(chunk) > len(potential_object)
                and chunk.startswith("{")
                and chunk.endswith("}")
            ):
                potential_object = chunk
            else:
                potential_object += chunk
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            final_obj = obj
            yield obj

        # Final validation: only validate if the JSON is structurally complete
        # If JSON is incomplete (stream ended mid-object), skip validation
        if final_obj is not None:
            original_model = getattr(cls, "_original_model", None)
            if original_model is not None:
                if is_json_complete(potential_object.strip() or "{}"):
                    original_model.model_validate(
                        final_obj.model_dump(exclude_none=True), **kwargs
                    )

    @classmethod
    def model_from_chunks(
        cls, json_chunks: Iterable[Any], **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        # Always use trailing-strings mode to preserve incomplete data during streaming
        # PartialLiteralMixin is deprecated - completeness-based validation handles Literals
        partial_mode = "trailing-strings"
        final_obj = None
        for chunk in json_chunks:
            if chunk is None:
                continue
            if not isinstance(chunk, str):
                try:
                    chunk = str(chunk)
                except Exception:
                    continue
            potential_object += remove_control_chars(chunk)
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            final_obj = obj
            yield obj

        # Final validation: only validate if the JSON is structurally complete
        # If JSON is incomplete (stream ended mid-object), skip validation
        if final_obj is not None:
            original_model = getattr(cls, "_original_model", None)
            if original_model is not None:
                if is_json_complete(potential_object.strip() or "{}"):
                    original_model.model_validate(
                        final_obj.model_dump(exclude_none=True), **kwargs
                    )

    @classmethod
    async def model_from_chunks_async(
        cls, json_chunks: AsyncGenerator[str, None], **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        potential_object = ""
        partial_model = cls.get_partial_model()
        # Always use trailing-strings mode to preserve incomplete data during streaming
        # PartialLiteralMixin is deprecated - completeness-based validation handles Literals
        partial_mode = "trailing-strings"
        final_obj = None
        async for chunk in json_chunks:
            if chunk is None:
                continue
            if not isinstance(chunk, str):
                try:
                    chunk = str(chunk)
                except Exception:
                    continue
            potential_object += remove_control_chars(chunk)
            obj = process_potential_object(
                potential_object, partial_mode, partial_model, **kwargs
            )
            final_obj = obj
            yield obj

        # Final validation: only validate if the JSON is structurally complete
        # If JSON is incomplete (stream ended mid-object), skip validation
        if final_obj is not None:
            original_model = getattr(cls, "_original_model", None)
            if original_model is not None:
                if is_json_complete(potential_object.strip() or "{}"):
                    original_model.model_validate(
                        final_obj.model_dump(exclude_none=True), **kwargs
                    )

    @staticmethod
    def extract_json(
        completion: Iterable[Any], mode: Mode
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from various LLM provider streaming responses.

        Each provider has a different structure for streaming responses that needs
        specific handling to extract the relevant JSON data."""
        json_started = False
        for chunk in completion:
            try:
                if mode in {Mode.COHERE_TOOLS, Mode.COHERE_JSON_SCHEMA}:
                    event_type = getattr(chunk, "event_type", None)
                    if event_type == "text-generation":
                        if text := getattr(chunk, "text", None):
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (text.find("{"), text.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                text = text[json_start:]
                            yield text
                    elif event_type == "tool-calls-chunk":
                        delta = getattr(chunk, "tool_call_delta", None)
                        args = getattr(delta, "parameters", None) or getattr(
                            delta, "text", None
                        )
                        if args:
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (args.find("{"), args.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                args = args[json_start:]
                            yield args
                        elif text := getattr(chunk, "text", None):
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (text.find("{"), text.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                text = text[json_start:]
                            yield text
                    elif event_type == "tool-calls-generation":
                        tool_calls = getattr(chunk, "tool_calls", None)
                        if tool_calls:
                            args = json.dumps(tool_calls[0].parameters)
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (args.find("{"), args.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                args = args[json_start:]
                            yield args
                        elif text := getattr(chunk, "text", None):
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (text.find("{"), text.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                text = text[json_start:]
                            yield text
                    else:
                        chunk_type = getattr(chunk, "type", None)
                        if chunk_type == "content-delta":
                            delta = getattr(chunk, "delta", None)
                            message = getattr(delta, "message", None)
                            content = getattr(message, "content", None)
                            if text := getattr(content, "text", None):
                                if not json_started:
                                    json_start = min(
                                        (
                                            pos
                                            for pos in (
                                                text.find("{"),
                                                text.find("["),
                                            )
                                            if pos != -1
                                        ),
                                        default=-1,
                                    )
                                    if json_start == -1:
                                        continue
                                    json_started = True
                                    text = text[json_start:]
                                yield text
                        elif chunk_type == "tool-call-delta":
                            delta = getattr(chunk, "delta", None)
                            message = getattr(delta, "message", None)
                            tool_calls = getattr(message, "tool_calls", None)
                            function = getattr(tool_calls, "function", None)
                            if args := getattr(function, "arguments", None):
                                if not json_started:
                                    json_start = min(
                                        (
                                            pos
                                            for pos in (
                                                args.find("{"),
                                                args.find("["),
                                            )
                                            if pos != -1
                                        ),
                                        default=-1,
                                    )
                                    if json_start == -1:
                                        continue
                                    json_started = True
                                    args = args[json_start:]
                                yield args
                if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
                    yield chunk.data.choices[0].delta.content
                if mode == Mode.MISTRAL_TOOLS:
                    if not chunk.data.choices[0].delta.tool_calls:
                        continue
                    yield chunk.data.choices[0].delta.tool_calls[0].function.arguments
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

                if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
                    try:
                        yield chunk.text
                    except ValueError as e:
                        if "valid `Part`" in str(e):
                            # Skip chunk with invalid Part (e.g., due to finish_reason=1 token limit)
                            continue
                        raise
                if mode == Mode.GENAI_TOOLS:
                    fc = chunk.candidates[0].content.parts[0].function_call.args
                    yield json.dumps(fc)
                if mode == Mode.GEMINI_JSON:
                    try:
                        yield chunk.text
                    except ValueError as e:
                        if "valid `Part`" in str(e):
                            # Skip chunk with invalid Part (e.g., due to finish_reason=1 token limit)
                            continue
                        raise
                if mode == Mode.GEMINI_TOOLS:
                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type:ignore
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
                elif mode in {
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
                            if json_chunk[0].function.arguments:
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
        json_started = False
        async for chunk in completion:
            try:
                if mode in {Mode.COHERE_TOOLS, Mode.COHERE_JSON_SCHEMA}:
                    event_type = getattr(chunk, "event_type", None)
                    if event_type == "text-generation":
                        if text := getattr(chunk, "text", None):
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (text.find("{"), text.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                text = text[json_start:]
                            yield text
                    elif event_type == "tool-calls-chunk":
                        delta = getattr(chunk, "tool_call_delta", None)
                        args = getattr(delta, "parameters", None) or getattr(
                            delta, "text", None
                        )
                        if args:
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (args.find("{"), args.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                args = args[json_start:]
                            yield args
                        elif text := getattr(chunk, "text", None):
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (text.find("{"), text.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                text = text[json_start:]
                            yield text
                    elif event_type == "tool-calls-generation":
                        tool_calls = getattr(chunk, "tool_calls", None)
                        if tool_calls:
                            args = json.dumps(tool_calls[0].parameters)
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (args.find("{"), args.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                args = args[json_start:]
                            yield args
                        elif text := getattr(chunk, "text", None):
                            if not json_started:
                                json_start = min(
                                    (
                                        pos
                                        for pos in (text.find("{"), text.find("["))
                                        if pos != -1
                                    ),
                                    default=-1,
                                )
                                if json_start == -1:
                                    continue
                                json_started = True
                                text = text[json_start:]
                            yield text
                    else:
                        chunk_type = getattr(chunk, "type", None)
                        if chunk_type == "content-delta":
                            delta = getattr(chunk, "delta", None)
                            message = getattr(delta, "message", None)
                            content = getattr(message, "content", None)
                            if text := getattr(content, "text", None):
                                if not json_started:
                                    json_start = min(
                                        (
                                            pos
                                            for pos in (
                                                text.find("{"),
                                                text.find("["),
                                            )
                                            if pos != -1
                                        ),
                                        default=-1,
                                    )
                                    if json_start == -1:
                                        continue
                                    json_started = True
                                    text = text[json_start:]
                                yield text
                        elif chunk_type == "tool-call-delta":
                            delta = getattr(chunk, "delta", None)
                            message = getattr(delta, "message", None)
                            tool_calls = getattr(message, "tool_calls", None)
                            function = getattr(tool_calls, "function", None)
                            if args := getattr(function, "arguments", None):
                                if not json_started:
                                    json_start = min(
                                        (
                                            pos
                                            for pos in (
                                                args.find("{"),
                                                args.find("["),
                                            )
                                            if pos != -1
                                        ),
                                        default=-1,
                                    )
                                    if json_start == -1:
                                        continue
                                    json_started = True
                                    args = args[json_start:]
                                yield args
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.MISTRAL_STRUCTURED_OUTPUTS:
                    yield chunk.data.choices[0].delta.content
                if mode == Mode.MISTRAL_TOOLS:
                    if not chunk.data.choices[0].delta.tool_calls:
                        continue
                    yield chunk.data.choices[0].delta.tool_calls[0].function.arguments
                if mode == Mode.VERTEXAI_JSON:
                    yield chunk.candidates[0].content.parts[0].text
                if mode == Mode.VERTEXAI_TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                if mode == Mode.GENAI_STRUCTURED_OUTPUTS:
                    try:
                        yield chunk.text
                    except ValueError as e:
                        if "valid `Part`" in str(e):
                            # Skip chunk with invalid Part (e.g., due to finish_reason=1 token limit)
                            continue
                        raise
                if mode == Mode.GENAI_TOOLS:
                    fc = chunk.candidates[0].content.parts[0].function_call.args
                    yield json.dumps(fc)
                if mode == Mode.GEMINI_JSON:
                    try:
                        yield chunk.text
                    except ValueError as e:
                        if "valid `Part`" in str(e):
                            # Skip chunk with invalid Part (e.g., due to finish_reason=1 token limit)
                            continue
                        raise
                if mode == Mode.GEMINI_TOOLS:
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
                            if json_chunk[0].function.arguments:
                                yield json_chunk[0].function.arguments
                    else:
                        raise NotImplementedError(
                            f"Mode {mode} is not supported for MultiTask streaming"
                        )
            except AttributeError:
                pass


class Partial(Generic[T_Model]):
    """Generate a new class which has PartialBase as a base class.

    Notes:
        This will enable partial validation of the model while streaming.

    Example:
        Partial[SomeModel]
    """

    def __new__(
        cls,
        *args: object,  # noqa
        **kwargs: object,  # noqa
    ) -> Partial[T_Model]:
        """Cannot instantiate.

        Raises:
            TypeError: Direct instantiation not allowed.
        """
        raise TypeError("Cannot instantiate abstract Partial class.")

    def __init_subclass__(
        cls,
        *args: object,
        **kwargs: object,
    ) -> NoReturn:
        """Cannot subclass.

        Raises:
           TypeError: Subclassing not allowed.
        """
        raise TypeError(f"Cannot subclass {cls.__module__}.Partial")

    def __class_getitem__(
        cls,
        wrapped_class: type[T_Model] | tuple[type[T_Model], type[MakeFieldsOptional]],
    ) -> type[T_Model]:
        """Convert model to one that inherits from PartialBase.

        We don't make the fields optional at this point, we just wrap them with `Partial` so the names of the nested models will be
        `Partial{ModelName}`. We want the output of `model_json_schema()` to
        reflect the name change, but everything else should be the same as the
        original model. During validation, we'll generate a true partial model
        to support partially defined fields.

        """

        make_fields_optional = None
        if isinstance(wrapped_class, tuple):
            wrapped_class, make_fields_optional = wrapped_class

        def _wrap_models(field: FieldInfo) -> tuple[object, FieldInfo]:
            tmp_field = deepcopy(field)

            annotation = field.annotation

            # Handle generics (like List, Dict, etc.)
            if get_origin(annotation) is not None:
                # Get the generic base (like List, Dict) and its arguments (like User in List[User])
                generic_base = get_origin(annotation)
                generic_args = get_args(annotation)

                modified_args = tuple(_process_generic_arg(arg) for arg in generic_args)

                # Reconstruct the generic type with modified arguments
                tmp_field.annotation = (
                    generic_base[modified_args] if generic_base else None
                )
            # If the field is a BaseModel, then recursively convert it's
            # attributes to optionals.
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                # Prevent infinite recursion for self-referential models
                if annotation in _processing_models:
                    tmp_field.annotation = (
                        annotation  # Already processing, keep unwrapped
                    )
                else:
                    _processing_models.add(annotation)
                    try:
                        tmp_field.annotation = Partial[annotation]
                    finally:
                        _processing_models.discard(annotation)
            return tmp_field.annotation, tmp_field

        model_name = (
            wrapped_class.__name__
            if wrapped_class.__name__.startswith("Partial")
            else f"Partial{wrapped_class.__name__}"
        )

        partial_model = create_model(
            model_name,
            __base__=(wrapped_class, PartialBase),  # type: ignore
            __module__=wrapped_class.__module__,
            **{
                field_name: (
                    _make_field_optional(field_info)
                    if make_fields_optional is not None
                    else _wrap_models(field_info)
                )
                for field_name, field_info in wrapped_class.model_fields.items()
            },  # type: ignore
        )

        # Store reference to original model for final validation
        partial_model._original_model = wrapped_class  # type: ignore[attr-defined]

        return partial_model
