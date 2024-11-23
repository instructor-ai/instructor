# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from __future__ import annotations

from typing import (
    Any,
    TypeVar,
    Generic,
    Type,
    get_args,
    get_origin,
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    runtime_checkable,
    Dict,
    Optional,
    Union,
)
from collections.abc import AsyncGenerator, Generator, Iterable
from functools import cache

if TYPE_CHECKING:
    from pydantic.v1 import BaseModel, Field, create_model, BaseConfig
    from pydantic.v1.fields import FieldInfo
else:
    from pydantic.v1 import BaseModel, Field, create_model, BaseConfig
    from pydantic.v1.fields import FieldInfo

from jiter import from_json

from instructor.mode import Mode
from instructor.utils import extract_json_from_stream, extract_json_from_stream_async

# Type definitions
T_Model = TypeVar('T_Model', bound='BaseModel')

# Type definitions
T_Model = TypeVar("T_Model", bound="BaseModel")
TVar = TypeVar("TVar")

@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol for JSON extraction from completion streams."""
    @classmethod
    def extract_json(
        cls,
        completion: Iterable[Any],
        mode: Mode
    ) -> Generator[str, None, None]: ...

    @classmethod
    def extract_json_async(
        cls,
        completion: AsyncGenerator[Any, None],
        mode: Mode
    ) -> AsyncGenerator[str, None]: ...

# Type aliases
FieldType = Type[FieldInfo]
ModelType = Type[BaseModel]

def create_field(
    annotation: Union[Type[Any], Any],
    default: Optional[Any] = None,
    **kwargs: Any
) -> FieldInfo:
    """Create a Pydantic field with proper typing."""
    if default is None and 'default_factory' not in kwargs:
        return Field(default=None, annotation=annotation, **kwargs)
    return Field(default=default, annotation=annotation, **kwargs)

class MakeFieldsOptional:
    """Marker class for making fields optional."""
    pass

class PartialLiteralMixin:
    """Mixin for partial literal types."""
    pass

def _make_field_optional(
    field_info: FieldInfo,
) -> tuple[Type[Any], FieldInfo]:
    """Make a field optional and handle nested models."""
    base_type = field_info.annotation if field_info.annotation is not None else Any
    base_type = cast(Type[Any], base_type)

    if get_origin(base_type) is not None:
        generic_base = get_origin(base_type)
        generic_args = get_args(base_type)
        modified_args = tuple(
            Partial[arg] if isinstance(arg, type) and issubclass(arg, BaseModel) else arg
            for arg in generic_args
        )
        new_type = generic_base[modified_args] if generic_base else Any
        return (new_type, create_field(new_type))
    elif isinstance(base_type, type) and issubclass(base_type, BaseModel):
        partial_type = Partial[base_type]
        return (partial_type, create_field(partial_type, default_factory=dict))
    else:
        return (base_type, create_field(base_type))

class PartialBase(Generic[T_Model]):
    """Base class for partial models with streaming support."""
    model_fields: ClassVar[Dict[str, FieldInfo]]

    @classmethod
    def extract_json(
        cls,
        completion: Iterable[Any],
        mode: Mode
    ) -> Generator[str, None, None]:
        """Extract JSON from completion stream."""
        for chunk in completion:
            yield str(chunk)

    @classmethod
    async def extract_json_async(
        cls,
        completion: AsyncGenerator[Any, None],
        mode: Mode
    ) -> AsyncGenerator[str, None]:
        """Extract JSON from completion stream asynchronously."""
        async for chunk in completion:
            yield str(chunk)

    @classmethod
    @cache
    def get_partial_model(cls: Type[PartialBase[T_Model]]) -> Type[T_Model]:
        """Return a partial model we can use to validate partial results."""
        if not issubclass(cls, BaseModel):
            raise TypeError(f"{cls.__name__} must be a subclass of BaseModel")

        model_name = (
            cls.__name__
            if cls.__name__.startswith("Partial")
            else f"Partial{cls.__name__}"
        )

        fields: Dict[str, tuple[Type[Any], FieldInfo]] = {}
        for field_name, field_info in cls.model_fields.items():
            field_type, field_info = _make_field_optional(field_info)
            fields[field_name] = (field_type, field_info)

        # Create model with proper type handling
        config = type('Config', (BaseConfig,), {'arbitrary_types_allowed': True})
        model_kwargs = {
            name: field_tuple
            for name, field_tuple in fields.items()
        }

        partial_model = create_model(
            model_name,
            __base__=cls,
            __module__=cls.__module__,
            __config__=config,
            **model_kwargs
        )

        return cast(Type[T_Model], partial_model)

        # Add fields after model creation
        for field_name, (_, field_info) in fields.items():
            partial_model.model_fields[field_name] = field_info

        return partial_model

    @classmethod
    def from_streaming_response(
        cls: Type[PartialBase[T_Model]],
        completion: Iterable[Any],
        mode: Mode,
        **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        """Process streaming response and yield models."""
        json_chunks: Generator[str, None, None] = cls.extract_json(completion, mode)

        if mode in {Mode.MD_JSON, Mode.GEMINI_TOOLS}:
            json_chunks = extract_json_from_stream(json_chunks)

        if mode == Mode.WRITER_TOOLS:
            yield from cls.writer_model_from_chunks(json_chunks, **kwargs)
        else:
            yield from cls.model_from_chunks(json_chunks, **kwargs)

    @classmethod
    async def from_streaming_response_async(
        cls: Type[PartialBase[T_Model]],
        completion: AsyncGenerator[Any, None],
        mode: Mode,
        **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        """Process streaming response asynchronously and yield models."""
        json_chunks: AsyncGenerator[str, None] = cls.extract_json_async(completion, mode)

        if mode == Mode.MD_JSON:
            json_chunks = extract_json_from_stream_async(json_chunks)
        elif mode == Mode.WRITER_TOOLS:
            async for model in cls.writer_model_from_chunks_async(json_chunks, **kwargs):
                yield model
            return

        async for model in cls.model_from_chunks_async(json_chunks, **kwargs):
            yield model

    @classmethod
    def writer_model_from_chunks(
        cls: Type[PartialBase[T_Model]],
        json_chunks: Iterable[str],
        **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        """Process chunks using writer mode and yield models."""
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        for chunk in json_chunks:
            if len(chunk) > len(potential_object):
                potential_object = chunk
            else:
                potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(),
                partial_mode=partial_mode
            )
            validated_obj = partial_model.model_validate(obj, context=kwargs)
            yield validated_obj

    @classmethod
    async def writer_model_from_chunks_async(
        cls: Type[PartialBase[T_Model]],
        json_chunks: AsyncGenerator[str, None],
        **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        """Process chunks asynchronously using writer mode and yield models."""
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        async for chunk in json_chunks:
            if len(chunk) > len(potential_object):
                potential_object = chunk
            else:
                potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(),
                partial_mode=partial_mode
            )
            validated_obj = partial_model.model_validate(obj, context=kwargs)
            yield validated_obj

    @classmethod
    def model_from_chunks(
        cls: Type[PartialBase[T_Model]],
        json_chunks: Iterable[str],
        **kwargs: Any
    ) -> Generator[T_Model, None, None]:
        """Process chunks and yield models."""
        potential_object = ""
        partial_model = cls.get_partial_model()
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        for chunk in json_chunks:
            potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(),
                partial_mode=partial_mode
            )
            validated_obj = partial_model.model_validate(obj, context=kwargs)
            yield validated_obj

    @classmethod
    async def model_from_chunks_async(
        cls: Type[PartialBase[T_Model]],
        json_chunks: AsyncGenerator[str, None],
        **kwargs: Any
    ) -> AsyncGenerator[T_Model, None]:
        """Process chunks asynchronously and yield models."""
        potential_object = ""
        partial_model = cast(Type[T_Model], cls.get_partial_model())
        partial_mode = (
            "on" if issubclass(cls, PartialLiteralMixin) else "trailing-strings"
        )
        async for chunk in json_chunks:
            potential_object += chunk
            obj = from_json(
                (potential_object.strip() or "{}").encode(),
                partial_mode=partial_mode
            )
            obj = partial_model.model_validate(obj, context=kwargs)
            yield obj

    @classmethod
    def extract_json(
        cls,
        completion: Iterable[Any],
        mode: Mode
    ) -> Generator[str, None, None]:
        """Extract JSON from completion chunks."""
        for chunk in completion:
            try:
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.GEMINI_JSON:
                    yield chunk.text
                if mode == Mode.GEMINI_TOOLS:
                    # Gemini seems to return the entire function_call and not a chunk?
                    import json

                    resp = chunk.candidates[0].content.parts[0].function_call
                    resp_dict = type(resp).to_dict(resp)  # type:ignore
                    if "args" in resp_dict:
                        yield json.dumps(resp_dict["args"])
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

    @classmethod
    async def extract_json_async(
        cls,
        completion: AsyncGenerator[Any, None],
        mode: Mode
    ) -> AsyncGenerator[str, None]:
        """Extract JSON from completion chunks asynchronously."""
        async for chunk in completion:
            try:
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
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
    """A class for creating partial versions of Pydantic models."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("Partial cannot be instantiated")

    def __init_subclass__(cls) -> None:
        raise TypeError("Partial cannot be subclassed")

    @classmethod
    def __class_getitem__(cls, model: Union[Type[T_Model], tuple[Type[T_Model], ...]]) -> Type[T_Model]:
        """Create a partial version of the model."""
        def _wrap_models(base_model: Type[T_Model]) -> Type[T_Model]:
            if not isinstance(base_model, type) or not issubclass(base_model, BaseModel):
                raise TypeError(f"Expected BaseModel subclass, got {base_model}")

            fields: Dict[str, tuple[Type[Any], FieldInfo]] = {}
            for field_name, field_info in base_model.model_fields.items():
                field_type, field_info = _make_field_optional(field_info)
                fields[field_name] = (field_type, field_info)

            model_name = (
                base_model.__name__
                if base_model.__name__.startswith("Partial")
                else f"Partial{base_model.__name__}"
            )

            return cast(Type[T_Model], _create_partial_model(base_model, model_name, fields))

        if isinstance(model, tuple):
            return cast(Type[T_Model], tuple(_wrap_models(m) for m in model))
        return _wrap_models(model)

        # Add fields after model creation
        for field_name, (field_type, field_info) in fields.items():
            setattr(model, field_name, field_info)
            model.model_fields[field_name] = field_info

        return model
