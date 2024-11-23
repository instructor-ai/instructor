# --------------------------------------------------------------------------------
# The following code is adapted from a comment on GitHub in the pydantic/pydantic repository by silviumarcu.
# Source: https://github.com/pydantic/pydantic/issues/6381#issuecomment-1831607091
#
# This code is used in accordance with the repository's license, and this reference
# serves as an acknowledgment of the original author's contribution to this project.
# --------------------------------------------------------------------------------

from __future__ import annotations

from copy import deepcopy
from functools import cache
from typing import (
    Any, TypeVar, Generic, Type, get_args, get_origin, TYPE_CHECKING,
    ClassVar, Protocol, runtime_checkable, Dict, Optional, Union, cast,
    Final
)
from collections.abc import AsyncGenerator, Generator, Iterable
import types
import sys

# Import types for type checking and runtime
if TYPE_CHECKING:
    from pydantic import BaseModel, Field, create_model, ConfigDict
    from pydantic.fields import FieldInfo
else:
    try:
        from pydantic.v2 import BaseModel, Field, create_model, ConfigDict
        from pydantic.v2.fields import FieldInfo
        PYDANTIC_V2: Final[bool] = True
    except ImportError:
        from pydantic import BaseModel, Field, create_model, ConfigDict  # type: ignore
        from pydantic.fields import FieldInfo  # type: ignore
        PYDANTIC_V2: Final[bool] = False

from jiter import from_json
from instructor.mode import Mode
from instructor.utils import extract_json_from_stream, extract_json_from_stream_async

# Type definitions
T_Model = TypeVar('T_Model', bound='BaseModel')

if sys.version_info >= (3, 10):
    UNION_ORIGINS = (Union, types.UnionType)
else:
    UNION_ORIGINS = (Union,)

class MakeFieldsOptional:
    """Marker class for making fields optional."""
    pass

class PartialLiteralMixin:
    """Mixin for partial literal types."""
    pass

def create_field(
    annotation: Union[Type[Any], Any],
    default: Optional[Any] = None,
    **kwargs: Any
) -> FieldInfo:
    """Create a Pydantic field with proper typing."""
    if default is None and 'default_factory' not in kwargs:
        return Field(default=None, annotation=annotation, **kwargs)
    return Field(default=default, annotation=annotation, **kwargs)

def _process_generic_arg(
    arg: Any,
    make_fields_optional: bool = False,
) -> Any:
    """Process generic type arguments for partial models."""
    arg_origin = get_origin(arg)
    if arg_origin is not None:
        nested_args = get_args(arg)
        modified_nested_args = tuple(
            _process_generic_arg(t, make_fields_optional=make_fields_optional)
            for t in nested_args
        )
        if arg_origin in UNION_ORIGINS:
            return Union[modified_nested_args]  # type: ignore
        return arg_origin[modified_nested_args]
    else:
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return (
                Partial[arg, MakeFieldsOptional]  # type: ignore[valid-type]
                if make_fields_optional
                else Partial[arg]
            )
        else:
            return arg

def _make_field_optional(
    field: FieldInfo,
) -> tuple[Any, FieldInfo]:
    """Make a field optional and handle nested models."""
    tmp_field = deepcopy(field)
    annotation = field.annotation

    if get_origin(annotation) is not None:
        generic_base = get_origin(annotation)
        generic_args = get_args(annotation)
        modified_args = tuple(
            _process_generic_arg(arg, make_fields_optional=True)
            for arg in generic_args
        )
        new_type = generic_base[modified_args] if generic_base else Any
        return (new_type, create_field(new_type))
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        partial_type = Partial[annotation]
        return (partial_type, create_field(partial_type, default_factory=dict))
    else:
        tmp_field.annotation = Optional[annotation]
        tmp_field.default = None
        return tmp_field.annotation, tmp_field

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
            try:
                if mode == Mode.ANTHROPIC_JSON:
                    if json_chunk := chunk.delta.text:
                        yield json_chunk
                if mode == Mode.ANTHROPIC_TOOLS:
                    yield chunk.delta.partial_json
                if mode == Mode.GEMINI_JSON:
                    yield chunk.text
                if mode == Mode.GEMINI_TOOLS:
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
        """Extract JSON from completion stream asynchronously."""
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
    def __class_getitem__(
        cls,
        wrapped_class: Union[Type[T_Model], tuple[Type[T_Model], Type[MakeFieldsOptional]]]
    ) -> Type[T_Model]:
        """Convert model to one that inherits from PartialBase.

        We don't make the fields optional at this point, we just wrap them with `Partial` so
        the names of the nested models will be `Partial{ModelName}`. During validation,
        we'll generate a true partial model to support partially defined fields.
        """
        make_fields_optional = None
        if isinstance(wrapped_class, tuple):
            wrapped_class, make_fields_optional = wrapped_class

        def _wrap_models(field: FieldInfo) -> tuple[object, FieldInfo]:
            tmp_field = deepcopy(field)
            annotation = field.annotation

            if get_origin(annotation) is not None:
                generic_base = get_origin(annotation)
                generic_args = get_args(annotation)
                modified_args = tuple(_process_generic_arg(arg) for arg in generic_args)
                tmp_field.annotation = generic_base[modified_args] if generic_base else None
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                tmp_field.annotation = Partial[annotation]
            return tmp_field.annotation, tmp_field
