from __future__ import annotations
from inspect import isclass
import typing
from pydantic import BaseModel, create_model
from enum import Enum


from instructor.dsl.partial import Partial
from instructor.function_calls import OpenAISchema


T = typing.TypeVar("T")


class AdapterBase(BaseModel):
    pass


class ModelAdapter(typing.Generic[T]):
    """
    Accepts a response model and returns a BaseModel with the response model as the content.
    """

    def __class_getitem__(cls, response_model: type[BaseModel]) -> type[BaseModel]:
        assert is_simple_type(response_model), "Only simple types are supported"
        return create_model(
            "Response",
            content=(response_model, ...),
            __doc__="Correctly Formatted and Extracted Response.",
            __base__=(AdapterBase, OpenAISchema),
        )


def validateIsSubClass(response_model: type):
    """
    Temporary guard against issues with generics in Python 3.9
    """
    import sys

    if sys.version_info < (3, 10):
        if len(typing.get_args(response_model)) == 0:
            return False
        return issubclass(typing.get_args(response_model)[0], BaseModel)
    try:
        # Add a guard here to prevent issues with GenericAlias
        import types

        if isinstance(response_model, types.GenericAlias):
            return False
    except Exception:
        pass

    return issubclass(response_model, BaseModel)


def is_simple_type(
    response_model: type[BaseModel] | str | int | float | bool | typing.Any,
) -> bool:
    # ! we're getting mixes between classes and instances due to how we handle some
    # ! response model types, we should fix this in later PRs

    # Special case for Python 3.9: Directly handle list[Union[int, str]] pattern
    import sys

    if sys.version_info < (3, 10):
        # Check if it's a list type with Union arguments using string representation
        if str(response_model).startswith("list[typing.Union[") or "list[Union[" in str(
            response_model
        ):
            return True

    try:
        if isclass(response_model) and validateIsSubClass(response_model):
            return False
    except TypeError:
        # ! In versions < 3.11, typing.Iterable is not a class, so we can't use isclass
        # ! for now if `response_model` is an Iterable isclass and issubclass will raise
        # ! TypeError, so we need to check if `response_model` is an Iterable
        # ! This is a workaround for now, we should fix this in later PRs
        return False

    # Get the origin of the response model
    origin = typing.get_origin(response_model)

    # Handle special case for list[int | str], list[Union[int, str]] or similar type patterns
    # Identify a list type by checking for various origins it might have
    if origin in {typing.Iterable, Partial, list}:
        # Always treat regular Python list as a simple type
        if origin is list:
            return True

        # Extract the inner types from the list
        args = typing.get_args(response_model)
        if args and len(args) == 1:
            inner_arg = args[0]
            # Special handling for Union types
            inner_origin = typing.get_origin(inner_arg)

            # Explicit check for Union types - try different patterns across Python versions
            if (
                inner_origin is typing.Union
                or inner_origin == typing.Union
                or str(inner_origin) == "typing.Union"
                or str(type(inner_arg)) == "<class 'typing._UnionGenericAlias'>"
            ):
                return True

            # Check for Python 3.10+ pipe syntax
            if hasattr(inner_arg, "__or__"):
                return True

            # For simple list with basic types, also return True
            if inner_arg in {str, int, float, bool}:
                return True

        # For other iterable patterns, return False (e.g., streaming types)
        return False

    if response_model in {
        str,
        int,
        float,
        bool,
    }:
        return True

    # If the response_model is a simple type like annotated
    if origin in {
        typing.Annotated,
        typing.Literal,
        typing.Union,
        list,  # origin of List[T] is list
    }:
        return True

    if isclass(response_model) and issubclass(response_model, Enum):
        return True

    return False
