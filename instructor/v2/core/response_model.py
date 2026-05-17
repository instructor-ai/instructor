"""Response-model preparation helpers owned by the v2 runtime."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, TypeVar, Union, cast, get_args, get_origin

from pydantic import BaseModel, create_model

T = TypeVar("T")


def is_typed_dict(cls: Any) -> bool:
    return (
        isinstance(cls, type)
        and issubclass(cls, dict)
        and hasattr(cls, "__annotations__")
    )


def is_simple_type(typehint: type[T]) -> bool:
    from instructor.v2.dsl.simple_type import is_simple_type as _is_simple_type

    return _is_simple_type(typehint)


def prepare_response_model(response_model: type[T] | None) -> type[T] | None:
    """Normalize user response-model inputs into runtime-ready model classes."""
    if response_model is None:
        return None

    origin = get_origin(response_model)
    if origin is list and is_simple_type(response_model):
        args = get_args(response_model)
        inner = args[0] if args else None

        def _is_model_type(candidate: Any) -> bool:
            if inspect.isclass(candidate) and issubclass(candidate, BaseModel):
                return True
            return get_origin(candidate) is Union and all(
                inspect.isclass(member) and issubclass(member, BaseModel)
                for member in get_args(candidate)
            )

        if inner is not None and _is_model_type(inner):
            origin = list
        else:
            from instructor.v2.dsl.simple_type import ModelAdapter

            response_model = ModelAdapter.__class_getitem__(response_model)  # type: ignore[arg-type]
            origin = get_origin(response_model)

    if is_typed_dict(response_model):
        model_name = getattr(response_model, "__name__", "TypedDictModel")
        annotations = getattr(response_model, "__annotations__", {})
        response_model = cast(
            type[BaseModel],
            create_model(
                model_name,
                **{name: (annotation, ...) for name, annotation in annotations.items()},
            ),
        )

    origin = get_origin(response_model)
    if origin in {Iterable, list}:
        from instructor.v2.dsl.iterable import IterableModel

        args = get_args(response_model)
        if not args or args[0] is None:
            raise ValueError(
                "response_model must be parameterized, e.g. list[User] or Iterable[User]"
            )
        iterable_element_class = args[0]
        if is_typed_dict(iterable_element_class):
            iterable_element_class = cast(
                type[BaseModel],
                create_model(
                    getattr(iterable_element_class, "__name__", "TypedDictModel"),
                    **{
                        name: (annotation, ...)
                        for name, annotation in getattr(
                            iterable_element_class,
                            "__annotations__",
                            {},
                        ).items()
                    },
                ),
            )
        response_model = IterableModel(cast(type[BaseModel], iterable_element_class))

    if is_simple_type(response_model):
        from instructor.v2.dsl.simple_type import ModelAdapter

        response_model = ModelAdapter.__class_getitem__(response_model)  # type: ignore[arg-type]

    from instructor.v2.core.function_calls import (
        ResponseSchema,
        openai_schema,
        response_schema,
    )

    if inspect.isclass(response_model) and not issubclass(
        response_model, ResponseSchema
    ):
        response_model = response_schema(response_model)  # type: ignore
    elif not inspect.isclass(response_model):
        response_model = openai_schema(response_model)  # type: ignore

    return response_model
