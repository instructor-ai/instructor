"""Async validation decorators owned by the v2 runtime."""

from __future__ import annotations

from inspect import signature
from typing import Any, Callable, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationInfo
from typing_extensions import TypeAliasType

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])


class AsyncValidationContext:
    """Carry context through async validation hooks."""

    context: dict[str, Any]

    def __init__(self, context: dict[str, Any]):
        self.context = context


def async_field_validator(field: str, *fields: str) -> Callable[[T], T]:
    """Mark an async field validator. Runtime response models with markers are rejected."""
    field_names = field, *fields

    def decorator(func: T) -> T:
        params = signature(func).parameters
        requires_validation_context = False
        if len(params) == 3:
            if "info" not in params:
                raise ValueError(
                    "Async validator can only have a value parameter and an optional info parameter"
                )
            if params["info"].annotation != ValidationInfo:
                raise ValueError(
                    "Async validator info parameter must be of type ValidationInfo"
                )
            requires_validation_context = True

        setattr(
            func, ASYNC_VALIDATOR_KEY, (field_names, func, requires_validation_context)
        )
        return func

    return decorator


def async_model_validator() -> Callable[[T], T]:
    """Mark an async model validator. Runtime response models with markers are rejected."""

    def decorator(func: T) -> T:
        params = signature(func).parameters
        requires_validation_context = False
        if len(params) > 2:
            raise ValueError("Invalid Parameter Count!")

        if len(params) == 2:
            if "info" not in params:
                raise ValueError(
                    "Async validator can only have a value parameter and an optional info parameter"
                )
            if params["info"].annotation != ValidationInfo:
                raise ValueError(
                    "Async validator info parameter must be of type ValidationInfo"
                )
            requires_validation_context = True

        setattr(
            func,
            ASYNC_MODEL_VALIDATOR_KEY,
            (func, requires_validation_context),
        )
        return func

    return decorator


def reject_async_validators(response_model: Any) -> None:
    """Fail closed: the runtime does not execute marked asynchronous validators."""
    seen: set[int] = set()

    def visit(model: Any) -> None:
        if id(model) in seen:
            return
        seen.add(id(model))
        if isinstance(model, TypeAliasType):
            visit(model.__value__)
        origin = get_origin(model)
        if isinstance(origin, TypeAliasType):
            visit(origin.__value__)
        for argument in get_args(model):
            visit(argument)
        if not isinstance(model, type):
            for child in getattr(model, "models", ()):
                visit(child)
            return
        if issubclass(model, BaseModel):
            # Inspect effective attributes, including inherited descriptors.
            members = {}
            for base in reversed(model.__mro__):
                members.update(vars(base))
            for name, member in members.items():
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if (
                    getattr(member, ASYNC_VALIDATOR_KEY, None) is not None
                    or getattr(member, ASYNC_MODEL_VALIDATOR_KEY, None) is not None
                ):
                    raise ValueError(
                        f"{model.__name__}.{name}: Instructor async validators are "
                        "not supported by the runtime. Use Pydantic validators or "
                        "explicitly await validation before consuming results."
                    )
            for field in model.model_fields.values():
                visit(field.annotation)
        else:
            for annotation in getattr(model, "__annotations__", {}).values():
                visit(annotation)

    visit(response_model)
