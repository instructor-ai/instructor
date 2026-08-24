"""Async validation decorators owned by the v2 runtime."""

from inspect import signature
from types import SimpleNamespace
from typing import Any, Callable, TypeVar

from pydantic import ValidationInfo

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])


class AsyncValidationContext:
    """Carry context through async validation hooks."""

    context: dict[str, Any]

    def __init__(self, context: dict[str, Any]):
        self.context = context


async def run_async_validators(
    value: Any, context: dict[str, Any] | None = None
) -> Any:
    """Run declared async validators on a parsed model and its nested models."""
    if isinstance(value, list):
        return [await run_async_validators(item, context) for item in value]
    if not hasattr(value, "__class__"):
        return value

    model_class = value.__class__
    info = SimpleNamespace(context=context or {})
    for member in model_class.__dict__.values():
        field_metadata = getattr(member, ASYNC_VALIDATOR_KEY, None)
        if field_metadata is not None:
            fields, validator, needs_info = field_metadata
            for field in fields:
                args = (
                    (model_class, getattr(value, field), info)
                    if needs_info
                    else (
                        model_class,
                        getattr(value, field),
                    )
                )
                await validator(*args)

        model_metadata = getattr(member, ASYNC_MODEL_VALIDATOR_KEY, None)
        if model_metadata is not None:
            validator, needs_info = model_metadata
            args = (value, info) if needs_info else (value,)
            result = await validator(*args)
            if result is not None:
                value = result

    for field in getattr(value.__class__, "model_fields", {}):
        nested = getattr(value, field, None)
        if isinstance(nested, (list, tuple)):
            for item in nested:
                await run_async_validators(item, context)
        elif hasattr(nested.__class__, "model_fields"):
            await run_async_validators(nested, context)
    return value


def async_field_validator(field: str, *fields: str) -> Callable[[T], T]:
    """Mark a callable as an async field validator."""
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
    """Mark a callable as an async model validator."""

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
