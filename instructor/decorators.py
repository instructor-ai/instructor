from typing import Callable, Any, TypeVar
from inspect import signature
from pydantic import ValidationInfo

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])


class AsyncValidationContext:
    context: dict[str, Any]

    def __init__(self, context: dict[str, Any]):
        self.context = context


class AsyncValidationError(ValueError):
    errors: list[ValueError]


def async_field_validator(field: str, *fields: str) -> Callable[[T], T]:
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
