"""Async validation decorators owned by the v2 runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, MutableMapping
from inspect import signature
from typing import Any, Callable, TypeVar
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ValidationInfo

from instructor.v2.core.errors import AsyncValidationError

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])
_DeclaredValidators = tuple[
    list[tuple[tuple[str, ...], Callable[..., Any], bool]],
    list[tuple[Callable[..., Any], bool]],
]


class AsyncValidationContext:
    """Carry context through async validation hooks."""

    context: dict[str, Any]

    def __init__(self, context: dict[str, Any]):
        self.context = context


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


_DECLARED_VALIDATORS: MutableMapping[type[BaseModel], _DeclaredValidators] = (
    WeakKeyDictionary()
)


def _declared_async_validators(model_class: type[BaseModel]) -> _DeclaredValidators:
    """Collect the async field and model validators declared on a model class."""
    cached = _DECLARED_VALIDATORS.get(model_class)
    if cached is not None:
        return cached

    field_validators: list[tuple[tuple[str, ...], Callable[..., Any], bool]] = []
    model_validators: list[tuple[Callable[..., Any], bool]] = []
    seen: set[str] = set()

    for klass in model_class.__mro__:
        for name, attribute in vars(klass).items():
            if name in seen:
                continue
            seen.add(name)
            # Unwrap classmethod/staticmethod so the decorator metadata stays visible.
            candidate = getattr(attribute, "__func__", attribute)
            field_metadata = getattr(candidate, ASYNC_VALIDATOR_KEY, None)
            if field_metadata is not None:
                field_validators.append(field_metadata)
                continue
            model_metadata = getattr(candidate, ASYNC_MODEL_VALIDATOR_KEY, None)
            if model_metadata is not None:
                model_validators.append(model_metadata)

    declared = (field_validators, model_validators)
    _DECLARED_VALIDATORS[model_class] = declared
    return declared


def _iter_models(
    value: Any,
    path: tuple[str, ...] = (),
    visited: set[int] | None = None,
) -> Iterator[tuple[BaseModel, tuple[str, ...]]]:
    """Yield every model reachable from ``value`` with its dotted field path."""
    visited = set() if visited is None else visited

    if isinstance(value, BaseModel):
        if id(value) in visited:
            return
        visited.add(id(value))
        yield value, path
        for field_name in type(value).model_fields:
            yield from _iter_models(
                getattr(value, field_name, None), path + (field_name,), visited
            )
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            yield from _iter_models(item, path + (str(index),), visited)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_models(item, path + (str(key),), visited)


def has_async_validators(value: Any) -> bool:
    """Return whether ``value`` or any nested model declares async validators."""
    for model, _ in _iter_models(value):
        field_validators, model_validators = _declared_async_validators(type(model))
        if field_validators or model_validators:
            return True
    return False


async def _run_validator(
    func: Callable[..., Any], args: tuple[Any, ...], location: tuple[str, ...]
) -> ValueError | None:
    try:
        await func(*args)
    except Exception as exception:
        suffix = f" at {'.'.join(location)}" if location else ""
        return ValueError(f"Exception of {exception} encountered{suffix}")
    return None


async def async_validate_model(
    value: Any, validation_context: dict[str, Any] | None = None
) -> None:
    """Await the async validators declared on ``value`` and its nested models.

    Validators run concurrently and every failure is collected, so a single
    ``AsyncValidationError`` reports all of them. The retry loop treats that
    error like any other validation failure and reasks the model.

    Args:
        value: A model, or a container of models, produced by a response parser.
        validation_context: Context exposed to validators that declare an
            ``info`` parameter, mirroring Pydantic's validation context.

    Raises:
        AsyncValidationError: If at least one async validator fails.
        ValueError: If a field validator names a field the model does not define.
    """
    info = AsyncValidationContext(context=validation_context or {})
    calls: list[tuple[Callable[..., Any], tuple[Any, ...], tuple[str, ...]]] = []

    for model, path in _iter_models(value):
        field_validators, model_validators = _declared_async_validators(type(model))
        for fields, func, requires_info in field_validators:
            for field in fields:
                if field not in type(model).model_fields:
                    raise ValueError(f"Invalid Field of {field} provided")
                args: tuple[Any, ...] = (model, getattr(model, field))
                if requires_info:
                    args += (info,)
                calls.append((func, args, path + (field,)))
        for func, requires_info in model_validators:
            args = (model, info) if requires_info else (model,)
            calls.append((func, args, path))

    if not calls:
        return

    results = await asyncio.gather(*(_run_validator(*call) for call in calls))
    errors = [error for error in results if error is not None]
    if errors:
        error = AsyncValidationError(f"Validation errors: {errors}")
        error.errors = errors
        raise error
