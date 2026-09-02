"""Async validation decorators owned by the v2 runtime."""

from inspect import signature
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationInfo

from instructor.v2.core.errors import AsyncValidationError

ASYNC_VALIDATOR_KEY = "__async_validator__"
ASYNC_MODEL_VALIDATOR_KEY = "__async_model_validator__"
T = TypeVar("T", bound=Callable[..., Any])


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


def _collect_markers(model_cls: type[BaseModel], key: str) -> list[Any]:
    """Walk the MRO (base classes first) collecting distinct decorator markers.

    A subclass redefining a validator under the same attribute name overrides
    the base class's version, matching normal Python method-override semantics.
    """
    markers: dict[str, Any] = {}
    for klass in reversed(model_cls.__mro__):
        for name, member in vars(klass).items():
            marker = getattr(member, key, None)
            if marker is not None:
                markers[name] = marker
    return list(markers.values())


def model_declares_async_validators(model_cls: Any) -> bool:
    """Return True if `model_cls` (or a base class) declares any async validator."""
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        return False
    return bool(
        _collect_markers(model_cls, ASYNC_VALIDATOR_KEY)
        or _collect_markers(model_cls, ASYNC_MODEL_VALIDATOR_KEY)
    )


async def run_async_validators(value: Any, *, context: dict[str, Any] | None) -> Any:
    """Recursively run declared async field/model validators over a parsed value.

    Nested `BaseModel` instances (directly, or inside lists/tuples/dicts) are
    validated depth-first so a parent model's async validators see already
    -validated children. Returns the (possibly updated) value; raises
    `AsyncValidationError` aggregating every failure found in the subtree.
    """
    if isinstance(value, BaseModel):
        return await _run_on_model(value, context=context)
    if isinstance(value, list):
        return [await run_async_validators(item, context=context) for item in value]
    if isinstance(value, tuple):
        return tuple(
            [await run_async_validators(item, context=context) for item in value]
        )
    if isinstance(value, dict):
        return {
            key: await run_async_validators(item, context=context)
            for key, item in value.items()
        }
    return value


async def _run_on_model(
    model: BaseModel, *, context: dict[str, Any] | None
) -> BaseModel:
    model_cls = type(model)
    info = AsyncValidationContext(context or {})
    errors: list[ValueError] = []
    updates: dict[str, Any] = {}

    for field_name in model_cls.model_fields:
        current = getattr(model, field_name)
        try:
            new_value = await run_async_validators(current, context=context)
        except AsyncValidationError as exc:
            errors.extend(exc.errors)
            continue
        if new_value is not current:
            updates[field_name] = new_value
    if updates:
        model = model.model_copy(update=updates)
        updates = {}

    for field_names, validator, needs_info in _collect_markers(
        model_cls, ASYNC_VALIDATOR_KEY
    ):
        for field_name in field_names:
            if field_name not in model_cls.model_fields:
                continue
            current = getattr(model, field_name)
            try:
                new_value = (
                    await validator(model_cls, current, info)
                    if needs_info
                    else await validator(model_cls, current)
                )
            except ValueError as exc:
                errors.append(exc)
                continue
            if new_value is not current:
                updates[field_name] = new_value
    if updates:
        model = model.model_copy(update=updates)

    for validator, needs_info in _collect_markers(model_cls, ASYNC_MODEL_VALIDATOR_KEY):
        try:
            result = (
                await validator(model, info) if needs_info else await validator(model)
            )
        except ValueError as exc:
            errors.append(exc)
            continue
        if result is not None:
            model = result

    if errors:
        summary = "; ".join(str(error) for error in errors)
        raise AsyncValidationError(
            f"Async validation failed for {model_cls.__name__}: {summary}",
            errors=errors,
        )

    return model
