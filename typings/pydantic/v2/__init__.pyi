"""Type stubs for pydantic v2."""
from typing import Any, TypeVar, Callable, overload
from .fields import FieldInfo

T = TypeVar('T', bound='BaseModel')

class BaseModel:
    """Base class for Pydantic v2 models."""
    model_fields: dict[str, FieldInfo]

    @classmethod
    def model_validate(
        cls: type[T],
        obj: Any,
        *,
        context: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> T: ...

    def __init__(self, **data: Any) -> None: ...

class ConfigDict:
    """Type stub for Pydantic v2 ConfigDict."""
    def __init__(self, arbitrary_types_allowed: bool = False, **kwargs: Any) -> None: ...

@overload
def Field(
    default: Any = None,
    *,
    default_factory: Callable[[], Any] | None = None,
    annotation: Any | None = None,
    **kwargs: Any,
) -> FieldInfo: ...

@overload
def Field(
    default_factory: Callable[[], Any],
    *,
    annotation: Any | None = None,
    **kwargs: Any,
) -> FieldInfo: ...

def create_model(
    model_name: str,
    *,
    __base__: type[T] | tuple[type[T], ...] = None,
    __module__: str | None = None,
    __config__: ConfigDict | None = None,
    __validators__: dict[str, Callable[..., Any]] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any,
) -> type[T]: ...
