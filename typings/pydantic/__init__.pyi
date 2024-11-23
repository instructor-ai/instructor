"""Type stubs for pydantic."""
from typing import Any, TypeVar, overload
from .fields import FieldInfo

T = TypeVar('T')
ModelT = TypeVar('ModelT', bound='BaseModel')

class BaseModel:
    """Base class for Pydantic models."""
    model_fields: dict[str, FieldInfo]
    __config__: dict[str, Any]

    @classmethod
    def model_validate(
        cls: type[ModelT],
        obj: Any,
        *,
        context: dict[str, Any] | None = None,
        strict: bool = False,
    ) -> ModelT: ...

    def __init__(self, **data: Any) -> None: ...

class ConfigDict(dict[str, Any]):
    """Type stub for Pydantic ConfigDict."""
    def __init__(self, **kwargs: Any) -> None: ...

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
    __base__: type[ModelT] | tuple[type[ModelT], ...] = None,
    __module__: str | None = None,
    __config__: dict[str, Any] | None = None,
    __doc__: str | None = None,
    __validators__: dict[str, Callable[..., Any]] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any,
) -> type[ModelT]: ...
