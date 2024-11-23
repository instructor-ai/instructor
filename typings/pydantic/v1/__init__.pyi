from typing import Any, TypeVar, Callable, overload
from .fields import FieldInfo

T = TypeVar('T', bound='BaseModel')

class BaseConfig:
    arbitrary_types_allowed: bool = False

class BaseModel:
    model_fields: dict[str, FieldInfo]

    @classmethod
    def model_validate(cls: type[T], obj: Any, context: dict[str, Any] | None = None) -> T: ...

class ConfigDict:
    def __init__(self, arbitrary_types_allowed: bool = False) -> None: ...

def Field(
    default: Any = None,
    *,
    default_factory: Any | None = None,
    annotation: Any | None = None,
    **kwargs: Any
) -> FieldInfo: ...

@overload
def create_model(
    __model_name: str,
    *,
    __config__: type[BaseConfig] | None = None,
    __base__: None = None,
    __module__: str = ...,
    __validators__: dict[str, Any] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any,
) -> type[BaseModel]: ...

@overload
def create_model(
    __model_name: str,
    *,
    __config__: type[BaseConfig] | None = None,
    __base__: type[T] | tuple[type[T], ...],
    __module__: str = ...,
    __validators__: dict[str, Any] | None = None,
    __cls_kwargs__: dict[str, Any] | None = None,
    **field_definitions: Any,
) -> type[T]: ...
