from typing import Any, Dict, Type, TypeVar, Optional, Union, overload
from .fields import FieldInfo

T = TypeVar('T', bound='BaseModel')

class BaseConfig:
    arbitrary_types_allowed: bool = False

class BaseModel:
    model_fields: Dict[str, FieldInfo]

    @classmethod
    def model_validate(cls: Type[T], obj: Any, context: Optional[Dict[str, Any]] = None) -> T: ...

class ConfigDict:
    def __init__(self, arbitrary_types_allowed: bool = False) -> None: ...

def Field(
    default: Any = None,
    *,
    default_factory: Optional[Any] = None,
    annotation: Optional[Any] = None,
    **kwargs: Any
) -> FieldInfo: ...

@overload
def create_model(
    __model_name: str,
    *,
    __config__: Optional[Type[BaseConfig]] = None,
    __base__: None = None,
    __module__: str = ...,
    __validators__: Optional[Dict[str, Any]] = None,
    __cls_kwargs__: Optional[Dict[str, Any]] = None,
    **field_definitions: Any,
) -> Type[BaseModel]: ...

@overload
def create_model(
    __model_name: str,
    *,
    __config__: Optional[Type[BaseConfig]] = None,
    __base__: Union[Type[T], tuple[Type[T], ...]],
    __module__: str = ...,
    __validators__: Optional[Dict[str, Any]] = None,
    __cls_kwargs__: Optional[Dict[str, Any]] = None,
    **field_definitions: Any,
) -> Type[T]: ...
