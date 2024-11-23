"""Type stubs for pydantic."""
from typing import Any, Type, TypeVar, Optional, Dict, Callable, Union, Tuple, Generic, overload
from .fields import FieldInfo

T = TypeVar('T')
ModelT = TypeVar('ModelT', bound='BaseModel')

class BaseModel:
    """Base class for Pydantic models."""
    model_fields: Dict[str, FieldInfo]
    __config__: Dict[str, Any]

    @classmethod
    def model_validate(
        cls: Type[ModelT],
        obj: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> ModelT: ...

    def __init__(self, **data: Any) -> None: ...

class ConfigDict(Dict[str, Any]):
    """Type stub for Pydantic ConfigDict."""
    def __init__(self, **kwargs: Any) -> None: ...

@overload
def Field(
    default: Any = None,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    annotation: Optional[Any] = None,
    **kwargs: Any,
) -> FieldInfo: ...

@overload
def Field(
    default_factory: Callable[[], Any],
    *,
    annotation: Optional[Any] = None,
    **kwargs: Any,
) -> FieldInfo: ...

def create_model(
    model_name: str,
    *,
    __base__: Union[Type[ModelT], Tuple[Type[ModelT], ...]] = None,
    __module__: Optional[str] = None,
    __config__: Optional[Dict[str, Any]] = None,
    __doc__: Optional[str] = None,
    __validators__: Optional[Dict[str, Callable[..., Any]]] = None,
    __cls_kwargs__: Optional[Dict[str, Any]] = None,
    **field_definitions: Any,
) -> Type[ModelT]: ...
