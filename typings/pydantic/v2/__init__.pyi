"""Type stubs for pydantic v2."""
from typing import Any, Dict, Type, TypeVar, Optional, Union, Callable, overload
from .fields import FieldInfo

T = TypeVar('T', bound='BaseModel')

class BaseModel:
    """Base class for Pydantic v2 models."""
    model_fields: Dict[str, FieldInfo]

    @classmethod
    def model_validate(
        cls: Type[T],
        obj: Any,
        *,
        context: Optional[Dict[str, Any]] = None,
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
    __base__: Union[Type[T], tuple[Type[T], ...]] = None,
    __module__: Optional[str] = None,
    __config__: Optional[ConfigDict] = None,
    __validators__: Optional[Dict[str, Callable[..., Any]]] = None,
    __cls_kwargs__: Optional[Dict[str, Any]] = None,
    **field_definitions: Any,
) -> Type[T]: ...
