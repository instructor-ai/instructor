"""Type stubs for pydantic.v2.fields."""
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, overload

T = TypeVar('T')

class FieldInfo:
    """Type stub for Pydantic v2 FieldInfo."""
    annotation: Any
    default: Any
    default_factory: Optional[Callable[[], Any]]
    model_config: Dict[str, Any]

    def __init__(
        self,
        *,
        annotation: Any = None,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __get__(self, obj: None, owner: Type[T]) -> 'FieldInfo': ...
    @overload
    def __get__(self, obj: T, owner: Type[T]) -> Any: ...

    def __set__(self, obj: Any, value: Any) -> None: ...
    def __delete__(self, obj: Any) -> None: ...
