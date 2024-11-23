"""Type stubs for pydantic.fields."""
from typing import Any, Callable, TypeVar, overload

T = TypeVar('T')

class FieldInfo:
    """Type stub for Pydantic FieldInfo."""
    annotation: Any
    default: Any
    default_factory: None | Callable[[], Any]
    model_config: dict[str, Any]

    def __init__(
        self,
        *,
        annotation: Any = None,
        default: Any = None,
        default_factory: None | Callable[[], Any] = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __get__(self, obj: None, owner: type[T]) -> FieldInfo: ...
    @overload
    def __get__(self, obj: T, owner: type[T]) -> Any: ...

    def __set__(self, obj: Any, value: Any) -> None: ...
    def __delete__(self, obj: Any) -> None: ...
