"""Type stubs for pydantic.v2.fields."""
from typing import Any, Callable, TypeVar, overload

T = TypeVar('T')

class FieldInfo:
    """Type stub for Pydantic v2 FieldInfo."""
    annotation: Any
    default: Any
    default_factory: Callable[[], Any] | None
    model_config: dict[str, Any]

    def __init__(
        self,
        *,
        annotation: Any = None,
        default: Any = None,
        default_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __get__(self, obj: None, owner: type[T]) -> FieldInfo: ...
    @overload
    def __get__(self, obj: T, owner: type[T]) -> Any: ...

    def __set__(self, obj: Any, value: Any) -> None: ...
    def __delete__(self, obj: Any) -> None: ...
