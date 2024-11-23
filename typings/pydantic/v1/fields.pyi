from typing import Any, Optional, Type, TypeVar

T = TypeVar('T')

class FieldInfo:
    annotation: Optional[Type[Any]]
    default: Any
    default_factory: Optional[Any]

    def __init__(
        self,
        default: Any = None,
        *,
        default_factory: Optional[Any] = None,
        annotation: Optional[Type[Any]] = None,
        **kwargs: Any
    ) -> None: ...
