from typing import Any, TypeVar

T = TypeVar('T')

class FieldInfo:
    annotation: type[Any] | None
    default: Any
    default_factory: Any | None

    def __init__(
        self,
        default: Any = None,
        *,
        default_factory: Any | None = None,
        annotation: type[Any] | None = None,
        **kwargs: Any
    ) -> None: ...
