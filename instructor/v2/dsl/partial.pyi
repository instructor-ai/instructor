from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from typing import Any, Generic, TypeVar, overload

from pydantic import BaseModel
from pydantic.fields import FieldInfo

T_Model = TypeVar("T_Model", bound=BaseModel)

class MakeFieldsOptional: ...

class PartialBase(BaseModel, Generic[T_Model]):
    @staticmethod
    def extract_json(
        completion: Iterable[Any],
        stream_extractor: Callable[[Iterable[Any]], Generator[str, None, None]] | Any,  # noqa: UP043
        on_event: Callable[..., Any] | None = None,
    ) -> Generator[str, None, None]: ...  # noqa: UP043
    @staticmethod
    def extract_json_async(
        completion: AsyncGenerator[Any, None],  # noqa: UP043
        stream_extractor: Callable[
            [AsyncGenerator[Any, None]], AsyncGenerator[str, None]  # noqa: UP043
        ]
        | Any,
        on_event: Callable[..., Any] | None = None,
    ) -> AsyncGenerator[str, None]: ...  # noqa: UP043

class PartialLiteralMixin: ...

def _make_field_optional(field: FieldInfo) -> tuple[Any, FieldInfo]: ...

class _PartialFactory:
    @overload
    def __getitem__(self, wrapped_class: type[T_Model]) -> type[T_Model]: ...
    @overload
    def __getitem__(
        self,
        wrapped_class: tuple[type[T_Model], type[MakeFieldsOptional]],
    ) -> type[T_Model]: ...

Partial: _PartialFactory
