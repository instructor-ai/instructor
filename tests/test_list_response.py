from __future__ import annotations

from collections.abc import Iterable as ABCIterable
from typing import Any

from pydantic import BaseModel

from instructor.dsl import ListResponse
from instructor.dsl.iterable import IterableBase
from instructor.mode import Mode
from instructor.processing.response import process_response
from instructor.utils.core import prepare_response_model


class User(BaseModel):
    name: str


def test_listresponse_preserves_raw_response_on_slice() -> None:
    raw: Any = {"provider": "test"}
    resp = ListResponse([User(name="a"), User(name="b")], _raw_response=raw)

    assert resp.get_raw_response() is raw
    assert resp[0].name == "a"

    sliced = resp[1:]
    assert isinstance(sliced, ListResponse)
    assert sliced.get_raw_response() is raw
    assert sliced[0].name == "b"


def test_process_response_wraps_iterablebase_tasks_with_raw_response() -> None:
    class FakeIterableResponse(BaseModel, IterableBase):
        tasks: list[User]

        @classmethod
        def from_response(  # type: ignore[override]
            cls, _response: Any, **_kwargs: Any
        ) -> FakeIterableResponse:
            return cls(tasks=[User(name="x"), User(name="y")])

    # `process_response()` is typed with a BaseModel-bounded type variable for `response`,
    # so use a BaseModel instance here to keep `ty` happy.
    raw_response: Any = User(name="raw")
    out = process_response(
        raw_response,
        response_model=FakeIterableResponse,
        stream=False,
        mode=Mode.TOOLS,
    )

    assert isinstance(out, ListResponse)
    assert [u.name for u in out] == ["x", "y"]
    assert out.get_raw_response() is raw_response


def test_prepare_response_model_supports_list_and_iterable() -> None:
    prepared_list = prepare_response_model(list[User])
    assert prepared_list is not None
    assert issubclass(prepared_list, IterableBase)

    prepared_iterable = prepare_response_model(ABCIterable[User])  # type: ignore[index]
    assert prepared_iterable is not None
    assert issubclass(prepared_iterable, IterableBase)
