from __future__ import annotations

from collections.abc import Iterable

import pytest
from pydantic import BaseModel

from instructor.v2.providers.anthropic import parallel


class Weather(BaseModel):
    city: str


class Score(BaseModel):
    value: int


def test_parallel_schema_generation_is_owned_by_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[type[BaseModel]] = []

    def fake_schema(model: type[BaseModel]) -> dict[str, str]:
        calls.append(model)
        return {"name": model.__name__}

    monkeypatch.setattr(parallel, "generate_anthropic_schema", fake_schema)

    schemas = parallel.handle_parallel_model(Iterable[Weather | Score])

    assert schemas == [{"name": "Weather"}, {"name": "Score"}]
    assert calls == [Weather, Score]
