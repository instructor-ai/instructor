from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from instructor.v2.providers.openai.handlers import OpenAIToolsHandler


class User(BaseModel):
    name: str


def test_openai_tools_streaming_iterable_not_parallel():
    handler = OpenAIToolsHandler()
    response_model, kwargs = handler.prepare_request(
        Iterable[User],
        {"stream": True},
    )

    assert kwargs["tool_choice"] != "auto"
    assert handler._consume_streaming_flag(response_model)
