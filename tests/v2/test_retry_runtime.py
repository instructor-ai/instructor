from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from instructor import Mode, Provider
from instructor.v2.core.errors import InstructorRetryException
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.retry import (
    _finalize_parsed_response,
    _initialize_usage,
    retry_async_v2,
    retry_sync_v2,
)
from instructor.v2.dsl.response_list import ListResponse


class Answer(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    try:
        Answer.model_validate({"value": "bad"})
    except ValidationError as exc:
        return exc
    raise AssertionError("Expected a validation error")


def test_finalize_parsed_response_wraps_plain_list_and_sets_raw_response() -> None:
    response = object()
    parsed = [Answer(value=1), Answer(value=2)]

    finalized = _finalize_parsed_response(parsed, response)

    assert isinstance(finalized, ListResponse)
    assert list(finalized) == parsed
    assert finalized._raw_response is response


def test_initialize_usage_returns_openai_usage_shape() -> None:
    usage = _initialize_usage(Provider.OPENAI)

    assert usage.completion_tokens == 0
    assert usage.prompt_tokens == 0
    assert usage.total_tokens == 0


def test_retry_sync_v2_returns_raw_result_when_no_response_model() -> None:
    def fake_func(*args: Any, **kwargs: Any) -> str:
        return f"{args[0]}:{kwargs['suffix']}"

    result = retry_sync_v2(
        func=fake_func,
        response_model=None,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context=None,
        max_retries=1,
        args=("hello",),
        kwargs={"suffix": "world"},
        strict=True,
        hooks=None,
    )

    assert result == "hello:world"


def test_retry_sync_v2_reasks_after_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    parser_calls: list[str] = []
    emitted: dict[str, list[Any]] = {"args": [], "responses": [], "errors": []}

    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {"payload": kwargs["messages"][-1]["content"]}

    def fake_parser(**kwargs: Any) -> Answer:
        parser_calls.append(kwargs["response"]["payload"])
        if len(parser_calls) == 1:
            raise _validation_error()
        return Answer(value=7)

    def fake_reask_handler(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response == {"payload": "first"}
        assert isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "second"}],
        }

    hooks = Hooks()
    hooks.on("completion:kwargs", lambda **kwargs: emitted["args"].append(kwargs))
    hooks.on(
        "completion:response",
        lambda response: emitted["responses"].append(response),
    )
    hooks.on("parse:error", lambda error: emitted["errors"].append(error))

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=fake_reask_handler,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    result = retry_sync_v2(
        func=fake_func,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.TOOLS,
        context={"tenant": "acme"},
        max_retries=Retrying(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type(ValidationError),
            reraise=True,
        ),
        args=(),
        kwargs={"messages": [{"role": "user", "content": "first"}]},
        strict=True,
        hooks=hooks,
    )

    assert result.value == 7
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["content"] == "second"
    assert parser_calls == ["first", "second"]
    assert len(emitted["args"]) == 2
    assert len(emitted["responses"]) == 2
    assert len(emitted["errors"]) == 1
    assert isinstance(emitted["errors"][0], ValidationError)


def test_retry_sync_v2_raises_instructor_retry_exception_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], _response: Any, _exception: ValidationError
    ) -> dict[str, Any]:
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=always_fail_parser,
            reask_handler=reask_kwargs,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=Retrying(
                stop=stop_after_attempt(2),
                retry=retry_if_exception_type(ValidationError),
                reraise=True,
            ),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=None,
        )

    error = exc_info.value
    assert error.n_attempts == 1
    assert error.last_completion == {"payload": "first"}
    assert error.create_kwargs is not None
    assert error.create_kwargs["messages"][-1]["content"] == "first"
    assert len(error.failed_attempts or []) == 1


@pytest.mark.asyncio
async def test_retry_async_v2_raises_retry_exception_after_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser_calls: list[str] = []

    async def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def fake_parser(**kwargs: Any) -> Answer:
        parser_calls.append(kwargs["response"]["payload"])
        if len(parser_calls) == 1:
            raise _validation_error()
        return Answer(value=9)

    def reask_kwargs(
        kwargs: dict[str, Any], _response: Any, _exception: ValidationError
    ) -> dict[str, Any]:
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=reask_kwargs,
        )

    def update_usage(response: Any, total_usage: Any) -> Any:
        assert total_usage == {"tokens": 0}
        return response

    def initialize_usage(_provider: Provider) -> dict[str, int]:
        return {"tokens": 0}

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        no_validate,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        get_handlers,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.update_total_usage",
        update_usage,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry._initialize_usage",
        initialize_usage,
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=AsyncRetrying(
                stop=stop_after_attempt(2),
                retry=retry_if_exception_type(ValidationError),
                reraise=True,
            ),
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=None,
        )

    assert exc_info.value.n_attempts == 1
    assert parser_calls == ["first"]
