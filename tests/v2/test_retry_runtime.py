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
    assert finalized._raw_response is response  # type: ignore[attr-defined]


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

    hooks = SimpleNamespace(
        emit_completion_arguments=lambda **kwargs: emitted["args"].append(kwargs),
        emit_completion_response=lambda response: emitted["responses"].append(response),
        emit_parse_error=lambda error: emitted["errors"].append(error),
        emit_completion_error=lambda _error, **_kw: None,
        emit_completion_last_attempt=lambda _error, **_kw: None,
    )

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


def _stub_retry_dependencies(
    monkeypatch: pytest.MonkeyPatch, parser: Any, reask: Any
) -> None:
    def no_validate(_provider: Provider, _mode: Mode) -> None:
        return None

    def get_handlers(_provider: Provider, _mode: Mode) -> SimpleNamespace:
        return SimpleNamespace(response_parser=parser, reask_handler=reask)

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


def test_retry_sync_v2_emits_completion_error_metadata_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_events: list[dict[str, Any]] = []

    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response is not None and isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    _stub_retry_dependencies(monkeypatch, always_fail_parser, reask_kwargs)

    hooks = SimpleNamespace(
        emit_completion_arguments=lambda **_kwargs: None,
        emit_completion_response=lambda _response: None,
        emit_parse_error=lambda _error: None,
        emit_completion_error=lambda error, **kw: error_events.append(
            {"error": error, **kw}
        ),
        emit_completion_last_attempt=lambda _error, **_kw: None,
    )

    with pytest.raises(InstructorRetryException):
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=3,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert [event["attempt_number"] for event in error_events] == [1, 2, 3]
    assert all(event["max_attempts"] == 3 for event in error_events)
    assert [event["is_last_attempt"] for event in error_events] == [False, False, True]
    assert all(isinstance(event["error"], ValidationError) for event in error_events)


def test_retry_sync_v2_emits_completion_last_attempt_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    last_attempt_events: list[dict[str, Any]] = []

    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response is not None and isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    _stub_retry_dependencies(monkeypatch, always_fail_parser, reask_kwargs)

    hooks = SimpleNamespace(
        emit_completion_arguments=lambda **_kwargs: None,
        emit_completion_response=lambda _response: None,
        emit_parse_error=lambda _error: None,
        emit_completion_error=lambda _error, **_kw: None,
        emit_completion_last_attempt=lambda error, **kw: last_attempt_events.append(
            {"error": error, **kw}
        ),
    )

    with pytest.raises(InstructorRetryException):
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert len(last_attempt_events) == 1
    event = last_attempt_events[0]
    assert event["attempt_number"] == 2
    assert event["max_attempts"] == 2
    assert event["is_last_attempt"] is True
    assert isinstance(event["error"], ValidationError)


def test_retry_sync_v2_emits_completion_error_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_events: list[dict[str, Any]] = []
    last_attempt_events: list[dict[str, Any]] = []

    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("network unreachable")

    def never_called_parser(**_kwargs: Any) -> Answer:
        raise AssertionError("parser must not be reached")

    def never_called_reask(
        _kwargs: dict[str, Any], _response: Any, _exception: ValidationError
    ) -> dict[str, Any]:
        raise AssertionError("reask must not be reached")

    _stub_retry_dependencies(monkeypatch, never_called_parser, never_called_reask)

    hooks = SimpleNamespace(
        emit_completion_arguments=lambda **_kwargs: None,
        emit_completion_response=lambda _response: None,
        emit_parse_error=lambda _error: None,
        emit_completion_error=lambda error, **kw: error_events.append(
            {"error": error, **kw}
        ),
        emit_completion_last_attempt=lambda error, **kw: last_attempt_events.append(
            {"error": error, **kw}
        ),
    )

    with pytest.raises(InstructorRetryException):
        retry_sync_v2(
            func=boom,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    # Non-validation errors are not retried by default; both hooks fire once each.
    assert len(error_events) == 1
    assert error_events[0]["attempt_number"] == 1
    assert error_events[0]["max_attempts"] == 2
    assert isinstance(error_events[0]["error"], RuntimeError)

    assert len(last_attempt_events) == 1
    assert last_attempt_events[0]["is_last_attempt"] is True
    assert isinstance(last_attempt_events[0]["error"], RuntimeError)


def test_retry_sync_v2_attempt_metadata_keeps_legacy_handler_signature_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instructor.v2.core.hooks import Hooks

    legacy_errors: list[Exception] = []
    legacy_last: list[Exception] = []

    def legacy_on_error(error: Exception) -> None:
        legacy_errors.append(error)

    def legacy_on_last_attempt(error: Exception) -> None:
        legacy_last.append(error)

    hooks = Hooks()
    hooks.on("completion:error", legacy_on_error)
    hooks.on("completion:last_attempt", legacy_on_last_attempt)

    def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response is not None and isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    _stub_retry_dependencies(monkeypatch, always_fail_parser, reask_kwargs)

    with pytest.raises(InstructorRetryException):
        retry_sync_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert len(legacy_errors) == 2
    assert len(legacy_last) == 1
    assert all(isinstance(err, ValidationError) for err in legacy_errors)
    assert isinstance(legacy_last[0], ValidationError)


@pytest.mark.asyncio
async def test_retry_async_v2_emits_attempt_metadata_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_events: list[dict[str, Any]] = []
    last_attempt_events: list[dict[str, Any]] = []

    async def fake_func(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"payload": kwargs["messages"][-1]["content"]}

    def always_fail_parser(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask_kwargs(
        kwargs: dict[str, Any], response: Any, exception: ValidationError
    ) -> dict[str, Any]:
        assert response is not None and isinstance(exception, ValidationError)
        return {
            **kwargs,
            "messages": [*kwargs["messages"], {"role": "user", "content": "retry"}],
        }

    _stub_retry_dependencies(monkeypatch, always_fail_parser, reask_kwargs)

    hooks = SimpleNamespace(
        emit_completion_arguments=lambda **_kwargs: None,
        emit_completion_response=lambda _response: None,
        emit_parse_error=lambda _error: None,
        emit_completion_error=lambda error, **kw: error_events.append(
            {"error": error, **kw}
        ),
        emit_completion_last_attempt=lambda error, **kw: last_attempt_events.append(
            {"error": error, **kw}
        ),
    )

    with pytest.raises(InstructorRetryException):
        await retry_async_v2(
            func=fake_func,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=2,
            args=(),
            kwargs={"messages": [{"role": "user", "content": "first"}]},
            strict=True,
            hooks=hooks,
        )

    assert [event["attempt_number"] for event in error_events] == [1, 2]
    assert all(event["max_attempts"] == 2 for event in error_events)
    assert [event["is_last_attempt"] for event in error_events] == [False, True]

    assert len(last_attempt_events) == 1
    assert last_attempt_events[0]["attempt_number"] == 2
    assert last_attempt_events[0]["is_last_attempt"] is True
