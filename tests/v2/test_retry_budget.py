from __future__ import annotations

import builtins
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types import CompletionUsage
from pydantic import BaseModel, ValidationError

from instructor import Mode, Provider
from instructor.v2.core.client import (
    AsyncInstructor,
    AsyncResponse,
    Instructor,
    Response,
)
from instructor.v2.core.errors import (
    InstructorRetryException,
    TokenBudgetExceeded,
    TokenUsageUnavailableError,
)
from instructor.v2.core.hooks import Hooks
from instructor.v2.core.patch import patch_v2
from instructor.v2.core.retry import (
    _budget_error,
    _finalize_parsed_response,
    _usage_snapshot,
    _usage_total_tokens,
    retry_async_v2,
    retry_sync_v2,
)
from instructor.v2.core.usage import has_compatible_usage
from instructor.v2.dsl.response_list import ListResponse


class Answer(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        Answer.model_validate({"value": "invalid"})
    return exc_info.value


def _response(tokens: int, *, value: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        value=value,
        usage=CompletionUsage(
            completion_tokens=tokens,
            prompt_tokens=0,
            total_tokens=tokens,
        ),
    )


def _install_handlers(
    monkeypatch: pytest.MonkeyPatch,
    parser: Callable[..., Answer],
    reask: Callable[..., dict[str, Any]],
) -> None:
    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda _provider, _mode: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda _provider, _mode: SimpleNamespace(
            response_parser=parser,
            reask_handler=reask,
        ),
    )


def test_retry_budget_stops_before_sync_reask_at_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls = 0
    reask_calls = 0
    response = _response(100, value=None)
    parse_events: list[dict[str, Any]] = []
    last_attempts: list[tuple[Exception, dict[str, Any]]] = []

    def create(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal provider_calls
        provider_calls += 1
        return response

    def parse(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask(**call: Any) -> dict[str, Any]:
        nonlocal reask_calls
        reask_calls += 1
        return cast(dict[str, Any], call["kwargs"])

    _install_handlers(monkeypatch, parse, reask)
    hooks = Hooks()
    hooks.on(
        "parse:error",
        lambda _error, **metadata: parse_events.append(metadata),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **metadata: last_attempts.append((error, metadata)),
    )

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={},
            strict=True,
            hooks=hooks,
            token_budget=100,
        )

    error = exc_info.value
    assert isinstance(error, InstructorRetryException)
    assert error.budget == 100
    assert error.n_attempts == 1
    assert error.last_completion is response
    assert error.total_usage.total_tokens == 100
    assert len(error.failed_attempts or []) == 1
    assert provider_calls == 1
    assert reask_calls == 0
    assert parse_events == [
        {"attempt_number": 1, "max_attempts": 4, "is_last_attempt": True}
    ]
    assert last_attempts == [
        (
            error,
            {"attempt_number": 1, "max_attempts": 4, "is_last_attempt": True},
        )
    ]


def test_retry_budget_returns_valid_response_that_crosses_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [_response(60, value=None), _response(60, value=7)]
    provider_calls = 0
    snapshots: list[tuple[CompletionUsage, int]] = []

    def create(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal provider_calls
        response = responses[provider_calls]
        provider_calls += 1
        return response

    def parse(*, response: SimpleNamespace, **_kwargs: Any) -> Answer:
        if response.value is None:
            raise _validation_error()
        return Answer(value=response.value)

    _install_handlers(
        monkeypatch,
        parse,
        lambda **call: cast(dict[str, Any], call["kwargs"]),
    )
    hooks = Hooks()

    def record_usage(usage: Any, *, attempt_number: int) -> None:
        snapshots.append((cast(CompletionUsage, usage), attempt_number))

    hooks.on("completion:usage", record_usage)

    result = retry_sync_v2(
        func=create,
        response_model=Answer,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
        context=None,
        max_retries=3,
        args=(),
        kwargs={},
        strict=True,
        hooks=hooks,
        token_budget=100,
    )

    assert result == Answer(value=7)
    assert provider_calls == 2
    assert [usage.total_tokens for usage, _ in snapshots] == [60, 120]
    assert [attempt for _, attempt in snapshots] == [1, 2]
    assert snapshots[0][0] is not snapshots[1][0]
    assert snapshots[0][0].total_tokens == 60
    assert cast(Any, result)._total_usage.total_tokens == 120


def test_retry_budget_fails_closed_when_usage_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls = 0
    reask_calls = 0

    def create(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal provider_calls
        provider_calls += 1
        return SimpleNamespace(value=None)

    def parse(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask(**call: Any) -> dict[str, Any]:
        nonlocal reask_calls
        reask_calls += 1
        return cast(dict[str, Any], call["kwargs"])

    _install_handlers(monkeypatch, parse, reask)

    with pytest.raises(TokenUsageUnavailableError) as exc_info:
        retry_sync_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={},
            strict=True,
            token_budget=100,
        )

    assert exc_info.value.n_attempts == 1
    assert provider_calls == 1
    assert reask_calls == 0


@pytest.mark.asyncio
async def test_retry_budget_stops_before_async_reask_at_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls = 0
    reask_calls = 0
    usage_events: list[tuple[CompletionUsage, int]] = []
    last_attempts: list[TokenBudgetExceeded] = []

    async def create(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        nonlocal provider_calls
        provider_calls += 1
        return _response(75, value=None)

    def parse(**_kwargs: Any) -> Answer:
        raise _validation_error()

    def reask(**call: Any) -> dict[str, Any]:
        nonlocal reask_calls
        reask_calls += 1
        return cast(dict[str, Any], call["kwargs"])

    _install_handlers(monkeypatch, parse, reask)
    hooks = Hooks()
    hooks.on(
        "completion:usage",
        lambda usage, *, attempt_number: usage_events.append(
            (cast(CompletionUsage, usage), attempt_number)
        ),
    )
    hooks.on(
        "completion:last_attempt",
        lambda error, **_metadata: last_attempts.append(
            cast(TokenBudgetExceeded, error)
        ),
    )

    with pytest.raises(TokenBudgetExceeded) as exc_info:
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={},
            strict=True,
            hooks=hooks,
            token_budget=75,
        )

    assert exc_info.value.total_usage.total_tokens == 75
    assert provider_calls == 1
    assert reask_calls == 0
    assert [(usage.total_tokens, attempt) for usage, attempt in usage_events] == [
        (75, 1)
    ]
    assert last_attempts == [exc_info.value]

    with pytest.raises(TokenBudgetExceeded):
        await retry_async_v2(
            func=create,
            response_model=Answer,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs={},
            strict=True,
            token_budget=75,
        )

    assert provider_calls == 2
    assert reask_calls == 0


@pytest.mark.parametrize(
    ("token_budget", "response_model", "kwargs", "error_type"),
    [
        (0, Answer, {}, ValueError),
        (-1, Answer, {}, ValueError),
        (True, Answer, {}, TypeError),
        (1.5, Answer, {}, TypeError),
        (100, None, {}, ValueError),
        (100, Answer, {"stream": True}, ValueError),
    ],
)
def test_retry_budget_rejects_unenforceable_configuration_before_provider_call(
    token_budget: Any,
    response_model: type[Answer] | None,
    kwargs: dict[str, Any],
    error_type: type[Exception],
) -> None:
    provider_calls = 0

    def create(*_args: Any, **_kwargs: Any) -> None:
        nonlocal provider_calls
        provider_calls += 1

    with pytest.raises(error_type):
        retry_sync_v2(
            func=create,
            response_model=response_model,
            provider=Provider.OPENAI,
            mode=Mode.JSON,
            context=None,
            max_retries=3,
            args=(),
            kwargs=kwargs,
            strict=True,
            token_budget=token_budget,
        )

    assert provider_calls == 0


def test_list_response_preserves_usage_snapshot_across_slices() -> None:
    raw_response = object()
    usage = CompletionUsage(
        completion_tokens=25,
        prompt_tokens=75,
        total_tokens=100,
    )

    result = _finalize_parsed_response(
        [Answer(value=1), Answer(value=2)],
        raw_response,
        total_usage=usage,
    )

    assert isinstance(result, ListResponse)
    assert result.get_total_usage() is not usage
    assert result.get_total_usage().total_tokens == 100
    assert result[:1].get_raw_response() is raw_response
    assert result[:1].get_total_usage().total_tokens == 100


def test_finalize_updates_existing_list_response_metadata() -> None:
    raw_response = object()
    usage = CompletionUsage(
        completion_tokens=25,
        prompt_tokens=75,
        total_tokens=100,
    )
    result = ListResponse([Answer(value=1)])

    finalized = _finalize_parsed_response(
        result,
        raw_response,
        total_usage=usage,
    )

    assert finalized is result
    assert result[0] == Answer(value=1)
    assert result.get_raw_response() is raw_response
    assert result.get_total_usage().total_tokens == 100
    assert _finalize_parsed_response("plain", raw_response) == "plain"


def test_anthropic_total_includes_cache_token_fields() -> None:
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=20,
        cache_creation_input_tokens=30,
        cache_read_input_tokens=40,
    )

    assert _usage_total_tokens(usage) == 100


def test_budget_error_rejects_usage_without_a_token_total() -> None:
    usage = SimpleNamespace(label="unsupported")

    error = _budget_error(
        token_budget=100,
        usage_available=True,
        total_usage=usage,
        attempt_number=1,
        response=object(),
        kwargs={},
        failed_attempts=[],
    )

    assert isinstance(error, TokenUsageUnavailableError)
    assert error.total_usage is not usage
    assert error.total_usage.label == "unsupported"
    assert _usage_snapshot(usage) is not usage


def test_usage_compatibility_handles_missing_anthropic_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_anthropic_types(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "anthropic.types":
            raise ImportError("anthropic is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_anthropic_types)

    assert not has_compatible_usage(SimpleNamespace(), SimpleNamespace())


def test_patched_create_validates_budget_before_request_processing() -> None:
    provider_calls = 0

    def create(*_args: Any, **_kwargs: Any) -> None:
        nonlocal provider_calls
        provider_calls += 1

    patched_create = patch_v2(
        func=create,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
    )

    with pytest.raises(ValueError, match="greater than zero"):
        patched_create(response_model=Answer, token_budget=0)

    assert provider_calls == 0


@pytest.mark.asyncio
async def test_patched_async_create_validates_budget_before_request_processing() -> (
    None
):
    provider_calls = 0

    async def create(*_args: Any, **_kwargs: Any) -> None:
        nonlocal provider_calls
        provider_calls += 1

    patched_create = patch_v2(
        func=create,
        provider=Provider.OPENAI,
        mode=Mode.JSON,
    )

    with pytest.raises(ValueError, match="greater than zero"):
        await patched_create(response_model=Answer, token_budget=0)

    assert provider_calls == 0


def test_sync_client_surfaces_forward_explicit_budget() -> None:
    received: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Answer:
        received.append(kwargs)
        return Answer(value=1)

    response = Response(cast(Any, SimpleNamespace(create=create)))
    instructor = Instructor(client=None, create=create)

    assert (
        response.create(messages=[], response_model=Answer, token_budget=10).value == 1
    )
    assert (
        instructor.create(messages=[], response_model=Answer, token_budget=20).value
        == 1
    )
    assert [call["token_budget"] for call in received] == [10, 20]


@pytest.mark.asyncio
async def test_async_client_surfaces_forward_explicit_budget() -> None:
    received: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Answer:
        received.append(kwargs)
        return Answer(value=1)

    response = AsyncResponse(cast(Any, SimpleNamespace(create=create)))
    instructor = AsyncInstructor(client=None, create=create)

    assert (
        await response.create(messages=[], response_model=Answer, token_budget=10)
    ).value == 1
    assert (
        await instructor.create(messages=[], response_model=Answer, token_budget=20)
    ).value == 1
    assert [call["token_budget"] for call in received] == [10, 20]
