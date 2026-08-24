from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from instructor.v2.core.errors import AsyncValidationError, InstructorRetryException
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.core.retry import retry_async_v2
from instructor.v2.validation import async_field_validator, async_model_validator
from instructor.v2.validation.async_validators import run_async_validators


class Nested(BaseModel):
    value: str


class Response(BaseModel):
    value: str
    nested: Nested

    @async_field_validator("value")
    async def validate_value(cls: type["Response"], value: str) -> str:
        if value != "valid":
            raise ValueError("value is invalid")
        return value

    @async_model_validator()
    async def normalize(model: "Response") -> "Response":
        return model.model_copy(update={"value": model.value.upper()})


@pytest.mark.asyncio
async def test_run_async_validators_runs_model_validators_and_nested_models() -> None:
    result = await run_async_validators(
        Response(value="valid", nested=Nested(value="child")), {"request_id": "1"}
    )

    assert result.value == "VALID"


@pytest.mark.asyncio
async def test_run_async_validators_propagates_field_validation_failure() -> None:
    with pytest.raises(ValueError, match="value is invalid"):
        await run_async_validators(
            Response(value="invalid", nested=Nested(value="child"))
        )


@pytest.mark.asyncio
async def test_async_retry_path_runs_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_func(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {"value": "invalid"}

    def fake_parser(**_kwargs: Any) -> Response:
        return Response(value="invalid", nested=Nested(value="child"))

    monkeypatch.setattr(
        "instructor.v2.core.retry.RegistryValidationMixin.validate_mode_registration",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "instructor.v2.core.retry.mode_registry.get_handlers",
        lambda *_args: SimpleNamespace(
            response_parser=fake_parser,
            reask_handler=lambda **kwargs: kwargs["kwargs"],
        ),
    )

    with pytest.raises(InstructorRetryException) as exc_info:
        await retry_async_v2(
            func=fake_func,
            response_model=Response,
            provider=Provider.OPENAI,
            mode=Mode.TOOLS,
            context=None,
            max_retries=0,
            args=(),
            kwargs={},
            strict=True,
        )

    assert isinstance(exc_info.value.failed_attempts[0].exception, AsyncValidationError)
