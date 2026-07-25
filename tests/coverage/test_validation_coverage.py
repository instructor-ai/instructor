from importlib import import_module
from types import SimpleNamespace
from typing import Annotated, ClassVar, cast

import pytest
from openai import OpenAI
from pydantic import (
    AfterValidator,
    BaseModel,
    ValidationError,
    ValidationInfo,
    field_validator,
)

import instructor.v2 as instructor_v2
from instructor.v2.core.client import Instructor
from instructor.v2.core.validators import Validator
from instructor.v2.validation import (
    ASYNC_MODEL_VALIDATOR_KEY,
    ASYNC_VALIDATOR_KEY,
    AsyncValidationContext,
    async_field_validator,
    async_model_validator,
    llm_validator,
    openai_moderation,
)


class ValidationInput(BaseModel):
    value: str
    last_info: ClassVar[ValidationInfo]

    @field_validator("value")
    @classmethod
    def remember_info(cls, value: str, info: ValidationInfo) -> str:
        cls.last_info = info
        return value.strip()


@pytest.mark.asyncio
async def test_async_field_validators_keep_fields_and_use_pydantic_context():
    payload = ValidationInput.model_validate(
        {"value": "  hello  "}, context={"suffix": "!"}
    )
    info = ValidationInput.last_info

    @async_field_validator("value", "fallback")
    async def add_suffix(cls, value: str, info: ValidationInfo) -> str:
        assert cls is ValidationInput
        assert isinstance(info.context, dict)
        return value + info.context["suffix"]

    fields, original, needs_info = getattr(add_suffix, ASYNC_VALIDATOR_KEY)
    assert fields == ("value", "fallback")
    assert original is add_suffix
    assert needs_info is True
    assert await add_suffix(ValidationInput, payload.value, info) == "hello!"

    @async_field_validator("value")
    async def uppercase(cls, value: str) -> str:
        assert cls is ValidationInput
        return value.upper()

    fields, original, needs_info = getattr(uppercase, ASYNC_VALIDATOR_KEY)
    assert (fields, original, needs_info) == (("value",), uppercase, False)
    assert await uppercase(ValidationInput, payload.value) == "HELLO"


def test_async_field_validator_rejects_invalid_info_signatures():
    async def wrong_name(_cls, value: str, _context: ValidationInfo) -> str:
        return value

    with pytest.raises(ValueError, match="optional info parameter"):
        async_field_validator("value")(wrong_name)
    assert not hasattr(wrong_name, ASYNC_VALIDATOR_KEY)

    async def wrong_type(_cls, value: str, info: dict) -> str:
        return value + info.get("suffix", "")

    with pytest.raises(ValueError, match="must be of type ValidationInfo"):
        async_field_validator("value")(wrong_type)
    assert not hasattr(wrong_type, ASYNC_VALIDATOR_KEY)


@pytest.mark.asyncio
async def test_async_model_validators_keep_callable_and_use_pydantic_context():
    payload = ValidationInput.model_validate(
        {"value": "  hello  "}, context={"suffix": "?"}
    )
    info = ValidationInput.last_info

    @async_model_validator()
    async def append_suffix(model: ValidationInput, info: ValidationInfo):
        assert isinstance(info.context, dict)
        return model.model_copy(update={"value": model.value + info.context["suffix"]})

    original, needs_info = getattr(append_suffix, ASYNC_MODEL_VALIDATOR_KEY)
    assert original is append_suffix
    assert needs_info is True
    assert (await append_suffix(payload, info)).value == "hello?"

    @async_model_validator()
    async def uppercase(model: ValidationInput):
        return model.model_copy(update={"value": model.value.upper()})

    original, needs_info = getattr(uppercase, ASYNC_MODEL_VALIDATOR_KEY)
    assert (original, needs_info) == (uppercase, False)
    assert (await uppercase(payload)).value == "HELLO"


def test_async_model_validator_rejects_invalid_signatures():
    async def too_many(model: ValidationInput, info: ValidationInfo, extra: str):
        assert isinstance(info.context, dict)
        return model.model_copy(
            update={"value": model.value + info.context["suffix"] + extra}
        )

    with pytest.raises(ValueError, match="Invalid Parameter Count"):
        async_model_validator()(too_many)
    assert not hasattr(too_many, ASYNC_MODEL_VALIDATOR_KEY)

    async def wrong_name(model: ValidationInput, context: ValidationInfo):
        assert isinstance(context.context, dict)
        return model.model_copy(
            update={"value": model.value + context.context["suffix"]}
        )

    with pytest.raises(ValueError, match="optional info parameter"):
        async_model_validator()(wrong_name)
    assert not hasattr(wrong_name, ASYNC_MODEL_VALIDATOR_KEY)

    async def wrong_type(model: ValidationInput, info: dict):
        return model.model_copy(update={"value": model.value + info.get("suffix", "")})

    with pytest.raises(ValueError, match="must be of type ValidationInfo"):
        async_model_validator()(wrong_type)
    assert not hasattr(wrong_type, ASYNC_MODEL_VALIDATOR_KEY)


def test_async_validation_context_preserves_context_object():
    context = {"request_id": "req-123", "attempt": 2}
    validation_context = AsyncValidationContext(context)

    assert validation_context.context is context
    assert validation_context.context == {"request_id": "req-123", "attempt": 2}


class RecordingCompletions:
    def __init__(self, response: Validator):
        self.response = response
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def make_validation_model(response: Validator, *, allow_override: bool):
    completions = RecordingCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    validator = llm_validator(
        "must be lowercase",
        client=cast(Instructor, client),
        allow_override=allow_override,
        model="test-model",
        temperature=0.25,
    )

    class Name(BaseModel):
        value: Annotated[str, AfterValidator(validator)]

    return Name, completions


@pytest.mark.parametrize(
    ("response", "allow_override", "value", "expected"),
    [
        (Validator(is_valid=True), False, "jason", "jason"),
        (
            Validator(is_valid=False, reason="not lowercase", fixed_value="jason"),
            True,
            "Jason",
            "jason",
        ),
    ],
)
def test_llm_validator_validates_and_repairs_through_pydantic(
    response: Validator, allow_override: bool, value: str, expected: str
):
    model, completions = make_validation_model(response, allow_override=allow_override)

    assert model.model_validate({"value": value}).value == expected
    assert completions.requests == [
        {
            "response_model": Validator,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a world class validation model. Capable to determine if the following value is valid for the statement, if it is not, explain why and suggest a new value.",
                },
                {
                    "role": "user",
                    "content": f"Does `{value}` follow the rules: must be lowercase",
                },
            ],
            "model": "test-model",
            "temperature": 0.25,
        }
    ]


@pytest.mark.parametrize(
    ("response", "allow_override"),
    [
        (
            Validator(is_valid=False, reason="not lowercase", fixed_value="jason"),
            False,
        ),
        (Validator(is_valid=False, reason="not lowercase"), True),
    ],
)
def test_llm_validator_returns_pydantic_error_for_invalid_unfixed_values(
    response: Validator, allow_override: bool
):
    model, completions = make_validation_model(response, allow_override=allow_override)

    with pytest.raises(ValidationError) as exc_info:
        model.model_validate({"value": "Jason"})

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "assertion_error"
    assert error["loc"] == ("value",)
    assert "not lowercase" in error["msg"]
    assert completions.requests[0]["messages"][1]["content"] == (
        "Does `Jason` follow the rules: must be lowercase"
    )


class ModerationCategories(BaseModel):
    violence: bool = False
    harassment: bool = False


class ModerationResult(BaseModel):
    flagged: bool
    categories: ModerationCategories


class RecordingModerations:
    def __init__(self, result: ModerationResult):
        self.result = result
        self.inputs = []

    def create(self, *, input: str):
        self.inputs.append(input)
        return SimpleNamespace(results=[self.result])


def test_openai_moderation_allows_unflagged_text_through_pydantic():
    moderations = RecordingModerations(
        ModerationResult(flagged=False, categories=ModerationCategories())
    )
    client = SimpleNamespace(moderations=moderations)

    class Message(BaseModel):
        value: Annotated[str, AfterValidator(openai_moderation(cast(OpenAI, client)))]

    assert Message.model_validate({"value": "hello"}).value == "hello"
    assert moderations.inputs == ["hello"]


def test_openai_moderation_reports_only_flagged_categories_through_pydantic():
    moderations = RecordingModerations(
        ModerationResult(
            flagged=True,
            categories=ModerationCategories(violence=True, harassment=False),
        )
    )
    client = SimpleNamespace(moderations=moderations)

    class Message(BaseModel):
        value: Annotated[str, AfterValidator(openai_moderation(cast(OpenAI, client)))]

    with pytest.raises(ValidationError) as exc_info:
        Message.model_validate({"value": "unsafe text"})

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "value_error"
    assert error["loc"] == ("value",)
    assert error["msg"] == "Value error, `unsafe text` was flagged for violence"
    assert moderations.inputs == ["unsafe text"]


def test_v2_rejects_an_unknown_public_attribute():
    with pytest.raises(AttributeError) as exc_info:
        instructor_v2.__getattr__("not_an_instructor_export")

    assert str(exc_info.value) == (
        "module 'instructor.v2' has no attribute 'not_an_instructor_export'"
    )


def test_v2_loads_and_caches_public_attributes_and_modules(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(instructor_v2.__dict__, "Mode", raising=False)
    monkeypatch.delitem(instructor_v2.__dict__, "providers", raising=False)

    mode = instructor_v2.__getattr__("Mode")
    providers = instructor_v2.__getattr__("providers")

    assert mode is import_module("instructor.v2.core.mode").Mode
    assert providers is import_module("instructor.v2.providers")
    assert instructor_v2.__dict__["Mode"] is mode
    assert instructor_v2.__dict__["providers"] is providers
