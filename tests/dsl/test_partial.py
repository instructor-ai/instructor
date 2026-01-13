# type: ignore[all]
from copy import deepcopy
from enum import Enum
from typing import Literal, Optional, Union

import pytest
from jiter import from_json
from pydantic import BaseModel, Field, ValidationError

import instructor
from instructor.dsl.partial import Partial, PartialLiteralMixin, _make_field_optional
import os
from openai import OpenAI, AsyncOpenAI

models = ["gpt-4o-mini"]
modes = [
    instructor.Mode.TOOLS,
]


class SampleNestedPartial(BaseModel):
    b: int


class SamplePartial(BaseModel):
    a: int
    b: SampleNestedPartial


class NestedA(BaseModel):
    a: str
    b: Optional[str]


class NestedB(BaseModel):
    c: str
    d: str
    e: list[Union[str, int]]
    f: str


class UnionWithNested(BaseModel):
    a: list[Union[NestedA, NestedB]]
    b: list[NestedA]
    c: NestedB


def test_partial():
    partial = Partial[SamplePartial]
    assert partial.model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {"b": {"title": "B", "type": "integer"}},
                "required": ["b"],
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {"title": "A", "type": "integer"},
            "b": {"$ref": "#/$defs/PartialSampleNestedPartial"},
        },
        "required": ["a", "b"],
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Wrapped model JSON schema has changed"
    assert partial.get_partial_model().model_json_schema() == {
        "$defs": {
            "PartialSampleNestedPartial": {
                "properties": {
                    "b": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "title": "B",
                    }
                },
                "title": "PartialSampleNestedPartial",
                "type": "object",
            }
        },
        "properties": {
            "a": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "A",
            },
            "b": {
                "anyOf": [
                    {"$ref": "#/$defs/PartialSampleNestedPartial"},
                    {"type": "null"},
                ],
                "default": {},
            },
        },
        "title": "PartialSamplePartial",
        "type": "object",
    }, "Partial model JSON schema has changed"


def test_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Get the actual models from chunks - must provide complete data for final validation
    models = list(partial.model_from_chunks(["\n", "\t", " ", '{"a": 42, "b": {"b": 1}}']))

    # Print actual values for debugging
    print(f"Number of models: {len(models)}")
    for i, model in enumerate(models):
        print(f"Model {i}: {model.model_dump()}")

    # Actual behavior: When whitespace chunks are processed, we may get models
    # First model has default values
    assert models[0].model_dump() == {"a": None, "b": {}}

    # Last model has all fields populated from JSON
    assert models[-1].model_dump() == {"a": 42, "b": {"b": 1}}

    # Check we have the expected number of models (2 instead of 4)
    assert len(models) == 2


@pytest.mark.asyncio
async def test_async_partial_with_whitespace():
    partial = Partial[SamplePartial]

    # Handle any leading whitespace from the model - must provide complete data for final validation
    async def async_generator():
        for chunk in ["\n", "\t", " ", '{"a": 42, "b": {"b": 1}}']:
            yield chunk

    expected_model_dicts = [
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": None, "b": {}},
        {"a": 42, "b": {"b": 1}},
    ]

    i = 0
    async for model in partial.model_from_chunks_async(async_generator()):
        assert model.model_dump() == expected_model_dicts[i]
        i += 1

    assert model.model_dump() == {"a": 42, "b": {"b": 1}}


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_summary_extraction():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = OpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_summary_extraction_async():
    class Summary(BaseModel, PartialLiteralMixin):
        summary: str = Field(description="A detailed summary")

    client = AsyncOpenAI()
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    extraction_stream = client.chat.completions.create_partial(
        model="gpt-4o",
        response_model=Summary,
        messages=[
            {"role": "system", "content": "You summarize text"},
            {"role": "user", "content": "Summarize: Mary had a little lamb"},
        ],
        stream=True,
    )

    previous_summary = None
    updates = 0
    async for extraction in extraction_stream:
        if previous_summary is not None and extraction:
            updates += 1
        previous_summary = extraction.summary

    assert updates == 1


def test_union_with_nested():
    partial = Partial[UnionWithNested]
    partial.get_partial_model().model_validate_json(
        '{"a": [{"b": "b"}, {"d": "d"}], "b": [{"b": "b"}], "c": {"d": "d"}, "e": [1, "a"]}'
    )


def test_partial_with_default_factory():
    """Test that Partial works with fields that have default_factory.

    This test ensures that when making fields optional, the default_factory
    is properly cleared to avoid Pydantic validation errors about having
    both default and default_factory set.
    """

    class ModelWithDefaultFactory(BaseModel):
        items: list[str] = Field(default_factory=list)
        tags: dict[str, str] = Field(default_factory=dict)
        name: str

    # This should not raise a validation error about both default and default_factory
    partial = Partial[ModelWithDefaultFactory]
    partial_model = partial.get_partial_model()

    # Verify we can instantiate and validate
    # In Partial models, all fields are made Optional with default=None
    instance = partial_model()
    assert instance.items is None
    assert instance.tags is None
    assert instance.name is None

    # Test with partial data
    instance2 = partial_model.model_validate({"items": ["a", "b"]})
    assert instance2.items == ["a", "b"]
    assert instance2.tags is None
    assert instance2.name is None


class TestMakeFieldOptionalWorksWithPydanticV2:
    """Tests proving that _make_field_optional with deepcopy works correctly in Pydantic v2.

    These tests refute the claim that deepcopy + setting default = None doesn't work
    in Pydantic v2. The implementation is correct and fields are properly made optional.

    See: https://github.com/instructor-ai/instructor/issues/XXXX
    """

    def test_deepcopy_approach_makes_field_optional(self):
        """Verify that deepcopy + default = None makes fields optional in Pydantic v2."""

        class Original(BaseModel):
            name: str  # Required field

        field = Original.model_fields["name"]
        assert field.is_required() is True, "Original field should be required"

        # This is what _make_field_optional does
        tmp = deepcopy(field)
        tmp.default = None
        tmp.annotation = Optional[str]

        assert tmp.is_required() is False, "Modified field should not be required"
        assert tmp.default is None, "Default should be None"

    def test_make_field_optional_function_works(self):
        """Verify _make_field_optional correctly transforms required fields."""

        class TestModel(BaseModel):
            name: str
            age: int

        for field_name, field_info in TestModel.model_fields.items():
            assert field_info.is_required() is True, f"{field_name} should be required"

            annotation, new_field = _make_field_optional(field_info)
            assert new_field.is_required() is False, (
                f"{field_name} should be optional after transformation"
            )
            assert new_field.default is None, f"{field_name} should have None default"

    def test_partial_model_validates_empty_dict(self):
        """Verify Partial models can validate empty dicts (all fields None)."""

        class MyModel(BaseModel):
            name: str
            age: int
            status: str

        PartialModel = Partial[MyModel]
        TruePartial = PartialModel.get_partial_model()

        # This should NOT raise ValidationError
        result = TruePartial.model_validate({})

        assert result.name is None
        assert result.age is None
        assert result.status is None

    def test_partial_validates_incremental_streaming_data(self):
        """Verify Partial models correctly handle incremental streaming data."""

        class MyModel(BaseModel):
            name: str
            age: int

        PartialModel = Partial[MyModel]
        TruePartial = PartialModel.get_partial_model()

        # Simulate streaming JSON chunks
        streaming_states = [
            ("{}", None, None),
            ('{"name": "Jo', "Jo", None),  # Partial string
            ('{"name": "John"}', "John", None),
            ('{"name": "John", "age": 25}', "John", 25),
        ]

        for json_str, expected_name, expected_age in streaming_states:
            obj = from_json(json_str.encode(), partial_mode="trailing-strings")
            result = TruePartial.model_validate(obj)
            assert result.name == expected_name, f"Failed for {json_str}"
            assert result.age == expected_age, f"Failed for {json_str}"

    def test_partial_with_all_field_types(self):
        """Verify _make_field_optional works with various field types."""

        class ComplexModel(BaseModel):
            string_field: str
            int_field: int
            float_field: float
            bool_field: bool
            list_field: list[str]
            optional_field: Optional[str]

        PartialModel = Partial[ComplexModel]
        TruePartial = PartialModel.get_partial_model()

        # All fields should validate with empty dict
        result = TruePartial.model_validate({})

        assert result.string_field is None
        assert result.int_field is None
        assert result.float_field is None
        assert result.bool_field is None
        assert result.list_field is None
        assert result.optional_field is None


class TestLiteralTypeStreaming:
    """Tests for Literal type handling during streaming.

    Without PartialLiteralMixin: uses partial_mode='trailing-strings', which keeps
    incomplete strings and causes validation errors for Literal/Enum fields.

    With PartialLiteralMixin: uses partial_mode='on', which drops incomplete strings
    so fields become None.
    """

    def test_literal_without_mixin_fails_on_incomplete_string(self):
        """Without PartialLiteralMixin, incomplete Literal strings cause validation errors."""

        class ModelWithLiteral(BaseModel):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = PartialModel.get_partial_model()

        # With partial_mode="trailing-strings", incomplete strings are kept
        partial_json = b'{"status": "act'
        obj = from_json(partial_json, partial_mode="trailing-strings")
        # obj is {"status": "act"} - a partial string that fails Literal validation

        with pytest.raises(ValidationError):
            TruePartial.model_validate(obj)

    def test_literal_with_mixin_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Literal strings are dropped."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = PartialModel.get_partial_model()

        # With partial_mode="on" (enabled by PartialLiteralMixin), incomplete strings are dropped
        partial_json = b'{"status": "act'
        obj = from_json(partial_json, partial_mode="on")
        # obj is {} because the incomplete string was dropped

        result = TruePartial.model_validate(obj)
        assert result.status is None

    def test_literal_accepts_valid_complete_value(self):
        """Literal fields should accept valid complete values."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = PartialModel.get_partial_model()

        result = TruePartial.model_validate({"status": "active"})
        assert result.status == "active"

        result = TruePartial.model_validate({"status": "inactive"})
        assert result.status == "inactive"

    def test_literal_with_missing_field_is_none(self):
        """Literal fields should be None when not present in data."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            name: str
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = PartialModel.get_partial_model()

        result = TruePartial.model_validate({"name": "John"})
        assert result.name == "John"
        assert result.status is None

    def test_literal_rejects_complete_invalid_value(self):
        """Complete but invalid Literal values should fail validation."""

        class ModelWithLiteral(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]

        PartialModel = Partial[ModelWithLiteral]
        TruePartial = PartialModel.get_partial_model()

        # "xyz" is a complete string but not a valid Literal value
        with pytest.raises(ValidationError):
            TruePartial.model_validate({"status": "xyz"})


class TestPartialStreamingWithComplexTypes:
    """Tests for streaming with complex Pydantic types using PartialLiteralMixin.

    With PartialLiteralMixin, partial_mode='on' is used, so incomplete values are dropped.
    """

    def test_enum_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Enum strings are dropped."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class ModelWithEnum(BaseModel, PartialLiteralMixin):
            status: Status

        PartialModel = Partial[ModelWithEnum]
        TruePartial = PartialModel.get_partial_model()

        # Incomplete string is dropped with partial_mode="on"
        obj = from_json(b'{"status": "act', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.status is None

    def test_enum_accepts_valid_complete_value(self):
        """Enum fields should accept valid complete values."""

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class ModelWithEnum(BaseModel, PartialLiteralMixin):
            status: Status

        PartialModel = Partial[ModelWithEnum]
        TruePartial = PartialModel.get_partial_model()

        result = TruePartial.model_validate({"status": "active"})
        assert result.status == Status.ACTIVE

    def test_optional_literal_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Optional[Literal] strings are dropped."""

        class ModelWithOptionalLiteral(BaseModel, PartialLiteralMixin):
            status: Optional[Literal["on", "off"]] = None

        PartialModel = Partial[ModelWithOptionalLiteral]
        TruePartial = PartialModel.get_partial_model()

        obj = from_json(b'{"status": "o', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.status is None

    def test_optional_literal_accepts_valid_value(self):
        """Optional[Literal] should accept valid complete values."""

        class ModelWithOptionalLiteral(BaseModel, PartialLiteralMixin):
            status: Optional[Literal["on", "off"]] = None

        PartialModel = Partial[ModelWithOptionalLiteral]
        TruePartial = PartialModel.get_partial_model()

        result = TruePartial.model_validate({"status": "on"})
        assert result.status == "on"

    def test_union_literal_incomplete_string_becomes_none(self):
        """With PartialLiteralMixin, incomplete Union[Literal, int] strings are dropped."""

        class ModelWithUnion(BaseModel, PartialLiteralMixin):
            value: Union[Literal["yes", "no"], int]

        PartialModel = Partial[ModelWithUnion]
        TruePartial = PartialModel.get_partial_model()

        # Incomplete string is dropped
        obj = from_json(b'{"value": "ye', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.value is None

    def test_union_literal_accepts_valid_values(self):
        """Union[Literal, int] should accept both valid Literal and int."""

        class ModelWithUnion(BaseModel, PartialLiteralMixin):
            value: Union[Literal["yes", "no"], int]

        PartialModel = Partial[ModelWithUnion]
        TruePartial = PartialModel.get_partial_model()

        result = TruePartial.model_validate({"value": "yes"})
        assert result.value == "yes"

        result = TruePartial.model_validate({"value": 42})
        assert result.value == 42

    def test_union_of_literals_matches_all_branches(self):
        """Union[Literal, Literal] should match values from all branches."""

        class ModelWithUnionLiterals(BaseModel, PartialLiteralMixin):
            value: Union[Literal["a", "b"], Literal["x", "y"]]

        PartialModel = Partial[ModelWithUnionLiterals]
        TruePartial = PartialModel.get_partial_model()

        # Both branches should work
        assert TruePartial.model_validate({"value": "a"}).value == "a"
        assert TruePartial.model_validate({"value": "b"}).value == "b"
        assert TruePartial.model_validate({"value": "x"}).value == "x"
        assert TruePartial.model_validate({"value": "y"}).value == "y"

    def test_list_literal_incomplete_item_dropped(self):
        """With PartialLiteralMixin, incomplete list items are dropped."""

        class ModelWithLiteralList(BaseModel, PartialLiteralMixin):
            tags: list[Literal["admin", "user", "guest"]]

        PartialModel = Partial[ModelWithLiteralList]
        TruePartial = PartialModel.get_partial_model()

        # Incomplete list item is dropped
        obj = from_json(b'{"tags": ["admin", "us', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.tags == ["admin"]

    def test_list_literal_accepts_valid_items(self):
        """list[Literal] should accept valid complete items."""

        class ModelWithLiteralList(BaseModel, PartialLiteralMixin):
            tags: list[Literal["admin", "user", "guest"]]

        PartialModel = Partial[ModelWithLiteralList]
        TruePartial = PartialModel.get_partial_model()

        result = TruePartial.model_validate({"tags": ["admin", "user"]})
        assert result.tags == ["admin", "user"]


class TestDiscriminatedUnionPartial:
    """Tests for discriminated unions with Partial streaming.

    KNOWN LIMITATION: Discriminated unions don't work with Partial because:
    - Partial makes all fields Optional
    - Pydantic requires discriminator fields to be strictly Literal, not Optional[Literal]

    Workaround: Use Union without the discriminator parameter.
    """

    def test_discriminated_union_not_compatible_with_partial(self):
        """Discriminated unions fail with Partial (known limitation)."""

        class Cat(BaseModel):
            pet_type: Literal["cat"]
            meows: int

        class Dog(BaseModel):
            pet_type: Literal["dog"]
            barks: int

        class PetContainer(BaseModel):
            pet: Union[Cat, Dog] = Field(discriminator="pet_type")

        # Fails because Partial makes pet_type Optional, but discriminators must be Literal
        from pydantic import PydanticUserError

        PartialModel = Partial[PetContainer]
        with pytest.raises(PydanticUserError):
            PartialModel.get_partial_model()

    def test_union_without_discriminator_works(self):
        """Union without discriminator works with Partial streaming."""

        class Cat(BaseModel):
            pet_type: Literal["cat"]
            meows: int

        class Dog(BaseModel):
            pet_type: Literal["dog"]
            barks: int

        class PetContainerNoDiscriminator(BaseModel):
            pet: Union[Cat, Dog]  # No discriminator - works with Partial

        PartialModel = Partial[PetContainerNoDiscriminator]
        TruePartial = PartialModel.get_partial_model()

        # Complete value works
        result = TruePartial.model_validate({"pet": {"pet_type": "cat", "meows": 5}})
        assert result.pet is not None
        assert result.pet.pet_type == "cat"

    def test_single_value_literal_incomplete_string(self):
        """Single-value Literals with incomplete strings become None."""

        class Cat(BaseModel):
            pet_type: Literal["cat"]

        PartialModel = Partial[Cat]
        TruePartial = PartialModel.get_partial_model()

        # Incomplete string is dropped
        obj = from_json(b'{"pet_type": "ca', partial_mode="on")
        result = TruePartial.model_validate(obj)
        assert result.pet_type is None

        # Complete value works
        result = TruePartial.model_validate({"pet_type": "cat"})
        assert result.pet_type == "cat"


class TestModelValidatorsDuringStreaming:
    """Tests for model validators during partial streaming.

    Model validators are automatically wrapped to skip during streaming
    (when context={"partial_streaming": True} is passed) and only run
    when validating without that context (final validation).
    """

    def test_model_validator_skipped_during_streaming(self):
        """Model validators should be skipped when streaming context is passed."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def validate_relationships(self):
                # This would fail during streaming without wrapping
                if self.status is not None and self.priority is None:
                    raise ValueError("If status is set, priority must also be set!")
                return self

        PartialModel = Partial[ModelWithValidator]
        TruePartial = PartialModel.get_partial_model()

        # During streaming, context={"partial_streaming": True} is passed
        # This skips model validators so incomplete data doesn't fail
        result = TruePartial.model_validate(
            {"status": "active"}, context={"partial_streaming": True}
        )
        assert result.status == "active"
        assert result.priority is None

    def test_model_validator_runs_when_complete(self):
        """Model validators should run when all fields are complete."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def validate_relationships(self):
                if self.status == "active" and self.priority == "low":
                    raise ValueError("Active status requires high priority!")
                return self

        PartialModel = Partial[ModelWithValidator]
        TruePartial = PartialModel.get_partial_model()

        # Valid complete data
        result = TruePartial.model_validate({"status": "active", "priority": "high"})
        assert result.status == "active"
        assert result.priority == "high"

        # Invalid complete data should fail
        with pytest.raises(ValidationError):
            TruePartial.model_validate({"status": "active", "priority": "low"})

    def test_multiple_model_validators(self):
        """Multiple model validators should all be wrapped and run when complete."""
        from pydantic import model_validator

        validator_calls = []

        class ModelWithMultipleValidators(BaseModel, PartialLiteralMixin):
            a: Literal["x", "y"]
            b: Literal["1", "2"]

            @model_validator(mode="after")
            def validator_one(self):
                validator_calls.append("one")
                return self

            @model_validator(mode="after")
            def validator_two(self):
                validator_calls.append("two")
                return self

        PartialModel = Partial[ModelWithMultipleValidators]
        TruePartial = PartialModel.get_partial_model()

        # During streaming (with context), validators should be skipped
        validator_calls.clear()
        TruePartial.model_validate({"a": "x"}, context={"partial_streaming": True})
        assert validator_calls == []

        # Final validation (without streaming context) - all validators should run
        validator_calls.clear()
        TruePartial.model_validate({"a": "x", "b": "1"})
        assert "one" in validator_calls
        assert "two" in validator_calls

    def test_validators_run_without_streaming_context(self):
        """Validators should run when no streaming context is passed (final validation)."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def validate_relationships(self):
                if self.status == "active" and self.priority == "low":
                    raise ValueError("Active requires high priority!")
                return self

        PartialModel = Partial[ModelWithValidator]
        TruePartial = PartialModel.get_partial_model()

        # Without streaming context, validators run even with incomplete data
        # This is the final validation scenario
        with pytest.raises(ValidationError):
            TruePartial.model_validate({"status": "active", "priority": "low"})

        # Valid complete data passes
        result = TruePartial.model_validate({"status": "active", "priority": "high"})
        assert result.status == "active"
        assert result.priority == "high"


class TestFinalValidationAfterStreaming:
    """Tests for final validation after streaming completes.

    When streaming ends, the final object is validated against the original
    model to enforce required fields and run validators without streaming context.
    """

    def test_final_validation_catches_missing_required_fields(self):
        """Final validation should fail if required fields are missing."""

        class ModelWithRequired(BaseModel):
            name: str  # Required
            age: int  # Required
            nickname: Optional[str] = None  # Optional

        PartialModel = Partial[ModelWithRequired]

        # Simulate streaming that doesn't provide all required fields
        chunks = ['{"name": "John"}']  # Missing 'age'

        with pytest.raises(ValidationError) as exc_info:
            list(PartialModel.model_from_chunks(iter(chunks)))

        # Should fail because 'age' is required but missing
        assert "age" in str(exc_info.value)

    def test_final_validation_passes_with_all_required_fields(self):
        """Final validation should pass when all required fields are present."""

        class ModelWithRequired(BaseModel):
            name: str
            age: int

        PartialModel = Partial[ModelWithRequired]

        # Simulate streaming that provides all required fields
        chunks = ['{"name": "John", "age": 30}']

        results = list(PartialModel.model_from_chunks(iter(chunks)))
        assert len(results) > 0
        final = results[-1]
        assert final.name == "John"
        assert final.age == 30

    def test_final_validation_runs_model_validators(self):
        """Final validation should run model validators without streaming context."""
        from pydantic import model_validator

        class ModelWithValidator(BaseModel, PartialLiteralMixin):
            status: Literal["active", "inactive"]
            priority: Literal["high", "low"]

            @model_validator(mode="after")
            def check_consistency(self):
                if self.status == "active" and self.priority == "low":
                    raise ValueError("Active tasks must have high priority")
                return self

        PartialModel = Partial[ModelWithValidator]

        # This should fail final validation due to the model validator
        chunks = ['{"status": "active", "priority": "low"}']

        with pytest.raises(ValidationError) as exc_info:
            list(PartialModel.model_from_chunks(iter(chunks)))

        assert "Active tasks must have high priority" in str(exc_info.value)

    def test_streaming_yields_partial_objects_before_final_validation(self):
        """Streaming should yield partial objects even if final validation will fail."""

        class ModelWithRequired(BaseModel):
            name: str
            age: int

        PartialModel = Partial[ModelWithRequired]

        # Stream that will fail final validation (missing 'age')
        chunks = ['{"na', 'me": "John"}']

        partial_objects = []
        try:
            for obj in PartialModel.model_from_chunks(iter(chunks)):
                partial_objects.append(obj)
        except ValidationError:
            pass  # Expected

        # Should have yielded at least one partial object before failing
        assert len(partial_objects) >= 1
        assert partial_objects[-1].name == "John"

    def test_original_model_reference_is_stored(self):
        """Partial model should store reference to original model."""

        class OriginalModel(BaseModel):
            name: str

        PartialModel = Partial[OriginalModel]

        assert hasattr(PartialModel, "_original_model")
        assert PartialModel._original_model is OriginalModel

    @pytest.mark.asyncio
    async def test_async_final_validation_catches_missing_required_fields(self):
        """Async streaming should also do final validation."""

        class ModelWithRequired(BaseModel):
            name: str
            age: int

        PartialModel = Partial[ModelWithRequired]

        async def async_chunks():
            yield '{"name": "John"}'  # Missing 'age'

        with pytest.raises(ValidationError) as exc_info:
            async for _ in PartialModel.model_from_chunks_async(async_chunks()):
                pass

        assert "age" in str(exc_info.value)
