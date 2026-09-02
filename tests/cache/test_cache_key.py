from instructor.cache import make_cache_key
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


messages = [
    {"role": "user", "content": "hello"},
]
model_name = "gpt-4.1-mini"


class UserV1(BaseModel):
    name: str = Field(..., description="User name")


class UserV1DiffDesc(BaseModel):
    name: str = Field(..., description="User full name")


class UserV1DiffField(BaseModel):
    name: str
    age: int


class UserDoc1(BaseModel):
    """First docstring"""

    name: str


class UserDoc2(BaseModel):
    """Second different docstring"""

    name: str


def test_cache_key_changes_on_description_change():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    k2 = make_cache_key(
        messages=messages, model=model_name, response_model=UserV1DiffDesc
    )
    assert k1 != k2, "Changing field description should bust the cache key"


def test_cache_key_changes_on_field_change():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    k2 = make_cache_key(
        messages=messages, model=model_name, response_model=UserV1DiffField
    )
    assert k1 != k2, "Adding or removing fields should bust the cache key"


def test_cache_key_same_for_identical_schema():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    k2 = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    assert k1 == k2, "Identical schemas should produce identical cache keys"


def test_cache_key_changes_on_docstring_change():
    k1 = make_cache_key(messages=messages, model=model_name, response_model=UserDoc1)
    k2 = make_cache_key(messages=messages, model=model_name, response_model=UserDoc2)
    assert k1 != k2, "Changing class docstring should bust the cache key"


def test_cache_key_changes_on_system_prompt_change():
    k1 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        system="Always answer in French.",
    )
    k2 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        system="Always answer in German.",
    )
    assert k1 != k2, "Changing the hoisted system prompt should bust the cache key"


def test_cache_key_unchanged_when_no_system_prompt():
    without = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    explicit_none = make_cache_key(
        messages=messages, model=model_name, response_model=UserV1, system=None
    )
    assert without == explicit_none, (
        "Keys must stay stable for providers without a hoisted system prompt"
    )


def test_cache_key_changes_on_temperature_change():
    k1 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        generation_kwargs={"temperature": 0.0},
    )
    k2 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        generation_kwargs={"temperature": 1.9},
    )
    assert k1 != k2, (
        "Calls that differ only in temperature must not collide in the cache"
    )


def test_cache_key_changes_on_seed_change():
    k1 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        generation_kwargs={"seed": 1},
    )
    k2 = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        generation_kwargs={"seed": 2},
    )
    assert k1 != k2, "Calls that differ only in seed must not collide in the cache"


def test_cache_key_unchanged_when_no_generation_kwargs():
    without = make_cache_key(messages=messages, model=model_name, response_model=UserV1)
    explicit_empty = make_cache_key(
        messages=messages,
        model=model_name,
        response_model=UserV1,
        generation_kwargs={},
    )
    assert without == explicit_empty, (
        "Keys must stay stable when no generation kwargs are supplied"
    )


def test_extract_generation_kwargs_drops_unset_and_irrelevant_keys():
    from instructor.cache import extract_generation_kwargs

    new_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "seed": None,  # unset -> must be dropped
        "tools": [{"type": "function"}],  # not a sampling param -> must be dropped
    }
    assert extract_generation_kwargs(new_kwargs) == {"temperature": 0.7}
