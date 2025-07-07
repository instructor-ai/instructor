from instructor.cache import make_cache_key
from pydantic import BaseModel, Field  # type: ignore[import-not-found]


messages = [
    {"role": "user", "content": "hello"},
]
model_name = "gpt-3.5-turbo"


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
