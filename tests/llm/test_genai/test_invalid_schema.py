import pytest
from typing import Optional, Union

import instructor
from google.genai import Client
from pydantic import BaseModel
from .util import models, modes
from itertools import product


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_nested(mode, model):
    """Test that nested schemas raise appropriate error with Gemini."""
    client = instructor.from_genai(Client(), mode=mode)

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Optional[Address] = None

    with pytest.raises(ValueError, match="Gemini does not support Optional types"):
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "John loves to go gardenning with his friends",
                }
            ],
            response_model=Person,
        )


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_union(mode, model):
    """Test that union types raise appropriate error with Gemini."""
    client = instructor.from_genai(Client(), mode=mode)

    class UserData(BaseModel):
        name: str
        id_value: Union[str, int]

    with pytest.raises(ValueError, match="Gemini does not support Optional types"):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "User name is Alice with ID 12345"}],
            response_model=UserData,
        )


@pytest.mark.parametrize("mode,model", product(modes, models))
def test_optional(mode, model):
    """Test that optional fields raise appropriate error with Gemini."""
    client = instructor.from_genai(Client(), mode=mode)

    class Profile(BaseModel):
        name: str
        age: int
        bio: Optional[str] = None

    with pytest.raises(ValueError, match="Gemini does not support Optional types"):
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Alice is 30 years old"}],
            response_model=Profile,
        )
