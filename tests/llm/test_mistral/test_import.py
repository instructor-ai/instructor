import os
import pytest
from pydantic import BaseModel
import instructor

try:
    # mistralai v2.0.0+
    from mistralai.client import Mistral
except ImportError:
    # mistralai v1.x
    from mistralai import Mistral


class UserDetail(BaseModel):
    name: str
    age: int


def test_mistral_client_init():
    """Test that we can initialize and patch the Mistral client successfully."""
    # We use a dummy API key for instantiation test if one isn't provided
    api_key = os.environ.get("MISTRAL_API_KEY", "dummy-key-for-test")

    mistral_client = Mistral(api_key=api_key)

    # We just want to ensure patching doesn't raise an exception
    # due to strict type checks returning False for Mistral v2.0 clients
    client = instructor.from_mistral(mistral_client)

    assert client is not None
    assert hasattr(client, "chat")


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY must be set"
)
def test_mistral_extraction():
    """Test a live extraction if the API key is provided."""
    mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
    client = instructor.from_mistral(mistral_client)

    user = client.chat.completions.create(
        model="mistral-large-latest",
        response_model=UserDetail,
        messages=[
            {
                "role": "user",
                "content": "Kitan is a 18 year old engineer living in Alapere.",
            }
        ],
    )

    assert user.name == "Kitan"
    assert user.age == 18
