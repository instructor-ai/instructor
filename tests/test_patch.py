from pydantic import BaseModel
import pytest
import openai
from instructor import patch


@pytest.mark.skip(reason="Needs openai call")
def test_runmodel():
    patch()

    class UserExtract(BaseModel):
        name: str
        age: int

    model = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtract), "Should be instance of UserExtract"
    assert model.name.lower() == "jason"
    assert hasattr(
        model, "_raw_response"
    ), "The raw response should be available from OpenAI"
