from typing import Literal

import pytest
from pydantic import BaseModel

from openai_function_call import openai_schema, OpenAISchema, openai_function


def test_openai_schema():
    @openai_schema
    class Dataframe(BaseModel):
        """
        Class representing a dataframe. This class is used to convert
        data into a frame that can be used by pandas.
        """

        data: str
        columns: str

        def to_pandas(self):
            pass

    assert hasattr(Dataframe, "openai_schema")
    assert hasattr(Dataframe, "from_response")
    assert hasattr(Dataframe, "to_pandas")
    assert Dataframe.openai_schema["name"] == "Dataframe"  # type: ignore


def test_openai_schema_raises_error():
    with pytest.raises(TypeError, match="must be a subclass of pydantic.BaseModel"):

        @openai_schema
        class Dummy:
            pass


def test_no_docstring():
    class Dummy(OpenAISchema):
        attr: str

    assert (
        Dummy.openai_schema["description"]
        == "Correctly extracted `Dummy` with all the required parameters with correct types"
    )


def test_openai_function():
    @openai_function
    def get_current_weather(
            location: str, format: Literal["celsius", "fahrenheit"] = "celsius"
    ):
        """
        Gets the current weather in a given location, use this function for any questions related to the weather

        Parameters
        ----------
        location
            The city to get the weather, e.g. San Francisco. Guess the location from user messages

        format
            A string with the full content of what the given role said
        """

    scheme = get_current_weather.openai_schema
    assert (
        scheme["parameters"]["properties"]["location"]["description"]
        ==
        "The city to get the weather, e.g. San Francisco. Guess the location from user messages"
    )
    assert (
        scheme["parameters"]["properties"]["format"]["enum"]
        ==
        ["celsius", "fahrenheit"]
    )
