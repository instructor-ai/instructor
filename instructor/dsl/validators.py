from pydantic import Field
from typing import Optional
import instructor
import openai


class Validator(instructor.OpenAISchema):
    """
    Validate if an attribute is correct and if not,
    return a new value with an error message
    """

    is_valid: bool = Field(
        default=True,
        description="Whether the attribute is valid based on the requirements",
    )
    reason: Optional[str] = Field(
        default=None,
        description="The error message if the attribute is not valid, otherwise None",
    )
    fixed_value: Optional[str] = Field(
        default=None,
        description="If the attribuet is not valid, suggest a new value for the attribute",
    )


def llm_validator(
    statement: str,
    allow_override: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
):
    """
    Create a validator that uses the LLM to validate an attribute

    ## Usage

    ```python
    from instructor import llm_validator
    from pydantic import BaseModel, Field, field_validator

    class User(BaseModel):
        name: str = Annotated[str, llm_validator("The name must be a full name all lowercase")]
        age: int = Field(description="The age of the person")

    try:
        user = User(name="Jason Liu", age=20)
    except ValidationError as e:
        print(e)
    ```

    ```
    1 validation error for User
    name
      The name is valid but not all lowercase (type=value_error.llm_validator)
    ```

    Note that there, the error message is written by the LLM, and the error type is `value_error.llm_validator`.

    Parameters:
        statement (str): The statement to validate
        model (str): The LLM to use for validation (default: "gpt-3.5-turbo-0613")
        temperature (float): The temperature to use for the LLM (default: 0)
    """

    def llm(v):
        resp = openai.ChatCompletion.create(
            functions=[Validator.openai_schema],
            function_call={"name": Validator.openai_schema["name"]},
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class validation model. Capable to determine if the following value is valid for the statement, if it is not, explain why and suggest a new value.",
                },
                {
                    "role": "user",
                    "content": f"Does `{v}` follow the rules: {statement}",
                },
            ],
            model=model,
            temperature=temperature,
        )  # type: ignore
        resp = Validator.from_response(resp)

        # If the response is  not valid, return the reason, this could be used in
        # the future to generate a better response, via reasking mechanism.
        assert resp.is_valid, resp.reason

        if allow_override and not resp.is_valid and resp.fixed_value is not None:
            # If the value is not valid, but we allow override, return the fixed value
            return resp.fixed_value
        return v

    return llm
