from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.moderation_create_response import ModerationCreateResponse

from pydantic import BaseModel, Field, ConfigDict

from instructor import Instructor
from instructor.function_calls import Mode


T = TypeVar('T')
ChatMessage = ChatCompletionMessageParam

class ValidatorProtocol(Protocol):
    """Protocol for validator objects."""
    def model_dump(self) -> dict[str, Any]: ...


class Validator(BaseModel):
    """Validate if an attribute is correct and if not, return a new value with an error message."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    is_valid: bool = Field(
        default=True,
        description="Whether the attribute is valid based on the requirements",
    )
    reason: str | None = Field(
        default=None,
        description="The error message if the attribute is not valid, otherwise None",
    )
    fixed_value: str | None = Field(
        default=None,
        description="If the attribute is not valid, suggest a new value for the attribute",
    )


def llm_validator(
    statement: str,
    client: Instructor,
    allow_override: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
) -> Callable[[str], str]:
    """
    Create a validator that uses the LLM to validate an attribute

    ## Usage

    ```python
    from instructor import llm_validator
    from pydantic import BaseModel, Field, field_validator

    class User(BaseModel):
        name: str = Annotated[str, llm_validator("The name must be a full name all lowercase")
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
        client (Instructor): The Instructor client to use for validation
        allow_override (bool): Whether to allow the LLM to override the value with a fixed value
        model (str): The LLM to use for validation (default: "gpt-3.5-turbo")
        temperature (float): The temperature to use for the LLM (default: 0)
    """

    def llm(v: str) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "You are a world class validation model. Capable to determine if the following value is valid for the statement, if it is not, explain why and suggest a new value.",
            },
            {
                "role": "user",
                "content": f"Does `{v}` follow the rules: {statement}",
            },
        ]

        # Use instructor client's create method which handles response_model
        resp: Validator = client.chat.completions.create(
            response_model=Validator,
            messages=messages,
            model=model,
            temperature=temperature,
            mode=Mode.JSON,
        )

        if not resp.is_valid:
            if resp.reason:
                raise ValueError(resp.reason)
            raise ValueError("Invalid value")

        if allow_override and resp.fixed_value is not None:
            return resp.fixed_value
        return v

    return llm


def openai_moderation(client: OpenAI) -> Callable[[str], str]:
    """
    Validates a message using OpenAI moderation model.

    Should only be used for monitoring inputs and outputs of OpenAI APIs
    Other use cases are disallowed as per:
    https://platform.openai.com/docs/guides/moderation/overview

    Example:
    ```python
    from instructor import OpenAIModeration

    class Response(BaseModel):
        message: Annotated[str, AfterValidator(OpenAIModeration(openai_client=client))]

    Response(message="I hate you")
    ```

    ```
     ValidationError: 1 validation error for Response
     message
    Value error, `I hate you.` was flagged for ['harassment'] [type=value_error, input_value='I hate you.', input_type=str]
    ```

    client (OpenAI): The OpenAI client to use, must be sync (default: None)
    """

    def validate_message_with_openai_mod(v: str) -> str:
        response: ModerationCreateResponse = client.moderations.create(input=v)
        result = response.results[0]

        if result.flagged:
            categories_dict: dict[str, bool] = dict(result.categories)
            flagged_categories: list[str] = [
                category for category, is_flagged in categories_dict.items()
                if is_flagged
            ]
            raise ValueError(f"`{v}` was flagged for {', '.join(flagged_categories)}")
        return v

    return validate_message_with_openai_mod
