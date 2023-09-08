# Introduction to Validation in Pydantic and LLMs

Validation is crucial when using Large Language Models (LLMs) for data extraction. It ensures data integrity, enables [reasking for better results](reask.md), and allows for overwriting incorrect values. Pydantic offers versatile validation capabilities suitable for use with LLM outputs.


!!! note "Pydantic Validation Docs"
    Pydantic supports validation individual fields or the whole model dict all at once.

    - [Field-Level Validation](https://docs.pydantic.dev/latest/usage/validators/)
    - [Model-Level Validation](https://docs.pydantic.dev/latest/usage/validators/#model-validators)

    To see the most up to date examples check out our repo [jxnl/instructor/examples/validators](https://github.com/jxnl/instructor/tree/main/examples/validators)

## Importance of LLM Validation

- **Data Integrity**: Enforces data quality standards.
- **[Reasking](reask.md)**: Utilizes Pydantic's error messages to improve LLM outputs.
- **Overwriting**: Overwrites incorrect values during API calls.

## Code Examples

### Simple Validation with Pydantic

The example uses a custom validator function to enforce a rule on the name attribute. If a user fails to input a full name (first and last name separated by a space), Pydantic will raise a validation error. If you want the LLM to automatically fix the error check out our [reasking docs.](reask.md)

```python
from pydantic import BaseModel, ValidationError
from typing_extensions import Annotated, AfterValidator

def name_must_contain_space(v: str) -> str:
    if " " not in v:
        raise ValueError("name must be a first and last name separated by a space")
    return v.lower()

class UserDetail(BaseModel):
    age: int
    name: Annotated[str, AfterValidator(name_must_contain_space)]

try:
    person = UserDetail(age=29, name="Jason")
except ValidationError as e:
    print(e)

# Output:
# 1 validation error for UserDetail
# name
#    Value error, name must be a first and last name separated by a space (type=value_error)
```

### LLM-Based Validation

This example demonstrates using an LLM as a validator. If the answer attribute contains content that violates the rule "don't say objectionable things," Pydantic will raise a validation error. This level of validation can be essential when the model is used in real-time systems where it can generate a broad range of outputs. Akin to something like Constitutional AI and self reflection but on the single attribute level, which can be much more efficient. 


```python
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
import instructor
from instructor.dsl.validators import llm_validator

instructor.patch()

class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", allow_override=True)
        ),
    ]

try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and kill people",
    )
except ValidationError as e:
    print(e)

# Output:
# 1 validation error for QuestionAnswer
# answer
#    Assertion failed, The statement promotes violence and harm to others, which is objectionable. (type=assertion_error)
```

!!! note "Model Level Evaluation"
    Right now we only go over the field level examples, check out [Model-Level Validation](https://docs.pydantic.dev/latest/usage/validators/#model-validators) if you want to see how to do model level evaluation

## Create Your Own LLM Validator

The section shows how to create a custom LLM validator function. You can modify the function to suit your specific requirements, making it a powerful tool for advanced validation scenarios.

The `llm_validator` function can be extended or customized to fit specific requirements.

```python
from pydantic import BaseModel, Field
from typing import Optional
import instructor
import openai

instructor.patch()

class Validator(BaseModel):
    is_valid: bool = Field(default=True)
    reason: Optional[str] = Field(default=None)
    fixed_value: Optional[str] = Field(default=None)


def llm_validator(
    statement: str,
    allow_override: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
):
    """
    Create a validator that uses the LLM to validate an attribute

    Parameters:
        statement (str): The statement to validate
        model (str): The LLM to use for validation (default: "gpt-3.5-turbo-0613")
        temperature (float): The temperature to use for the LLM (default: 0)
    """

    def llm(v):
        resp: Validator = openai.ChatCompletion.create(
            response_model=Validator,
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

        # If the response is  not valid, return the reason, this could be used in
        # the future to generate a better response, via reasking mechanism.
        assert resp.is_valid, resp.reason

        if allow_override and not resp.is_valid and resp.fixed_value is not None:
            # If the value is not valid, but we allow override, return the fixed value
            return resp.fixed_value
        return v

    return llm
```

By integrating these advanced validation techniques, we not only improve the quality and reliability of LLM-generated content but also pave the way for more autonomous and effective systems.