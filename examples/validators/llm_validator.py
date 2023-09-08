from typing_extensions import Annotated
from pydantic import (
    BaseModel,
    BeforeValidator,
    ValidationError,
)
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


# Example 1) Valid input name is long
qa = QuestionAnswer(
    question="What is the meaning of life?",
    answer="The meaning of life is to be happy",
)
print(qa.model_dump_json(indent=2))
"""
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life is to be happy"
}
"""

# Example 2) Invalid input, we'll get a validation error
try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and kill people",
    )
except ValidationError as e:
    print(e)
    """
    {
        "is_valid": false,
        "reason": "The statement promotes violence and harm to others, which is objectionable.",
        "fixed_value": null
    }

    1 validation error for QuestionAnswer
    answer
        Assertion failed, The statement promotes violence and harm to others, which is objectionable. [type=assertion_error, input_value='The meaning of life is t...be evil and kill people', input_type=str]
        For further information visit https://errors.pydantic.dev/2.3/v/assertion_error
        """
