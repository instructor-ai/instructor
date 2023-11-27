import pytest

import instructor

from typing_extensions import Annotated
from pydantic import BaseModel, AfterValidator, BeforeValidator, ValidationError
from openai import OpenAI

from instructor.dsl.validators import llm_validator

client = instructor.patch(OpenAI())


def test_patch_completes_successfully():
    class Response(BaseModel):
        message: Annotated[
            str, AfterValidator(instructor.openai_moderation(client=client))
        ]

    with pytest.raises(ValidationError):
        Response(message="I want to make them suffer the consequences")


def test_runmodel_validator_error():
    class QuestionAnswerNoEvil(BaseModel):
        question: str
        answer: Annotated[
            str,
            BeforeValidator(
                llm_validator("don't say objectionable things", openai_client=client)
            ),
        ]

    with pytest.raises(ValidationError):
        QuestionAnswerNoEvil(
            question="What is the meaning of life?",
            answer="The meaning of life is to be evil and steal",
        )
