from itertools import product
import pytest

import instructor

from typing import Annotated
from pydantic import BaseModel, BeforeValidator, ValidationError

from instructor.dsl.validators import llm_validator
import google.generativeai as genai
from .util import models, modes


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_runmodel_validator_error(model, mode):
    client = instructor.from_gemini(
        genai.GenerativeModel(
            model,
        ),
        mode=mode,
    )

    class QuestionAnswerNoEvil(BaseModel):
        question: str
        answer: Annotated[
            str,
            BeforeValidator(
                llm_validator(
                    "don't say objectionable things", model=model, client=client
                )
            ),
        ]

    with pytest.raises(ValidationError):
        QuestionAnswerNoEvil(
            question="What is the meaning of life?",
            answer="The meaning of life is to be evil and steal",
        )