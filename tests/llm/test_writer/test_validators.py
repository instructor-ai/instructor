from itertools import product
import pytest

import instructor

from typing import Annotated
from pydantic import BaseModel, BeforeValidator, ValidationError
from writerai import Writer

from instructor.dsl.validators import llm_validator
from .util import models, modes


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_writer_runmodel_validator_error(model: str, mode: instructor.Mode):
    client = instructor.from_writer(client=Writer(), mode=mode)

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


@pytest.mark.parametrize("model", models)
def test_writer_runmodel_validator(model: str):
    client = instructor.from_writer(client=Writer(), mode=instructor.Mode.WRITER_TOOLS)

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
