from typing_extensions import Annotated
from pydantic import (
    BaseModel,
    BeforeValidator,
)

from instructor import llm_validator, patch
import openai

patch()


class QuestionAnswer(BaseModel):
    question: str
    answer: str


question = "What is the meaning of life?"
context = "The according to the devil is to live a life of sin and debauchery."

qa: QuestionAnswer = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswer,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)  # type: ignore

print("Before validation with `llm_validator`")
print(qa.model_dump_json(indent=2), end="\n\n")
"""
After validation with `llm_validator`
{
    "question": "What is the meaning of life?",
    "answer": "The meaning of life, according to the context, is to live a life of sin and debauchery.",
}
"""


class QuestionAnswerNoEvil(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("don't say objectionable things", allow_override=True)
        ),
    ]


try:
    qa: QuestionAnswerNoEvil = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        response_model=QuestionAnswerNoEvil,
        messages=[
            {
                "role": "system",
                "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
            },
            {
                "role": "user",
                "content": f"using the context: {context}\n\nAnswer the following question: {question}",
            },
        ],
    )  # type: ignore
except Exception as e:
    print(e, end="\n\n")
    """
    1 validation error for QuestionAnswerNoEvil
    answer
        Assertion failed, The statement promotes sin and debauchery, which is objectionable. [type=assertion_error, input_value='The meaning of life is t... of sin and debauchery.', input_type=str]
        For further information visit https://errors.pydantic.dev/2.3/v/assertion_error
    """

qa: QuestionAnswerNoEvil = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=QuestionAnswerNoEvil,
    max_retries=1,
    messages=[
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {context}\n\nAnswer the following question: {question}",
        },
    ],
)  # type: ignore

print("After validation with `llm_validator` with `max_retries=1`")
print(qa.model_dump_json(indent=2), end="\n\n")
"""
After validation with `llm_validator` with `max_retries=1`
{
  "question": "What is the meaning of life?",
  "answer": "The meaning of life is subjective and can vary depending on individual beliefs and philosophies."
}
"""
