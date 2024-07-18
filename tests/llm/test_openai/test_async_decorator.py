from itertools import product
from pydantic import BaseModel, ValidationInfo, field_validator
import pytest
import instructor
from instructor.decorators import async_field_validator
from openai import AsyncOpenAI
from instructor import from_openai
from .util import models, modes


class UserExtractValidated(BaseModel):
    name: str
    age: int

    @async_field_validator("name")
    async def validate_name(cls, v: str) -> str:
        if not v.isupper():
            raise ValueError(
                f"All Letters in the name must be uppercased. {v} is not a valid response. Eg JASON, TOM not jason, tom"
            )
        return v


# @pytest.mark.parametrize("model, mode", product(models, modes))
# @pytest.mark.asyncio
async def test_async_validator(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    model = await aclient.chat.completions.create(
        model=model,
        response_model=UserExtractValidated,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    assert isinstance(model, UserExtractValidated), "Should be instance of UserExtract"
    assert model.name == "JASON"


class ValidationResult(BaseModel):
    chain_of_thought: str
    is_valid: bool


class ExtractedContent(BaseModel):
    relevant_question: str

    @async_field_validator("relevant_question")
    async def validate_relevant_question(cls, v: str, info: ValidationInfo) -> str:
        client = from_openai(AsyncOpenAI())
        if info.context and "content" in info.context:
            original_source = info.context["content"]
            evaluation = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Evaluate and determine if the question is a valid question well supported from the text",
                    },
                    {
                        "role": "user",
                        "content": f"The question is {v} and the source is {original_source}",
                    },
                ],
                response_model=ValidationResult,
            )
            if not evaluation.is_valid:
                raise ValueError(f"{v} is an invalid question!")
            return v

        raise ValueError("Invalid Response!")


@pytest.mark.parametrize("model, mode", product(models, modes))
@pytest.mark.asyncio
async def test_async_validator(model, mode, aclient):
    aclient = instructor.from_openai(aclient, mode=mode)
    content = """
    From President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. 

    Groups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland. 

    In this struggle as President Zelenskyy said in his speech to the European Parliament “Light will win over darkness.” The Ukrainian Ambassador to the United States is here tonight. 

    Let each of us here tonight in this Chamber send an unmistakable signal to Ukraine and to the world. 

    Please rise if you are able and show that, Yes, we the United States of America stand with the Ukrainian people. 

    Throughout our history we’ve learned this lesson when dictators do not pay a price for their aggression they cause more chaos.   

    They keep moving.
    """
    model = await aclient.chat.completions.create(
        model=model,
        response_model=ExtractedContent,
        max_retries=2,
        messages=[
            {
                "role": "user",
                "content": f"Generate a question from the context of {content}",
            },
        ],
        validation_context={"content": content},
    )
    assert isinstance(
        model, ExtractedContent
    ), "Should be instance of Extracted Content"
