from typing import Annotated, cast
from pydantic import AfterValidator, BaseModel, Field
import instructor
import vertexai.generative_models as gm #type: ignore[reportMissingTypeStubs]
from .util import model, mode


def uppercase_validator(v: str):
    if v.islower():
        raise ValueError("Name must be ALL CAPS")
    return v


class UserDetail(BaseModel):
    name: Annotated[str, AfterValidator(uppercase_validator)] = Field(
        ..., description="The name of the user"
    )
    age: int


def test_upper_case():
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode=mode)
    response = client.create(
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=3,
    )
    assert response.name == "JASON"


def test_upper_case_tenacity():
    client = instructor.from_vertexai(gm.GenerativeModel(model), mode=mode)
    from tenacity import Retrying, stop_after_attempt, wait_fixed

    retries = Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
    )

    retries = cast(int, retries)

    response = client.create(
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract `jason is 12`"},
        ],
        max_retries=retries,
    )
    assert response.name == "JASON"
