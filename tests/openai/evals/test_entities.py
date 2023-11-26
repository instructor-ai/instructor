from itertools import product
from typing import List
from pydantic import BaseModel, Field
import pytest

import instructor
from openai import OpenAI


class Property(BaseModel):
    key: str
    value: str
    resolved_absolute_value: str


class Entity(BaseModel):
    id: int = Field(
        ...,
        description="Unique identifier for the entity, used for deduplication, design a scheme allows multiple entities",
    )
    subquote_string: List[str] = Field(
        ...,
        description="Correctly resolved value of the entity, if the entity is a reference to another entity, this should be the id of the referenced entity, include a few more words before and after the value to allow for some context to be used in the resolution",
    )
    entity_title: str
    properties: List[Property] = Field(
        ..., description="List of properties of the entity"
    )
    dependencies: List[int] = Field(
        ...,
        description="List of entity ids that this entity depends  or relies on to resolve it",
    )


class DocumentExtraction(BaseModel):
    entities: List[Entity] = Field(
        ...,
        description="Body of the answer, each fact should be its seperate object with a body and a list of sources",
    )


def ask_ai(content, model, mode, client) -> DocumentExtraction:
    resp: DocumentExtraction = client.chat.completions.create(
        model="gpt-4",
        response_model=DocumentExtraction,
        messages=[
            {
                "role": "system",
                "content": "You are a perfect entity resolution system that extracts facts from the document. Extract and resolve a list of entities from the following document:",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )  # type: ignore
    return resp


content = """
Sample Legal Contract
Agreement Contract

This Agreement is made and entered into on 2020-01-01 by and between Company A ("the Client") and Company B ("the Service Provider").

Article 1: Scope of Work

The Service Provider will deliver the software product to the Client 30 days after the agreement date.

Article 2: Payment Terms

The total payment for the service is $50,000.
An initial payment of $10,000 will be made within 7 days of the the signed date.
The final payment will be due 45 days after [SignDate].

Article 3: Confidentiality

The parties agree not to disclose any confidential information received from the other party for 3 months after the final payment date.

Article 4: Termination

The contract can be terminated with a 30-day notice, unless there are outstanding obligations that must be fulfilled after the [DeliveryDate].
"""



models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]
modes = [instructor.Mode.FUNCTIONS, instructor.Mode.JSON, instructor.Mode.TOOLS]


@pytest.mark.parametrize("model, mode", product(models, modes))
def test_extract(model, mode):
    client = instructor.patch(OpenAI(), mode=mode)
    if mode == instructor.Mode.JSON and model in {"gpt-3.5-turbo", "gpt-4"}:
        pytest.skip(
            "JSON mode is not supported for gpt-3.5-turbo and gpt-4, skipping test"
        )

    # Honestly, if there are no errors, then it's a pass
    extract = ask_ai(content, model, mode, client)
    assert len(extract.entities) > 0
