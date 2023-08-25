from typing import List
from instructor import patch
from pydantic import BaseModel, Field

import openai

patch()


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


def ask_ai(content) -> DocumentExtraction:
    resp: DocumentExtraction = openai.ChatCompletion.create(
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

model = ask_ai(content)
print(model.model_dump_json(indent=2))

"""
{
  "entities": [
    {
      "id": 1,
      "subquote_string": [
        "This Agreement is made and entered into on 2020-01-01 by and between Company A (\"the Client\") and Company B (\"the Service Provider\")."
      ],
      "entity_title": "Agreement between Company A and Company B",
      "properties": [
        {
          "key": "Date",
          "value": "2020-01-01",
          "resolved_absolute_value": "2020-01-01"
        },
        {
          "key": "Party 1",
          "value": "Company A",
          "resolved_absolute_value": "Company A"
        },
        {
          "key": "Party 2",
          "value": "Company B",
          "resolved_absolute_value": "Company B"
        }
      ],
      "dependencies": []
    },
    {
      "id": 2,
      "subquote_string": [
        "The Service Provider will deliver the software product to the Client 30 days after the agreement date."
      ],
      "entity_title": "Scope of Work",
      "properties": [
        {
          "key": "Delivery Date",
          "value": "30 days after the agreement date",
          "resolved_absolute_value": "2020-01-31"
        }
      ],
      "dependencies": [
        1
      ]
    },
    {
      "id": 3,
      "subquote_string": [
        "The total payment for the service is $50,000.",
        "An initial payment of $10,000 will be made within 7 days of the the signed date.",
        "The final payment will be due 45 days after [SignDate]."
      ],
      "entity_title": "Payment Terms",
      "properties": [
        {
          "key": "Total Payment",
          "value": "$50,000",
          "resolved_absolute_value": "50000"
        },
        {
          "key": "Initial Payment",
          "value": "$10,000",
          "resolved_absolute_value": "10000"
        },
        {
          "key": "Final Payment Due Date",
          "value": "45 days after [SignDate]",
          "resolved_absolute_value": "2020-02-15"
        }
      ],
      "dependencies": [
        1
      ]
    },
    {
      "id": 4,
      "subquote_string": [
        "The parties agree not to disclose any confidential information received from the other party for 3 months after the final payment date."
      ],
      "entity_title": "Confidentiality Terms",
      "properties": [
        {
          "key": "Confidentiality Duration",
          "value": "3 months after the final payment date",
          "resolved_absolute_value": "2020-05-15"
        }
      ],
      "dependencies": [
        3
      ]
    },
    {
      "id": 5,
      "subquote_string": [
        "The contract can be terminated with a 30-day notice, unless there are outstanding obligations that must be fulfilled after the [DeliveryDate]."
      ],
      "entity_title": "Termination",
      "properties": [
        {
          "key": "Termination Notice",
          "value": "30-day",
          "resolved_absolute_value": "30 days"
        }
      ],
      "dependencies": [
        2
      ]
    }
  ]
}
"""
