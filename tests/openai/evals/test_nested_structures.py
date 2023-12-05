from typing import Iterable
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

import pytest

import instructor
from instructor.function_calls import Mode


class Item(BaseModel):
    name: str
    price: float


class Order(BaseModel):
    items: List[Item] = Field(..., default_factory=list)
    customer: str


@pytest.mark.parametrize("mode", [Mode.FUNCTIONS, Mode.JSON, Mode.TOOLS, Mode.MD_JSON])
def test_nested(mode):
    client = instructor.patch(OpenAI(), mode=mode)

    content = """
    Order Details:
    Customer: Jason
    Items:

    Name: Apple, Price: 0.50
    Name: Bread, Price: 2.00
    Name: Milk, Price: 1.50
    """

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=Order,
        messages=[
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    assert len(resp.items) == 3
    assert {x.name.lower() for x in resp.items} == {"apple", "bread", "milk"}
    assert {x.price for x in resp.items} == {0.5, 2.0, 1.5}
    assert resp.customer.lower() == "jason"
