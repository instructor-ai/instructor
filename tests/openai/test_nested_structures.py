from typing import Iterable
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

import instructor
from instructor.function_calls import Mode


class Item(BaseModel):
    name: str
    price: float

class Order(BaseModel):
    items: List[Item] = Field(..., default_factory=list)
    customer: str

content = """
Order Details:
Customer: Jason
Items:

Name: Apple, Price: 0.50
Name: Bread, Price: 2.00
Name: Milk, Price: 1.50
"""

def test_nested_structures_function_mode():
    client = instructor.patch(OpenAI(), mode=Mode.FUNCTIONS)

    resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=Order,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )

    assert len(resp.items) == 3
    assert resp.items[0].name.lower() == "apple"
    assert resp.items[1].name.lower() == "bread"
    assert resp.items[2].name.lower() == "milk"
    assert resp.items[0].price == 0.5
    assert resp.items[1].price == 2.0
    assert resp.items[2].price == 1.5
    assert resp.customer.lower() == "jason"



def test_nested_structures_json_mode():
    client = instructor.patch(OpenAI(), mode=Mode.JSON)

    resp = client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_model=Order,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )

    assert len(resp.items) == 3
    assert resp.items[0].name.lower() == "apple"
    assert resp.items[1].name.lower() == "bread"
    assert resp.items[2].name.lower() == "milk"
    assert resp.items[0].price == 0.5
    assert resp.items[1].price == 2.0
    assert resp.items[2].price == 1.5
    assert resp.customer.lower() == "jason"


def test_nested_structures_tools_mode():
    client = instructor.patch(OpenAI(), mode=Mode.TOOLS)

    resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=Order,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )

    assert len(resp.items) == 3
    assert resp.items[0].name.lower() == "apple"
    assert resp.items[1].name.lower() == "bread"
    assert resp.items[2].name.lower() == "milk"
    assert resp.items[0].price == 0.5
    assert resp.items[1].price == 2.0
    assert resp.items[2].price == 1.5
    assert resp.customer.lower() == "jason"