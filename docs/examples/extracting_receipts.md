---
title: Extracting Receipt Data with GPT-4 and Python
description: Learn how to use Python and GPT-4 to extract and validate receipt data from images for efficient expense tracking.
---

# Extracting Receipt Data using GPT-4 and Python

This post demonstrates how to use Python's Pydantic library and OpenAI's GPT-4 model to extract receipt data from images and validate the total amount. This method is particularly useful for automating expense tracking and financial analysis tasks.

## Defining the Item and Receipt Classes

First, we define two Pydantic models, `Item` and `Receipt`, to structure the extracted data. The `Item` class represents individual items on the receipt, with fields for name, price, and quantity. The `Receipt` class contains a list of `Item` objects and the total amount.

```python
from pydantic import BaseModel, field_validator, ConfigDict

class Item(BaseModel):
    name: str
    price: float
    quantity: int
    model_config = ConfigDict(validate_default=True)

class Receipt(BaseModel):
    items: list[Item]
    total: float
    model_config = ConfigDict(validate_default=True)

    @field_validator("total", mode="after")
    @classmethod
    def check_total(cls, total: float, info) -> float:
        items = info.data.get('items', [])
        calculated_total = round(sum(item.price * item.quantity for item in items), 2)
        if calculated_total != total:
            raise ValueError(
                f"Total {total} does not match the sum of item prices {calculated_total}"
            )
        return total
```

## Validating the Total Amount

To ensure the accuracy of the extracted data, we use Pydantic's `field_validator` decorator to define a custom validation function, `check_total`. This function calculates the sum of item prices and compares it to the extracted total amount. If there's a discrepancy, it raises a `ValueError`.

```python
from pydantic import field_validator, ConfigDict

@field_validator("total", mode="after")
@classmethod
def check_total(cls, total: float, info) -> float:
    items = info.data.get('items', [])
    calculated_total = sum(item.price * item.quantity for item in items)
    if calculated_total != total:
        raise ValueError(
            f"Total {total} does not match the sum of item prices {calculated_total}"
        )
    return total
```

## Extracting Receipt Data from Images

The `extract_receipt` function uses OpenAI's GPT-4 model to process an image URL and extract receipt data. We utilize the `instructor` library to configure the OpenAI client for this purpose.

```python
import instructor
from openai import OpenAI

# <%hide%>
from pydantic import BaseModel, field_validator, ConfigDict

class Item(BaseModel):
    name: str
    price: float
    quantity: int
    model_config = ConfigDict(validate_default=True)

class Receipt(BaseModel):
    items: list[Item]
    total: float
    model_config = ConfigDict(validate_default=True)

    @field_validator("total", mode="after")
    @classmethod
    def check_total(cls, total: float, info) -> float:
        items = info.data.get('items', [])
        calculated_total = sum(item.price * item.quantity for item in items)
        if calculated_total != total:
            raise ValueError(
                f"Total {total} does not match the sum of item prices {calculated_total}"
            )
        return total

# <%hide%>

client = instructor.from_openai(OpenAI())

def extract(url: str) -> Receipt:
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        max_tokens=4000,
        response_model=Receipt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": "Analyze the image and return the items in the receipt and the total amount.",
                    },
                ],
            }
        ],
    )
```

## Practical Examples

In these examples, we apply the method to extract receipt data from two different images. The custom validation function ensures that the extracted total amount matches the sum of item prices.

```python
# <%hide%>
from pydantic import BaseModel, field_validator, ConfigDict
import instructor
from openai import OpenAI

class Item(BaseModel):
    name: str
    price: float
    quantity: int
    model_config = ConfigDict(validate_default=True)

class Receipt(BaseModel):
    items: list[Item]
    total: float
    model_config = ConfigDict(validate_default=True)

    @field_validator("total", mode="after")
    @classmethod
    def check_total(cls, total: float, info) -> float:
        items = info.data.get('items', [])
        calculated_total = round(sum(item.price * item.quantity for item in items), 2)
        if calculated_total != total:
            raise ValueError(
                f"Total {total} does not match the sum of item prices {calculated_total}"
            )
        return total
```

client = instructor.from_openai(OpenAI())


def extract(url: str) -> Receipt:
    return client.chat.completions.create(
        model="gpt-4-turbo-preview",
        max_tokens=4000,
        response_model=Receipt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    },
                    {
                        "type": "text",
                        "text": "Analyze the image and return the items in the receipt and the total amount.",
                    },
                ],
            }
        ],
    )


# <%hide%>
url = "https://templates.mediamodifier.com/645124ff36ed2f5227cbf871/supermarket-receipt-template.jpg"


receipt = extract(url)
print(receipt)
```

By combining the power of GPT-4 and Python's Pydantic library, we can accurately extract and validate receipt data from images, streamlining expense tracking and financial analysis tasks.
