# Simple Object Extraction

This guide covers extracting a simple object with defined fields from text - the most common pattern in structured data extraction.

## Basic Example

```python
from pydantic import BaseModel
import instructor
from openai import OpenAI

# Define the structure you want to extract
class Person(BaseModel):
    name: str
    age: int
    occupation: str

# Extract the structured data
client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "John Smith is a 35-year-old software engineer."}
    ],
    response_model=Person
)

print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
```

```
┌───────────────┐            ┌───────────────┐
│ Define Model  │            │ Extracted     │
│ name: str     │  Extract   │ name: "John"  │
│ age: int      │ ─────────> │ age: 35       │
│ occupation: str│            │ occupation:   │
└───────────────┘            │ "software..." │
                             └───────────────┘
```

## Using Field Descriptions

Adding descriptions helps the model understand what to extract:

```python
from pydantic import BaseModel, Field

class Book(BaseModel):
    title: str = Field(description="The full title of the book")
    author: str = Field(description="The author's full name")
    publication_year: int = Field(description="The year the book was published")
```

Field descriptions act like instructions for the extraction process.

## Handling Optional Fields

Sometimes the text won't contain all information:

```python
from typing import Optional
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    director: Optional[str] = None  # Optional field
    rating: float
```

By using `Optional` and providing a default value, fields can be missing without causing errors.

## Adding Simple Validation

You can add basic validation rules:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="The product price in USD")
    in_stock: bool
```

This example ensures `price` must be greater than zero.

## Real-world Example

Here's a more practical example:

```python
from pydantic import BaseModel
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[Address] = None

# Extract structured data
client = instructor.from_openai(OpenAI())
contact = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": """
        Contact information:
        Name: Sarah Johnson
        Email: sarah.j@example.com
        Phone: (555) 123-4567
        Address: 123 Main St, Boston, MA 02108
        """}
    ],
    response_model=ContactInfo
)

print(f"Name: {contact.name}")
print(f"Email: {contact.email}")
```

## Next Steps

- Try [List Extraction](list_extraction.md) to extract multiple objects
- Learn about [Nested Structure](nested_structure.md) for more complex data
- Check out [Field Validation](field_validation.md) for validation techniques 