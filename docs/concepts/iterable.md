# Working with Iterables in Instructor

This guide explains how to work with iterable outputs in Instructor, allowing you to process lists and sequences of structured data from language models.

## Basic Iterable Usage

Instructor supports iterating over structured outputs using standard Python list types:

```python
from pydantic import BaseModel
from typing import List
from instructor import patch
from openai import OpenAI

class Item(BaseModel):
    name: str
    quantity: int

class ShoppingList(BaseModel):
    items: List[Item]

client = patch(OpenAI())

shopping = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=ShoppingList,
    messages=[{"role": "user", "content": "Create a shopping list: milk, bread, eggs"}]
)

for item in shopping.items:
    print(f"Need to buy {item.quantity} {item.name}")
```

## Streaming Iterables

You can stream iterable responses for real-time processing:

```python
from instructor import patch
from openai import OpenAI
from typing import Iterator

client = patch(OpenAI())

def stream_items() -> Iterator[Item]:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Item,
        stream=True,
        messages=[{"role": "user", "content": "List grocery items one by one"}]
    )
    for item in response:
        yield item

for item in stream_items():
    print(f"Received item: {item.name}")
```

## Nested Iterables

Handle complex nested structures with multiple levels of iteration:

```python
class Category(BaseModel):
    name: str
    items: List[Item]

class Inventory(BaseModel):
    categories: List[Category]

inventory = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Inventory,
    messages=[{"role": "user", "content": "Create a categorized inventory"}]
)

for category in inventory.categories:
    print(f"Category: {category.name}")
    for item in category.items:
        print(f"  - {item.name}: {item.quantity}")
```

## Best Practices

1. **Type Hints**: Always use proper type hints (List[T] or list[T])
2. **Validation**: Add validators for list contents
3. **Streaming**: Consider streaming for large lists
4. **Error Handling**: Handle partial results gracefully

## Common Patterns

### List Validation
```python
from pydantic import BaseModel, field_validator
from typing import List

class TeamRoster(BaseModel):
    members: List[str]

    @field_validator('members')
    def validate_team_size(cls, v):
        if len(v) < 2:
            raise ValueError('Team must have at least 2 members')
        return v
```

### Unique Items
```python
from pydantic import BaseModel
from typing import Set

class UniqueItems(BaseModel):
    tags: Set[str]  # Automatically ensures uniqueness
```

### Optional Lists
```python
from typing import Optional, List

class Document(BaseModel):
    title: str
    tags: Optional[List[str]] = None
```

## Integration with Other Features

### Partial Results
```python
from instructor import patch
from openai import OpenAI

client = patch(OpenAI())

def stream_partial_list():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=ShoppingList,
        stream=True,
        messages=[{"role": "user", "content": "Create a shopping list"}]
    )
    for partial in response:
        # Access partial results as they arrive
        if partial.items:
            print(f"Items so far: {len(partial.items)}")
```

### Validation Hooks
```python
from instructor import patch
from openai import OpenAI

client = patch(OpenAI())

def validate_items(items: List[Item]) -> bool:
    return all(item.quantity > 0 for item in items)

shopping = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=ShoppingList,
    validation_hook=validate_items,
    messages=[{"role": "user", "content": "Create a shopping list"}]
)
```

## Error Handling

Handle validation errors for lists:

```python
from pydantic import ValidationError

try:
    shopping = ShoppingList(items=[
        {"name": "milk", "quantity": -1}  # Invalid quantity
    ])
except ValidationError as e:
    print(f"Validation error: {e}")
```

For more information about working with lists and iterables, check out the [Pydantic documentation on containers](https://docs.pydantic.dev/latest/concepts/types/#iterables).
