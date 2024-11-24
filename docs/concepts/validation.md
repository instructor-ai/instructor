# Validation in Instructor

This guide covers validation concepts and best practices when using Instructor for structured outputs.

## Overview

Validation in Instructor ensures that the output from language models matches your expected schema. This is crucial for:
- Data consistency
- Error handling
- Type safety
- Business logic enforcement

## Basic Validation

Instructor uses Pydantic for validation, which provides:
1. Type checking
2. Data coercion
3. Custom validators
4. Field constraints

```python
from pydantic import BaseModel, Field, validator
from typing import List


class User(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=0, le=150)
    emails: List[str]

    @validator('emails')
    def validate_emails(cls, v):
        if not all('@' in email for email in v):
            raise ValueError('Invalid email format')
        return v
```

## Validation Strategies

### 1. Field Validation

Use Field() for basic constraints:
```python
class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=0)
```

### 2. Custom Validators

Use @validator for complex validation:
```python
class Order(BaseModel):
    items: List[str]
    total: float

    @validator('total')
    def validate_total(cls, v, values):
        if v < 0:
            raise ValueError('Total cannot be negative')
        return v
```

### 3. Pre-validation Hooks

Use pre-validation hooks for data transformation:
```python
class UserProfile(BaseModel):
    username: str

    @validator('username', pre=True)
    def lowercase_username(cls, v):
        return v.lower()
```

## Error Handling

Instructor provides robust error handling for validation failures:

```python
from instructor import patch
import openai

client = patch(openai.OpenAI())

try:
    user = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=User,
        messages=[{"role": "user", "content": "Extract: John Doe, age: -5"}],
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Best Practices

1. **Start Simple**: Begin with basic type validation before adding complex rules
2. **Use Type Hints**: Always specify types for better code clarity
3. **Document Constraints**: Add clear descriptions to Field() definitions
4. **Handle Errors**: Implement proper error handling for validation failures
5. **Test Edge Cases**: Verify validation works with unexpected inputs

## Common Patterns

### Optional Fields
```python
class Profile(BaseModel):
    name: str
    bio: Optional[str] = None
```

### Nested Validation
```python
class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    addresses: List[Address]
```

### Complex Validation
```python
class Transaction(BaseModel):
    amount: float
    currency: str
    timestamp: datetime

    @validator('currency')
    def validate_currency(cls, v):
        valid_currencies = ['USD', 'EUR', 'GBP']
        if v not in valid_currencies:
            raise ValueError(f'Currency must be one of {valid_currencies}')
        return v
```

## Related Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/gpt/function-calling)
- [Instructor Examples](../examples/index.md)

## Updates and Compatibility

- Works with all supported LLM providers
- Compatible with latest Pydantic versions
- Regular updates for new validation features
