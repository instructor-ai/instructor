# Validation in Instructor

This guide covers validation concepts and best practices when using Instructor for structured outputs.

## Overview

Validation in Instructor ensures that the output from language models matches your expected schema. This is crucial for:
- Data consistency
- Error handling
- Type safety
- Business logic enforcement

## Basic Validation

Instructor uses Pydantic v2 for validation, which provides:
1. Type checking
2. Data coercion
3. Custom field validators
4. Field constraints

The main validation tools in Pydantic v2 are:
- `field_validator` decorator for custom validation logic
- `Field` for basic constraints and metadata
- `ValidationInfo` for accessing model context during validation

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class User(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=0, le=150)
    emails: List[str]

    @field_validator('emails')
    @classmethod
    def validate_emails(cls, v):
        if not all('@' in email for email in v):
            raise ValueError('Invalid email format')
        return v

## Validation Strategies

### 1. Field Validation

Use Field() for basic constraints:
```python
class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=0)
```

### 2. Custom Field Validators

Use `@field_validator` with appropriate mode for complex validation:
```python
class Order(BaseModel):
    items: List[str]
    total: float

    @field_validator('total')
    @classmethod
    def validate_total(cls, v, info: ValidationInfo):
        if v < 0:
            raise ValueError('Total cannot be negative')
        return v
```

### 3. Validation Modes

Pydantic v2 provides different validation modes:
- `mode='before'`: Run before type coercion (pre-validation)
- `mode='after'`: Run after type coercion (post-validation)

Example using before mode for data transformation:
```python
class UserProfile(BaseModel):
    username: str

    @field_validator('username', mode='before')
    @classmethod
    def lowercase_username(cls, v):
        return v.lower()
```

## Error Handling

Instructor provides robust error handling for validation failures:

```python
from instructor import patch
import openai

client = patch(openai.OpenAI())

from pydantic import BaseModel
from typing import Optional

try:
    user = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_model=User,
        messages=[{"role": "user", "content": "Extract: John Doe, age: -5"}],
    )
except ValueError as e:
    print(f"Validation error: {e}")

## Best Practices

1. **Use Field Validators**: Prefer `field_validator` over older validation patterns
2. **Choose Appropriate Mode**: Use `mode='before'` or `mode='after'` based on needs
3. **Use ValidationInfo**: Access model context in validators when needed
4. **Add @classmethod**: Always include @classmethod with field_validator
5. **Handle Errors**: Implement proper error handling for validation failures
6. **Test Edge Cases**: Verify validation works with unexpected inputs

## Common Patterns

### Optional Fields
```python
from pydantic import BaseModel
from typing import Optional

class Profile(BaseModel):
    name: str
    bio: Optional[str] = None
```

### Nested Validation
```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    addresses: List[Address]

### Complex Validation with ValidationInfo
from pydantic import BaseModel, field_validator, ValidationInfo
from datetime import datetime

class Transaction(BaseModel):
    amount: float
    currency: str
    timestamp: datetime

    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v, info: ValidationInfo):
        valid_currencies = ['USD', 'EUR', 'GBP']
        if v not in valid_currencies:
            raise ValueError('Currency must be one of {valid_currencies}')
        # Can access other fields via info.data
        return v

## Related Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/gpt/function-calling)
- [Instructor Examples](../examples/index.md)

## Updates and Compatibility

- Works with all supported LLM providers
- Compatible with latest Pydantic versions
- Regular updates for new validation features
