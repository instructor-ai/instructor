# Validation Basics

Validation ensures that the data extracted by LLMs meets your requirements. This guide covers the essentials of validation with Instructor.

## Why Validation Matters

Validation helps ensure:

1. **Data Integrity**: All required fields are present and formatted correctly
2. **Consistency**: Data follows your business rules
3. **Quality**: Outputs meet specific criteria for your application

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ LLM         │ -> │ Instructor   │ -> │ Validated   │
│ Generates   │    │ Validates    │    │ Structured  │
│ Response    │    │ Structure    │    │ Data        │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          │ If validation fails
                          ▼
                   ┌─────────────┐
                   │ Retry with  │
                   │ Feedback    │
                   └─────────────┘
```

## Simple Example

Here's a basic example with validation:

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

# Define a model with validation
class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=13, description="User's age in years")

# Extract validated data
client = instructor.from_openai(OpenAI())
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "My name is Jane Smith and I'm 25 years old."}
    ],
    response_model=UserProfile
)

print(f"User: {response.name}, Age: {response.age}")
```

In this example:
- The `age` field has a validation constraint (`ge=13`) ensuring users are at least 13 years old
- If validation fails, Instructor will automatically retry with feedback

## Common Validation Types

Here are the most common validations you can use:

| Validation | Example | What It Does |
|------------|---------|-------------|
| Type checking | `age: int` | Ensures value is an integer |
| Required fields | `name: str` | Field must be present |
| Optional fields | `middle_name: Optional[str] = None` | Field can be missing |
| Minimum value | `age: int = Field(ge=18)` | Value must be ≥ 18 |
| Maximum value | `rating: float = Field(le=5.0)` | Value must be ≤ 5.0 |
| String length | `username: str = Field(min_length=3)` | String must be at least 3 chars |

## How Validation Works

When using validation with Instructor:

1. The LLM generates a response based on your prompt
2. Instructor tries to fit the response into your model
3. If validation fails, Instructor captures the errors
4. The errors are sent back to the LLM for a retry
5. This continues until validation passes or max retries is reached

## Adding Custom Error Messages

For clearer feedback, you can add custom error messages:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str
    price: float = Field(
        gt=0, 
        description="Product price in USD",
        json_schema_extra={"error_msg": "Price must be greater than zero"}
    )
```

## Next Steps

- Learn about [Custom Validators](custom_validators.md)
- Explore [Retry Mechanisms](retry_mechanisms.md)
- Try [Field-level Validation](field_level_validation.md) 