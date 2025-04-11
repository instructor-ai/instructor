# Retry Mechanisms

Retry mechanisms in Instructor handle validation failures by giving the LLM another chance to generate valid responses. This guide explains how retries work and how to customize them for your use case.

## How Retries Work

When validation fails, Instructor:

1. Captures the validation error(s)
2. Formats them as feedback
3. Adds the feedback to the prompt context
4. Asks the LLM to try again with this new information

This creates a feedback loop that helps the LLM correct its output until it produces a valid response.

## Basic Retry Example

Here's a simple example showing retries in action:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# Initialize the client with max_retries
client = instructor.from_openai(
    OpenAI(),
    max_retries=2  # Will try up to 3 times (initial + 2 retries)
)

class Product(BaseModel):
    name: str
    price: float = Field(..., gt=0)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError("Product name must be at least 3 characters")
        return v

# This will automatically retry if validation fails
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Product: Pen, Price: -5"}
    ],
    response_model=Product
)
```

In this example, the initial response will likely fail validation because:
- The price is negative (violating the `gt=0` constraint)
- Instructor will automatically retry with feedback about these issues

For more details on max_retries configuration, see the [Retrying](/concepts/retrying.md) concepts page.

## Customizing Retry Behavior

You can customize retry behavior when initializing the Instructor client:

```python
import instructor
from openai import OpenAI

# Customize retry behavior
client = instructor.from_openai(
    OpenAI(),
    max_retries=3,                   # Maximum number of retries
    retry_if_parsing_fails=True,     # Retry on JSON parsing failures
    throw_error=True                 # Throw an error if all retries fail
)
```

### Retry Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_retries` | Maximum number of retry attempts | 0 |
| `retry_if_parsing_fails` | Whether to retry if JSON parsing fails | True |
| `throw_error` | Whether to throw an error if all retries fail | True |

## Handling Retry Failures

When all retries fail, depending on your configuration:

1. With `throw_error=True` (default): An exception is raised
2. With `throw_error=False`: The last failed response is returned, and you can handle it gracefully

For more on handling validation failures, see [Fallback Strategies](fallback_strategies.md).

## Error Messages and Feedback

Instructor provides detailed error messages to the LLM during retries:

```
The following errors occurred during validation:
- price: ensure this value is greater than 0
- name: Product name must be at least 3 characters

Please fix these errors and ensure the response is valid.
```

This feedback helps the LLM understand exactly what needs to be fixed.

## Retry Limitations

While retries are powerful, they have some limitations:

1. **Retry Budget**: Each retry consumes tokens and time
2. **Persistent Errors**: Some errors might not be fixable by the LLM
3. **Model Limitations**: Some models may consistently struggle with certain validations

For complex validation scenarios, consider implementing [Custom Validators](custom_validators.md) or [Field-level Validation](field_level_validation.md).

## Advanced Retry Pattern: Progressive Validation

For complex schemas, you can implement a progressive validation pattern:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# Initialize with moderate retries
client = instructor.from_openai(
    OpenAI(),
    max_retries=2
)

# Basic validation first
class BasicProduct(BaseModel):
    name: str
    price: float = Field(..., gt=0)

# Advanced validation second
class DetailedProduct(BasicProduct):
    description: str = Field(..., min_length=10)
    category: str
    in_stock: bool

# Two-step extraction with validation
try:
    # First get basic fields
    basic = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Product: Mini Pen, Price: $2.50"}
        ],
        response_model=BasicProduct
    )
    
    # Then get full details with context from the first step
    detailed = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Provide more details about {basic.name} which costs ${basic.price}"}
        ],
        response_model=DetailedProduct
    )
except Exception as e:
    # Handle validation failures
    print(f"Validation failed: {e}")
```

## Related Resources

- [Retrying](/concepts/retrying.md) - Core retry concepts
- [Validation](/concepts/validation.md) - Main validation documentation
- [Custom Validators](custom_validators.md) - Creating custom validation logic
- [Fallback Strategies](fallback_strategies.md) - Handling persistent validation failures
- [Self Critique](/examples/self_critique.md) - Example of model self-correction

## Next Steps

- Learn about [Field-level Validation](field_level_validation.md)
- Implement [Custom Validators](custom_validators.md) 