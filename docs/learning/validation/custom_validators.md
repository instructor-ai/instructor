# Custom Validators

Custom validators allow you to implement specialized validation logic for your structured data extraction. This tutorial will show you how to create and use custom validators with Instructor.

## Basic Custom Validator

Custom validators are functions that validate field values and can be applied using Pydantic's field validators.

```python
from pydantic import BaseModel, field_validator
import instructor
from openai import OpenAI

# Initialize the client
client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    age: int
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, value):
        if value < 0 or value > 120:
            raise ValueError("Age must be between 0 and 120")
        return value

# Extract data with validation
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "The person's name is John and they are 150 years old."}
    ],
    response_model=Person
)
```

If the model returns an age outside the valid range, Instructor will retry the request with specific feedback about the validation failure.

For more information on how Instructor handles validation and retries, see [Validation Basics](basics.md) and the [Retrying](/concepts/retrying.md) concepts page.

## Complex Validation

You can create more complex validators that check multiple fields or have conditional logic:

```python
from pydantic import BaseModel, field_validator, model_validator
import instructor
from openai import OpenAI
from typing import List, Optional
from datetime import date

client = instructor.from_openai(OpenAI())

class Employee(BaseModel):
    name: str
    hire_date: date
    termination_date: Optional[date] = None
    skills: List[str]
    
    @field_validator('skills')
    @classmethod
    def validate_skills(cls, skills):
        if len(skills) < 1:
            raise ValueError("Employee must have at least one skill")
        return skills
    
    @model_validator(mode='after')
    def validate_dates(self):
        if self.termination_date and self.termination_date < self.hire_date:
            raise ValueError("Termination date cannot be before hire date")
        return self
```

For more advanced validation approaches, check out [Field-level Validation](field_level_validation.md) and the [Validators](/concepts/reask_validation.md) concepts page.

## Handling Complex Data Types

Custom validators can also process more complex data types and perform transformations:

```python
from pydantic import BaseModel, field_validator
import instructor
from openai import OpenAI
import re

client = instructor.from_openai(OpenAI())

class Contact(BaseModel):
    name: str
    email: str
    phone: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, value):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, value):
            raise ValueError("Invalid email format")
        return value
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, value):
        # Remove non-digit characters and validate
        digits_only = re.sub(r'\D', '', value)
        if len(digits_only) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return digits_only  # Return the cleaned version
```

For a practical example of extraction with validation, see the [Contact Information Extraction](/examples/extract_contact_info.md) example.

## Using External Services for Validation

You can also use external services or APIs for validation:

```python
from pydantic import BaseModel, field_validator
import instructor
from openai import OpenAI
import requests

client = instructor.from_openai(OpenAI())

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    
    @field_validator('zip_code')
    @classmethod
    def validate_zip_code(cls, value):
        # Example of validation using an external service (simplified)
        # In a real app, you might use a postal code validation API
        if not (value.isdigit() and len(value) == 5):
            raise ValueError("Zip code must be 5 digits")
        return value
```

## Handling Validation Failures

When validation fails, Instructor can handle it in different ways. Learn more about:

- [Retry Mechanisms](retry_mechanisms.md) for automatic retries with feedback
- [Self-Correction](/examples/self_critique.md) for AI model self-correction techniques

## Best Practices for Custom Validators

1. **Be specific in error messages**: Provide clear error messages that explain exactly what went wrong
2. **Validate early**: Apply validators to individual fields when possible before model-level validation
3. **Keep validators focused**: Each validator should have a single responsibility
4. **Use type hints**: Proper type hints help both Pydantic and Instructor understand your data better
5. **Consider both validation and transformation**: Validators can both validate and transform data

For more information on validation in general, check out the [Validation](/concepts/validation.md) concepts page.

## Related Resources

- [Fields](/concepts/fields.md) - Learn about field definitions and properties
- [Models](/concepts/models.md) - Understand model creation and configuration
- [Types](/concepts/types.md) - Explore the different data types you can use

Custom validators are a powerful way to ensure the data you extract meets your specific requirements, improving the reliability and quality of structured outputs from LLMs. 