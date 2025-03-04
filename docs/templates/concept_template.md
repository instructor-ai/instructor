---
title: [Concept Name]
description: Learn about [Concept Name] in Instructor and how to use it effectively
---

# [Concept Name]

[One paragraph introduction to the concept, what it is, and why it's important]

## Overview

[Detailed explanation of the concept and its purpose in Instructor]

## When to Use

- [Scenario 1 where this concept is useful]
- [Scenario 2 where this concept is useful]
- [Additional scenarios]

## Basic Usage

Here's how to use [Concept Name] in a basic scenario:

```python
# Standard library imports
import os
from typing import List, Optional

# Third-party imports
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Define your model with proper annotations
class ExampleModel(BaseModel):
    """Model that demonstrates the concept in action."""
    field1: str = Field(description="Description of field1")
    field2: int = Field(description="Description of field2")
    # Additional fields demonstrating the concept

# Initialize client with explicit mode
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.JSON  # Always specify mode explicitly
)

# Use the concept with proper error handling
try:
    result = client.chat.completions.create(
        model="gpt-4o",  # Use latest stable model
        messages=[
            {"role": "system", "content": "Generate structured data based on the user request."},
            {"role": "user", "content": "Example prompt"}
        ],
        response_model=ExampleModel,
        # Concept-specific parameters
    )
    
    print(result.model_dump_json(indent=2))
    # Expected output:
    # {
    #   "field1": "example value",
    #   "field2": 42
    # }
except instructor.exceptions.InstructorError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Usage

[Explanation of more complex use cases]

```python
# More advanced example code
```

## Working with Different Providers

[How this concept works across different LLM providers]

### OpenAI

```python
# OpenAI specific example
```

### Anthropic

```python
# Anthropic specific example if different
```

## Common Patterns and Best Practices

- [Best practice 1]
- [Best practice 2]
- [Additional best practices]

## Related Concepts

- [Link to related concept 1](../concepts/related1.md)
- [Link to related concept 2](../concepts/related2.md)

## Examples

For complete examples, see:

- [Example 1](../examples/example1.md)
- [Example 2](../examples/example2.md)