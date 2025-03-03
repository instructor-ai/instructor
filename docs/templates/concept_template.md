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
import instructor
from openai import OpenAI
from pydantic import BaseModel

# Example code showing the concept in action
client = instructor.from_openai(OpenAI())

# Define your model
class ExampleModel(BaseModel):
    # Model fields that demonstrate the concept
    field1: str
    field2: int

# Use the concept
result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Example prompt"}],
    response_model=ExampleModel,
    # Concept-specific parameters
)

print(result)
# Expected output
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