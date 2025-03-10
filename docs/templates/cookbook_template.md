---
title: [Example Name]
description: Learn how to [what the example accomplishes] using Instructor
---

# [Example Name]

[Introduction explaining the practical problem this example solves and why it's useful]

## Use Case

[Detailed explanation of when and why you would use this approach]

## Prerequisites

Before you begin, make sure you have:

- Instructor installed: `pip install instructor`
- Required dependencies: `pip install [any additional libraries]`
- API keys configured:
  ```bash
  export OPENAI_API_KEY=your_api_key_here
  # Any other environment variables
  ```

## Implementation

### Step 1: [First step description]

[Brief explanation of what this step accomplishes]

```python
# Standard library imports
import os
from typing import List, Optional, Dict, Any

# Third-party imports
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Step 1 code with comments explaining key parts
```

### Step 2: [Second step description]

[Brief explanation of what this step accomplishes]

```python
# Step 2 code with comments explaining key parts
```

### Step 3: [Third step description]

[Brief explanation of what this step accomplishes]

```python
# Step 3 code with comments explaining key parts
```

## Complete Solution

Here's the complete code:

```python
# Standard library imports
import os
from typing import List, Optional, Dict, Any

# Third-party imports
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# Local imports (if any)
# from my_module import my_function

# Set up environment
# os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Define your models with proper type annotations
class MyResponseModel(BaseModel):
    """Model representing the structured output from the LLM."""
    field1: str = Field(description="Description of field1")
    field2: int = Field(description="Description of field2")
    optional_field: Optional[List[str]] = Field(None, description="Optional field example")

# Initialize the client with explicit mode
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.JSON  # Always specify mode explicitly
)

def process_data(input_text: str) -> MyResponseModel:
    """
    Process input text and return structured data.
    
    Args:
        input_text: The text to be processed
        
    Returns:
        A structured MyResponseModel object
    """
    try:
        result = client.chat.completions.create(
            model="gpt-4o",  # Use a consistent, current model version
            messages=[
                {"role": "system", "content": "Extract structured information from the user input."},
                {"role": "user", "content": input_text}
            ],
            response_model=MyResponseModel,
        )
        return result
    except instructor.exceptions.InstructorError as e:
        # Handle validation/extraction errors
        print(f"Extraction error: {e}")
        raise
    except Exception as e:
        # Handle other errors (API, network, etc.)
        print(f"Unexpected error: {e}")
        raise

# Example usage
if __name__ == "__main__":
    input_text = "Field1 is example data and field2 is 42."
    result = process_data(input_text)
    print(result.model_dump_json(indent=2))

# Expected output:
# {
#   "field1": "example data",
#   "field2": 42,
#   "optional_field": null
# }
```

## How It Works

[Detailed explanation of the key concepts and techniques used]

## Customization Options

Here are some ways you can customize this solution:

- [Option 1]: [How to customize this aspect]
- [Option 2]: [How to customize this aspect]
- [Advanced option]: [More complex customization]

## Limitations and Considerations

- [Limitation 1]
- [Limitation 2]
- [Important consideration]

## Related Concepts

- [Link to related concept 1](../concepts/related1.md)
- [Link to related concept 2](../concepts/related2.md)

## Next Steps

Now that you've mastered this technique, you might want to explore:

- [Link to related example 1](../examples/related1.md)
- [Link to related example 2](../examples/related2.md)