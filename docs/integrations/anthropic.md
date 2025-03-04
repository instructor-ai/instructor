---
title: "Structured outputs with Anthropic, a complete guide w/ instructor"
description: Learn how to combine Anthropic and Instructor clients to create user models with complex properties in Python.
---

# Structured outputs with Anthropic, a complete guide w/ instructor

Now that we have a [Anthropic](https://www.anthropic.com/) client, we can use it with the `instructor` client to make requests.

Let's first install the instructor client with anthropic support

```
pip install "instructor[anthropic]"
```

Once we've done so, getting started is as simple as using our `from_anthropic` method to patch the client up.

### Basic Usage

```python
# Standard library imports
import os
from typing import List

# Third-party imports
import anthropic
import instructor
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Define your models with proper type annotations
class Properties(BaseModel):
    """Model representing a key-value property."""
    name: str = Field(description="The name of the property")
    value: str = Field(description="The value of the property")


class User(BaseModel):
    """Model representing a user with properties."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    properties: List[Properties] = Field(description="List of user properties")

# Initialize the client with explicit mode
client = instructor.from_anthropic(
    anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
    mode=instructor.Mode.ANTHROPIC_TOOLS  # Using Anthropic's tool calling API
)

try:
    # Extract structured data
    user_response = client.chat.completions.create(
        model="claude-3-haiku-20240307",  # Use latest stable model
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": "Extract structured information based on the user's request."
            },
            {
                "role": "user",
                "content": "Create a user for a model with a name, age, and properties.",
            }
        ],
        response_model=User,
    )
    
    # Print the result as formatted JSON
    print(user_response.model_dump_json(indent=2))
    
    # Expected output:
    # {
    #   "name": "John Doe",
    #   "age": 35,
    #   "properties": [
    #     {
    #       "name": "City",
    #       "value": "New York"
    #     },
    #     {
    #       "name": "Occupation",
    #       "value": "Software Engineer"
    #     }
    #   ]
    # }
except instructor.exceptions.InstructorError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
```

### Beta API Features

Anthropic provides beta features through their beta API endpoint. To use these features, enable the beta parameter when creating the client:

```python
client = instructor.from_anthropic(
    anthropic.Anthropic(),
    beta=True  # Enable beta API features
)
```

This allows you to access beta features like PDF support and other experimental capabilities. For more information about available beta features, refer to [Anthropic's documentation](https://docs.anthropic.com/claude/docs/beta-features).

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

You can use our `create_partial` method to stream a single object. Note that validators should not be declared in the response model when streaming objects because it will break the streaming process.

```python
# Standard library imports
import os

# Third-party imports
import anthropic
from instructor import from_anthropic
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Initialize client with explicit mode
client = from_anthropic(
    anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
    mode=instructor.Mode.ANTHROPIC_TOOLS
)

# Define your model with proper annotations
class User(BaseModel):
    """Model representing a user profile."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")
    bio: str = Field(description="A biographical description of the user")

try:
    # Stream partial objects as they're generated
    for partial_user in client.chat.completions.create_partial(
        model="claude-3-haiku-20240307",  # Use latest stable model
        messages=[
            {"role": "system", "content": "Create a detailed user profile based on the information provided."},
            {"role": "user", "content": "Create a user profile for Jason, age 25"},
        ],
        response_model=User,
        max_tokens=4096,
    ):
        print(f"Current state: {partial_user}")
        
    # Expected output:
    # > Current state: name='Jason' age=None bio=None
    # > Current state: name='Jason' age=25 bio='Jason is a 25-year-old with an adventurous spirit and a love for technology. He is'
    # > Current state: name='Jason' age=25 bio='Jason is a 25-year-old with an adventurous spirit and a love for technology. He is always on the lookout for new challenges and opportunities to grow both personally and professionally.'
except Exception as e:
    print(f"Error during streaming: {e}")
```

### Iterable Example

You can also use our `create_iterable` method to stream a list of objects. This is helpful when you'd like to extract multiple instances of the same response model from a single prompt.

```python
# Standard library imports
import os

# Third-party imports
import anthropic
from instructor import from_anthropic
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Initialize client with explicit mode
client = from_anthropic(
    anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
    mode=instructor.Mode.ANTHROPIC_TOOLS
)

# Define your model with proper annotations
class User(BaseModel):
    """Model representing a basic user."""
    name: str = Field(description="The user's full name")
    age: int = Field(description="The user's age in years")

try:
    # Create an iterable of user objects
    users = client.chat.completions.create_iterable(
        model="claude-3-haiku-20240307",  # Use latest stable model
        messages=[
            {
                "role": "system", 
                "content": "Extract all users from the provided text into structured format."
            },
            {
                "role": "user",
                "content": """
                Extract users:
                1. Jason is 25 years old
                2. Sarah is 30 years old
                3. Mike is 28 years old
                """,
            },
        ],
        max_tokens=4096,
        response_model=User,
    )

    # Process each user as it's extracted
    for user in users:
        print(user)
        
    # Expected output:
    # > name='Jason' age=25
    # > name='Sarah' age=30
    # > name='Mike' age=28
except Exception as e:
    print(f"Error during iteration: {e}")
```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that Anthropic supports

1. `instructor.Mode.ANTHROPIC_JSON` : This uses the text completion API from the Anthropic API and then extracts out the desired response model from the text completion model
2. `instructor.Mode.ANTHROPIC_TOOLS` : This uses Anthropic's [tools calling API](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) to return structured outputs to the client

In general, we recommend using `Mode.ANTHROPIC_TOOLS` because it's the best way to ensure you have the desired response schema that you want.

## Caching

If you'd like to use caching with the Anthropic Client, we also support it for images and text input.

### Caching Text Input

Here's how you can implement caching for text input ( assuming you have a giant `book.txt` file that you read in).

We've written a comprehensive walkthrough of how to use caching to implement Anthropic's new Contextual Retrieval method that gives a significant bump to retrieval accuracy.

```python
# Standard library imports
import os

# Third-party imports
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Define your Pydantic model with proper annotations
class Character(BaseModel):
    """Model representing a character extracted from text."""
    name: str = Field(description="The character's full name")
    description: str = Field(description="A description of the character")

# Initialize client with explicit mode and prompt caching
client = instructor.from_anthropic(
    Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
    mode=instructor.Mode.ANTHROPIC_TOOLS,
    enable_prompt_caching=True  # Enable prompt caching
)

try:
    # Load your large context
    with open("./book.txt", "r") as f:
        book = f.read()

    # Make multiple calls using the cached context
    for _ in range(2):
        # The first time processes the large text, subsequent calls use the cache
        resp, completion = client.chat.completions.create_with_completion(
            model="claude-3-haiku-20240307",  # Use latest stable model
            messages=[
                {
                    "role": "system",
                    "content": "Extract character information from the provided text."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<book>" + book + "</book>",
                            "cache_control": {"type": "ephemeral"},  # Mark for caching
                        },
                        {
                            "type": "text",
                            "text": "Extract a character from the text given above",
                        },
                    ],
                },
            ],
            response_model=Character,
            max_tokens=1000,
        )
        
        # Process the result
        print(f"Character: {resp.name}")
        print(f"Description: {resp.description}")
        
        # The completion contains the raw response
        print(f"Raw completion length: {len(completion)}")
        
    # Note: Second iteration should be faster due to cache hit
    
except Exception as e:
    print(f"Error: {e}")
```

### Caching Images

We also support caching for images. This helps significantly, especially if you're using images repeatedly to save on costs. Read more about it [here](../concepts/caching.md)

```python
# Standard library imports
import os

# Third-party imports
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field

# Set up environment (typically handled before script execution)
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key"  # Uncomment and replace with your API key if not set

# Define your model for image analysis
class ImageAnalyzer(BaseModel):
    """Model for analyzing image content."""
    content_description: str = Field(description="Description of what appears in the images")
    objects: list[str] = Field(description="List of objects visible in the images")
    scene_type: str = Field(description="Type of scene shown in the images (indoor, outdoor, etc.)")

# Initialize client with explicit mode and image caching enabled
client = instructor.from_anthropic(
    Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")),
    mode=instructor.Mode.ANTHROPIC_TOOLS,
    enable_prompt_caching=True  # Enable prompt caching
)

try:
    # Configure cache control for images
    cache_control = {"type": "ephemeral"}
    
    # Make a request with cached images
    response = client.chat.completions.create(
        model="claude-3-haiku-20240307",  # Use latest stable model
        response_model=ImageAnalyzer,
        messages=[
            {
                "role": "system",
                "content": "Analyze the content of the provided images in detail."
            },
            {
                "role": "user",
                "content": [
                    "What is in these two images?",
                    # Remote image with caching
                    {
                        "type": "image", 
                        "source": "https://example.com/image.jpg", 
                        "cache_control": cache_control
                    },
                    # Local image with caching
                    {
                        "type": "image", 
                        "source": "path/to/image.jpg", 
                        "cache_control": cache_control
                    },
                ]
            }
        ],
        autodetect_images=True  # Automatically handle image content
    )
    
    # Process the results
    print(f"Description: {response.content_description}")
    print(f"Objects: {', '.join(response.objects)}")
    print(f"Scene type: {response.scene_type}")
    
    # Subsequent identical requests will use cached images
    
except Exception as e:
    print(f"Error during image analysis: {e}")
```
