---
title: "Structured outputs with OpenAI, a complete guide with instructor"
description: "Learn how to use Instructor with OpenAI's models for type-safe, structured outputs. Complete guide with examples and best practices for GPT-4 and other OpenAI models."
---

# Structured outputs with OpenAI, a complete guide with instructor

OpenAI is the primary integration for Instructor, offering robust support for structured outputs with GPT-3.5, GPT-4, and future models. This guide covers everything you need to know about using OpenAI with Instructor for type-safe, validated responses.

## Quick Start

Instructor comes with support for OpenAI out of the box, so you don't need to install anything extra.

```bash
pip install "instructor"
```

⚠️ **Important**: You must set your OpenAI API key before using the client. You can do this in two ways:

1. Set the environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

2. Or provide it directly to the client:

```python
import os
from openai import OpenAI
client = OpenAI(api_key='your-api-key-here')
```

## Simple User Example (Sync)

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Enable instructor patches for OpenAI client
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
    response_model=User,
)

print(user)
#> User(name='Jason', age=25)
```

## Simple User Example (Async)

```python
import os
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel
import asyncio

# Initialize with API key
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Enable instructor patches for async OpenAI client
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

async def extract_user():
    user = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Extract: Jason is 25 years old"},
        ],
        response_model=User,
    )
    return user

# Run async function
user = asyncio.run(extract_user())
print(user)
#> User(name='Jason', age=25)
```

## Nested Example

```python
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]

# Initialize with API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Enable instructor patches for OpenAI client
client = instructor.from_openai(client)
# Create structured output with nested objects
user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": """
            Extract: Jason is 25 years old.
            He lives at 123 Main St, New York, USA
            and has a summer house at 456 Beach Rd, Miami, USA
        """},
    ],
    response_model=User,
)

print(user)
#> {
#>     'name': 'Jason',
#>     'age': 25,
#>     'addresses': [
#>         {
#>             'street': '123 Main St',
#>             'city': 'New York',
#>             'country': 'USA'
#>         },
#>         {
#>             'street': '456 Beach Rd',
#>             'city': 'Miami',
#>             'country': 'USA'
#>         }
#>     ]
#> }
```

## Multimodal

> We've provided a few different sample files for you to use to test out these new features. All examples below use these files.
>
> - (Audio) : A Recording of the Original Gettysburg Address : [gettysburg.wav](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav)
> - (Image) : An image of some blueberry plants [image.jpg](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg)
> - (PDF) : A sample PDF file which contains a fake invoice [invoice.pdf](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf)

Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images, PDFs, and audio files. With Instructor's multimodal objects, you can easily load media from URLs, local files, or base64 strings using a consistent API that works across different AI providers (OpenAI, Anthropic, Mistral, etc.).

Instructor handles all the provider-specific formatting requirements behind the scenes, ensuring your code remains clean and future-proof as provider APIs evolve.

Let's see how to use the Image, Audio and PDF classes.

### Image

> For a more in-depth walkthrough of the Image component, check out the [docs here](../concepts/multimodal.md)

Instructor makes it easy to analyse and extract semantic information from images using OpenAI's GPT-4o models. [Click here](https://platform.openai.com/docs/models) to check if the model you'd like to use has vision capabilities.

Let's see an example below with the sample image above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path` and the `from_base64` class methods.

```python
from instructor.multimodal import Image
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI


class ImageDescription(BaseModel):
    objects: list[str] = Field(..., description="The objects in the image")
    scene: str = Field(..., description="The scene of the image")
    colors: list[str] = Field(..., description="The colors in the image")


client = instructor.from_openai(OpenAI())
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"
# Multiple ways to load an image:
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                # Option 1: Direct URL with autodetection
                Image.from_url(url),
                # Option 2: Local file
                # Image.from_path("path/to/local/image.jpg")
                # Option 3: Base64 string
                # Image.from_base64("base64_encoded_string_here")
                # Option 4: Autodetect
                # Image.autodetect(<url|path|base64>)
            ],
        },
    ],
)

print(response)
# Example output:
# ImageDescription(
#     objects=['blueberries', 'leaves'],
#     scene='A blueberry bush with clusters of ripe blueberries and some unripe ones against a cloudy sky',
#     colors=['green', 'blue', 'purple', 'white']
# )
```

### PDF

Instructor makes it easy to analyse and extract semantic information from PDFs using OpenAI's GPT-4o models.

Let's see an example below with the sample PDF above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path` and the `from_base64` class methods.

```python
from instructor.multimodal import PDF
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI


class Receipt(BaseModel):
    total: int
    items: list[str]


client = instructor.from_openai(OpenAI())
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
# Multiple ways to load an PDF:
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Receipt,
    messages=[
        {
            "role": "user",
            "content": [
                "Extract out the total and line items from the invoice",
                # Option 1: Direct URL
                PDF.from_url(url),
                # Option 2: Local file
                # PDF.from_path("path/to/local/invoice.pdf"),
                # Option 3: Base64 string
                # PDF.from_base64("base64_encoded_string_here")
                # Option 4: Autodetect
                # PDF.autodetect(<url|path|base64>)
            ],
        },
    ],
)

print(response)
# > Receipt(total=220, items=['English Tea', 'Tofu'])
```

### Audio

Instructor makes it easy to analyse and extract semantic information from Audio files using OpenAI's GPT-4o models. Let's see an example below with the sample Audio file above where we'll load it in using our `from_url` method.

Note that we support local files and base64 strings too with the `from_path`

```python
from instructor.multimodal import Audio
from pydantic import BaseModel
import instructor
from openai import OpenAI


class AudioDescription(BaseModel):
    transcript: str
    summary: str
    speakers: list[str]
    key_points: list[str]


url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"

client = instructor.from_openai(OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    response_model=AudioDescription,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Please transcribe and analyze this audio:",
                # Multiple loading options:
                Audio.from_url(url),
                # Option 2: Local file
                # Audio.from_path("path/to/local/audio.mp3")
            ],
        },
    ],
)

print(response)
# > transcript='Four score and seven years ago our fathers..."]
```

## Streaming Support

Instructor has two main ways that you can use to stream responses out

1. **Iterables**: These are useful when you'd like to stream a list of objects of the same type (Eg. use structured outputs to extract multiple users)
2. **Partial Streaming**: This is useful when you'd like to stream a single object and you'd like to immediately start processing the response as it comes in.

### Partials

```python
from instructor import from_openai
import openai
from pydantic import BaseModel

client = from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int
    bio: str


user = client.chat.completions.create_partial(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Create a user profile for Jason, age 25"},
    ],
    response_model=User,
)

for user_partial in user:
    print(user_partial)

# > name='Jason' age=None bio='None'
# > name='Jason' age=25 bio='A tech'
# > name='Jason' age=25 bio='A tech enthusiast'
# > name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new'
# > name='Jason' age=25 bio='A tech enthusiast who loves coding, gaming, and exploring new technologies'

```

### Iterable Example

```python
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Extract multiple users from text
users = client.chat.completions.create_iterable(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": """
            Extract users:
            1. Jason is 25 years old
            2. Sarah is 30 years old
            3. Mike is 28 years old
        """},
    ],
    response_model=User,
)

for user in users:
    print(user)
    #> name='Jason' age=25
    #> name='Sarah' age=30
    #> name='Mike' age=28
```

## Instructor Modes

We provide several modes to make it easy to work with the different response models that OpenAI supports

1. `instructor.Mode.TOOLS` : This uses the [tool calling API](https://platform.openai.com/docs/guides/function-calling) to return structured outputs to the client
2. `instructor.Mode.JSON` : This forces the model to return JSON by using [OpenAI's JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode).
3. `instructor.Mode.FUNCTIONS` : This uses OpenAI's function calling API to return structured outputs and will be deprecated in the future.
4. `instructor.Mode.PARALLEL_TOOLS` : This uses the [parallel tool calling API](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling) to return structured outputs to the client. This allows the model to generate multiple calls in a single response.
5. `instructor.Mode.MD_JSON` : This makes a simple call to the OpenAI chat completion API and parses the raw response as JSON.
6. `instructor.Mode.TOOLS_STRICT` : This uses the new Open AI structured outputs API to return structured outputs to the client using constrained grammar sampling. This restricts users to a subset of the JSON schema.
7. `instructor.Mode.JSON_O1` : This is a mode for the `O1` model. We created a new mode because `O1` doesn't support any system messages, tool calling or streaming so you need to use this mode to use Instructor with `O1`.

In general, we recommend using `Mode.Tools` because it's the most flexible and future-proof mode. It has the largest set of features that you can specify your schema in and makes things significantly easier to work with.

## Batch API

We also support batching requests using the `create_batch` method. This is helpful if your request is not time sensitive because you'll get a 50% discount on the token cost.

Read more about how to use it [here](../examples/batch_job_oai.md)

## Best Practices

1. **Model Selection** : We recommend using gpt-4o-mini for simpler use cases because it's cheap and works well with a clearly defined objective for structured outputs. When the task is more ambigious, consider upgrading to `4o` or even `O1` depending on your needs

2. **Performance Optimization** : Streaming a response model is faster and should be done from the get-go. This is especially true if you're using a simple response model.

## Common Use Cases

- Data Extraction
- Form Parsing
- API Response Structuring
- Document Analysis
- Configuration Generation

## Related Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Instructor Core Concepts](../concepts/index.md)
- [Type Validation Guide](../concepts/validation.md)
- [Advanced Usage Examples](../examples/index.md)

## Updates and Compatibility

Instructor maintains compatibility with the latest OpenAI API versions and models. Check the [changelog](https://github.com/jxnl/instructor/blob/main/CHANGELOG.md) for updates.
