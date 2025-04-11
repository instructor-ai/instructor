---
title: Seamless Multimodal Interactions with Instructor
description: Learn how the Image, PDF and Audio class in Instructor enables seamless handling of multimodal content across different AI models.
---

# Multimodal

> We've provided a few different sample files for you to use to test out these new features. All examples below use these files.
>
> - (Image) : An image of some blueberry plants [image.jpg](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg)
> - (Audio) : A Recording of the Original Gettysburg Address : [gettysburg.wav](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav)
> - (PDF) : A sample PDF file which contains a fake invoice [invoice.pdf](https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf)
>   Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images and PDFs.

Instructor provides a unified, provider-agnostic interface for working with multimodal inputs like images, PDFs, and audio files.

With Instructor's multimodal objects, you can easily load media from URLs, local files, or base64 strings using a consistent API that works across different AI providers (OpenAI, Anthropic, Mistral, etc.).

Instructor handles all the provider-specific formatting requirements behind the scenes, ensuring your code remains clean and future-proof as provider APIs evolve. Let's see how to use the Image, Audio and PDF classes.

## `Image`

This class represents an image that can be loaded from a URL or file path. It provides a set of methods to create `Image` instances from different sources (Eg. URLs, paths and base64 strings). The following shows which methods are supported for the individual providers.

| Method          | OpenAI | Anthropic | Google GenAI |
| --------------- | ------ | --------- | ------------ |
| `from_url()`    | ✅     | ✅        | ✅           |
| `from_path()`   | ✅     | ✅        | ✅           |
| `from_base64()` | ✅     | ✅        | ✅           |
| `autodetect()`  | ✅     | ✅        | ✅           |

We also support Anthropic Prompt Caching for images with the `ImageWith

### Usage

By using the `Image` class, we can abstract away the differences between the different formats, allowing you to work with a unified interface.

You can create an `Image` instance from a URL or file path using the `from_url` or `from_path` methods. The `Image` class will automatically convert the image to a base64-encoded string and include it in the API request.

```python
import instructor
from instructor.multimodal import Image
import openai
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Use our sample image provided above.
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

client = instructor.from_openai(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                Image.from_url(url),
            ],
        }
    ],
)

print(response)
# > description='A bush with numerous clusters of blueberries surrounded by green leaves, under a cloudy sky.' items=['blueberries', 'green leaves', 'cloudy sky']
```

We also provide a `autodetect_image` keyword argument that allows you to provide URLs or file paths as normal strings when you set it to true.

You can see an example below.

```python
import instructor
from instructor.multimodal import Image
import openai
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Download a sample image for demonstration
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

client = instructor.from_openai(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ImageDescription,
    autodetect_images=True,  # Set this to True
    messages=[
        {
            "role": "user",
            "content": ["What is in this image?", url],
        }
    ],
)

print(response)
# > description='A bush with numerous clusters of blueberries surrounded by green leaves, under a cloudy sky.' items=['blueberries', 'green leaves', 'cloudy sky']
```

If you'll like to support Anthropic prompt caching with images, we provide the `ImageWithCacheControl` Object to do so. Simply use the `from_image_params` method and you'll be able to leverage Anthropic's prompt caching.

```python
import instructor
from instructor.multimodal import ImageWithCacheControl
import anthropic
from pydantic import BaseModel


class ImageDescription(BaseModel):
    description: str
    items: list[str]


# Download a sample image for demonstration
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/image.jpg"

client = instructor.from_anthropic(anthropic.Anthropic())

response, completion = client.chat.completions.create_with_completion(
    model="claude-3-5-sonnet-20240620",
    response_model=ImageDescription,
    autodetect_images=True,  # Set this to True
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this image?",
                ImageWithCacheControl.from_image_params(
                    {
                        "source": url,
                        "cache_control": {
                            "type": "ephemeral",
                        },
                    }
                ),
            ],
        }
    ],
    max_tokens=1000,
)

print(response)
# > description='A bush with numerous clusters of blueberries surrounded by green leaves, under a cloudy sky.' items=['blueberries', 'green leaves', 'cloudy sky']

print(completion.usage.cache_creation_input_tokens)
# > 1820
```

By leveraging Instructor's multimodal capabilities, you can focus on building your application logic without worrying about the intricacies of each provider's image handling format. This not only saves development time but also makes your code more maintainable and adaptable to future changes in AI provider APIs.

## `Audio`

> Note : Only OpenAI and Gemini support audio files at the moment. For Gemini, we're passing in the raw bytes as bytes for this feature. If you'd like to use the `Files` API instead, we also support it, [read more at](../integrations/genai.md) to see how to do so.

Similar to the Image class, we provide methods to create `Audio` instances.

| Method        | OpenAI | Google GenAI |
| ------------- | ------ | ------------ |
| `from_url()`  | ✅     | ✅           |
| `from_path()` | ✅     | ✅           |

The `Audio` class represents an audio file that can be loaded from a URL or file path. It provides methods to create `Audio` instances using the `from_path` and `from_url` methods.

The `Audio` class will automatically convert it to a the right format and include it in the API request.

```python
from openai import OpenAI
from pydantic import BaseModel
import instructor
from instructor.multimodal import Audio
import base64

# Initialize the client
client = instructor.from_openai(OpenAI())


# Define our response model
class AudioDescription(BaseModel):
    summary: str
    transcript: str


url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/gettysburg.wav"

# Make the API call with the audio file
resp = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    response_model=AudioDescription,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Extract the following information from the audio:",
                Audio.from_url(url),
            ],
        },
    ],
)

print(resp)
```

## `PDF`

The `PDF` class represents a PDF file that can be loaded from a URL or file path.

It provides methods to create `PDF` instances and is currently supported for OpenAI, Mistral, GenAI and Anthropic client integrations.

| Method          | OpenAI | Anthropic | Google GenAI | Mistral |
| --------------- | ------ | --------- | ------------ | ------- |
| `from_url()`    | ✅     | ✅        | ✅           | ✅      |
| `from_path()`   | ✅     | ✅        | ✅           | ❎      |
| `from_base64()` | ✅     | ✅        | ✅           | ❎      |
| `autodetect()`  | ✅     | ✅        | ✅           | ✅      |

For Gemini, we also provide two additional methods that make working with the google-genai files package easy which you can access in the `PDFWithGenaiFile` object.

For Anthropic, you can enable caching with the `PDFWithCacheControl` object. Note that this has caching configured by default for easy usage.

We provide examples of how to use all three object classes below.

### Usage

```python
 from openai import OpenAI
 import instructor
 from pydantic import BaseModel
 from instructor.multimodal import PDF

 # Set up the client
 url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
 client = instructor.from_openai(OpenAI())


 # Create a model for analyzing PDFs
 class Invoice(BaseModel):
     total: float
     items: list[str]


 # Load and analyze a PDF
 response = client.chat.completions.create(
     model="gpt-4o-mini",
     response_model=Invoice,
     messages=[
         {
             "role": "user",
             "content": [
                 "Analyze this document",
                 PDF.from_url(url),
             ],
         }
     ],
 )

 print(response)
 # > Total = 220, items = ['English Tea', 'Tofu']
```

### Caching

If you'd like to cache the PDF for Anthropic, we provide the `PDFWithCacheControl` class which has caching configured by default.

```python
from anthropic import Anthropic
import instructor
from pydantic import BaseModel
from instructor.multimodal import PDFWithCacheControl

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_anthropic(Anthropic())


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
response, completion = client.chat.completions.create_with_completion(
    model="claude-3-5-sonnet-20240620",
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDFWithCacheControl.from_url(url),
            ],
        }
    ],
    max_tokens=1000,
)

print(response)
# > Total = 220, items = ['English Tea', 'Tofu']

print(completion.usage.cache_creation_input_tokens)
# > 2091
```

### Using Files

We also provide a convinient wrapper around the Files API - allowing you to use both uploaded files and to block the main thread while your file is uploading.

In this example below, we download the sample PDF and then upload it using the `Files` api provided by the `google.genai` sdk.

```python
from google.genai import Client
import instructor
from pydantic import BaseModel
from instructor.multimodal import PDFWithGenaiFile
import requests

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_genai(Client())

with requests.get(url) as response:
    pdf_data = response.content
    with open("./invoice.pdf", "wb") as f:
        f.write(pdf_data)


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
response = client.chat.completions.create(
    model="gemini-2.0-flash",
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDFWithGenaiFile.from_new_genai_file(
                    file_path="./invoice.pdf",
                    retry_delay=10,
                    max_retries=20,
                ),
            ],
        }
    ],
)

print(response)
# > Total = 220, items = ['English Tea', 'Tofu']
```

If you've already uploaded your file ahead of time, we also support it. Just provide us with the file name as seen below

```python
from google.genai import Client
import instructor
from pydantic import BaseModel
from instructor.multimodal import PDFWithGenaiFile
import requests

# Set up the client
url = "https://raw.githubusercontent.com/instructor-ai/instructor/main/tests/assets/invoice.pdf"
client = instructor.from_genai(Client())

with requests.get(url) as response:
    pdf_data = response.content
    with open("./invoice.pdf", "wb") as f:
        f.write(pdf_data)

file = client.files.upload(
    file="invoice.pdf",
)


# Create a model for analyzing PDFs
class Invoice(BaseModel):
    total: float
    items: list[str]


# Load and analyze a PDF
response = client.chat.completions.create(
    model="gemini-2.0-flash",
    response_model=Invoice,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                PDFWithGenaiFile.from_existing_genai_file(file_name=file.name),
            ],
        }
    ],
)

print(response)
# > Total = 220, items = ['English Tea', 'Tofu']
```

This way you have more granular control over how the file is uploaded, potentially also processing multiple file uploads at once too.
