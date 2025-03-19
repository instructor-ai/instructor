---
title: Seamless Multimodal Interactions with Instructor
description: Learn how the Image and Audio class in Instructor enables seamless handling of images, audio and text across different AI models.
---

# Multimodal

Instructor supports multimodal interactions by providing helper classes that are automatically converted to the correct format for different providers, allowing you to work with both text and images in your prompts and responses. This functionality is implemented in the `multimodal.py` module and provides a seamless way to handle images alongside text for various AI models.

## `Image`

The core of multimodal support in Instructor is the `Image` class. This class represents an image that can be loaded from a URL or file path. It provides methods to create `Image` instances and convert them to formats compatible with different AI providers.

It's important to note that Anthropic and OpenAI have different formats for handling images in their API requests. The `Image` class in Instructor abstracts away these differences, allowing you to work with a unified interface.

### Usage

You can create an `Image` instance from a URL or file path using the `from_url` or `from_path` methods. The `Image` class will automatically convert the image to a base64-encoded string and include it in the API request.

```python
import instructor
import openai
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
import os


class ImageAnalyzer(BaseModel):
    """A model for analyzing image content."""

    description: str = Field(
        description="A detailed description of what's in the image"
    )
    objects: List[str] = Field(description="List of objects identified in the image")
    colors: List[str] = Field(description="Dominant colors in the image")
    text: Optional[str] = Field(
        description="Any text visible in the image", default=None
    )


# Download a sample image for demonstration
url = "https://static01.nyt.com/images/2017/04/14/dining/14COOKING-RITZ-MUFFINS/14COOKING-RITZ-MUFFINS-jumbo.jpg"
response = requests.get(url)

# Save the image locally
with open("muffin.jpg", "wb") as file:
    file.write(response.content)

# Create Image objects from URL and file path
image1 = instructor.Image.from_url(url)
image2 = instructor.Image.from_path("muffin.jpg")

client = instructor.from_openai(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ImageAnalyzer,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this two images?",
                image1,
                image2,
            ],
        }
    ],
)

print(response.model_dump_json())
"""
{"description":"A tray filled with several blueberry muffins, with one muffin prominently in the foreground. The muffins have a golden-brown top and are surrounded by a beige paper liners. Some muffins are partially visible, and fresh blueberries are scattered around the tray.", "objects": ["muffins", "blueberries", "tray", "paper liners"], "colors": ["golden-brown", "blue", "beige"], "text": null}
"""

With autodetect_images=True, you can directly provide URLs or file paths
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ImageAnalyzer,
    messages=[
        {
            "role": "user",
            "content": [
                "What is in this two images?",
                "https://static01.nyt.com/images/2017/04/14/dining/14COOKING-RITZ-MUFFINS/14COOKING-RITZ-MUFFINS-jumbo.jpg",
                "muffin.jpg",  # Using the file we downloaded in the previous example
            ],
        }
    ],
    autodetect_images=True,
)

print(response.model_dump_json())
"""
{"description":"A tray of freshly baked blueberry muffins with golden-brown tops in paper liners.", "objects":["muffins","blueberries","tray","paper liners"], "colors":["golden-brown","blue","beige"], "text":null}
"""

By leveraging Instructor's multimodal capabilities, you can focus on building your application logic without worrying about the intricacies of each provider's image handling format. This not only saves development time but also makes your code more maintainable and adaptable to future changes in AI provider APIs.

### Anthropic Prompt Caching
Instructor supports Anthropic prompt caching with images. To activate prompt caching, you can pass image content as a dictionary of the form
```python
{"type": "image", "source": <path_or_url_or_base64_encoding>, "cache_control": True}
```
and set `autodetect_images=True`, or flag it within a constructor such as `instructor.Image.from_path("path/to/image.jpg", cache_control=True)`. For example:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
import instructor
from anthropic import Anthropic
from instructor import Image
import requests
import os


class ImageAnalyzer(BaseModel):
    """A model for analyzing image content."""

    description: str = Field(
        description="A detailed description of what's in the image"
    )
    objects: List[str] = Field(description="List of objects identified in the image")
    colors: List[str] = Field(description="Dominant colors in the image")
    text: Optional[str] = Field(
        description="Any text visible in the image", default=None
    )


client = instructor.from_anthropic(Anthropic(), enable_prompt_caching=True)

# Download sample images for demonstration
url1 = "https://static01.nyt.com/images/2017/04/14/dining/14COOKING-RITZ-MUFFINS/14COOKING-RITZ-MUFFINS-jumbo.jpg"
response = requests.get(url1)

# Save the image locally
with open("image1.jpg", "wb") as file:
    file.write(response.content)

# Create a copy for the second image for demonstration purposes
with open("image2.jpg", "wb") as file:
    file.write(response.content)

# Create an image from a file path with caching enabled
image1 = Image.from_path("image1.jpg", cache_control=True)
image2 = Image.from_path("image2.jpg", cache_control=True)

response = client.messages.create(
    model="claude-3-opus-20240229",
    response_model=ImageAnalyzer,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in these images?"},
                image1,
                image2,
            ],
        }
    ],
)

print(response.model_dump_json())
"""
{"description":"A tray of freshly baked blueberry muffins with golden-brown tops in paper liners.", "objects":["muffins","blueberries","tray","paper liners"], "colors":["golden-brown","blue","beige"], "text":null}
"""

## `Audio`

The `Audio` class represents an audio file that can be loaded from a URL or file path. It provides methods to create `Audio` instances but currently only OpenAI supports it. You can create an instance using the `from_path` and `from_url` methods. The `Audio` class will automatically convert it to a base64-encoded image and include it in the API request.

### Usage

```python
from openai import OpenAI
from pydantic import BaseModel
import instructor
from instructor.multimodal import Audio
import base64

# Initialize the client
client = instructor.from_openai(OpenAI())


# Define our response model
class User(BaseModel):
    name: str
    age: int


# For testing, you'd need an actual audio file
# Option 1: Create a sample audio file with text-to-speech
# import gtts
# tts = gtts.gTTS("My name is Jason and I am 20 years old")
# tts.save("output.wav")

# Option 2: Use a mock audio object for documentation/testing purposes
# This creates a small placeholder audio to avoid the file not found error
sample_audio_base64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
sample_audio_path = "./sample_audio.wav"
with open(sample_audio_path, "wb") as f:
    f.write(base64.b64decode(sample_audio_base64))

# Make the API call with the audio file
resp = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    response_model=User,
    modalities=["text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                "Extract the following information from the audio:",
                Audio.from_path(sample_audio_path),  # Use our sample audio
            ],
        },
    ],
)

print(resp)
#> name='Jason' age=20
```
