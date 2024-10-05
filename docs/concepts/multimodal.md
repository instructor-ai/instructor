# Multimodal

Instructor supports multimodal interactions by providing helper classes that are automatically converted to the correct format for different providers, allowing you to work with both text and images in your prompts and responses. This functionality is implemented in the `multimodal.py` module and provides a seamless way to handle images alongside text for various AI models.

## `Image`

The core of multimodal support in Instructor is the `Image` class. This class represents an image that can be loaded from a URL or file path. It provides methods to create `Image` instances and convert them to formats compatible with different AI providers.

It's important to note that Anthropic and OpenAI have different formats for handling images in their API requests. The `Image` class in Instructor abstracts away these differences, allowing you to work with a unified interface.

### Anthropic Format

Anthropic uses a specific format where images are represented as base64-encoded strings with metadata:

"""python
anthropic_format = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "<base64_encoded_image_data>"
    }
}
"""

### OpenAI Format

OpenAI, on the other hand, uses a different format where images are represented as URL strings or base64-encoded data:

"""python
openai_format = {
    "type": "image_url",
    "image_url": {
        "url": "data:image/jpeg;base64,<base64_encoded_image_data>"
    }
}
"""

One of the key advantages of using Instructor's `Image` class is that it allows for seamless model switching without changing your code. This is particularly useful when you want to experiment with different AI providers or models.

## Example

Here's an example demonstrating how you can use the same code structure for both Anthropic and OpenAI, allowing for easy model switching:

"""python
import instructor
from pydantic import BaseModel

class ImageAnalyzer(BaseModel):
    caption: str
    objects: list[str]

def analyze_image_from_path(client, model: str, image_path: str, prompt: str) -> ImageAnalyzer:
    return client.chat.completions.create(
            model=model,
            response_model=ImageAnalyzer,
            messages=[
                {"role": "user", "content": [
                    "What is in this image?",
                    instructor.Image.from_path(image_path)
                ]}
            ]
        )


def analyze_image_from_url(client, model: str, image_url: str, prompt: str) -> ImageAnalyzer:
    return client.chat.completions.create(
            model=model,
            response_model=ImageAnalyzer,
            messages=[
                {"role": "user", "content": [
                    "What is in this image?",
                    instructor.Image.from_url(image_url)
                ]}
            ]
        )
"""

As you can see, we handle the cases of reading from paths and URLs and converting to base64 when appropriate for any given model. This abstraction allows for a consistent interface regardless of the underlying AI provider, making it easier to switch between different models or providers without significant code changes.

The `Image` class takes care of the necessary conversions and formatting, ensuring that your code remains clean and provider-agnostic. This flexibility is particularly valuable when you're experimenting with different models or when you need to switch providers based on specific project requirements.

By leveraging Instructor's multimodal capabilities, you can focus on building your application logic without worrying about the intricacies of each provider's image handling format. This not only saves development time but also makes your code more maintainable and adaptable to future changes in AI provider APIs.