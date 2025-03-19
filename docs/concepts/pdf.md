---
title: PDF Support
description: Learn how to work with PDF documents in Instructor
---

# PDF Support

Instructor makes it easy to work with PDF documents in your AI applications. The `PDF` class handles loading PDFs from different sources and works with multiple AI providers.

!!! note "Provider Documentation"
    For more details on how different providers handle PDFs, see:
    
    - [OpenAI PDF Files Guide](https://platform.openai.com/docs/guides/pdf-files?api-mode)
    - [Anthropic Claude PDF Support](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support)

## Basic Usage

Here's how to use PDFs with OpenAI:

```python
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Set up the client
client = instructor.patch(OpenAI())

# Create a model for analyzing PDFs
class DocumentAnalysis(BaseModel):
    summary: str
    key_points: list[str]

# Load and analyze a PDF
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    response_model=DocumentAnalysis,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                instructor.PDF.from_url("https://example.com/document.pdf")
            ]
        }
    ]
)
```

## Loading PDFs

You can load PDFs in several ways:

```python
from instructor import PDF

# From a URL
pdf = PDF.from_url("https://example.com/document.pdf")

# From a local file
pdf = PDF.from_path("path/to/document.pdf")

# From base64 data
pdf = PDF.from_base64("data:application/pdf;base64,...")

# From Google Cloud Storage
pdf = PDF.from_gs_url("gs://bucket/document.pdf")

# Automatically detect the source type
pdf = PDF.autodetect("https://example.com/document.pdf")  # URL
pdf = PDF.autodetect("path/to/document.pdf")  # Local file
pdf = PDF.autodetect("data:application/pdf;base64,...")  # Base64
```

## Using with Anthropic

Here's how to use PDFs with Anthropic's Claude:

```python
from anthropic import Anthropic
import instructor
from pydantic import BaseModel

# Set up the client
client = instructor.patch(Anthropic())

class DocumentAnalysis(BaseModel):
    summary: str
    key_points: list[str]

# Load and analyze a PDF
response = client.messages.create(
    model="claude-3-opus-20240229",
    response_model=DocumentAnalysis,
    messages=[
        {
            "role": "user",
            "content": [
                "Analyze this document",
                instructor.PDF.from_url("https://example.com/document.pdf")
            ]
        }
    ]
)
```

## Prompt Caching with Anthropic

When working with Anthropic's Claude, you can enable prompt caching to improve performance:

```python
pdf = {
    "type": "document",
    "source": "https://example.com/document.pdf",
    "cache_control": {"type": "ephemeral"}
}
```

## Error Handling

The PDF class includes built-in error handling for common issues:

```python
from instructor import PDF

try:
    # Try to load a PDF
    pdf = PDF.from_url("https://example.com/document.pdf")
except ValueError as e:
    print("Invalid PDF format:", e)
except FileNotFoundError as e:
    print("PDF file not found:", e)
except Exception as e:
    print("Error loading PDF:", e)
```

## Supported MIME Types

The PDF class supports the following MIME type:
- `application/pdf`

## Provider-Specific Details

### OpenAI
- Supports both URL and base64-encoded PDFs
- Works with GPT-4 Vision models
- Best for analyzing document content and structure

### Anthropic
- Supports both URL and base64-encoded PDFs
- Works with Claude 3 models
- Includes prompt caching for better performance
- Excellent for document understanding and analysis

## Best Practices

1. File Size: Keep PDF files reasonably sized to avoid timeouts
2. Enable Caching: Use prompt caching with Anthropic for better performance
3. Error Handling: Always implement error handling for file operations
4. Format Verification: Check that your PDFs are in the correct format
5. URL Access: Ensure URLs are publicly accessible when using them 