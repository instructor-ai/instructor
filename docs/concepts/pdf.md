---
title: PDF Support
description: Learn how to work with PDF documents in Instructor
---

# PDF Support

> **Note** : If you don't have a PDF on hand, you can download a sample PDF [here](https://github.com/instructor-ai/instructor/blob/main/tests/assets/invoice.pdf) from our github repo.

Instructor provides a unified way to work with PDF documents across different AI providers through the `PDF` class. This allows users to load in PDFs from base64 strings, urls and even local files.

## Basic Usage

Here's a simple example of how to use PDFs with instructor where we'll extract out the line items and the total cost from a reciept invoice.

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

## Methods Provided

The `PDF` class provides three main methods for loading PDFs, here's a breakdown of the support across different providers.

| Method      | OpenAI | Anthropic | Mistral | Google.GenAI |
| ----------- | ------ | --------- | ------- | ------------ |
| from_url    | ✅     | ✅        | ✅      | ✅           |
| from_path   | ✅     | ✅        | ❌      | ✅           |
| from_base64 | ✅     | ✅        | ❌      | ✅           |

```python
from instructor import PDF

# Load from a URL
pdf = PDF.from_url("https://example.com/document.pdf")

# Load from a local file
pdf = PDF.from_path("path/to/document.pdf")

# Load from base64 data
pdf = PDF.from_base64("data:application/pdf;base64,...")

# Automatically detect the source type
pdf = PDF.autodetect("https://example.com/document.pdf")
```
