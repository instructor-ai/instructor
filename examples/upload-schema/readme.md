# Instructor CLI Tool: Schema Catalog Management

## Overview

The Instructor CLI tool is designed for managing and uploading schema catalogs to OpenAI's servers. Leveraging the power of Pydantic models, this CLI tool allows for dynamic schema generation and upload.

## Features

- **Dynamic Model Import**: Specify Pydantic models at runtime for schema creation.
- **Directory-Agnostic**: Run the CLI tool from any directory containing your Pydantic model file.
- **User-Friendly**: Intuitive command-line interface for quick and easy schema management.

## Installation

If you haven't installed the CLI tool yet, you can install it using pip:

```bash
pip install instructor
```

## Usage

### Uploading a Schema Catalog

To upload a schema catalog, navigate to the directory that contains your Pydantic model (e.g., `model.py`). Then execute the following command:

```bash
instructor schema-catalog upload model:Article
```

On successful execution, you should see an output similar to:

```plaintext
Creating Schema Catalog for model:Article using Instructor API Key None
{
  "$defs": {
    "Topic": {
      "enum": [
        "science",
        "technology",
        "art"
      ],
      "title": "Topic",
      "type": "string"
    }
  },
  ... (rest of the schema)
}
```

### Example `model.py`

Below is an example `model.py` that defines an `Article` model with an enumerated `Topic`:

```python
from pydantic import BaseModel
from enum import Enum

class Topic(Enum):
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ART = "art"

class Article(BaseModel):
    title: str
    content: str
    topic: Topic
```

## Troubleshooting

If you encounter the error "No module named 'model'", ensure you are running the command in the directory containing your `model.py`.
