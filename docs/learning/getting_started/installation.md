# Installing Instructor

Instructor is a Python library that works with various LLM providers to extract structured outputs. This guide covers installation and setting up API keys for different providers.

## Basic Installation

Install the core Instructor package with pip:

```shell
pip install instructor
```

Instructor requires Pydantic for defining data models:

```shell
pip install pydantic
```

## Setting Up with Different LLM Providers

### OpenAI

OpenAI is the default provider and works out of the box:

```shell
pip install instructor
```

Set up your OpenAI API key:

```shell
export OPENAI_API_KEY=your_openai_key
```

### Anthropic (Claude)

To use with Anthropic's Claude models:

```shell
pip install "instructor[anthropic]"
```

Set up your Anthropic API key:

```shell
export ANTHROPIC_API_KEY=your_anthropic_key
```

### Google Gemini

To use with Google's Gemini models:

```shell
pip install "instructor[google-generativeai]"
```

Set up your Google API key:

```shell
export GOOGLE_API_KEY=your_google_key
```

### Cohere

To use with Cohere's models:

```shell
pip install "instructor[cohere]"
```

Set up your Cohere API key:

```shell
export COHERE_API_KEY=your_cohere_key
```

### Mistral

To use with Mistral AI's models:

```shell
pip install "instructor[mistralai]"
```

Set up your Mistral API key:

```shell
export MISTRAL_API_KEY=your_mistral_key
```

### LiteLLM (Multiple Providers)

To use LiteLLM for accessing multiple providers:

```shell
pip install "instructor[litellm]"
```

Set up API keys for the providers you want to use.

## Verifying Your Installation

You can verify your installation by running a simple extraction:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Person,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old"}
    ]
)

print(f"Name: {person.name}, Age: {person.age}")
```

## Next Steps

Now that you've installed Instructor, you can:

1. Create your first extraction with a simple model
2. Understand the different response models available
3. Set up clients for your preferred LLM provider

Check the [Client Setup](client_setup.md) guide to learn how to configure clients for different providers. 