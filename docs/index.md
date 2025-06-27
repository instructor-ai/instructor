---
title: "Instructor - Python Library for Structured LLM Outputs | OpenAI, Anthropic, Google"
description: "Get structured data from LLMs with Instructor - the most popular Python library for LLM validation. Supports OpenAI, Anthropic, Google, Ollama, DeepSeek, and 15+ providers. Built on Pydantic with automatic retries, streaming, and type safety."
keywords: "LLM structured outputs, OpenAI structured data, Pydantic LLM validation, Python LLM library, Anthropic structured outputs, GPT structured data extraction, LLM response validation, AI data extraction, Ollama structured outputs, open source LLM, DeepSeek validation"
---

# Instructor: Python's Top Library for Structured LLM Outputs

_Extract structured data from any LLM with type safety, validation, and automatic retries._

[![PyPI - Version](https://img.shields.io/pypi/v/instructor?style=flat-square&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/instructor/)
[![License](https://img.shields.io/github/license/instructor-ai/instructor?style=flat-square&color=blue)](https://github.com/instructor-ai/instructor/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square&logo=github&logoColor=white)](https://github.com/instructor-ai/instructor)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square&logo=pypi&logoColor=white&label=Downloads)](https://pypi.org/project/instructor/)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square&logo=discord&logoColor=white&label=Discord)](https://discord.gg/bD9YE9JArw)
[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/jxnlco)

## What is Instructor?

Instructor is the **most popular Python library** for extracting structured data from Large Language Models (LLMs). With over **3 million monthly downloads, 11k stars, and 100+ contributors**, it's the go-to solution for developers who need reliable, validated outputs from AI models.

Built on top of **Pydantic**, Instructor provides type-safe data extraction with automatic validation, retries, and streaming support. Whether you're using OpenAI's GPT models, Anthropic's Claude, Google's Gemini, **open source models with Ollama**, **DeepSeek**, or any of 15+ supported providers, Instructor ensures your LLM outputs are always structured and validated.

## Key Features for LLM Data Extraction

- **Structured Outputs**: Define Pydantic models to specify exactly what data you want from your LLM
- **Automatic Retries**: Built-in retry logic when validation fails - no more manual error handling
- **Data Validation**: Leverage Pydantic's powerful validation to ensure response quality
- **Streaming Support**: Real-time processing of partial responses and lists
- **Multi-Provider**: Works with OpenAI, Anthropic, Google, Mistral, Cohere, Ollama, DeepSeek, and 15+ LLM providers
- **Type Safety**: Full IDE support with proper type inference and autocompletion
- **Open Source Support**: Run any open source model locally with Ollama, llama-cpp-python, or vLLM

## Quick Start: Extract Structured Data in 3 Lines

Install Instructor and start extracting structured data immediately:

=== "pip"
    ```bash
    pip install instructor
    ```

=== "uv"
    ```bash
    uv add instructor
    ```

=== "poetry"
    ```bash
    poetry add instructor
    ```

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI


class Person(BaseModel):
    name: str
    age: int
    occupation: str


client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Person,
    messages=[
        {"role": "user", "content": "Extract: John is a 30-year-old software engineer"}
    ],
)
print(person)  # Person(name='John', age=30, occupation='software engineer')
```

## Universal Provider API - One Interface for All LLMs

Instructor's **`from_provider`** function provides a unified interface to work with any LLM provider. Switch between OpenAI, Anthropic, Google, Ollama, DeepSeek, and 15+ providers with the same code:

```python
import instructor
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


# Works with any provider - same interface everywhere
client = instructor.from_provider("openai/gpt-4")  # OpenAI
client = instructor.from_provider("anthropic/claude-3")  # Anthropic
client = instructor.from_provider("google/gemini-pro")  # Google
client = instructor.from_provider("ollama/llama3")  # Ollama (local)
client = instructor.from_provider("deepseek/deepseek-chat")  # DeepSeek

# Same extraction code works with all providers
user = client.chat.completions.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)
```

The **`from_provider`** API supports both sync and async usage (`async_client=True`) and automatically handles provider-specific configurations, making it effortless to switch between different LLM services.

## Supported LLM Providers

Instructor works seamlessly with **15+ popular LLM providers**, giving you the flexibility to use any model while maintaining consistent structured output handling. From OpenAI's GPT models to **open source alternatives with Ollama**, **DeepSeek models**, and local inference, get validated data extraction everywhere.

It stands out for its simplicity, transparency, and user-centric design, built on top of Pydantic. Instructor helps you manage [validation context](./concepts/reask_validation.md), retries with [Tenacity](./concepts/retrying.md), and streaming [Lists](./concepts/lists.md) and [Partial](./concepts/partial.md) responses.

[:material-star: Star the Repo](https://github.com/jxnl/instructor){: .md-button .md-button--primary } [:material-book-open-variant: Cookbooks](./examples/index.md){: .md-button } [:material-lightbulb: Prompting Guide](./prompting/index.md){: .md-button }

If you ever get stuck, you can always run `instructor docs` to open the documentation in your browser. It even supports searching for specific topics.

```bash
instructor docs [QUERY]
```

### OpenAI GPT Models - Structured Outputs

Get structured data from OpenAI's most powerful models including GPT-4, GPT-4 Turbo, and GPT-3.5.

=== "OpenAI"
    ```bash
    pip install instructor
    ```

    !!! info "Using OpenAI's Structured Output Response"

        You can now use OpenAI's structured output response with Instructor. This feature combines the strengths of Instructor with OpenAI's precise sampling.

        ```python
        client = instructor.from_openai(OpenAI(), mode=Mode.TOOLS_STRICT)
        ```

    ```python
    import instructor
    from pydantic import BaseModel
    from openai import OpenAI


    # Define your desired output structure
    class ExtractUser(BaseModel):
        name: str
        age: int


    # Patch the OpenAI client
    client = instructor.from_openai(OpenAI())

    # Extract structured data from natural language
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ExtractUser,
        messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    )

    assert res.name == "John Doe"
    assert res.age == 30
    ```

    [See more :material-arrow-right:](./integrations/openai.md){: .md-button }

=== "Ollama"

    ```bash
    pip install "instructor[ollama]"
    ```

    ```python
    from openai import OpenAI
    from pydantic import BaseModel
    import instructor


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        mode=instructor.Mode.JSON,
    )

    resp = client.chat.completions.create(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=ExtractUser,
    )
    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/ollama.md){: .md-button }

=== "llama-cpp-python"
    ```bash
    pip install "instructor[llama-cpp-python]"
    ```

    ```python
    import llama_cpp
    import instructor
    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
    from pydantic import BaseModel

    llama = llama_cpp.Llama(
        model_path="../../models/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        n_gpu_layers=-1,
        chat_format="chatml",
        n_ctx=2048,
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
        logits_all=True,
        verbose=False,
    )

    create = instructor.patch(
        create=llama.create_chat_completion_openai_v1,
        mode=instructor.Mode.JSON_SCHEMA,
    )


    class ExtractUser(BaseModel):
        name: str
        age: int


    user = create(
        messages=[
            {
                "role": "user",
                "content": "Extract `Jason is 30 years old`",
            }
        ],
        response_model=ExtractUser,
    )

    assert user.name == "Jason"
    assert user.age == 30
    ```

    [See more :material-arrow-right:](./integrations/llama-cpp-python.md){: .md-button }

=== "Anthropic"
    ```bash
    pip install "instructor[anthropic]"
    ```

    ```python
    import instructor
    from anthropic import Anthropic
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_anthropic(Anthropic())

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=ExtractUser,
    )

    assert isinstance(resp, ExtractUser)
    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/anthropic.md){: .md-button }

=== "Gemini"
    ```bash
    pip install "instructor[google-generativeai]"
    ```

    ```python
    import instructor
    import google.generativeai as genai
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",
        ),
        mode=instructor.Mode.GEMINI_JSON,
    )

    # note that client.chat.completions.create will also work
    resp = client.messages.create(
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=ExtractUser,
    )

    assert isinstance(resp, ExtractUser)
    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/google.md){: .md-button }

=== "Vertex AI"
    ```bash
    pip install "instructor[vertexai]"
    ```

    ```python
    import instructor
    import vertexai  # type: ignore
    from vertexai.generative_models import GenerativeModel  # type: ignore
    from pydantic import BaseModel

    vertexai.init()


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_vertexai(
        client=GenerativeModel("gemini-1.5-pro-preview-0409"),
        mode=instructor.Mode.VERTEXAI_TOOLS,
    )

    # note that client.chat.completions.create will also work
    resp = client.create(
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=ExtractUser,
    )

    assert isinstance(resp, ExtractUser)
    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/vertex.md){: .md-button }

=== "Groq"
    ```bash
    pip install "instructor[groq]"
    ```

    ```python
    import instructor
    from groq import Groq
    from pydantic import BaseModel

    client = instructor.from_groq(Groq())


    class ExtractUser(BaseModel):
        name: str
        age: int


    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        response_model=ExtractUser,
        messages=[{"role": "user", "content": "Extract Jason is 25 years old."}],
    )

    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/groq.md){: .md-button }

=== "Litellm"
    ```bash
    pip install "instructor[litellm]"
    ```

    ```python
    import instructor
    from litellm import completion
    from pydantic import BaseModel


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_litellm(completion)

    resp = client.chat.completions.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
        response_model=ExtractUser,
    )

    assert isinstance(resp, ExtractUser)
    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/litellm.md){: .md-button }

=== "Cohere"
    ```bash
    pip install "instructor[cohere]"
    ```

    ```python
    import instructor
    from pydantic import BaseModel
    from cohere import Client


    class ExtractUser(BaseModel):
        name: str
        age: int


    client = instructor.from_cohere(Client())

    resp = client.chat.completions.create(
        response_model=ExtractUser,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
    )

    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/cohere.md){: .md-button }

=== "Cerebras"
    ```bash
    pip install "instructor[cerebras]"
    ```

    ```python
    from cerebras.cloud.sdk import Cerebras
    import instructor
    from pydantic import BaseModel
    import os

    client = Cerebras(
        api_key=os.environ.get("CEREBRAS_API_KEY"),
    )
    client = instructor.from_cerebras(client)


    class ExtractUser(BaseModel):
        name: str
        age: int


    resp = client.chat.completions.create(
        model="llama3.1-70b",
        response_model=ExtractUser,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
    )

    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/cerebras.md){: .md-button }

=== "Fireworks"
    ```bash
    pip install "instructor[fireworks]"
    ```

    ```python
    from fireworks.client import Fireworks
    import instructor
    from pydantic import BaseModel
    import os

    client = Fireworks(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
    )
    client = instructor.from_fireworks(client)


    class ExtractUser(BaseModel):
        name: str
        age: int


    resp = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p2-1b-instruct",
        response_model=ExtractUser,
        messages=[
            {
                "role": "user",
                "content": "Extract Jason is 25 years old.",
            }
        ],
    )

    assert resp.name == "Jason"
    assert resp.age == 25
    ```

    [See more :material-arrow-right:](./integrations/fireworks.md){: .md-button }

## Citation

If you use Instructor in your research or project, please cite it using:

```bibtex
@software{liu2024instructor,
  author = {Jason Liu and Contributors},
  title = {Instructor: A library for structured outputs from large language models},
  url = {https://github.com/instructor-ai/instructor},
  year = {2024},
  month = {3}
}
```

## Why use Instructor?

<div class="grid cards" markdown>

- :material-code-tags: **Simple API with Full Prompt Control**

    Instructor provides a straightforward API that gives you complete ownership and control over your prompts. This allows for fine-tuned customization and optimization of your LLM interactions.

    [:octicons-arrow-right-16: Explore Concepts](./concepts/models.md)

- :material-translate: **Multi-Language Support**

    Simplify structured data extraction from LLMs with type hints and validation.

    [:simple-python: Python](https://python.useinstructor.com) · [:simple-typescript: TypeScript](https://js.useinstructor.com) · [:simple-ruby: Ruby](https://ruby.useinstructor.com) · [:simple-go: Go](https://go.useinstructor.com) · [:simple-elixir: Elixir](https://hex.pm/packages/instructor) · [:simple-rust: Rust](https://rust.useinstructor.com)

- :material-refresh: **Reasking and Validation**

    Automatically reask the model when validation fails, ensuring high-quality outputs. Leverage Pydantic's validation for robust error handling.

    [:octicons-arrow-right-16: Learn about Reasking](./concepts/reask_validation.md)

- :material-repeat-variant: **Streaming Support**

    Stream partial results and iterables with ease, allowing for real-time processing and improved responsiveness in your applications.

    [:octicons-arrow-right-16: Learn about Streaming](./concepts/partial.md)

- :material-code-braces: **Powered by Type Hints**

    Leverage Pydantic for schema validation, prompting control, less code, and IDE integration.

    [:octicons-arrow-right-16: Learn more](https://docs.pydantic.dev/)

- :material-lightning-bolt: **Simplified LLM Interactions**

    Support for [OpenAI](./integrations/openai.md), [Anthropic](./integrations/anthropic.md), [Google](./integrations/google.md), [Vertex AI](./integrations/vertex.md), [Mistral/Mixtral](./integrations/together.md), [Ollama](./integrations/ollama.md), [llama-cpp-python](./integrations/llama-cpp-python.md), [Cohere](./integrations/cohere.md), [LiteLLM](./integrations/litellm.md).

    [:octicons-arrow-right-16: See Hub](./integrations/index.md)

</div>


### Using Hooks

Instructor includes a hooks system that lets you manage events during the language model interaction process. Hooks allow you to intercept, log, and handle events at different stages, such as when completion arguments are provided or when a response is received. This system is based on the `Hooks` class, which handles event registration and emission. You can use hooks to add custom behavior like logging or error handling. Here's a simple example demonstrating how to use hooks:

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


# Initialize the OpenAI client with Instructor
client = instructor.from_openai(OpenAI())


# Define hook functions
def log_kwargs(**kwargs):
    print(f"Function called with kwargs: {kwargs}")


def log_exception(exception: Exception):
    print(f"An exception occurred: {str(exception)}")


client.on("completion:kwargs", log_kwargs)
client.on("completion:error", log_exception)

user_info = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserInfo,
    messages=[
        {"role": "user", "content": "Extract the user name: 'John is 20 years old'"}
    ],
)

"""
{
        'args': (),
        'kwargs': {
            'messages': [
                {
                    'role': 'user',
                    'content': "Extract the user name: 'John is 20 years old'",
                }
            ],
            'model': 'gpt-3.5-turbo',
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'UserInfo',
                        'description': 'Correctly extracted `UserInfo` with all the required parameters with correct types',
                        'parameters': {
                            'properties': {
                                'name': {'title': 'Name', 'type': 'string'},
                                'age': {'title': 'Age', 'type': 'integer'},
                            },
                            'required': ['age', 'name'],
                            'type': 'object',
                        },
                    },
                }
            ],
            'tool_choice': {'type': 'function', 'function': {'name': 'UserInfo'}},
        },
    }
"""

print(f"Name: {user_info.name}, Age: {user_info.age}")
#> Name: John, Age: 20
```

This example demonstrates:
1. A pre-execution hook that logs all kwargs passed to the function.
2. An exception hook that logs any exceptions that occur during execution.

The hooks provide valuable insights into the function's inputs and any errors,
enhancing debugging and monitoring capabilities.

[Learn more about hooks :octicons-arrow-right:](./concepts/hooks.md){: .md-button .md-button-primary }

## Correct Type Inference

This was the dream of instructor but due to the patching of openai, it wasnt possible for me to get typing to work well. Now, with the new client, we can get typing to work well! We've also added a few `create_*` methods to make it easier to create iterables and partials, and to access the original completion.

### Calling `create`

```python
import openai
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_openai(openai.OpenAI())

user = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)
```

Now if you use a IDE, you can see the type is correctly infered.

![type](./blog/posts/img/type.png)

### Handling async: `await create`

This will also work correctly with asynchronous clients.

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.AsyncOpenAI())


class User(BaseModel):
    name: str
    age: int


async def extract():
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=User,
    )
```

Notice that simply because we return the `create` method, the `extract()` function will return the correct user type.

![async](./blog/posts/img/async_type.png)

### Returning the original completion: `create_with_completion`

You can also return the original completion object

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


user, completion = client.chat.completions.create_with_completion(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)
```

![with_completion](./blog/posts/img/with_completion.png)

### Streaming Partial Objects: `create_partial`

In order to handle streams, we still support `Iterable[T]` and `Partial[T]` but to simply the type inference, we've added `create_iterable` and `create_partial` methods as well!

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


user_stream = client.chat.completions.create_partial(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create a user"},
    ],
    response_model=User,
)

for user in user_stream:
    print(user)
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=None
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name=None age=25
    #> name='John Doe' age=25
    # name=None age=None
    # name='' age=None
    # name='John' age=None
    # name='John Doe' age=None
    # name='John Doe' age=30
```

Notice now that the type infered is `Generator[User, None]`

![generator](./blog/posts/img/generator.png)

### Streaming Iterables: `create_iterable`

We get an iterable of objects when we want to extract multiple objects.

```python
import openai
import instructor
from pydantic import BaseModel


client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


users = client.chat.completions.create_iterable(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": "Create 2 users"},
    ],
    response_model=User,
)

for user in users:
    print(user)
    #> name='John Doe' age=30
    #> name='Jane Doe' age=28
    # User(name='John Doe', age=30)
    # User(name='Jane Smith', age=25)
```

![iterable](./blog/posts/img/iterable.png)

## Templating

Instructor supports templating with Jinja, which lets you create dynamic prompts. This is useful when you want to fill in parts of a prompt with data. Here's a simple example:

```python
import openai
import instructor
from pydantic import BaseModel

client = instructor.from_openai(openai.OpenAI())


class User(BaseModel):
    name: str
    age: int


# Create a completion using a Jinja template in the message content
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """Extract the information from the
            following text: {{ data }}`""",
        },
    ],
    response_model=User,
    context={"data": "John Doe is thirty years old"},
)

print(response)
#> User(name='John Doe', age=30)
```

[Learn more about templating :octicons-arrow-right:](./concepts/templating.md){: .md-button .md-button-primary }
## Validation

You can also use Pydantic to validate your outputs and get the llm to retry on failure. Check out our docs on [retrying](./concepts/retrying.md) and [validation context](./concepts/reask_validation.md).

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator

# Apply the patch to the OpenAI client
client = instructor.from_openai(OpenAI())


class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(llm_validator("don't say objectionable things", client=client)),
    ]


try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and steal",
    )
except ValidationError as e:
    print(e)
    """
    1 validation error for QuestionAnswer
    answer
      Assertion failed, The statement promotes objectionable behavior by encouraging evil and stealing. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str]
    """
```

## Contributing

If you want to help out, checkout some of the issues marked as `good-first-issue` or `help-wanted`. Found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cook book.

## License

This project is licensed under the terms of the MIT License.

## Frequently Asked Questions

### How do I get structured data from OpenAI GPT models?

Use Instructor with OpenAI to automatically extract structured data with Pydantic models. Simply define your data structure and let Instructor handle validation and retries.

### What LLM providers work with Instructor?

Instructor supports 15+ providers including OpenAI (GPT-4, ChatGPT), Anthropic (Claude), Google (Gemini), Mistral, Cohere, Groq, **Ollama for open source models**, **DeepSeek**, and many more.

### How does Instructor handle LLM validation errors?

Instructor automatically retries failed requests with detailed error messages, ensuring your structured outputs always match your Pydantic schema.

### Can I use Instructor with local LLM models?

Yes! Instructor works with local models through **Ollama**, llama-cpp-python, and other local inference frameworks. Perfect for running **open source models** like Llama, Mistral, or CodeLlama locally.

### Does Instructor support Ollama for open source models?

Absolutely! Instructor has first-class support for **Ollama**, making it easy to run open source models locally while getting the same structured output validation. Simply point Instructor to your Ollama endpoint.

### How do I use DeepSeek models with Instructor?

Instructor supports **DeepSeek** models through the OpenAI-compatible API. You can use DeepSeek's powerful reasoning capabilities while maintaining full validation and retry logic.

### Is Instructor compatible with async Python code?

Absolutely. Instructor fully supports asyncio with async clients for OpenAI, Anthropic, and other providers.

## Why Choose Instructor for LLM Structured Outputs?

- **🏆 Industry Standard**: Over 1M monthly downloads make it the most trusted Python library for LLM data extraction
- **⚡ Production Ready**: Built-in error handling, retries, and validation for reliable production deployments
- **🔧 Developer Friendly**: Full IDE support with type hints, autocompletion, and comprehensive documentation
- **🌐 Provider Agnostic**: Switch between LLM providers without changing your code structure
- **📈 Scalable**: From simple scripts to enterprise applications, Instructor scales with your needs

## Related Topics

- [LLM Response Validation](./concepts/reask_validation.md) - Ensure quality outputs
- [Streaming LLM Outputs](./concepts/partial.md) - Real-time data processing
- [Multi-Provider Setup](./integrations/index.md) - Use any LLM service
- [Production Deployment](./concepts/retrying.md) - Error handling and retries
- [Type Safety with Pydantic](./concepts/models.md) - Schema validation
