# Getting Started with Instructor

_Structured extraction in Python, powered by OpenAI's function calling api, designed for simplicity, transparency, and control._

---

[Star us on Github!](https://jxnl.github.io/instructor).

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-yellow)](https://www.buymeacoffee.com/jxnlco)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)
[![GitHub stars](https://img.shields.io/github/stars/jxnl/instructor.svg)](https://github.com/jxnl/instructor/stargazers)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://jxnl.github.io/instructor)
[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![GitHub issues](https://img.shields.io/github/issues/jxnl/instructor.svg)](https://github.com/jxnl/instructor/issues)
[![GitHub license](https://img.shields.io/github/license/jxnl/instructor.svg)](https://github.com/jxnl/instructor/blob/main/LICENSE)
[![Github discussions](https://img.shields.io/github/discussions/jxnl/instructor)](https:github.com/jxnl/instructor/discussions)
[![PyPI version](https://img.shields.io/pypi/v/instructor.svg)](https://pypi.python.org/pypi/instructor)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/instructor.svg)](https://pypi.python.org/pypi/instructor)

Built to interact solely with openai's function calling api from python. It's designed to be intuitive, easy to use, and provide great visibility into your prompts.

## Usage

```py hl_lines="5 13"
from openai import OpenAI
import instructor

# Enables `response_model`
client = instructor.patch(OpenAI())

class UserDetail(BaseModel):
    name: str
    age: int

user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25
```

!!! note "Using `openai<1.0.0`"

    If you're using `openai<1.0.0` then make sure you `pip install instructor<0.3.0`
    where you can patch a global client like so:

    ```python hl_lines="4 8"
    import openai
    import instructor

    instructor.patch()

    user = openai.ChatCompletion.create(
        ...,
        response_model=UserDetail,
    )
    ```

## Installation

To get started you need to install it using `pip`. Run the following command in your terminal:

```sh
$ pip install instructor
```

## Quick Start

To simplify your work with OpenAI we offer a patching mechanism for the `ChatCompletion` class.
The patch introduces 3 features to the `ChatCompletion` class:

1. The `response_model` parameter, which allows you to specify a Pydantic model to extract data into.
2. The `max_retries` parameter, which allows you to specify the number of times to retry the request if it fails.
3. The `validation_context` parameter, which allows you to specify a context object that validators have access to.

!!! note "Using Validators"

    Learn more about validators checkout our blog post [Good llm validation is just good validation](https://jxnl.github.io/instructor/blog/2023/10/23/good-llm-validation-is-just-good-validation/)

### Step 1: Patch the client

First, import the required libraries and apply the patch function to the OpenAI module. This exposes new functionality with the response_model parameter.

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

# This enables response_model keyword
# from client.chat.completions.create
client = instructor.patch(OpenAI())
```

### Step 2: Define the Pydantic Model

Create a Pydantic model to define the structure of the data you want to extract. This model will map directly to the information in the prompt.

```python
from pydantic import BaseModel

class UserDetail(BaseModel):
    name: str
    age: int
```

### Step 3: Extract

Use the `client.chat.completions.create` method to send a prompt and extract the data into the Pydantic object. The response_model parameter specifies the Pydantic model to use for extraction. Its helpful to annotate the variable with the type of the response model.
which will help your IDE provide autocomplete and spell check.

```python
user: UserDetail = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)

assert user.name == "Jason"
assert user.age == 25
```

## Pydantic Validation

Validation can also be plugged into the same Pydantic model. Here, if the answer attribute contains content that violates the rule "don't say objectionable things," Pydantic will raise a validation error.

```python hl_lines="9 15"
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator

class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(llm_validator("don't say objectionable things"))
    ]

try:
    qa = QuestionAnswer(
        question="What is the meaning of life?",
        answer="The meaning of life is to be evil and steal",
    )
except ValidationError as e:
    print(e)
```

Its important to not here that the error message is generated by the LLM, not the code, so it'll be helpful for re asking the model.

```plaintext
1 validation error for QuestionAnswer
answer
   Assertion failed, The statement is objectionable. (type=assertion_error)
```

## Reask on validation error

Here, the `UserDetails` model is passed as the `response_model`, and `max_retries` is set to 2.

```python
from openai import OpenAI
import instructor

from pydantic import BaseModel, field_validator

# Apply the patch to the OpenAI client
client = instructor.patch(OpenAI())

class UserDetails(BaseModel):
    name: str
    age: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v.upper() != v:
            raise ValueError("Name must be in uppercase.")
        return v

model = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetails,
    max_retries=2,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert model.name == "JASON"
```

## License

This project is licensed under the terms of the MIT License.
