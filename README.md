# Welcome to Instructor - Your Gateway to Structured Outputs with OpenAI

_Structured extraction in Python, powered by OpenAI's function calling api, designed for simplicity, transparency, and control._

---

[Star us on Github!](www.github.com/jxnl/instructor).

[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)
[![GitHub stars](https://img.shields.io/github/stars/jxnl/instructor.svg)](https://github.com/jxnl/instructor/stargazers)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://jxnl.github.io/instructor)
[![GitHub issues](https://img.shields.io/github/issues/jxnl/instructor.svg)](https://github.com/jxnl/instructor/issues)
[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)

Dive into the world of Python-based structured extraction, empowered by OpenAI's cutting-edge function calling API. Instructor stands out for its simplicity, transparency, and user-centric design. Whether you're a seasoned developer or just starting out, you'll find Instructor's approach intuitive and its results insightful.

## Get Started in Moments

Installing Instructor is a breeze. Just run `pip install instructor` in your terminal and you're on your way to a smoother data handling experience.

## How Instructor Enhances Your Workflow

Our `instructor.patch` for the `OpenAI` class introduces three key enhancements:

- **Response Mode:** Specify a Pydantic model to streamline data extraction.
- **Max Retries:** Set your desired number of retry attempts for requests.
- **Validation Context:** Provide a context object for enhanced validator access.
  A Glimpse into Instructor's Capabilities

!!! note "Using Validators"

    Learn more about validators checkout our blog post [Good llm validation is just good validation](https://jxnl.github.io/instructor/blog/2023/10/23/good-llm-validation-is-just-good-validation/)

With Instructor, your code becomes more efficient and readable. Here’s a quick peek:

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

**"Using `openai<1.0.0`"**

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

**"Using async clients"**

For async clients you must use apatch vs patch like so:

```py
import instructor
from openai import AsyncOpenAI

aclient = instructor.apatch(AsyncOpenAI())

class UserExtract(BaseModel):
    name: str
    age: int

model = await aclient.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

assert isinstance(model, UserExtract)
```

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
import instructor

from openai import OpenAI
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

## Contributing

If you want to help out checkout some of the issues marked as `good-first-issue` or `help-wanted`. Found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cook book.

## License

This project is licensed under the terms of the MIT License.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/jxnl/instructor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jxnl/instructor" />
</a>
