# 🎓 Instructor: Your Friendly Guide to Structured LLM Outputs

Instructor is a Python library that makes it a breeze to work with structured outputs from large language models (LLMs). Built on top of Pydantic, it provides a simple, transparent, and user-friendly API to manage validation, retries, and streaming responses. Get ready to supercharge your LLM workflows!

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/CV8sPM5k5Y)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)


## 🌟 Key Features

- 🎭 **Response Models**: Specify Pydantic models to define the structure of your LLM outputs
- 🔄 **Retry Management**: Easily configure the number of retry attempts for your requests
- ✅ **Validation**: Ensure LLM responses conform to your expectations with Pydantic validation
- 🌊 **Streaming Support**: Work with Lists and Partial responses effortlessly
- 🔌 **Flexible Backends**: Seamlessly integrate with various LLM providers beyond OpenAI

## 🚀 Get Started in Minutes

Install Instructor with a single command:

```bash
pip install -U instructor
```

Now, let's see Instructor in action with a simple example:

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    name: str
    age: int


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserInfo,
    messages=[{"role": "user", "content": "John Doe is 30 years old."}],
)

print(user_info.name)  # "John Doe"
#> John Doe
print(user_info.age)  # 30
#> 30
```

## 🎯 Validation Made Easy

Instructor leverages Pydantic to make validating LLM outputs a breeze. Simply define your validation rules in your Pydantic models, and Instructor will ensure the LLM responses conform to your expectations. No more manual checking or parsing!

```python
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator

import instructor
import openai

client = instructor.from_openai(openai.OpenAI())


class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(llm_validator("Don't say objectionable things", client=client)),
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
      Assertion failed, The statement promotes evil behavior, which is objectionable. [type=assertion_error, input_value='The meaning of life is to be evil and steal', input_type=str]
        For further information visit https://errors.pydantic.dev/2.6/v/assertion_error
    """
```

## 📖 Learn More

Dive deeper into Instructor's concepts and features:

- [Validation Context](./docs/concepts/reask_validation.md)
- [Retrying](./docs/concepts/retrying.md) 
- [Lists](./docs/concepts/lists.md)
- [Partial Responses](./docs/concepts/partial.md)
- [Patching](./docs/concepts/patching.md)

## 🤝 Join the Community

Have questions? Want to share your Instructor projects? Join our vibrant community on [Discord](https://discord.gg/CV8sPM5k5Y)! We're here to help you get the most out of Instructor and celebrate your successes.


## 🎉 Start Building

Instructor is your friendly companion on the exciting journey of working with LLMs. Install it now and unlock the full potential of structured outputs in your projects. Happy building! 🚀

---

We can't wait to see the amazing things you create with Instructor. If you have any questions, ideas, or just want to say hello, don't hesitate to reach out on [Twitter](https://twitter.com/jxnlco) or [Discord](https://discord.gg/CV8sPM5k5Y). Let's build the future together! 🌟

---

## Using Anthropic Models

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_anthropic(Anthropic())

# note that client.chat.completions.create will also work
resp = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Extract Jason is 25 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

## Using Litellm

```python
import instructor
from litellm import completion
from pydantic import BaseModel


class User(BaseModel):
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
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "Jason"
assert resp.age == 25
```

## [Evals](https://github.com/jxnl/instructor/tree/main/tests/openai/evals)

We invite you to contribute to evals in `pytest` as a way to monitor the quality of the OpenAI models and the `instructor` library. To get started check out the [jxnl/instructor/tests/evals](https://github.com/jxnl/instructor/tree/main/tests/openai/evals) and contribute your own evals in the form of pytest tests. These evals will be run once a week and the results will be posted.

## Contributing

If you want to help, checkout some of the issues marked as `good-first-issue` or `help-wanted` found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cookbook.

## CLI

We also provide some added CLI functionality for easy convinience:

- `instructor jobs` : This helps with the creation of fine-tuning jobs with OpenAI. Simple use `instructor jobs create-from-file --help` to get started creating your first fine-tuned GPT3.5 model

- `instructor files` : Manage your uploaded files with ease. You'll be able to create, delete and upload files all from the command line

- `instructor usage` : Instead of heading to the OpenAI site each time, you can monitor your usage from the cli and filter by date and time period. Note that usage often takes ~5-10 minutes to update from OpenAI's side

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
