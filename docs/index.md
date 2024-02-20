# Instructor

_Structured outputs powered by llms. Designed for simplicity, transparency, and control._

---

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/CV8sPM5k5Y)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)

Instructor stands out for its simplicity, transparency, and user-centric design. We leverage Pydantic to do the heavy lifting, and we've built a simple, easy-to-use API on top of it by helping you manage [validation context](./concepts/reask_validation.md), retries with [Tenacity](./concepts/retrying.md), and streaming [Lists](./concepts/lists.md) and [Partial](./concepts/partial.md) responses.

Check us out in [Typescript](https://instructor-ai.github.io/instructor-js/) and [Elixir](https://github.com/thmsmlr/instructor_ex/).

!!! tip "Not limited to the OpenAI API"

    Instructor is not limited to the OpenAI API, we have support for many other backends that via patching. Check out more on [patching](./concepts/patching.md).

    1. Wrap OpenAI's SDK
    2. Wrap the create method

    Including but not limited to:

    - [Together](./blog/posts/together.md)
    - [Ollama](./blog/posts/ollama.md)
    - [AnyScale](./blog/posts/anyscale.md)
    - [llama-cpp-python](./blog/posts/llama-cpp-python.md)

## Usage

```py
import instructor
from openai import OpenAI
from pydantic import BaseModel

# This enables response_model keyword
# from client.chat.completions.create
client = instructor.patch(OpenAI())


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25
print(user.model_dump_json(indent=2))
"""
{
  "name": "Jason",
  "age": 25
}
"""
```

**Using async clients**

For async clients you must use `apatch` vs `patch` like so:

```py
import asyncio
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel

aclient = instructor.apatch(AsyncOpenAI())


class UserExtract(BaseModel):
    name: str
    age: int


task = aclient.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserExtract,
    messages=[
        {"role": "user", "content": "Extract jason is 25 years old"},
    ],
)

response = asyncio.run(task)
print(response.model_dump_json(indent=2))
"""
{
  "name": "Jason",
  "age": 25
}
"""
```

!!! note "Accessing the original response and usage tokens"

    If you want to access anything like usage or other metadata, the original response is available on the `Model._raw_response` attribute.

    ```python
    import openai
    import instructor
    from pydantic import BaseModel

    client = instructor.patch(openai.OpenAI())


    class UserDetail(BaseModel):
        name: str
        age: int


    user = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserDetail,
        messages=[
            {"role": "user", "content": "Extract Jason is 25 years old"},
        ],
    )

    print(user._raw_response.model_dump_json(indent=2))
    """
    {
      "id": "chatcmpl-8u9e2TV3ehCgLsRxNLLeAbzpEmBuZ",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "logprobs": null,
          "message": {
            "content": null,
            "role": "assistant",
            "function_call": null,
            "tool_calls": [
              {
                "id": "call_3ZuQhfteTLEy7CUokjwnLBHr",
                "function": {
                  "arguments": "{\"name\":\"Jason\",\"age\":25}",
                  "name": "UserDetail"
                },
                "type": "function"
              }
            ]
          }
        }
      ],
      "created": 1708394134,
      "model": "gpt-3.5-turbo-0125",
      "object": "chat.completion",
      "system_fingerprint": "fp_69829325d0",
      "usage": {
        "completion_tokens": 9,
        "prompt_tokens": 81,
        "total_tokens": 90
      }
    }
    """
    ```

## Why use Instructor?

The question of using Instructor is fundamentally a question of why to use Pydantic.

1. **Powered by type hints** — Instructor is powered by Pydantic, which is powered by type hints. Schema validation, prompting is controlled by type annotations; less to learn, less code to write, and integrates with your IDE.

2. **Powered by OpenAI** — Instructor is powered by OpenAI's function calling API. This means you can use the same API for both prompting and extraction.

3. **Customizable** — Pydantic is highly customizable. You can define your own validators, custom error messages, and more.

4. **Ecosystem** Pydantic is the most widely used data validation library for Python. It's used by FastAPI, Typer, and many other popular libraries.

5. **Battle Tested** — Pydantic is downloaded over 100M times per month, and supported by a large community of contributors.

6. **Easy Integration with CLI** - We offer a variety of CLI tools like `instructor jobs`, `instructor files` and `instructor usage` to track your OpenAI usage, fine-tuning jobs and more, just check out our [CLI Documentation](cli/index.md) to find out more.

## More Examples

If you'd like to see more check out our [cookbook](examples/index.md).

[Installing Instructor](installation.md) is a breeze. Just run `pip install instructor`.

## Contributing

If you want to help out, checkout some of the issues marked as `good-first-issue` or `help-wanted`. Found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cook book.

## License

This project is licensed under the terms of the MIT License.
