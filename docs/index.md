# Instructor

_Structured outputs powered by llms. Designed for simplicity, transparency, and control._

---

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/CV8sPM5k5Y)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)

Dive into the world of Python-based structured extraction, by OpenAI's function calling API and Pydantic, the most widely used data validation library for Python. Instructor stands out for its simplicity, transparency, and user-centric design. Whether you're a seasoned developer or just starting out, you'll find Instructor's approach intuitive and steerable.

!!! tip "Support in other languages"

    Check out ports to other languages below:

    - [Typescript / Javascript](https://instructor-ai.github.io/instructor-js/)
    - [Elixir](https://github.com/thmsmlr/instructor_ex/)

    If you want to port Instructor to another language, please reach out to us on [Twitter](https://twitter.com/jxnlco) we'd love to help you get started!

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
      "id": "chatcmpl-8p19pXZ5BIqGtaPhVLo184UuQNH3v",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "logprobs": null,
          "message": {
            "content": null,
            "role": "assistant",
            "function_call": {
              "arguments": "{\n  \"name\": \"Jason\",\n  \"age\": 25\n}",
              "name": "UserDetail"
            },
            "tool_calls": null
          }
        }
      ],
      "created": 1707169869,
      "model": "gpt-3.5-turbo-0613",
      "object": "chat.completion",
      "system_fingerprint": null,
      "usage": {
        "completion_tokens": 16,
        "prompt_tokens": 72,
        "total_tokens": 88
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
