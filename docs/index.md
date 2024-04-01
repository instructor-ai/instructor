# Instructor, Generating Structure from LLMs

_Structured outputs powered by llms. Designed for simplicity, transparency, and control._

---

[![Twitter Follow](https://img.shields.io/twitter/follow/jxnlco?style=social)](https://twitter.com/jxnlco)
[![Discord](https://img.shields.io/discord/1192334452110659664?label=discord)](https://discord.gg/CV8sPM5k5Y)
[![Downloads](https://img.shields.io/pypi/dm/instructor.svg)](https://pypi.python.org/pypi/instructor)
[![GPT](https://img.shields.io/badge/docs-InstructorGPT-blue)](https://chat.openai.com/g/g-EvZweRWrE-instructor-gpt)

Instructor makes it easy to reliably get structured data like JSON from Large Language Models (LLMs) like GPT-3.5, GPT-4, GPT-4-Vision, including open source models like Mistral/Mixtral from [Together](./hub/together.md), [Anyscale](./hub/anyscale.md), [Ollama](./hub/ollama.md), and [llama-cpp-python](./hub/llama-cpp-python.md).

By leveraging various modes like Function Calling, Tool Calling and even constrained sampling modes like JSON mode, JSON Schema; Instructor stands out for its simplicity, transparency, and user-centric design. We leverage Pydantic to do the heavy lifting, and we've built a simple, easy-to-use API on top of it by helping you manage [validation context](./concepts/reask_validation.md), retries with [Tenacity](./concepts/retrying.md), and streaming [Lists](./concepts/lists.md) and [Partial](./concepts/partial.md) responses.


We also provide a library in [Typescript](https://instructor-ai.github.io/instructor-js/), [Elixir](https://github.com/thmsmlr/instructor_ex/) and [PHP](https://github.com/cognesy/instructor-php/).

## Why use Instructor?

The question of using Instructor is fundamentally a question of why to use Pydantic.

1. **Powered by type hints** — Instructor is powered by Pydantic, which is powered by type hints. Schema validation, prompting is controlled by type annotations; less to learn, less code to write, and integrates with your IDE.

2. **Customizable** — Pydantic is highly customizable. You can define your own validators, custom error messages, and more.

3. **Ecosystem** Pydantic is the most widely used data validation library for Python with over 100M downloads a month. It's used by FastAPI, Typer, and many other popular libraries.


## Getting Started

```
pip install -U instructor
```

If you ever get stuck, you can always run `instructor docs` to open the documentation in your browser. It even supports searching for specific topics.

```
instructor docs [QUERY]
```

You can also check out our [cookbooks](./examples/index.md) and [concepts](./concepts/models.md) to learn more about how to use Instructor.

Now, let's see Instructor in action with a simple example:

### Using OpenAI

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

print(user_info.name)
#> John Doe
print(user_info.age)
#> 30
```

### Using Anthropic

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

### Using Litellm

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

## Correct Typing

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
    # User(name='John Doe', age=30)
    # User(name='Jane Smith', age=25)
```

![iterable](./blog/posts/img/iterable.png)

## Validation

You can also use Pydantic to validate your outputs and get the llm to retry on failure. Check out our docs on [retrying](./concepts/retrying.md) and [validation context](./concepts/reask_validation.md).

## More Examples

If you'd like to see more check out our [cookbook](examples/index.md).

[Installing Instructor](installation.md) is a breeze. Just run `pip install instructor`.

## Contributing

If you want to help out, checkout some of the issues marked as `good-first-issue` or `help-wanted`. Found [here](https://github.com/jxnl/instructor/labels/good%20first%20issue). They could be anything from code improvements, a guest blog post, or a new cook book.

## License

This project is licensed under the terms of the MIT License.
