# Welcome to OpenAI Function Call

OpenAI Function Call is a library that provides a minimal and non-intrusive extension to the `Pydantic.BaseModel` class called `OpenAISchema`. It offers two main methods: `openai_schema` for generating the correct schema and `from_response` for creating an instance of the class from the completion result.

The library primarily focuses on showcasing examples and providing a helper class, so I'll keep the example as a simple structured extraction.

If OpenAI is like a chef's knife for code, I aim to provide you with a nice handle and a little booklet of cutting techniques. OpenAI Function Call leverages the data validation capabilities of the Pydantic library to handle output parsing in a structured and reliable manner.

If you have any feedback or need assistance, feel free to leave an issue or reach out to me on [Twitter](https://twitter.com/jxnlco).

If you're looking for a more comprehensive solution with batteries included, I highly recommend [MarvinAI](https://www.askmarvin.ai/). MarvinAI provides a high-level API but doesn't offer as much access to prompting.

!!! tip "Just rip it out!"
    If you don't want to install dependencies, you can literally take the `function_calls.py` file from the library's source code and add it to your own codebase. You can find the [source code here](https://github.com/jxnl/openai_function_call/blob/main/openai_function_call/function_calls.py).

## Installation

You can install OpenAI Function Call using pip:

```sh
pip install openai_function_call
```

## Usage

Below are some examples that demonstrate the usage of function calls and schemas with OpenAI and Pydantic. In subsequent documentation, we will explore more creative use cases.

### Example 1: Extraction

Prompts are now sourced from docstrings and field descriptions, so it's important to write clear and descriptive documentation for your schemas.

```python
import openai
from openai_function_call import OpenAISchema

from pydantic import Field

class UserDetails(OpenAISchema):
    """Details of a user"""
    name: str = Field(..., description="User's full name")
    age: int

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    functions=[UserDetails.openai_schema],
    function_call={"name": UserDetails.openai_schema["name"]},
    messages=[
        {"role": "system", "content": "Extract user details from my requests"},
        {"role": "user", "content": "My name is John Doe and I'm 30 years old."},
    ],
)

user_details = UserDetails.from_response(completion)
print(user_details)  # UserDetails(name='John Doe', age=30)
```

### Example 2: Function Calls

```python
import openai
from openai_function_call import openai_function

@openai_function
def sum(a:int, b:int) -> int:
    """Sum description adds a + b"""
    return a + b

completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        functions=[sum.openai_schema],
        function_call={"name": sum.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You must use the `sum` function instead of adding yourself.",
            },
            {
                "role": "user",
                "content": "What is 6+3",
            },
        ],
    )

result = sum.from_response(completion)
print(result)  # 9
```
