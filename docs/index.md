# Welcome to OpenAI Function Call

We offer a minially invasive extention of `Pydantic.BaseModel` named `OpenAISchema`. It only has two methods, one to generate the correct schema, and one to produce the class from the completion.

This library is more, so a list of examples and a helper class so I'll keep the example as just structured extraction.

If the OpenAI is a chef's knife of code, I hope you sell you a nice handle which comes with a little pamplet of cuttign techniques.

It leverages the data validation capabilities of the Pydantic library to handle output parsing in a more structured and reliable manner.

If you have any feedback, leave an issue or hit me up on [twitter](https://twitter.com/jxnlco).

If you're looking for something more batteries included I strongly recommend [MarvinAI](https://www.askmarvin.ai/) which offers a high level api but does not provide as much access to prompting.

!!! tip "Just rip it out!"
    If you don't want to install dependencies. I recommend literally ripping the `function_calls.py` into your own codebase. [[source code]](https://github.com/jxnl/openai_function_call/blob/main/openai_function_call/function_calls.py)

## Installation

```python
pip install openai_function_call
```

## Usage

Below are examples showcasing the use of function calls and schemas with OpenAI and Pydantic. In later docs we'll go over a wide array of more creative uses.

### Example 1: Extraction

!!! Tip
    Prompt are now sourced from docstrings and descriptions, so write clear and descriptive documentation!

```python
import openai
from openai_function_call import OpenAISchema

from pydantic import Field

class UserDetails(OpenAISchema):
    """Details of a user"""
    name: str = Field(..., description="users's full name")
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
print(user_details)  # name="John Doe", age=30
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
