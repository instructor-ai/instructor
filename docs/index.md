# Welcome to OpenAI Function Call

We try to provides a powerful and efficient approach to output parsing when interacting with OpenAI's Function Call API. One that is framework agnostic and minimizes any dependencies. It leverages the data validation capabilities of the Pydantic library to handle output parsing in a more structured and reliable manner. If you have any feedback, leave an issue or hit me up on [twitter](https://twitter.com/jxnlco). 

## Installation

```python
pip install openai_function_call
```

## Usage

This module simplifies the interaction with the OpenAI API, enabling a more structured outputs. Below are examples showcasing the use of function calls and schemas with OpenAI and Pydantic. In later modoules we'll go over a wide array of more creative uses.

### Example 1: Function Calls

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

### Example 2: Schema Extraction

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

# Code 

::: openai_function_call