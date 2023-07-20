# Pydantic is all you need, for openai function calls.

Check out the [docs](https://openai-function-call.onrender.com/)!

We try to provides a powerful and efficient approach to output parsing when interacting with OpenAI's Function Call API. One that is framework agnostic and minimizes any dependencies. It leverages the data validation capabilities of the Pydantic library to handle output parsing in a more structured and reliable manner.
If you have any feedback, leave an issue or hit me up on [twitter](https://twitter.com/jxnlco).

This repo also contains a range of examples I've used in experimentation and in production and I welcome new contributions for different types of schemas.

## Support

Follow me on twitter and consider helping pay for openai tokens!

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/jxnlco.svg?style=social&label=Follow%20%40jxnlco)](https://twitter.com/jxnlco) [!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/jxnl)

## Installation

Ensure you have Python version 3.9 or above.

```python
pip install openai-function-call
```

## Contributing

To get started, clone the repository

```bash
git clone https://github.com/jxnl/openai_function_call.git
```

Next, install the necessary Python packages from the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Poetry

We also use poetry if you'd like

```bash
poetry build
```

Your contributions are welcome! If you have great examples or find neat patterns, clone the repo and add another example.
Check out the issues for any ideas if you want to learn. The goal is to find great patterns and cool examples to highlight.

If you encounter any issues or want to provide feedback, you can create an issue in this repository. You can also reach out to me on Twitter at @jxnlco.

## Usage

This module simplifies the interaction with the OpenAI API, enabling a more structured and predictable conversation with the AI. Below are examples showcasing the use of function calls and schemas with OpenAI and Pydantic.

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
        messages=[
            {
                "role": "system",
                "content": "You must use the `sum` function instead of adding yourself.",
            },
            {
                "role": "user",
                "content": "What is 6+3 use the `sum` function",
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
    """User Details"""
    name: str = Field(..., description="User's name")
    age: int = Field(..., description="User's age")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    functions=[UserDetails.openai_schema],
    messages=[
        {"role": "system", "content": "I'm going to ask for user details. Use UserDetails to parse this data."},
        {"role": "user", "content": "My name is John Doe and I'm 30 years old."},
    ],
)

user_details = UserDetails.from_response(completion)
print(user_details)  # UserDetails(name="John Doe", age=30)
```

### Example 2.1: Using the Decorator

The following will also work but we're having issues with propogating type hints
so language services throw errors for methods like `.openai_schema`. We'd welcome a PR to fix this! 

```python
from openai_function_call import openai_schema

@openai_schema
class UserDetails(BaseModel):
    """User Details"""
    name: str = Field(..., description="User's name")
    age: int = Field(..., description="User's age")
```

### Example 3: Using the DSL

```python
from pprint import pprint

from openai_function_call import OpenAISchema
from openai_function_call.dsl import ChatCompletion, MultiTask, messages as m
from openai_function_call.dsl.messages import SystemIdentity, SystemTask, SystemStyle, SystemGuidelines, SystemTips

# Define a subtask you'd like to extract from then,
# We'll use MultTask to easily map it to a List[Search]
# so we can extract more than one
class Search(OpenAISchema):
    id: int
    query: str

tasks = (
    ChatCompletion(name="Acme Inc Email Segmentation", model="gpt-3.5-turbo-0613")
    | SystemIdentity(identity="World class state of the art agent") # if no identity is provided, this is the default one
    | SystemTask(task="Segment emails into search queries")
    | SystemStyle(style="Professional, clear and concise")
    | SystemGuidelines(guidelines=[
        'You never swear',
        'You are polite',
        'You say please and thank you often.'
    ])
    | SystemTips(tips=[
        "When unsure about the correct segmentation, try to think about the task as a whole",
        "If acronyms are used expand them to their full form",
        "Use multiple phrases to describe the same thing"]
                  )
    | MultiTask(subtask_class=Search)
    | m.TaggedMessage(
        tag="email",
        content="Can you find the video I sent last week and also the post about dogs",
    )
    | m.ChainOfThought()
)
# Its important that this just builds you request,
# all these | operators are overloaded and all we do is compile
# it to the openai kwargs
# Also note that the System components are combined sequentially
# so the order matters!
assert isinstance(tasks, ChatCompletion)
pprint(tasks.kwargs, indent=3)
"""
{
    "messages": [
        {
            "role": "system",
            "content": "You are a world class state of the art agent.\n\nYour purpose is to correctly complete this task:
                        `Segment emails into search queries`.\n\nYour style when answering is professional, clear and concise\n\n
                        These are the guidelines you consider when completing your task:\n\n* You never swear\n* You are polite\n* You say please and thank you often.\n\nHere are some tips to help you complete the task:\n\n* When unsure about the correct segmentation, try to think about the task as a whole\n* If acronyms are used expand them to their full form\n* Use multiple phrases to describe the same thing"
        },
        ...
        {
            "role": "user",
            "content": "<email>Can you find the video I sent last week and also the post about dogs</email>"
        },
        {
            "role": "assistant",
            "content": "Lets think step by step to get the correct answer:"
        }
    ],
    "functions": [
        {
            "name": "MultiSearch",
            "description": "Correct segmentation of `Search` tasks",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "description": "Correctly segmented list of `Search` tasks",
                        "type": "array",
                        "items": {"$ref": "#/definitions/Search"}
                    }
                },
                "definitions": {
                    "Search": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "query": {"type": "string"}
                        },
                        "required": ["id", "query"]
                    }
                },
                "required": ["tasks"]
            }
        }
    ],
    "function_call": {"name": "MultiSearch"},
    "max_tokens": 1000,
    "temperature": 0.1,
    "model": "gpt-3.5-turbo-0613"
"""

# Once we call .create we'll be returned with a multitask object that contains our list of task
result = tasks.create()

for task in result.tasks:
    # We can now extract the list of tasks as we could normally
    assert isinstance(task, Search)
```

## Advanced Usage

If you want to see more examples checkout the examples folder!

## License

This project is licensed under the terms of the MIT license.

For more details, refer to the LICENSE file in the repository.
