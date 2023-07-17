# OpenAI Function Call

*OpenAISchema, structured extraction in Python, powered by OpenAI, designed for simplicity, transparency, and control.*

-----

This library is build to interact with openai's function call api from python code, with python objects. It's designed to be intuitive, easy to use, but give great visibily in how we call openai.

**OpenAISchema** is based on Python type annotations, and powered by Pydantic.

The key features are:

* **Intuitive to write**: Great support for editors, completions. Spend less time debugging.
* **Writing prompts as code**: Collocate docstrings and descriptions as part of your prompting.
* **Extensible**: Bring your own kitchen sink without being weighted down by abstractions.

## Structured Extraction with `openai`

Welcome to the Quick Start Guide for OpenAI Function Call. This guide will walk you through the installation process and provide examples demonstrating the usage of function calls and schemas with OpenAI and Pydantic.

### Requirements

This library depends on **Pydantic** an **OpenAI** that's all.

### Installation

To get started with OpenAI Function Call, you need to install it using `pip`. Run the following command in your terminal:

!!! note Requirement
    Ensure you have Python version 3.9 or above.


<!-- termynal -->
```
$ pip install openai_function_call
```

## Quick Start

This quick start guide contains the follow sections:

1. Defining a schema 
2. Adding Additional Prompting
3. Calling the ChatCompletion
4. Deserializing back to the instance

OpenAI Function Call allows you to leverage OpenAI's powerful language models for function calls and schema extraction. This guide provides a quick start for using OpenAI Function Call.

### Section 1: Defining a Schema

To begin, let's define a schema using OpenAI Function Call. A schema describes the structure of the input and output data for a function. In this example, we'll define a simple schema for a `User` object:

```python
from openai_function_call import OpenAISchema

class UserDetails(OpenAISchema):
    name: str
    age: int
```

In this schema, we define a `UserDetails` class that extends `OpenAISchema`. We declare two fields, `name` and `age`, of type `str` and `int` respectively. 

### Section 2: Adding Additional Prompting

To enhance the performance of the OpenAI language model, you can add additional prompting in the form of docstrings and field descriptions. They can provide context and guide the model on how to process the data.

```python hl_lines="5 6"
from openai_function_call import OpenAISchema
from pydantic import Field

class UserDetails(OpenAISchema):
    "Correctly extracted user information"
    name: str = Field(..., description="User's full name")
    age: int
```

In this updated schema, we use the `Field` class from `pydantic` to add descriptions to the `name` field. The description provides information about the field, giving even more context to the language model.


!!! note "Code, schema, and prompt"
     We can run `openai_schema` to see exactly what the API will see, notice how the docstrings, attributes, types, and field descriptions are now part of the schema. This describes on this library's core philosophies.

    ```python hl_lines="2 3"
    class UserDetails(OpenAISchema):
        "Correctly extracted user information"
        name: str = Field(..., description="User's full name")
        age: int

    UserDetails.openai_schema
    ```

    ```json hl_lines="3 8"
    {
    "name": "UserDetails",
    "description": "Correctly extracted user information",
    "parameters": {
        "type": "object",
        "properties": {
        "name": {
            "description": "User's full name",
            "type": "string"
        },
        "age": {
            "type": "integer"
        }
        },
        "required": [
        "age",
        "name"
        ]
    }
    }
    ```

### Section 3: Calling the ChatCompletion

With the schema defined, let's proceed with calling the `ChatCompletion` API using the defined schema and messages.

```python hl_lines="11 12 15"
from openai_function_call import OpenAISchema
from pydantic import Field

class UserDetails(OpenAISchema):
    "Correctly extracted user information"
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
```

In this example, we make a call to the `ChatCompletion` API by providing the model name (`gpt-3.5-turbo-0613`) and a list of messages. The messages consist of a system message and a user message. The system message sets the context by requesting user details, while the user message provides the input with the user's name and age.

Note that we have omitted the additional parameters that can be included in the API request, such as `temperature`, `max_tokens`, and `n`. These parameters can be customized according to your requirements.

### Section 4: Deserializing Back to the Instance

To deserialize the response from the `ChatCompletion` API back into an instance of the `UserDetails` class, we can use the `from_response` method.

```python hl_lines="1"
user = UserDetails.from_response(completion)
print(user.name)  # Output: John Doe
print(user.age)   # Output: 30
```

By calling `UserDetails.from_response`, we create an instance of the `UserDetails` class using the response from the API call. Subsequently, we can access the extracted user details through the `name` and `age` attributes of the `user` object.

## IDE Support 

Everything is designed for you to get the best developer experience possible, with the best editor support.

Including **autocompletion**:

![autocomplete](img/ide_support.png)

And even **inline errors**

![errors](img/error2.png)

## OpenAI Schema and Pydantic

This quick start guide provided you with a basic understanding of how to use OpenAI Function Call for schema extraction and function calls. You can now explore more advanced use cases and creative applications of this library.

Since `UserDetails` is a `OpenAISchems` and a `pydantic.BaseModel` you can use inheritance and nesting to create more complex emails while avoiding code duplication

```python
class UserDetails(OpenAISchema):
    name: str = Field(..., description="User's full name")
    age: int

class UserWithAddress(UserDetails):
    address: str 

class UserWithFriends(UserDetails):
    best_friend: UserDetails
    friends: List[UserDetails]
```

If you have any questions, feel free to leave an issue or reach out to the library's author on [Twitter](https://twitter.com/jxnlco). For a more comprehensive solution with additional features, consider checking out [MarvinAI](https://www.askmarvin.ai/).

To see more examples of how we can create interesting models check out some [examples.](examples/index.md)

## License

This project is licensed under ther terms of the MIT License.