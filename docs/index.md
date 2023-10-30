# Instructor (openai_function_call)

!!! note "Renaming from openai_function_call"
    This library used to be called `openai_function_call` simply change the import and you should be good to go!

    ```sh
    find /path/to/dir -type f -exec sed -i 's/openai_function_call/instructor/g' {} \;
    ```

*Structured extraction in Python, powered by OpenAI's function calling api, designed for simplicity, transparency, and control.*

-----

This library is built to interact with openai's function call api from python code, with python structs / objects. It's designed to be intuitive, easy to use, but give great visibily in how we call openai.

The approach of combining a human prompt and a "response schema" is not necessarily unique; however, it shows great promise. As we have been concentrating on translating user intent into structured data, we have discovered that Python with Pydantic is exceptionally well-suited for this task. 

**OpenAISchema** is based on Python type annotations, and powered by Pydantic.

The key features are:

* **Intuitive to write**: Great support for editors, completions. Spend less time debugging.
* **Writing prompts as code**: Collocate docstrings and descriptions as part of your prompting.
* **Extensible**: Bring your own kitchen sink without being weighted down by abstractions.

## Structured Extraction with `openai`

Welcome to the Quick Start Guide for OpenAI Function Call. This guide will walk you through the installation process and provide examples demonstrating the usage of function calls and schemas with OpenAI and Pydantic.

### Requirements

This library depends on **Pydantic** and **OpenAI** that's all.

### Installation

To get started with OpenAI Function Call, you need to install it using `pip`. Run the following command in your terminal:

!!! note Requirement
    Ensure you have Python version 3.9 or above.

```sh
$ pip install instructor
```

## Quick Start with Patching ChatCompletion

To simplify your work with OpenAI models and streamline the extraction of Pydantic objects from prompts, we offer a patching mechanism for the `ChatCompletion`` class. Here's a step-by-step guide:

### Step 1: Import and Patch the Module

First, import the required libraries and apply the patch function to the OpenAI module. This exposes new functionality with the response_model parameter.

```python
import openai
import instructor
from pydantic import BaseModel

# This enables response_model keyword
# from openai.ChatCompletion.create
instructor.patch()
```

### Step 2: Define the Pydantic Model

Create a Pydantic model to define the structure of the data you want to extract. This model will map directly to the information in the prompt.

```python
class UserDetail(BaseModel):
    name: str
    age: int
```

### Step 3: Extract Data with ChatCompletion

Use the openai.ChatCompletion.create method to send a prompt and extract the data into the Pydantic object. The response_model parameter specifies the Pydantic model to use for extraction.

```python
user: UserDetail = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ]
)
```

### Step 4: Validate the Extracted Data

You can then validate the extracted data by asserting the expected values. By adding the type things you also get a bunch of nice benefits with your IDE like spell check and auto complete!

```python
assert user.name == "Jason"
assert user.age == 25
```

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
