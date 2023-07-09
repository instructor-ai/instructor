# Example: Creating Multiple Files Program

This example shows how to create a multiple files program based on specifications by utilizing the OpenAI Function Call. We will define the necessary data structures using Pydantic and demonstrate how to convert a specification (prompt) into multiple files.


!!! note "Motivation"
    Creating multiple file programs based on specifications is a challenging and rewarding skill that can help you build complex and scalable applications. 
    With OpenAI Function Call, you can leverage the power of language models to generate an entire codebase and code snippets that match your specifications.

## Defining the Data Structures

Let's start by defining the data structure of `File` and `Program`.

```python
from typing import List
from pydantic import Field
from openai_function_call import OpenAISchema


class File(OpenAISchema):
    """
    Correctly named file with contents.
    """

    file_name: str = Field(
        ..., description="The name of the file including the extension"
    )
    body: str = Field(..., description="Correct contents of a file")

    def save(self):
        with open(self.file_name, "w") as f:
            f.write(self.body)


class Program(OpenAISchema):
    """
    Set of files that represent a complete and correct program
    """

    files: List[File] = Field(..., description="List of files")
```

The `File` class represents a single file or script, and it contains a `name` attribute and `body` for the text content of the file. 
Notice that we added the `save` method to the `File` class. This method is used to writes the body of the file to disk using the name as path.

The `Program` class represents a collection of files that form a complete and correct program. 
It contains a list of `File` objects in the `files` attribute.

## Calling Completions

To create the files, we will use the base `openai` API. 
We can define a function that takes in a string and returns a `Program` object.

```python
import openai

def segment(data: str) -> Program:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.1,
        functions=[Program.openai_schema],
        function_call={"name": Program.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a world class programming AI capable of writing correct python scripts and modules. You will name files correct, include __init__.py files and write correct python code with correct imports.",
            },
            {
                "role": "user",
                "content": data,
            },
        ],
        max_tokens=1000,
    )
    return Program.from_response(completion)
```

## Evaluating an Example

Let's evaluate the example by specifying the program to create and print the resulting files.

```python
queries = segment(
        """
        Create a fastapi app with a readme.md file and a main.py file with 
        some basic math functions. the datamodels should use pydantic and 
        the main.py should use fastapi. the readme.md should have a title 
        and a description. The readme should contain some helpful infromation 
        and a curl example"""
    )

for file in queries.files:
    print(file.file_name)
    print("-")
    print(file.body)
    print("\n\n\n")
```

The output will be:
```markdown
# readme.md
-
    # FastAPI App

    This is a FastAPI app that provides some basic math functions.

    ## Usage

    To use this app, follow the instructions below:

    1. Install the required dependencies by running `pip install -r requirements.txt`.
    2. Start the app by running `uvicorn main:app --reload`.
    3. Open your browser and navigate to `http://localhost:8000/docs` to access the Swagger UI documentation.

    ## Example

    You can use the following curl command to test the `/add` endpoint:

    ```bash
    $ curl -X POST -H "Content-Type: application/json" -d '{"a": 2, "b": 3}' http://localhost:8000/add
    ```
```
```python
# main.py
-
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()


    class Numbers(BaseModel):
        a: int
        b: int


    @app.post('/add')
    def add_numbers(numbers: Numbers):
        return {'result': numbers.a + numbers.b}


    @app.post('/subtract')
    def subtract_numbers(numbers: Numbers):
        return {'result': numbers.a - numbers.b}


    @app.post('/multiply')
    def multiply_numbers(numbers: Numbers):
        return {'result': numbers.a * numbers.b}


    @app.post('/divide')
    def divide_numbers(numbers: Numbers):
        if numbers.b == 0:
            return {'error': 'Cannot divide by zero'}
        return {'result': numbers.a / numbers.b}
```
```markdown
# requirements.txt
-
    fastapi
    uvicorn
    pydantic
```