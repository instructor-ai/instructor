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
from instructor import BaseModel


class File(BaseModel):
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


class Program(BaseModel):
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
import instructor
from openai import OpenAI

# Apply the patch to the OpenAI client
# enables response_model keyword
client = instructor.patch(OpenAI())


def develop(data: str) -> Program:
    return client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.1,
        response_model=Program,
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
```

## Evaluating an Example

Let's evaluate the example by specifying the program to create and print the resulting files.

```python
program = develop(
    """
        Create a fastapi app with a readme.md file and a main.py file with
        some basic math functions. the datamodels should use pydantic and
        the main.py should use fastapi. the readme.md should have a title
        and a description. The readme should contain some helpful infromation
        and a curl example"""
)

for file in program.files:
    print(file.file_name)
    print("-")
    print(file.body)
    print("\n\n\n")
```

The output will be:

````markdown
# readme.md

- # FastAPI App

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
````

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

- fastapi
  uvicorn
  pydantic
```

## Add Refactoring Capabilities

This second part of the example shows how OpenAI API can be used to update the multiples files previously created, based on new specifications.

In order to do that, we'll rely on the standard [unidiff](https://en.wikipedia.org/wiki/Diff#Unified_format) format.

This will be our definition for a change in our code base:

```python
from pydantic import Field
from instructor import BaseModel

class Diff(BaseModel):
    """
    Changes that must be correctly made in a program's code repository defined as a
    complete diff (Unified Format) file which will be used to `patch` the repository.

    Example:
      --- /path/to/original	timestamp
      +++ /path/to/new	timestamp
      @@ -1,3 +1,9 @@
      +This is an important
      +notice! It should
      +therefore be located at
      +the beginning of this
      +document!
      +
       This part of the
       document has stayed the
       same from version to
      @@ -8,13 +14,8 @@
       compress the size of the
       changes.
      -This paragraph contains
      -text that is outdated.
      -It will be deleted in the
      -near future.
      -
       It is important to spell
      -check this dokument. On
      +check this document. On
       the other hand, a
       misspelled word isn't
       the end of the world.
      @@ -22,3 +23,7 @@
       this paragraph needs to
       be changed. Things can
       be added after it.
      +
      +This paragraph contains
      +important new additions
      +to this document.
    """

    diff: str = Field(
        ...,
        description=(
            "Changes in a code repository correctly represented in 'diff' format, "
            "correctly escaped so it could be used in a JSON"
        ),
    )
```

The `diff` class represents a _diff_ file, with a set of changes that can be applied to our program using a tool like patch or Git.

## Calling Refactor Completions

We'll define a function that will pass the program and the new specifications to the OpenAI API:

```python
from generate import Program

def refactor(new_requirements: str, program: Program) -> Diff:
    program_description = "\n".join(
        [f"{code.file_name}\n[[[\n{code.body}\n]]]\n" for code in program.files]
    )
    return client.chat.completions.create(
        # model="gpt-3.5-turbo-0613",
        model="gpt-4",
        temperature=0,
        response_model=Diff,
        messages=[
            {
                "role": "system",
                "content": "You are a world class programming AI capable of refactor "
                "existing python repositories. You will name files correct, include "
                "__init__.py files and write correct python code, with correct imports. "
                "You'll deliver your changes in valid 'diff' format so that they could "
                "be applied using the 'patch' command. "
                "Make sure you put the correct line numbers, "
                "and that all lines that must be changed are correctly marked.",
            },
            {
                "role": "user",
                "content": new_requirements,
            },
            {
                "role": "user",
                "content": program_description,
            },
        ],
        max_tokens=1000,
    )
```

Notice we're using here the version `gpt-4` of the model, which is more powerful but, also, more expensive.

## Creating an Example Refactoring

To tests these refactoring, we'll use the `program` object, generated in the first part of this example.

```python
changes = refactor(
    new_requirements="Refactor this code to use flask instead.",
    program=program,
)
print(changes.diff)
```

The output will be this:

````diff
--- readme.md
+++ readme.md
@@ -1,9 +1,9 @@
 # FastAPI App

-This is a FastAPI app that provides some basic math functions.
+This is a Flask app that provides some basic math functions.

 ## Usage

 To use this app, follow the instructions below:

 1. Install the required dependencies by running `pip install -r requirements.txt`.
-2. Start the app by running `uvicorn main:app --reload`.
+2. Start the app by running `flask run`.
 3. Open your browser and navigate to `http://localhost:5000/docs` to access the Swagger UI documentation.

 ## Example

 To perform a basic math operation, you can use the following curl command:

 ```bash
-curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:8000/calculate
+curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:5000/calculate
````

--- main.py
+++ main.py
@@ -1,29 +1,29 @@
-from fastapi import FastAPI
-from pydantic import BaseModel
+from flask import Flask, request, jsonify

-app = FastAPI()
+app = Flask(**name**)

-class Operation(BaseModel):

- operation: str
- operands: list
  +@app.route('/calculate', methods=['POST'])
  +def calculate():

* data = request.get_json()
* operation = data.get('operation')
* operands = data.get('operands')

-@app.post('/calculate')
-async def calculate(operation: Operation):

- if operation.operation == 'add':
-        result = sum(operation.operands)
- elif operation.operation == 'subtract':
-        result = operation.operands[0] - sum(operation.operands[1:])
- elif operation.operation == 'multiply':

* if operation == 'add':
*        result = sum(operands)
* elif operation == 'subtract':
*        result = operands[0] - sum(operands[1:])
* elif operation == 'multiply':
  result = 1

-        for operand in operation.operands:

*        for operand in operands:
             result *= operand

- elif operation.operation == 'divide':
-        result = operation.operands[0]
-        for operand in operation.operands[1:]:

* elif operation == 'divide':
*        result = operands[0]
*        for operand in operands[1:]:
             result /= operand
  else:
  result = None

- return {'result': result}

* return jsonify({'result': result})

--- requirements.txt
+++ requirements.txt
@@ -1,3 +1,2 @@
-fastapi
-uvicorn
-pydantic
+flask
+flask-cors

```

```
