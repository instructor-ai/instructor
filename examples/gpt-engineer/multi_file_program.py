import openai

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


class Diff(OpenAISchema):
    """
    Changes in a program's code repository
    """
    diff: str = Field(..., description="Changes in a code repository in 'diff' format, correctly escaped so it could be used in a JSON")


def develop(data: str) -> Program:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.1,
        functions=[Program.openai_schema],
        function_call={"name": Program.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a world class programming AI capable of writing correct python scripts and modules. You will name files correct, include __init__.py files and write correct python code. with correct imports.",
            },
            {
                "role": "user",
                "content": data,
            },
        ],
        max_tokens=1000,
    )
    return Program.from_response(completion)


def refactor(new_requirements: str, program: Program) -> Diff:
    program_description = "\n".join([
            f"{code.file_name}\n[[[\n{code.body}\n]]]\n"
            for code in program.files
    ])
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0613",
        model="gpt-4",
        temperature=0,
        functions=[Diff.openai_schema],
        function_call={"name": Diff.openai_schema["name"]},
        messages=[
            {
                "role": "system",
                "content": "You are a world class programming AI capable of refactor "
                "existing python repositories. You will name files correct, include "
                "__init__.py files and write correct python code. with correct imports. "
                "You'll deliver your changes in valid 'diff' format so that they could "
                "be applied using the 'patch' command.",
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
    return Diff.from_response(completion)


if __name__ == "__main__":
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
    """
    readme.md
    -
    # FastAPI App

    This is a FastAPI app that provides some basic math functions.

    ## Usage

    To use this app, follow the instructions below:

    1. Install the required dependencies by running `pip install -r requirements.txt`.
    2. Start the app by running `uvicorn main:app --reload`.
    3. Open your browser and navigate to `http://localhost:8000/docs` to access the Swagger UI documentation.

    ## Example

    To perform a basic math operation, you can use the following curl command:

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:8000/calculate
    ```





    main.py
    -
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()


    class Operation(BaseModel):
        operation: str
        operands: list[float]


    @app.post('/calculate')
    async def calculate(operation: Operation):
        if operation.operation == 'add':
            result = sum(operation.operands)
        elif operation.operation == 'subtract':
            result = operation.operands[0] - sum(operation.operands[1:])
        elif operation.operation == 'multiply':
            result = 1
            for operand in operation.operands:
                result *= operand
        elif operation.operation == 'divide':
            result = operation.operands[0]
            for operand in operation.operands[1:]:
                result /= operand
        else:
            return {'error': 'Invalid operation'}
        return {'result': result}





    requirements.txt
    -
    fastapi
    uvicorn
    pydantic
    """

    changes = refactor(
            new_requirements="Refactor this code to use flask instead.",
            program=program,
    )
    print(changes.diff)
    """
    --- readme.md
    +++ readme.md
    @@ -1,9 +1,9 @@
    -# FastAPI App
    +# Flask App

     This is a Flask app that provides some basic math functions.

     ## Usage

     To use this app, follow the instructions below:

    -1. Install the required dependencies by running `pip install -r requirements.txt`.
    -2. Start the app by running `uvicorn main:app --reload`.
    -3. Open your browser and navigate to `http://localhost:8000/docs` to access the Swagger UI documentation.
    +1. Install the required dependencies by running `pip install -r requirements.txt`.
    +2. Start the app by running `flask run`.
    +3. Open your browser and navigate to `http://localhost:5000` to access the app.

     ## Example

     To perform a basic math operation, you can use the following curl command:

     ```bash
    -curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:8000/calculate
    +curl -X POST -H "Content-Type: application/json" -d '{"operation": "add", "operands": [2, 3]}' http://localhost:5000/calculate
     ```

    --- main.py
    +++ main.py
    @@ -1,23 +1,23 @@
    -from fastapi import FastAPI
    -from pydantic import BaseModel
    +from flask import Flask, request, jsonify

    -app = FastAPI()
    +app = Flask(__name__)

    -class Operation(BaseModel):
    -    operation: str
    -    operands: list[float]
    +@app.route('/calculate', methods=['POST'])
    +def calculate():
    +    data = request.get_json()
    +    operation = data.get('operation')
    +    operands = data.get('operands')

    -@app.post('/calculate')
    -async def calculate(operation: Operation):
    -    if operation.operation == 'add':
    -        result = sum(operation.operands)
    -    elif operation.operation == 'subtract':
    -        result = operation.operands[0] - sum(operation.operands[1:])
    -    elif operation.operation == 'multiply':
    +    if operation == 'add':
    +        result = sum(operands)
    +    elif operation == 'subtract':
    +        result = operands[0] - sum(operands[1:])
    +    elif operation == 'multiply':
             result = 1
    -        for operand in operation.operands:
    +        for operand in operands:
                 result *= operand
    -    elif operation.operation == 'divide':
    -        result = operation.operands[0]
    -        for operand in operation.operands[1:]:
    +    elif operation == 'divide':
    +        result = operands[0]
    +        for operand in operands[1:]:
                 result /= operand
         else:
    -        return {'error': 'Invalid operation'}
    -    return {'result': result}
    +        return jsonify({'error': 'Invalid operation'}), 400
    +    return jsonify({'result': result}), 200

    --- requirements.txt
    +++ requirements.txt
    @@ -1,3 +1,2 @@
    -fastapi
    -uvicorn
    -pydantic
    +flask
    """
