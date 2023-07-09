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


def segment(data: str) -> Program:
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


if __name__ == "__main__":
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

    You can use the following curl command to test the `/add` endpoint:

    ```bash
    $ curl -X POST -H "Content-Type: application/json" -d '{"a": 2, "b": 3}' http://localhost:8000/add
    ```

    main.py
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


    requirements.txt
    -
    fastapi
    uvicorn
    pydantic
    """
