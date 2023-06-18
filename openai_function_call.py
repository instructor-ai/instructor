
# MIT License
#
# Copyright (c) 2023 Jason Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from functools import wraps
from typing import Any, Callable
from pydantic import validate_arguments, BaseModel


class openai_function:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.validate_func = validate_arguments(func)
        self.openai_schema = {
            "name": self.func.__name__,
            "description": self.func.__doc__,
            "parameters": self.validate_func.model.schema(),
        }
        self.model = self.validate_func.model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.validate_func(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def from_response(self, completion, throw_error=True):
        """Execute the function from the response of an openai chat completion"""
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == self.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"])
        return self.validate_func(**arguments)


class OpenAISchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        schema = cls.schema()
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": schema,
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        message = completion.choices[0].message

        if throw_error:
            assert "function_call" in message, "No function call detected"
            assert (
                message["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = message["function_call"]
        arguments = json.loads(function_call["arguments"])
        return cls(**arguments)
