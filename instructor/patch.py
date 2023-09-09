from functools import wraps
from json import JSONDecodeError
from pydantic import ValidationError
import openai
import inspect
from typing import Callable, Type

from pydantic import BaseModel
from .function_calls import OpenAISchema, openai_schema

OVERRIDE_DOCS = """
Creates a new chat completion for the provided messages and parameters.

See: https://platform.openai.com/docs/api-reference/chat-completions/create

Additional Notes:

Using the `response_model` parameter, you can specify a response model to use for parsing the response from OpenAI's API. If its present, the response will be parsed using the response model, otherwise it will be returned as is. 

If `stream=True` is specified, the response will be parsed using the `from_stream_response` method of the response model, if available, otherwise it will be parsed using the `from_response` method.

If need to obtain the raw response from OpenAI's API, you can access it using the `_raw_response` attribute of the response model.

Parameters:

    response_model (Union[Type[BaseModel], Type[OpenAISchema]]): The response model to use for parsing the response from OpenAI's API, if available (default: None)
    max_retries (int): The maximum number of retries to attempt if the response is not valid (default: 0)
"""


def handle_response_model(response_model: Type[BaseModel], kwargs):
    new_kwargs = kwargs.copy()
    if response_model is not None:
        if not issubclass(response_model, OpenAISchema):
            response_model = openai_schema(response_model)  # type: ignore
        new_kwargs["functions"] = [response_model.openai_schema]  # type: ignore
        new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}  # type: ignore

    if new_kwargs.get("stream", False) and response_model is not None:
        import warnings

        warnings.warn(
            "stream=True is not supported when using response_model parameter"
        )

    return response_model, new_kwargs


def process_response(response, response_model, validation_context=None):  # type: ignore
    if response_model is not None:
        model = response_model.from_response(
            response, validation_context=validation_context
        )
        model._raw_response = response
        return model
    return response


async def retry_async(
    func, response_model, validation_context, args, kwargs, max_retries
):
    retries = 0
    while retries <= max_retries:
        try:
            response = await func(*args, **kwargs)
            return process_response(response, response_model, validation_context), None
        except (ValidationError, JSONDecodeError) as e:
            kwargs["messages"].append(dict(**response.choices[0].message))  # type: ignore
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, exceptions found\n{e}",
                }
            )
            retries += 1
            if retries > max_retries:
                raise e


def retry_sync(func, response_model, validation_context, args, kwargs, max_retries):
    retries = 0
    new_kwargs = kwargs.copy()
    while retries <= max_retries:
        # Excepts ValidationError, and JSONDecodeError
        try:
            response = func(*args, **kwargs)
            return process_response(response, response_model, validation_context), None
        except (ValidationError, JSONDecodeError) as e:
            kwargs["messages"].append(dict(**response.choices[0].message))  # type: ignore
            kwargs["messages"].append(
                {
                    "role": "user",
                    "content": f"Recall the function correctly, exceptions found\n{e}",
                }
            )
            retries += 1
            if retries > max_retries:
                raise e


def wrap_chatcompletion(func: Callable) -> Callable:
    is_async = inspect.iscoroutinefunction(func)

    @wraps(func)
    async def new_chatcompletion_async(
        response_model=None, valiation_context=None, *args, max_retries=0, **kwargs
    ):
        response_model, new_kwargs = handle_response_model(response_model, kwargs)  # type: ignore
        response, error = await retry_async(
            func=func,
            response_model=response_model,
            valiation_context=valiation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
        )  # type: ignore
        if error:
            raise ValueError(error)
        return (response,)

    @wraps(func)
    def new_chatcompletion_sync(
        response_model=None, validation_context=None, *args, max_retries=0, **kwargs
    ):
        response_model, new_kwargs = handle_response_model(response_model, kwargs)  # type: ignore
        response, error = retry_sync(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
        )  # type: ignore
        if error:
            raise ValueError(error)
        return response

    wrapper_function = new_chatcompletion_async if is_async else new_chatcompletion_sync
    wrapper_function.__doc__ = OVERRIDE_DOCS
    return wrapper_function


def process_response(response, response_model):
    if response_model is not None:
        model = response_model.from_response(response)
        model._raw_response = response
        return model
    return response


original_chatcompletion = openai.ChatCompletion.create
original_chatcompletion_async = openai.ChatCompletion.acreate


def patch():
    openai.ChatCompletion.create = wrap_chatcompletion(original_chatcompletion)
    openai.ChatCompletion.acreate = wrap_chatcompletion(original_chatcompletion_async)


def unpatch():
    openai.ChatCompletion.create = original_chatcompletion
    openai.ChatCompletion.acreate = original_chatcompletion_async
