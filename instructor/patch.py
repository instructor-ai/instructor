from functools import wraps
import openai
import inspect
from typing import Callable, Optional, Type, Union

from pydantic import BaseModel
from .function_calls import OpenAISchema, openai_schema


def wrap_chatcompletion(func: Callable) -> Callable:
    is_async = inspect.iscoroutinefunction(func)
    if is_async:

        @wraps(func)
        async def new_chatcompletion(
            *args,
            response_model: Optional[Union[Type[BaseModel], Type[OpenAISchema]]] = None,
            **kwargs
        ):  # type: ignore
            if response_model is not None:
                if not issubclass(response_model, OpenAISchema):
                    response_model = openai_schema(response_model)
                kwargs["functions"] = [response_model.openai_schema]
                kwargs["function_call"] = {"name": response_model.openai_schema["name"]}

            if kwargs.get("stream", False) and response_model is not None:
                import warnings

                warnings.warn(
                    "stream=True is not supported when using response_model parameter"
                )

            response = await func(*args, **kwargs)

            if response_model is not None:
                model = response_model.from_response(response)
                model._raw_response = response
                return model
            return response

    else:

        @wraps(func)
        def new_chatcompletion(
            *args,
            response_model: Optional[Union[Type[BaseModel], Type[OpenAISchema]]] = None,
            **kwargs
        ):
            if response_model is not None:
                if not issubclass(response_model, OpenAISchema):
                    response_model = openai_schema(response_model)
                kwargs["functions"] = [response_model.openai_schema]
                kwargs["function_call"] = {"name": response_model.openai_schema["name"]}

            if kwargs.get("stream", False) and response_model is not None:
                import warnings

                warnings.warn(
                    "stream=True is not supported when using response_model parameter"
                )

            response = func(*args, **kwargs)
            if response_model is not None:
                model = response_model.from_response(response)
                model._raw_response = response
                return model
            return response

    new_chatcompletion.__doc__ = """
Creates a new chat completion for the provided messages and parameters.

See: https://platform.openai.com/docs/api-reference/chat-completions/create

Additional Notes:

Using the `response_model` parameter, you can specify a response model to use for parsing the response from OpenAI's API. If its present, the response will be parsed using the response model, otherwise it will be returned as is. 

If `stream=True` is specified, the response will be parsed using the `from_stream_response` method of the response model, if available, otherwise it will be parsed using the `from_response` method.

If need to obtain the raw response from OpenAI's API, you can access it using the `_raw_response` attribute of the response model.

Parameters:

    response_model (Union[Type[BaseModel], Type[OpenAISchema]]): The response model to use for parsing the response from OpenAI's API, if available (default: None)
"""
    return new_chatcompletion


original_chatcompletion = openai.ChatCompletion.create
original_chatcompletion_async = openai.ChatCompletion.acreate


def patch():
    openai.ChatCompletion.create = wrap_chatcompletion(original_chatcompletion)
    openai.ChatCompletion.acreate = wrap_chatcompletion(original_chatcompletion_async)


def unpatch():
    openai.ChatCompletion.create = original_chatcompletion
    openai.ChatCompletion.acreate = original_chatcompletion_async
