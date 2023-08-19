from functools import wraps
import openai
import inspect
from typing import Callable, Optional, Type, Union

from sqlmodel import Session
from models import Completion, Messages,
from sqlmodels import engine, session, ChatCompletion, Message

def wrap_chatcompletion(func: Callable) -> Callable:
    is_async = inspect.iscoroutinefunction(func)
    if is_async:

        @wraps(func)
        async def new_chatcompletion(
            *args,
            **kwargs
        ):  # type: ignore
            response = await func(*args, **kwargs)

            with Session(engine) as session:
                chat_completion = ChatCompletion(
                    messages=[Message(**message, index=ii) for ii, message in enumerate(kwargs["messages"])],
                    temperature=kwargs.get("temperature", None),
                    model=kwargs.get("model", None),
                    max_tokens=kwargs.get("max_tokens", None),
                    prompt_tokens=response.usage.get("prompt_tokens", None),
                    completion_tokens=response.usage.get("completion_tokens", None),
                    total_tokens=response.usage.get("total_tokens", None),
                    finish_reason=response.choices[0].finish_reason,
                    response=Message(**response.choices[0]),
                )
                session.add(chat_completion)


            return response

    else:

        @wraps(func)
        def new_chatcompletion(
            *args,
            **kwargs
        ):
            response = func(*args, **kwargs)
            return response

    return new_chatcompletion


original_chatcompletion = openai.ChatCompletion.create
original_chatcompletion_async = openai.ChatCompletion.acreate


def patch():
    openai.ChatCompletion.create = wrap_chatcompletion(original_chatcompletion)
    openai.ChatCompletion.acreate = wrap_chatcompletion(original_chatcompletion_async)


def unpatch():
    openai.ChatCompletion.create = original_chatcompletion
    openai.ChatCompletion.acreate = original_chatcompletion_async
