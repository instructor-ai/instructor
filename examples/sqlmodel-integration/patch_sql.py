try:
    from sqlalchemy.orm import Session
except ImportError:
    import warnings

    warnings.warn("SQLAlchemy is not installed. Please install it to use this feature `pip install sqlalchemy`")

import time
from functools import wraps
import openai
import inspect
import json
from typing import Callable

from sa import ChatCompletion, Message


def sql_message(index, message, is_response=False):
    return Message(
        index=index,
        content=message.get("content", None),
        role=message["role"],
        arguments=message.get("function_call", {}).get("arguments", None),
        name=message.get("function_call", {}).get("name", None),
        is_function_call="function_call" in message,
        is_response=is_response,
    )


# Synchronous function to insert chat completion
def sync_insert_chat_completion(
    engine,
    messages: list[dict],
    responses: list[dict] = [],
    **kwargs,
):
    with Session(engine) as session: 
        chat = ChatCompletion(
            id=kwargs.pop("id", None),
            created_at=kwargs.pop("created", None),
            functions=json.dumps(kwargs.pop("functions", None)),
            function_call=json.dumps(kwargs.pop("function_call", None)),
            latency_ms=kwargs.pop("latency_ms", None),
            messages=[
                sql_message(index=ii, message=message)
                for (ii, message) in enumerate(messages)
            ],
            responses=[
                sql_message(index=resp["index"], message=resp.message, is_response=True) # type: ignore
                for resp in responses
            ],
            **kwargs,
        )
        session.add(chat)
        session.commit()


def patch_with_engine(engine):
    def add_sql_alchemy(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)
        if is_async:

            @wraps(func)
            async def new_chatcompletion(*args, **kwargs):  # type: ignore

                start_ms = time.time()
                response = await func(*args, **kwargs)
                latency_ms = round((time.time() - start_ms) * 1000)
                sync_insert_chat_completion(
                    engine,
                    messages=kwargs.pop("messages", []),
                    responses=response.choices,
                    latency_ms=latency_ms,
                    id=response["id"],
                    **response["usage"],
                    **kwargs,
                )
                return response

        else:

            @wraps(func)
            def new_chatcompletion(*args, **kwargs):
                start_ms = time.time()
                response = func(*args, **kwargs)
                latency_ms = round((time.time() - start_ms) * 1000)

                sync_insert_chat_completion(
                    engine,
                    messages=kwargs.pop("messages", []),
                    responses=response.choices,
                    id=response["id"],
                    latency_ms=latency_ms,
                    **response["usage"],
                    **kwargs,
                )
                return response

        return new_chatcompletion

    return add_sql_alchemy


def instrument_chat_completion_sa(engine):
    patcher = patch_with_engine(engine)
    original_chatcompletion = openai.ChatCompletion.create
    original_chatcompletion_async = openai.ChatCompletion.acreate
    openai.ChatCompletion.create = patcher(original_chatcompletion)
    openai.ChatCompletion.acreate = patcher(original_chatcompletion_async)
