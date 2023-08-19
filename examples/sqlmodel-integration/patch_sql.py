"""
TODO: add the created_at, id and called it completion vs response
"""

try:
    import importlib

    importlib.import_module("sqlalchemy")
except ImportError:
    import warnings

    warnings.warn("SQLAlchemy is not installed. Please install it to use this feature.")

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

import asyncio
from functools import wraps
import openai
import inspect
import json
from typing import Callable

from sa import ChatCompletion, Message


# Check if the engine is asynchronous
def is_async_engine(engine) -> bool:
    return isinstance(engine, AsyncEngine)


# Async function to insert chat completion
async def async_insert_chat_completion(
    engine: AsyncEngine,
    messages: list[dict],
    responses: list[dict] = [],
    **kwargs,
):
    # Create a custom Session class
    AsyncSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with AsyncSessionLocal() as session:
        chat = ChatCompletion(
            id=kwargs.pop("id", None),
            messages=[
                Message(**message, index=ii) for (ii, message) in enumerate(messages)
            ],
            responses=[Message(**response) for response in responses],
            **kwargs,
        )
        session.add(chat)
        await session.commit()


# Synchronous function to insert chat completion
def sync_insert_chat_completion(
    engine,
    messages: list[dict],
    responses: list[dict] = [],
    **kwargs,
):
    with Session(engine) as session:
        print(responses[0])

        chat = ChatCompletion(
            id=kwargs.pop("id", None),
            functions=json.dumps(kwargs.pop("functions", None)),
            function_call=json.dumps(kwargs.pop("function_call", None)),
            messages=[
                Message(
                    index=ii,
                    content=message["content"],
                    role=message["role"],
                    arguments=message.get("function_call", {}).get("arguments", None),
                    name=message.get("function_call", {}).get("name", None),
                )
                for (ii, message) in enumerate(messages)
            ],
            responses=[
                Message(
                    index=resp["index"],
                    content=resp.message.get("content", None),
                    role=resp.message.get("role", None),
                    arguments=resp.message.get("function_call", {}).get(
                        "arguments", None
                    ),
                    name=resp.message.get("function_call", {}).get("name", None),
                    is_response=True,
                )
                for resp in responses
            ],
            **kwargs,
        )
        session.add(chat)
        session.commit()


def patch_with_engine(engine):
    # Check if the engine is asynchronous
    if is_async_engine(engine):

        def save_chat_completion(messages: list[dict], responses: list[dict], **kwargs):
            asyncio.run(
                async_insert_chat_completion(engine, messages, responses, **kwargs)
            )

    else:

        def save_chat_completion(messages: list[dict], responses: list[dict], **kwargs):
            sync_insert_chat_completion(engine, messages, responses, **kwargs)

    def add_sql_alchemy(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)
        if is_async:

            @wraps(func)
            async def new_chatcompletion(*args, **kwargs):  # type: ignore
                response = await func(*args, **kwargs)
                save_chat_completion(
                    messages=kwargs.pop("messages", []),
                    responses=response.choices,
                    id=response["id"],
                    **response["usage"],
                    **kwargs,
                )
                return response

        else:

            @wraps(func)
            def new_chatcompletion(*args, **kwargs):
                response = func(*args, **kwargs)

                print(response)

                save_chat_completion(
                    messages=kwargs.pop("messages", []),
                    responses=response.choices,
                    id=response["id"],
                    **response["usage"],
                    **kwargs,
                )
                response._completion_id = response["id"]
                return response

        return new_chatcompletion

    return add_sql_alchemy


def instrument_with_sqlalchemy(engine):
    patcher = patch_with_engine(engine)
    original_chatcompletion = openai.ChatCompletion.create
    original_chatcompletion_async = openai.ChatCompletion.acreate
    openai.ChatCompletion.create = patcher(original_chatcompletion)
    openai.ChatCompletion.acreate = patcher(original_chatcompletion_async)
