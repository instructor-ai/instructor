import asyncio
from functools import wraps
import openai
import inspect
from typing import Callable, Optional, Type, Union

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

from sa import engine, ChatCompletion, Message


# Check if the engine is asynchronous
def is_async_engine(engine) -> bool:
    return isinstance(engine, AsyncEngine)


# Async function to insert chat completion
async def async_insert_chat_completion(
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
    messages: list[dict],
    responses: list[dict] = [],
    **kwargs,
):
    with Session(engine) as session:
        chat = ChatCompletion(
            messages=[
                Message(**message, index=ii) for (ii, message) in enumerate(messages)
            ],
            responses=[Message(**response) for response in responses],
            **kwargs,
        )
        session.add(chat)
        session.commit()


def patch_with_engine(engine):
    # Check if the engine is asynchronous
    if is_async_engine(engine):

        def save_chat_completion(
            messages: list[dict], responses: list[dict], **kwargs
        ):
            asyncio.run(async_insert_chat_completion(messages, responses, **kwargs))

    else:

        def save_chat_completion(
            messages: list[dict], responses: list[dict], **kwargs
        ):
            sync_insert_chat_completion(messages, responses, **kwargs)

    def add_sql_alchemy(func: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(func)
        if is_async:

            @wraps(func)
            async def new_chatcompletion(*args, **kwargs):  # type: ignore
                response = await func(*args, **kwargs)
                save_chat_completion(
                    messages=response.messages,
                    responses=response.responses,
                    **response["usage"],
                    **kwargs,
                )
                return response

        else:

            @wraps(func)
            def new_chatcompletion(*args, **kwargs):
                response = func(*args, **kwargs)
                save_chat_completion(
                    messages=response.messages,
                    responses=response.responses,
                    **response["usage"],
                    **kwargs,
                )
                return response

        return new_chatcompletion

    return add_sql_alchemy


original_chatcompletion = openai.ChatCompletion.create
original_chatcompletion_async = openai.ChatCompletion.acreate


def patch(engine):
    patcher = patch_with_engine(engine)
    openai.ChatCompletion.create = patcher(original_chatcompletion)
    openai.ChatCompletion.acreate = patcher(original_chatcompletion_async)


def unpatch():
    openai.ChatCompletion.create = original_chatcompletion
    openai.ChatCompletion.acreate = original_chatcompletion_async
