from typing import List, Callable
from openai.types.chat import ChatCompletionMessageParam
from abc import ABC, abstractmethod
from pydantic import BaseModel


class MessageMiddleware(BaseModel, ABC):

    @abstractmethod
    def __call__(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        pass


class AsyncMessageMiddleware(MessageMiddleware):

    @abstractmethod
    async def __call__(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        pass


def messages_middleware(func: Callable) -> MessageMiddleware:

    class _Middleware(MessageMiddleware):
        def __call__(
            self, messages: List[ChatCompletionMessageParam]
        ) -> List[ChatCompletionMessageParam]:
            return func(messages)

    return _Middleware()
