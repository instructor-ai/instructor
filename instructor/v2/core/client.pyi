from collections.abc import AsyncGenerator, Coroutine, Generator
from typing import Any, TypeVar, overload

from openai.types.chat import ChatCompletionMessageParam
T = TypeVar("T")

class Response:
    @overload
    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> T: ...
    @overload
    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> Any: ...
    @overload
    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...
    @overload
    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> tuple[Any, Any]: ...
    @overload
    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...
    @overload
    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> Generator[Any, None, None]: ...
    @overload
    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...
    @overload
    def create_partial(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> Generator[Any, None, None]: ...

class AsyncResponse(Response):
    @overload
    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, T]: ...
    @overload
    def create(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, Any]: ...
    @overload
    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, tuple[T, Any]]: ...
    @overload
    def create_with_completion(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, tuple[Any, Any]]: ...
    @overload
    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: type[T] = ...,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, AsyncGenerator[T, None]]: ...
    @overload
    def create_iterable(
        self,
        messages: str | list[ChatCompletionMessageParam] | None = ...,
        response_model: None = ...,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, AsyncGenerator[Any, None]]: ...

class Instructor:
    chat: Any
    def create_iterable(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...
    def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

class AsyncInstructor(Instructor):
    def create_iterable(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...
    def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...
