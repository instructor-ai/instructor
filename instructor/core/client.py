from __future__ import annotations

import openai
import inspect
from functools import partial
import instructor
from ..utils.providers import Provider, get_provider
from openai.types.chat import ChatCompletionMessageParam
from typing import (
    TypeVar,
    Callable,
    overload,
    Union,
    Literal,
    Any,
    get_origin,
    get_args,
)
from tenacity import (
    AsyncRetrying,
    Retrying,
)
from collections.abc import Generator, Iterable, Awaitable, AsyncGenerator
from typing_extensions import Self
from pydantic import BaseModel
from ..dsl.partial import Partial
from .hooks import Hooks, HookName


T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class Response:
    def __init__(
        self,
        client: Instructor,
    ):
        self.client = client

    def create(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T] | None = None,
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T | Any:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return self.client.create(
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            strict=strict,
            messages=input,
            **kwargs,
        )

    def create_with_completion(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        **kwargs,
    ) -> tuple[T, Any]:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return self.client.create_with_completion(
            messages=input,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    def create_iterable(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        **kwargs,
    ) -> Generator[T, None, None]:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return self.client.create_iterable(
            messages=input,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    def create_partial(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        **kwargs,
    ) -> Generator[T, None, None]:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return self.client.create_partial(
            messages=input,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )


class AsyncResponse(Response):
    def __init__(self, client: AsyncInstructor):
        self.client = client

    async def create(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T] | None = None,
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T | Any:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return await self.client.create(
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            strict=strict,
            messages=input,
            **kwargs,
        )

    async def create_with_completion(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        **kwargs,
    ) -> tuple[T, Any]:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return await self.client.create_with_completion(
            messages=input,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )

    async def create_iterable(
        self,
        input: str | list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        **kwargs,
    ) -> AsyncGenerator[T, None]:
        if isinstance(input, str):
            input = [
                {
                    "role": "user",
                    "content": input,
                }
            ]

        return self.client.create_iterable(
            messages=input,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )


class Instructor:
    client: Any | None
    create_fn: Callable[..., Any]
    mode: instructor.Mode
    default_model: str | None = None
    provider: Provider
    hooks: Hooks

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        if mode == instructor.Mode.FUNCTIONS:
            instructor.Mode.warn_mode_functions_deprecation()

        self.kwargs = kwargs
        self.provider = provider
        self.hooks = hooks or Hooks()

        if mode in {
            instructor.Mode.RESPONSES_TOOLS,
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }:
            assert isinstance(client, (openai.OpenAI, openai.AsyncOpenAI))
            self.responses = Response(client=self)

    def on(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        self.hooks.on(hook_name, handler)

    def off(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        ),
        handler: Callable[[Any], None],
    ) -> None:
        self.hooks.off(hook_name, handler)

    def clear(
        self,
        hook_name: (
            HookName
            | Literal[
                "completion:kwargs",
                "completion:response",
                "completion:error",
                "completion:last_attempt",
                "parse:error",
            ]
        )
        | None = None,
    ) -> None:
        self.hooks.clear(hook_name)

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    @property
    def messages(self) -> Self:
        return self

    @overload
    def create(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Awaitable[T]: ...

    @overload
    def create(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self: AsyncInstructor,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Awaitable[Any]: ...

    @overload
    def create(
        self: Self,
        response_model: None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Any: ...

    def create(
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> T | Any | Awaitable[T] | Awaitable[Any]:
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        return self.create_fn(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    @overload
    def create_partial(
        self: AsyncInstructor,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,  # {{ edit_1 }}
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_partial(
        self: Self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_partial(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        kwargs["stream"] = True

        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        response_model = instructor.Partial[response_model]  # type: ignore
        return self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    @overload
    def create_iterable(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]: ...

    @overload
    def create_iterable(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None]: ...

    def create_iterable(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Generator[T, None, None] | AsyncGenerator[T, None]:
        kwargs["stream"] = True
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        response_model = Iterable[response_model]  # type: ignore
        return self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    @overload
    def create_with_completion(
        self: AsyncInstructor,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> Awaitable[tuple[T, Any]]: ...

    @overload
    def create_with_completion(
        self: Self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    def create_with_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | Retrying | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any] | Awaitable[tuple[T, Any]]:
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        model = self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            context=context,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )
        return model, model._raw_response

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Handle and process keyword arguments for the API call.

        This method merges the provided kwargs with the default kwargs stored in the instance.
        It ensures that any kwargs passed to the method call take precedence over the default ones.
        """
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs

    def __getattr__(self, attr: str) -> Any:
        if attr not in {"create", "chat", "messages"}:
            return getattr(self.client, attr)

        return getattr(self, attr)


class AsyncInstructor(Instructor):
    client: Any | None
    create_fn: Callable[..., Any]
    mode: instructor.Mode
    default_model: str | None = None
    provider: Provider
    hooks: Hooks

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs
        self.provider = provider
        self.hooks = hooks or Hooks()

        if mode in {
            instructor.Mode.RESPONSES_TOOLS,
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }:
            assert isinstance(client, (openai.OpenAI, openai.AsyncOpenAI))
            self.responses = AsyncResponse(client=self)

    async def create(  # type: ignore[override]
        self,
        response_model: type[T] | None,
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> T | Any:
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        # Check if the response model is an iterable type
        if (
            get_origin(response_model) in {Iterable}
            and get_args(response_model)
            and get_args(response_model)[0] is not None
            and self.mode
            not in {
                instructor.Mode.PARALLEL_TOOLS,
                instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
                instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
            }
        ):
            return self.create_iterable(
                messages=messages,
                response_model=get_args(response_model)[0],
                max_retries=max_retries,
                validation_context=validation_context,
                context=context,
                strict=strict,
                hooks=hooks,  # Pass the per-call hooks to create_iterable
                **kwargs,
            )

        return await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )

    async def create_partial(  # type: ignore[override]
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        async for item in await self.create_fn(
            response_model=instructor.Partial[response_model],  # type: ignore
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        ):
            yield item

    async def create_iterable(  # type: ignore[override]
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[T, None]:
        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        async for item in await self.create_fn(
            response_model=Iterable[response_model],
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        ):
            yield item

    async def create_with_completion(  # type: ignore[override]
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[T],
        max_retries: int | AsyncRetrying = 3,
        validation_context: dict[str, Any] | None = None,  # Deprecate in 2.0
        context: dict[str, Any] | None = None,
        strict: bool = True,
        hooks: Hooks | None = None,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        kwargs = self.handle_kwargs(kwargs)

        # Combine client hooks with per-call hooks
        combined_hooks = self.hooks
        if hooks is not None:
            combined_hooks = self.hooks + hooks

        response = await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            context=context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            hooks=combined_hooks,
            **kwargs,
        )
        return response, response._raw_response


@overload
def from_openai(
    client: openai.OpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor:
    pass


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor:
    pass


def map_chat_completion_to_response(messages, client, *args, **kwargs) -> Any:
    return client.responses.create(
        *args,
        input=messages,
        **kwargs,
    )


async def async_map_chat_completion_to_response(
    messages, client, *args, **kwargs
) -> Any:
    return await client.responses.create(
        *args,
        input=messages,
        **kwargs,
    )


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    if hasattr(client, "base_url"):
        provider = get_provider(str(client.base_url))
    else:
        provider = Provider.OPENAI

    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        import warnings

        warnings.warn(
            "Client should be an instance of openai.OpenAI or openai.AsyncOpenAI. Unexpected behavior may occur with other client types.",
            stacklevel=2,
        )

    if provider in {Provider.OPENROUTER}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS,
            instructor.Mode.JSON,
        }

    if provider in {Provider.ANYSCALE, Provider.TOGETHER}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.JSON_SCHEMA,
            instructor.Mode.MD_JSON,
        }

    if provider in {Provider.OPENAI, Provider.DATABRICKS}:
        assert mode in {
            instructor.Mode.TOOLS,
            instructor.Mode.JSON,
            instructor.Mode.FUNCTIONS,
            instructor.Mode.PARALLEL_TOOLS,
            instructor.Mode.MD_JSON,
            instructor.Mode.TOOLS_STRICT,
            instructor.Mode.JSON_O1,
            instructor.Mode.RESPONSES_TOOLS,
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        }

    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=instructor.patch(
                create=(
                    client.chat.completions.create
                    if mode
                    not in {
                        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                        instructor.Mode.RESPONSES_TOOLS,
                    }
                    else partial(map_chat_completion_to_response, client=client)
                ),
                mode=mode,
            ),
            mode=mode,
            provider=provider,
            **kwargs,
        )

    if isinstance(client, openai.AsyncOpenAI):
        return AsyncInstructor(
            client=client,
            create=instructor.patch(
                create=(
                    client.chat.completions.create
                    if mode
                    not in {
                        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
                        instructor.Mode.RESPONSES_TOOLS,
                    }
                    else partial(async_map_chat_completion_to_response, client=client)
                ),
                mode=mode,
            ),
            mode=mode,
            provider=provider,
            **kwargs,
        )


@overload
def from_litellm(
    completion: Callable[..., Awaitable[Any]],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_litellm(
    completion: Callable[..., Any],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


def from_litellm(
    completion: Callable[..., Any] | Callable[..., Awaitable[Any]],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    is_async = inspect.iscoroutinefunction(completion)

    if not is_async:
        return Instructor(
            client=None,
            create=instructor.patch(create=completion, mode=mode),
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=None,
            create=instructor.patch(create=completion, mode=mode),
            mode=mode,
            **kwargs,
        )
