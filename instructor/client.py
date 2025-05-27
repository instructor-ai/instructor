from __future__ import annotations

import inspect
import warnings
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import instructor
from instructor.dsl.iterable import IterableModel
from instructor.dsl.partial import Partial
from instructor.exceptions import HookError
from instructor.function_calls import OpenAISchema
from instructor.hooks import Hook
from instructor.mode import Mode
from instructor.patch import apatch, patch
from instructor.utils import Provider, get_provider

T = TypeVar("T")
R = TypeVar("R")
M = TypeVar("M", bound=OpenAISchema)


class Response(Generic[T]):
    """A response from the API."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> T:
        """Create a response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API.
        """
        from instructor.process_response import process_response

        return process_response(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            stream=stream,
            **kwargs,
        )

    def create_with_completion(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Create a response from the API and return the completion.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API and the completion.
        """
        from instructor.process_response import process_response_with_completion

        return process_response_with_completion(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            stream=stream,
            **kwargs,
        )

    def create_iterable(
        self,
        model: Type[IterableModel[T]],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> IterableModel[T]:
        """Create an iterable response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The iterable response from the API.
        """
        from instructor.process_response import process_iterable_response

        return process_iterable_response(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )

    def create_partial(
        self,
        model: Type[Partial[T]],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Partial[T]:
        """Create a partial response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The partial response from the API.
        """
        from instructor.process_response import process_partial_response

        return process_partial_response(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )


class AsyncResponse(Generic[T]):
    """An async response from the API."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> T:
        """Create a response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API.
        """
        from instructor.process_response import process_response_async

        return await process_response_async(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            stream=stream,
            **kwargs,
        )

    async def create_with_completion(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Create a response from the API and return the completion.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API and the completion.
        """
        from instructor.process_response import process_response_with_completion_async

        return await process_response_with_completion_async(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            stream=stream,
            **kwargs,
        )

    async def create_iterable(
        self,
        model: Type[IterableModel[T]],
        messages: List[Dict[str, Any]],
        *,
        max_retries: int = 3,
        validation_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> IterableModel[T]:
        """Create an iterable response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            max_retries: The maximum number of retries to attempt.
            validation_context: The context to use for validation.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The iterable response from the API.
        """
        from instructor.process_response import process_iterable_response_async

        return await process_iterable_response_async(
            model=model,
            messages=messages,
            client=self.client,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )


class Instructor:
    """A client for the API."""

    def __init__(
        self,
        client: Any,
        create: Callable[..., Any],
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        **kwargs: Any,
    ) -> None:
        """Initialize the client.

        Args:
            client: The client to use for the API.
            create: The function to use for creating completions.
            mode: The mode to use for the client.
            provider: The provider to use for the client.
            **kwargs: Additional arguments to pass to the client.
        """
        self.client = client
        self.create = create
        self.mode = mode
        self.provider = provider
        self.kwargs = kwargs
        self.hooks: List[Hook] = []
        self.responses = Response(client)

    def on(self, hook: Hook) -> None:
        """Add a hook to the client.

        Args:
            hook: The hook to add.

        Raises:
            HookError: If the hook is already registered.
        """
        if hook in self.hooks:
            raise HookError(f"Hook {hook} already registered")
        self.hooks.append(hook)

    def off(self, hook: Hook) -> None:
        """Remove a hook from the client.

        Args:
            hook: The hook to remove.

        Raises:
            HookError: If the hook is not registered.
        """
        if hook not in self.hooks:
            raise HookError(f"Hook {hook} not registered")
        self.hooks.remove(hook)

    def clear(self) -> None:
        """Clear all hooks from the client."""
        self.hooks = []

    @overload
    def chat(self) -> Any: ...

    def chat(self) -> Any:
        return self.client.chat

    @overload
    def completions(self) -> Any: ...

    def completions(self) -> Any:
        return self.client.completions

    @overload
    def messages(self) -> Any: ...

    def messages(self) -> Any:
        return self.client.messages

    @overload
    def create(
        self, model: Type[T], messages: List[Dict[str, Any]], **kwargs: Any
    ) -> T: ...

    @overload
    def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: Literal[False],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: Literal[True],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: bool,
        **kwargs: Any,
    ) -> T: ...

    def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> T:
        """Create a response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API.
        """
        kwargs = self.handle_kwargs(kwargs)
        return self.responses.create(
            model=model, messages=messages, stream=stream, **kwargs
        )

    @overload
    def create_partial(
        self, model: Type[Partial[T]], messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Partial[T]: ...

    @overload
    def create_partial(
        self,
        model: Type[Partial[T]],
        messages: List[Dict[str, Any]],
        stream: Literal[False],
        **kwargs: Any,
    ) -> Partial[T]: ...

    def create_partial(
        self,
        model: Type[Partial[T]],
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Partial[T]:
        """Create a partial response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The partial response from the API.
        """
        kwargs = self.handle_kwargs(kwargs)
        return self.responses.create_partial(model=model, messages=messages, **kwargs)

    @overload
    def create_iterable(
        self,
        model: Type[IterableModel[T]],
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> IterableModel[T]: ...

    @overload
    def create_iterable(
        self,
        model: Type[IterableModel[T]],
        messages: List[Dict[str, Any]],
        stream: Literal[False],
        **kwargs: Any,
    ) -> IterableModel[T]: ...

    def create_iterable(
        self,
        model: Type[IterableModel[T]],
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> IterableModel[T]:
        """Create an iterable response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The iterable response from the API.
        """
        kwargs = self.handle_kwargs(kwargs)
        return self.responses.create_iterable(model=model, messages=messages, **kwargs)

    @overload
    def create_with_completion(
        self, model: Type[T], messages: List[Dict[str, Any]], **kwargs: Any
    ) -> tuple[T, Any]: ...

    @overload
    def create_with_completion(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: Literal[False],
        **kwargs: Any,
    ) -> tuple[T, Any]: ...

    def create_with_completion(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Create a response from the API and return the completion.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API and the completion.
        """
        kwargs = self.handle_kwargs(kwargs)
        return self.responses.create_with_completion(
            model=model, messages=messages, stream=stream, **kwargs
        )

    def handle_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the kwargs for the client.

        Args:
            kwargs: The kwargs to handle.

        Returns:
            The handled kwargs.
        """
        kwargs = {**self.kwargs, **kwargs}
        kwargs["hooks"] = self.hooks
        return kwargs

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the client.

        Args:
            name: The name of the attribute.

        Returns:
            The attribute.
        """
        return getattr(self.client, name)


class AsyncInstructor:
    """An async client for the API."""

    def __init__(
        self,
        client: Any,
        create: Callable[..., Awaitable[Any]],
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        **kwargs: Any,
    ) -> None:
        """Initialize the client.

        Args:
            client: The client to use for the API.
            create: The function to use for creating completions.
            mode: The mode to use for the client.
            provider: The provider to use for the client.
            **kwargs: Additional arguments to pass to the client.
        """
        self.client = client
        self.create = create
        self.mode = mode
        self.provider = provider
        self.kwargs = kwargs
        self.hooks: List[Hook] = []
        self.responses = AsyncResponse(client)

    async def create(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> T:
        """Create a response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API.
        """
        kwargs = {**self.kwargs, **kwargs}
        kwargs["hooks"] = self.hooks
        return await self.responses.create(
            model=model, messages=messages, stream=stream, **kwargs
        )

    async def create_partial(
        self,
        model: Type[Partial[T]],
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Partial[T]:
        """Create a partial response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The partial response from the API.
        """
        kwargs = {**self.kwargs, **kwargs}
        kwargs["hooks"] = self.hooks
        return await self.responses.create_partial(model=model, messages=messages, **kwargs)

    async def create_iterable(
        self,
        model: Type[IterableModel[T]],
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> IterableModel[T]:
        """Create an iterable response from the API.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The iterable response from the API.
        """
        kwargs = {**self.kwargs, **kwargs}
        kwargs["hooks"] = self.hooks
        return await self.responses.create_iterable(model=model, messages=messages, **kwargs)

    async def create_with_completion(
        self,
        model: Type[T],
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[T, Any]:
        """Create a response from the API and return the completion.

        Args:
            model: The model to use for the response.
            messages: The messages to send to the API.
            stream: Whether to stream the response.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            The response from the API and the completion.
        """
        kwargs = {**self.kwargs, **kwargs}
        kwargs["hooks"] = self.hooks
        return await self.responses.create_with_completion(
            model=model, messages=messages, stream=stream, **kwargs
        )


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
    """Create an Instructor instance from an OpenAI client.

    .. deprecated:: 0.5.0
       Use :func:`instructor.providers.openai.from_openai` instead.

    Args:
        client: An instance of OpenAI client (sync or async)
        mode: The mode to use for the client
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        AssertionError: If mode is not compatible with the provider
    """
    warnings.warn(
        "from_openai in client.py is deprecated. Use instructor.providers.openai.from_openai instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    # Import here to avoid circular imports
    from instructor.providers.openai import from_openai as new_from_openai
    
    return new_from_openai(client, mode, **kwargs)


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
