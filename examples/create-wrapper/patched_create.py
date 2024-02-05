
import openai
import instructor
import functools

from typing import Coroutine, Any, Callable, Type, TypeVar, ParamSpec, Concatenate, cast

from openai import AsyncOpenAI
from instructor import Mode


T = TypeVar('T')
M = TypeVar('M')
P = ParamSpec('P')


def patched_create_delay_response_model(mode: Mode = Mode.FUNCTIONS, **client_kwargs: Any):
    """
    Creates a patched version of the `openai_client.chat.completions.create` method.
    This patched version requires the `response_model` to be provided as the first positional argument.
    The `response_model` is the type that the coroutine will return an instance of.

    Args:
        mode (instructor.Mode): The mode to use for the patched client.
        **client_kwargs: Arbitrary keyword arguments that will be passed to the `openai.AsyncClient` constructor.

    Returns:
        A callable that represents the patched `create` method. This callable can be awaited to obtain
        an instance of the specified `response_model`.
    """
    # Initialize the OpenAI async client with the provided keyword arguments
    _client = openai.AsyncClient(**client_kwargs)
    
    # Patch the client using `instructor.patch`
    client: AsyncOpenAI = instructor.patch(client=_client, mode=mode)

    # Get the original `create` method from the patched client
    original_create = client.chat.completions.create

    # Define a wrapper function that will concatenate the `response_model` as a positional arguments
    def create_wrapper(original_create: Callable[P, Coroutine[Any, Any, T]]) -> Callable[Concatenate[Type[M], P], Coroutine[Any, Any, M]]:
        """
        A wrapper coroutine for the `openai_client.chat.completions.create` method.

        This coroutine requires that the `response_model` is passed as the first argument `__p0` to the patched
        `create` method and returns an instance of the `response_model` via `instructor`.

        Args:
            response_model: The type of the model that the coroutine will return an instance of.
            *args: ParamSpec.args
            **kwargs: ParamSpec.kwargs
        Returns:
            An instance of the specified `response_model` via `instructor`.
        """
        async def wrapper(response_model: Type[M] | None = None, *args: P.args, **kwargs: P.kwargs) -> M:
            # Potential catch for future if `response_model` is passed as a keyword argument
            if untyped_response_model := kwargs.pop("response_model", response_model):
                response_model = cast(Type[M], untyped_response_model)
            
            # Cast the original function to accept the `response_model` as the first argument
            typed_original_create = cast(Callable[Concatenate[Type[M] | None, P], Coroutine[Any, Any, M]], original_create)

            # Call and await the function including the `response_model` as the first argument and return the result
            response_model_instance = await typed_original_create(response_model, *args, **kwargs)
            return response_model_instance
        
        # Return the wrapper coroutine function
        return wrapper
    
    # Return the wrapped `create` function
    return create_wrapper(original_create)





def patched_create(response_model: Type[M], mode: Mode = Mode.FUNCTIONS, **client_kwargs: Any):
    """
    Creates a patched version of the `openai_client.chat.completion.create` method.

    This function takes a `response_model` class and additional `client_kwargs`, and returns
    a new `create` function that automatically includes the `response_model` as the first
    argument when called. This allows the caller to use the `create` function without
    specifying the `response_model` each time.

    Args:
        response_model: The model class that the API response should be deserialized into.
        mode (instructor.Mode): The mode to use for the patched client.
        **client_kwargs: Additional keyword arguments to be passed to the `AsyncClient`.

    Returns:
        A coroutine function that, when awaited, sends a request to the OpenAI API and
        returns an instance of the `response_model` via `instructor`.
    """
    # Initialize the OpenAI async client with the provided keyword arguments
    _client = openai.AsyncClient(**client_kwargs)
    
    # Patch the client using `instructor.patch`
    client: AsyncOpenAI = instructor.patch(client=_client, mode=mode)
    
    # Get the original `create` method from the patched client
    original_create = client.chat.completions.create
    
    # Define a wrapper function that will prepend the `response_model` to the arguments
    def create_wrapper(original_create: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, M]]:
        # Cast the original function to accept the `response_model` as the first argument
        typed_original_create = cast(Callable[Concatenate[Type[M], P], Coroutine[Any, Any, M]], original_create)

        # Create a partial function that includes the `response_model` as the first argument
        partial_create = functools.partial(typed_original_create, response_model)

        # Cast the partial function back to the expected type
        typed_partial_create = cast(Callable[P, Coroutine[Any, Any, T]], partial_create)
        
        # Define the actual wrapper coroutine that will be called by the user
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> M:
            # Cast the partial function to return the `response_model`
            typed_partial_create_response = cast(Callable[P, Coroutine[Any, Any, M]], typed_partial_create)

            # Await the partial function with the provided arguments and return the result
            response_model_instance = await typed_partial_create_response(*args, **kwargs)
            return response_model_instance
        
        # Return the wrapper coroutine function
        return wrapper
    
    # Return the wrapped `create` function
    return create_wrapper(original_create)

