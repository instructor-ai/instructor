import inspect
import json
import logging
from collections.abc import Iterable
from functools import wraps
from tenacity import Retrying, AsyncRetrying, stop_after_attempt, RetryError
from json import JSONDecodeError
from typing import (
    Callable,
    Optional,
    ParamSpec,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, ValidationError

from instructor.dsl.iterable import IterableModel, IterableBase
from instructor.dsl.parallel import ParallelBase, ParallelModel, handle_parallel_model
from instructor.dsl.partial import PartialBase

from .function_calls import Mode, OpenAISchema, openai_schema

logger = logging.getLogger("instructor")
T = TypeVar("T")


T_Model = TypeVar("T_Model", bound=BaseModel)
T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


def dump_message(message: ChatCompletionMessage) -> ChatCompletionMessageParam:
    """Dumps a message to a dict, to be returned to the OpenAI API.
    Workaround for an issue with the OpenAI API, where the `tool_calls` field isn't allowed to be present in requests
    if it isn't used.
    """
    ret: ChatCompletionMessageParam = {
        "role": message.role,
        "content": message.content or "",
    }
    if hasattr(message, "tool_calls") and message.tool_calls is not None:
        ret["tool_calls"] = message.model_dump()["tool_calls"]
    if hasattr(message, "function_call") and message.function_call is not None:
        ret["content"] += json.dumps(message.model_dump()["function_call"])
    return ret


def handle_response_model(
    response_model: T, mode: Mode = Mode.TOOLS, **kwargs
) -> Union[Type[OpenAISchema], dict]:
    """Prepare the response model type hint, and returns the response_model
    along with the new modified kwargs needed to be able to use the response_model
    parameter with the patch function.


    Args:
        response_model (T): The response model to use for parsing the response
        mode (Mode, optional): The openai completion mode. Defaults to Mode.TOOLS.

    Raises:
        NotImplementedError: When using stream=True with a non-iterable response_model
        ValueError: When using an invalid patch mode

    Returns:
        Union[Type[OpenAISchema], dict]: The response model to use for parsing the response
    """
    new_kwargs = kwargs.copy()
    if response_model is not None:
        # This a special case for parallel tools
        if mode == Mode.PARALLEL_TOOLS:
            assert (
                new_kwargs.get("stream", False) is False
            ), "stream=True is not supported when using PARALLEL_TOOLS mode"
            new_kwargs["tools"] = handle_parallel_model(response_model)
            new_kwargs["tool_choice"] = "auto"

            # This is a special case for parallel models
            response_model = ParallelModel(typehint=response_model)
            return response_model, new_kwargs

        # This is for all other single model cases
        if get_origin(response_model) is Iterable:
            iterable_element_class = get_args(response_model)[0]
            response_model = IterableModel(iterable_element_class)
        if not issubclass(response_model, OpenAISchema):
            response_model = openai_schema(response_model)  # type: ignore

        if new_kwargs.get("stream", False) and not issubclass(
            response_model, (IterableBase, PartialBase)
        ):
            raise NotImplementedError(
                "stream=True is not supported when using response_model parameter for non-iterables"
            )

        if mode == Mode.FUNCTIONS:
            new_kwargs["functions"] = [response_model.openai_schema]  # type: ignore
            new_kwargs["function_call"] = {"name": response_model.openai_schema["name"]}  # type: ignore
        elif mode == Mode.TOOLS:
            new_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": response_model.openai_schema,
                }
            ]
            new_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": response_model.openai_schema["name"]},
            }
        elif mode in {Mode.JSON, Mode.MD_JSON, Mode.JSON_SCHEMA}:
            # If its a JSON Mode we need to massage the prompt a bit
            # in order to get the response we want in a json format
            message = f"""
                As a genius expert, your task is to understand the content and provide
                the parsed objects in json that match the following json_schema:\n
                {response_model.model_json_schema()['properties']}
                """
            # Check for nested models
            if "$defs" in response_model.model_json_schema():
                message += f"\nHere are some more definitions to adhere too:\n{response_model.model_json_schema()['$defs']}"

            if mode == Mode.JSON:
                new_kwargs["response_format"] = {"type": "json_object"}

            elif mode == Mode.JSON_SCHEMA:
                new_kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": response_model.model_json_schema(),
                }

            elif mode == Mode.MD_JSON:
                new_kwargs["messages"].append(
                    {
                        "role": "assistant",
                        "content": "Here is the perfectly correctly formatted JSON\n```json",
                    },
                )
                new_kwargs["stop"] = "```"
            # check that the first message is a system message
            # if it is not, add a system message to the beginning
            if new_kwargs["messages"][0]["role"] != "system":
                new_kwargs["messages"].insert(
                    0,
                    {
                        "role": "system",
                        "content": message,
                    },
                )

            # if the first message is a system append the schema to the end
            if new_kwargs["messages"][0]["role"] == "system":
                new_kwargs["messages"][0]["content"] += f"\n\n{message}"
        else:
            raise ValueError(f"Invalid patch mode: {mode}")
    return response_model, new_kwargs


def process_response(
    response: T,
    *,
    response_model: Type[T_Model],
    stream: bool,
    validation_context: dict = None,
    strict=None,
    mode: Mode = Mode.FUNCTIONS,
) -> Union[T_Model, T]:
    """Processes a OpenAI response with the response model, if available.

    Args:
        response (T): The response from OpenAI's API
        response_model (Type[T_Model]): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (_type_, optional): Whether to use strict json parsing. Defaults to None.
        mode (Mode, optional): The openai completion mode. Defaults to Mode.FUNCTIONS.

    Returns:
        Union[T_Model, T]: The parsed response, if a response model is available, otherwise the response as is from the SDK
    """
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = response_model.from_streaming_response(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        return model

    model._raw_response = response
    return model


async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: Type[T_Model],
    stream: bool = False,
    validation_context: dict = None,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T:
    """Processes a OpenAI response with the response model, if available.
    It can use `validation_context` and `strict` to validate the response
    via the pydantic model

    Args:
        response (ChatCompletion): The response from OpenAI's API
        response_model (BaseModel): The response model to use for parsing the response
        stream (bool): Whether the response is a stream
        validation_context (dict, optional): The validation context to use for validating the response. Defaults to None.
        strict (bool, optional): Whether to use strict json parsing. Defaults to None.
    """
    if response_model is None:
        return response

    if (
        inspect.isclass(response_model)
        and issubclass(response_model, (IterableBase, PartialBase))
        and stream
    ):
        model = await response_model.from_streaming_response_async(
            response,
            mode=mode,
        )
        return model

    model = response_model.from_response(
        response,
        validation_context=validation_context,
        strict=strict,
        mode=mode,
    )

    # ? This really hints at the fact that we need a better way of
    # ? attaching usage data and the raw response to the model we return.
    if isinstance(model, IterableBase):
        #! If the response model is a multitask, return the tasks
        return [task for task in model.tasks]

    if isinstance(response_model, ParallelBase):
        return model

    model._raw_response = response
    return model


async def retry_async(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: Type[T],
    validation_context,
    args,
    kwargs,
    max_retries: int | AsyncRetrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> T:
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    # If max_retries is int, then create a AsyncRetrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries = AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (AsyncRetrying, Retrying)):
        raise ValueError(
            "max_retries must be an `int` or a `tenacity.AsyncRetrying` object"
        )

    try:
        async for attempt in max_retries:
            logger.debug(f"Retrying, attempt: {attempt}")
            with attempt:
                try:
                    response: ChatCompletion = await func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    if (
                        isinstance(response, ChatCompletion)
                        and response.usage is not None
                    ):
                        total_usage.completion_tokens += (
                            response.usage.completion_tokens or 0
                        )
                        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                        total_usage.total_tokens += response.usage.total_tokens or 0
                        response.usage = total_usage  # Replace each response usage with the total usage
                    return await process_response_async(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}")
                    kwargs["messages"].append(dump_message(response.choices[0].message))  # type: ignore
                    if mode == Mode.TOOLS:
                        kwargs["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": response.choices[0]
                                .message.tool_calls[0]
                                .id,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                                "content": "failure",
                            }
                        )
                    kwargs["messages"].append(
                        {
                            "role": "user",
                            "content": f"Recall the function correctly, fix the errors, exceptions found\n{e}",
                        }
                    )
                    if mode == Mode.MD_JSON:
                        kwargs["messages"].append(
                            {
                                "role": "assistant",
                                "content": "```json",
                            },
                        )
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e


def retry_sync(
    func: Callable[T_ParamSpec, T_Retval],
    response_model: Type[T],
    validation_context: dict,
    args,
    kwargs,
    max_retries: int | Retrying = 1,
    strict: Optional[bool] = None,
    mode: Mode = Mode.FUNCTIONS,
):
    total_usage = CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    # If max_retries is int, then create a Retrying object
    if isinstance(max_retries, int):
        logger.debug(f"max_retries: {max_retries}")
        max_retries: Retrying = Retrying(
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
    if not isinstance(max_retries, (Retrying, AsyncRetrying)):
        raise ValueError("max_retries must be an int or a `tenacity.Retrying` object")

    try:
        for attempt in max_retries:
            with attempt:
                try:
                    response = func(*args, **kwargs)
                    stream = kwargs.get("stream", False)
                    if (
                        isinstance(response, ChatCompletion)
                        and response.usage is not None
                    ):
                        total_usage.completion_tokens += (
                            response.usage.completion_tokens or 0
                        )
                        total_usage.prompt_tokens += response.usage.prompt_tokens or 0
                        total_usage.total_tokens += response.usage.total_tokens or 0
                        response.usage = total_usage  # Replace each response usage with the total usage
                    return process_response(
                        response,
                        response_model=response_model,
                        stream=stream,
                        validation_context=validation_context,
                        strict=strict,
                        mode=mode,
                    )
                except (ValidationError, JSONDecodeError) as e:
                    logger.debug(f"Error response: {response}")
                    kwargs["messages"].append(dump_message(response.choices[0].message))
                    # ! How do we handle this for parallel tools in the future?
                    if mode == Mode.TOOLS:
                        kwargs["messages"].append(
                            {
                                "role": "tool",
                                "tool_call_id": response.choices[0]
                                .message.tool_calls[0]
                                .id,
                                "name": response.choices[0]
                                .message.tool_calls[0]
                                .function.name,
                                "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                            }
                        )
                    else:
                        kwargs["messages"].append(
                            {
                                "role": "user",
                                "content": f"Recall the function correctly, fix the errors and exceptions found\n{e}",
                            }
                        )
                    if mode == Mode.MD_JSON:
                        kwargs["messages"].append(
                            {
                                "role": "assistant",
                                "content": "```json",
                            },
                        )
                    raise e
    except RetryError as e:
        logger.exception(f"Failed after retries: {e.last_attempt.exception}")
        raise e.last_attempt.exception from e


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    return inspect.iscoroutinefunction(func) or (
        hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(func.__wrapped__)
    )


OVERRIDE_DOCS = """
Creates a new chat completion for the provided messages and parameters.

See: https://platform.openai.com/docs/api-reference/chat-completions/create

Additional Notes:

Using the `response_model` parameter, you can specify a response model to use for parsing the response from OpenAI's API. If its present, the response will be parsed using the response model, otherwise it will be returned as is.

If `stream=True` is specified, the response will be parsed using the `from_stream_response` method of the response model, if available, otherwise it will be parsed using the `from_response` method.

If need to obtain the raw response from OpenAI's API, you can access it using the `_raw_response` attribute of the response model. The `_raw_response.usage` attribute is modified to reflect the token usage from the last successful response as well as from any previous unsuccessful attempts.

Parameters:
    response_model (Union[Type[BaseModel], Type[OpenAISchema]]): The response model to use for parsing the response from OpenAI's API, if available (default: None)
    max_retries (int): The maximum number of retries to attempt if the response is not valid (default: 0)
    validation_context (dict): The validation context to use for validating the response (default: None)
"""


class InstructorChatCompletionCreate(Protocol):
    def __call__(
        self,
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        ...


@overload
def patch(
    client: OpenAI,
    mode: Mode = Mode.FUNCTIONS,
) -> OpenAI:
    ...


@overload
def patch(
    client: AsyncOpenAI,
    mode: Mode = Mode.FUNCTIONS,
) -> AsyncOpenAI:
    ...


@overload
def patch(
    create: Callable[T_ParamSpec, T_Retval],
    mode: Mode = Mode.FUNCTIONS,
) -> InstructorChatCompletionCreate:
    ...


def patch(
    client: Union[OpenAI, AsyncOpenAI] = None,
    create: Callable[T_ParamSpec, T_Retval] = None,
    mode: Mode = Mode.FUNCTIONS,
) -> Union[OpenAI, AsyncOpenAI]:
    """
    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """

    logger.debug(f"Patching `client.chat.completions.create` with {mode=}")

    if create is not None:
        func = create
    elif client is not None:
        func = client.chat.completions.create
    else:
        raise ValueError("Either client or create must be provided")

    func_is_async = is_async(func)

    @wraps(func)
    async def new_create_async(
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
        response = await retry_async(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )  # type: ignore
        return response

    @wraps(func)
    def new_create_sync(
        response_model: Type[T_Model] = None,
        validation_context: dict = None,
        max_retries: int = 1,
        *args: T_ParamSpec.args,
        **kwargs: T_ParamSpec.kwargs,
    ) -> T_Model:
        response_model, new_kwargs = handle_response_model(
            response_model=response_model, mode=mode, **kwargs
        )
        response = retry_sync(
            func=func,
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            args=args,
            kwargs=new_kwargs,
            mode=mode,
        )
        return response

    new_create = new_create_async if func_is_async else new_create_sync
    new_create.__doc__ = OVERRIDE_DOCS

    if client is not None:
        client.chat.completions.create = new_create
        return client
    else:
        return new_create


def apatch(client: AsyncOpenAI, mode: Mode = Mode.FUNCTIONS):
    """
    No longer necessary, use `patch` instead.

    Patch the `client.chat.completions.create` method

    Enables the following features:

    - `response_model` parameter to parse the response from OpenAI's API
    - `max_retries` parameter to retry the function if the response is not valid
    - `validation_context` parameter to validate the response using the pydantic model
    - `strict` parameter to use strict json parsing
    """
    import warnings

    warnings.warn(
        "apatch is deprecated, use patch instead", DeprecationWarning, stacklevel=2
    )
    return patch(client, mode=mode)
