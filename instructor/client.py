import openai
import inspect
import instructor
import anthropic
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from anthropic.types import Message
from typing import (
    Type,
    TypeVar,
    Generator,
    Iterable,
    Tuple,
    Callable,
    List,
    overload,
    Union,
    Awaitable,
    AsyncGenerator,
    Any,
)
from typing_extensions import Self
from pydantic import BaseModel
from enum import Enum


T = TypeVar("T", bound=BaseModel)


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"


class Instructor:
    client: openai.OpenAI | anthropic.Anthropic | None
    create_fn: Any
    mode: instructor.Mode
    default_model: str | None = None

    def __init__(
        self,
        client: openai.OpenAI | anthropic.Anthropic | None,
        create: Callable,
        mode: instructor.Mode = instructor.Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        **kwargs,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs
        self.provider = provider

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    @property
    def messages(self) -> Self:
        return self

    def create(
        self,
        response_model: Type[T],
        messages: List[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict | None = None,
        **kwargs,
    ) -> T:
        kwargs = self.handle_kwargs(kwargs)

        return self.create_fn(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )

    def create_partial(
        self,
        response_model: Type[T],
        messages: List[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict | None = None,
        **kwargs,
    ) -> Generator[T, None, None]:
        assert self.provider != Provider.ANTHROPIC, "Anthropic doesn't support partial"

        kwargs["stream"] = True

        kwargs = self.handle_kwargs(kwargs)

        response_model = instructor.Partial[response_model]  # type: ignore
        return self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )

    def create_iterable(
        self,
        messages: List[ChatCompletionMessageParam],
        response_model: Type[T],
        max_retries: int = 3,
        validation_context: dict | None = None,
        **kwargs,
    ) -> Iterable[T]:
        assert self.provider != Provider.ANTHROPIC, "Anthropic doesn't support iterable"

        kwargs["stream"] = True
        kwargs = self.handle_kwargs(kwargs)

        response_model = Iterable[response_model]  # type: ignore
        return self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )

    def create_with_response(
        self,
        messages: List[ChatCompletionMessageParam],
        response_model: Type[T],
        max_retries: int = 3,
        validation_context: dict | None = None,
        **kwargs,
    ) -> Tuple[T, ChatCompletion | Message]:
        kwargs = self.handle_kwargs(kwargs)
        model = self.create_fn(
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            validation_context=validation_context,
            **kwargs,
        )
        return model, model._raw_response

    def handle_kwargs(self, kwargs: dict):
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs


class AsyncInstructor(Instructor):
    client: openai.AsyncOpenAI | anthropic.AsyncAnthropic | None
    create_fn: Any
    mode: instructor.Mode
    default_model: str | None = None

    def __init__(
        self,
        client: openai.AsyncOpenAI | anthropic.AsyncAnthropic | None,
        create: Callable,
        mode: instructor.Mode = instructor.Mode.TOOLS,
        **kwargs,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs

    async def create(
        self,
        messages: List[ChatCompletionMessageParam],
        response_model: Type[T],
        validation_context: dict | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> T:
        kwargs = self.handle_kwargs(kwargs)
        return await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            messages=messages,
            **kwargs,
        )

    async def create_partial(
        self,
        response_model: Type[T],
        messages: List[ChatCompletionMessageParam],
        validation_context: dict | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> AsyncGenerator[T, None]:
        assert self.provider != Provider.ANTHROPIC, "Anthropic doesn't support partial"

        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True
        async for item in await self.create_fn(
            response_model=instructor.Partial[response_model],
            validation_context=validation_context,
            max_retries=max_retries,
            messages=messages,
            **kwargs,
        ):
            yield item

    async def create_iterable(
        self,
        response_model: Type[T],
        messages: List[ChatCompletionMessageParam],
        validation_context: dict | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> AsyncGenerator[T, None]:
        assert self.provider != Provider.ANTHROPIC, "Anthropic doesn't support iterable"

        kwargs = self.handle_kwargs(kwargs)
        kwargs["stream"] = True
        async for item in await self.create_fn(
            response_model=Iterable[response_model],
            validation_context=validation_context,
            max_retries=max_retries,
            messages=messages,
            **kwargs,
        ):
            yield item

    async def create_with_response(
        self,
        response_model: Type[T],
        messages: List[ChatCompletionMessageParam],
        validation_context: dict | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> Tuple[T, dict]:
        kwargs = self.handle_kwargs(kwargs)
        response = await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            messages=messages,
            **kwargs,
        )
        return response, response._raw_response


@overload
def from_openai(
    client: openai.OpenAI, mode: instructor.Mode = instructor.Mode.TOOLS, **kwargs
) -> Instructor:
    pass


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs,
) -> AsyncInstructor:
    pass


def from_openai(
    client: Union[openai.OpenAI, openai.AsyncOpenAI],
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs,
) -> Instructor | AsyncInstructor:
    assert mode in {
        instructor.Mode.TOOLS,
        instructor.Mode.MD_JSON,
        instructor.Mode.JSON,
        instructor.Mode.FUNCTIONS,
    }, "Mode be one of {instructor.Mode.TOOLS, instructor.Mode.MD_JSON, instructor.Mode.JSON, instructor.Mode.FUNCTIONS}"

    assert isinstance(
        client, (openai.OpenAI, openai.AsyncOpenAI)
    ), "Client must be an instance of openai.OpenAI or openai.AsyncOpenAI"

    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create),
            mode=mode,
            **kwargs,
        )

    if isinstance(client, openai.AsyncOpenAI):
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.chat.completions.create),
            mode=mode,
            **kwargs,
        )


@overload
def from_litellm(
    completion: Callable,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> Instructor: ...


@overload
def from_litellm(
    completion: Awaitable,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> AsyncInstructor:
    pass


def from_litellm(
    completion: Callable | Awaitable,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> Instructor | AsyncInstructor:
    is_async = inspect.isawaitable(completion)

    if not is_async:
        return Instructor(
            client=None,
            create=instructor.patch(create=completion),
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=None,
            create=instructor.patch(create=completion),
            mode=mode,
            **kwargs,
        )


@overload
def from_anthropic(
    client: anthropic.Anthropic,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> Instructor: ...


@overload
def from_anthropic(
    client: anthropic.AsyncAnthropic,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_TOOLS,
    **kwargs,
) -> Instructor: ...


def from_anthropic(
    client: anthropic.Anthropic | anthropic.AsyncAnthropic,
    mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
    **kwargs,
) -> Instructor | AsyncInstructor:
    assert mode in {
        instructor.Mode.ANTHROPIC_JSON,
        instructor.Mode.ANTHROPIC_TOOLS,
    }, "Mode be one of {instructor.Mode.ANTHROPIC_JSON, instructor.Mode.ANTHROPIC_TOOLS}"

    assert isinstance(
        client, (anthropic.Anthropic, anthropic.AsyncAnthropic)
    ), "Client must be an instance of anthropic.Anthropic or anthropic.AsyncAnthropic"

    if isinstance(client, anthropic.Anthropic):
        return Instructor(
            client=client,
            create=instructor.patch(create=client.messages.create),
            mode=mode,
            **kwargs,
        )

    else:
        return AsyncInstructor(
            client=client,
            create=instructor.patch(create=client.messages.create),
            mode=mode,
            **kwargs,
        )