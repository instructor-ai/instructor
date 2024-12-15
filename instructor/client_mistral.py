# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

from typing import Any, Dict, Literal, Protocol, overload, TYPE_CHECKING, Callable, TypeVar, Union

if TYPE_CHECKING:
    from mistralai.client import MistralClient as Mistral
    from mistralai.models.chat_completion import ChatCompletionResponse
    from instructor import Instructor, AsyncInstructor
    from instructor.patch import patch, PatchedFunctionReturn
else:
    from mistralai import Mistral
    from instructor import Instructor, AsyncInstructor
    from instructor.patch import patch

from instructor.mode import Mode
from instructor.utils import Provider

T = TypeVar("T")
PatchFunction = Callable[..., Union[T, PatchedFunctionReturn]]

class MistralChatProtocol(Protocol):
    def complete(self, **kwargs: Dict[str, Any]) -> ChatCompletionResponse: ...
    async def complete_async(self, **kwargs: Dict[str, Any]) -> ChatCompletionResponse: ...

class MistralChat:
    client: Mistral
    complete: Callable[..., ChatCompletionResponse]
    complete_async: Callable[..., ChatCompletionResponse]

    def __init__(self, client: Mistral) -> None:
        self.client = client
        self.complete = client.chat.complete
        self.complete_async = client.chat.complete_async

@overload
def from_mistral(
    client: Mistral,
    mode: Mode = Mode.MISTRAL_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_mistral(
    client: Mistral,
    mode: Mode = Mode.MISTRAL_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_mistral(
    client: Mistral,
    mode: Mode = Mode.MISTRAL_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    assert mode in {
        Mode.MISTRAL_TOOLS,
        Mode.MISTRAL_JSON,
    }, f"Mode must be one of {Mode.MISTRAL_TOOLS}, {Mode.MISTRAL_JSON}"

    assert isinstance(
        client, Mistral
    ), "Client must be an instance of mistralai.Mistral"

    chat_client = MistralChat(client)

    if not use_async:
        return Instructor(
            client=client,
            create=patch(
                create=chat_client.complete,
                mode=mode,
                provider=Provider.MISTRAL,
            ),
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=client,
            create=patch(
                create=chat_client.complete_async,
                mode=mode,
                provider=Provider.MISTRAL,
            ),
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )
