# Future imports to ensure compatibility with Python 3.9
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload, cast
from typing_extensions import ParamSpec
from mistralai.models.chat_completion import ChatCompletionResponse, ChatCompletionStreamResponse

from instructor import Mode, Provider
from instructor.client import ChatCompletionProtocol

if TYPE_CHECKING:
    from mistralai.client import MistralClient
    from instructor import Instructor, AsyncInstructor

T = TypeVar("T")
P = ParamSpec("P")

class MistralChatAdapter(ChatCompletionProtocol):
    """Adapter to make MistralClient compatible with ChatCompletionProtocol."""
    def __init__(self, client: MistralClient):
        self.client = client

    def create(self, *args: Any, **kwargs: Any) -> ChatCompletionResponse:
        return self.client.chat(*args, **kwargs)

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        response = self.client.chat_stream(*args, **kwargs)
        return cast(ChatCompletionStreamResponse, response)

@overload
def from_mistral(
    client: MistralClient,
    mode: Mode = Mode.MISTRAL_TOOLS,
    *,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> Instructor: ...

@overload
def from_mistral(
    client: MistralClient,
    mode: Mode = Mode.MISTRAL_TOOLS,
    *,
    use_async: Literal[True],
    **kwargs: Any,
) -> AsyncInstructor: ...

def from_mistral(
    client: MistralClient,
    mode: Mode = Mode.MISTRAL_TOOLS,
    use_async: bool = False,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from a Mistral client.

    Args:
        client: A Mistral client instance
        mode: The mode to use for function calling
        use_async: Whether to use async client
        **kwargs: Additional arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance configured with the Mistral client
    """
    assert mode in {
        Mode.MISTRAL_TOOLS,
    }, "Mode must be Mode.MISTRAL_TOOLS"

    # Create adapter that implements ChatCompletionProtocol
    adapted_client = MistralChatAdapter(client)

    if not use_async:
        return Instructor(
            client=adapted_client,
            create=adapted_client.create,
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=adapted_client,
            create=adapted_client.acreate,
            provider=Provider.MISTRAL,
            mode=mode,
            **kwargs,
        )
