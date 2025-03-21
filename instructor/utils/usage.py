import logging
from typing import TYPE_CHECKING, TypeVar, Protocol, Union, Optional

from openai.types import CompletionUsage as OpenAIUsage

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage


logger = logging.getLogger("instructor")
T_Model = TypeVar("T_Model", bound="Response")

T = TypeVar("T")


class Response(Protocol):
    usage: Union[OpenAIUsage, "AnthropicUsage"]


def update_total_usage(
    response: Optional[T_Model],
    total_usage: Optional[OpenAIUsage] = None,
) -> Optional[OpenAIUsage]:
    """Update the total usage with the response usage."""
    if total_usage is None:
        total_usage = OpenAIUsage(
            completion_tokens=0,
            prompt_tokens=0,
            total_tokens=0,
        )

    if response is None:
        return total_usage

    try:
        if isinstance(response.usage, OpenAIUsage):
            total_usage.completion_tokens += response.usage.completion_tokens
            total_usage.prompt_tokens += response.usage.prompt_tokens
            total_usage.total_tokens += response.usage.total_tokens
        else:
            # Anthropic usage
            total_usage.completion_tokens += response.usage.output_tokens
            total_usage.prompt_tokens += response.usage.input_tokens
            total_usage.total_tokens += (
                response.usage.output_tokens + response.usage.input_tokens
            )
    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to update usage: {e}")

    return total_usage
