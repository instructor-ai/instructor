"""v2 Azure OpenAI / AI Foundry client factory.

Azure OpenAI is fully OpenAI-compatible, so this factory delegates to the
shared ``_from_openai_compat`` helper with ``Provider.AZURE_OPENAI``.
This enables ``from_azure()`` and ``from_provider("azure/...")`` ergonomics,
automatic ``AZURE_OPENAI_*`` env detection, and proper provider attribution.
"""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor.v2.core.client import AsyncInstructor, Instructor
from instructor.v2.core.mode import Mode
from instructor.v2.core.providers import Provider
from instructor.v2.providers.openai.client import _from_openai_compat

# Ensure OpenAI handlers are registered (decorators auto-register on import).
from instructor.v2.providers.openai import handlers  # noqa: F401


@overload
def from_azure(
    client: openai.AzureOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_azure(
    client: openai.AsyncAzureOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_azure(
    client: openai.AzureOpenAI | openai.AsyncAzureOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an Azure OpenAI client.

    Azure OpenAI (and Azure AI Foundry-hosted models) uses an
    OpenAI-compatible API, so all OpenAI modes are supported:
    ``TOOLS``, ``JSON``, ``JSON_SCHEMA``, ``MD_JSON``, ``PARALLEL_TOOLS``.

    Args:
        client: An ``openai.AzureOpenAI`` or ``openai.AsyncAzureOpenAI`` instance.
            Also accepts plain ``openai.OpenAI``/``AsyncOpenAI`` for Foundry
            endpoints that use key-based auth without the Azure SDK wrapper.
        mode: The structured-output mode to use. Defaults to ``Mode.TOOLS``.
        model: Optional default model name (deployment name for Azure).
        **kwargs: Additional keyword arguments forwarded to the Instructor constructor.

    Returns:
        An ``Instructor`` or ``AsyncInstructor`` instance.

    Examples:
        >>> from openai import AzureOpenAI
        >>> import instructor
        >>> client = AzureOpenAI(
        ...     api_key="your-key",
        ...     api_version="2024-02-01",
        ...     azure_endpoint="https://my-resource.openai.azure.com",
        ... )
        >>> ic = instructor.from_azure(client)
    """
    return _from_openai_compat(
        client=client,
        provider=Provider.AZURE_OPENAI,
        mode=mode,
        model=model,
        **kwargs,
    )


__all__ = ["from_azure"]
