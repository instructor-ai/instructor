from __future__ import annotations
from typing import Any, Union, Literal, overload
from instructor.client import AsyncInstructor, Instructor
import instructor
from instructor.mode import Mode

# Type alias for the return type
InstructorType = Union[Instructor, AsyncInstructor]

# Default modes for each provider
DEFAULT_MODES: dict[str, Mode] = {
    "openai": Mode.OPENAI_FUNCTIONS,
    "anthropic": Mode.ANTHROPIC_TOOLS,
    "google": Mode.GEMINI_JSON,
    "mistral": Mode.MISTRAL_TOOLS,
    "cohere": Mode.COHERE_TOOLS,
    "perplexity": Mode.JSON,
    "groq": Mode.GROQ_TOOLS,
    "writer": Mode.WRITER_JSON,
    "bedrock": Mode.ANTHROPIC_TOOLS,  # Default for Claude on Bedrock
    "cerebras": Mode.CEREBRAS_TOOLS,
    "fireworks": Mode.TOOLS,
    "vertexai": Mode.VERTEXAI_TOOLS,
    "genai": Mode.GENAI_STRUCTURED,
}


@overload
def from_provider(
    model: str, async_client: Literal[True], **kwargs: Any
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: str, async_client: Literal[False], **kwargs: Any
) -> Instructor: ...


def from_provider(
    model: str,
    async_client: bool = False,
    mode: instructor.Mode | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor client from a model string.

    Args:
        model: String in format "provider/model-name"
              (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro")
        async_client: Whether to return an async client
        mode: Override the default mode for the provider. If not specified, uses the
              recommended default mode for each provider.
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        Instructor or AsyncInstructor instance

    Raises:
        ValueError: If provider is not supported or model string is invalid
        ImportError: If required package for provider is not installed

    Examples:
        >>> import instructor
        >>> # Sync clients
        >>> client = instructor.from_provider("openai/gpt-4")
        >>> client = instructor.from_provider("anthropic/claude-3-sonnet")
        >>> # Async clients
        >>> async_client = instructor.from_provider("openai/gpt-4", async_client=True)
    """
    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        format_err = ValueError(
            'Model string must be in format "provider/model-name" '
            '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
        )
        raise format_err from None

    # Use default mode if not specified
    if mode is None and provider in DEFAULT_MODES:
        mode = DEFAULT_MODES[provider]

    if provider == "openai":
        try:
            import openai
            from instructor import from_openai

            client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
            return from_openai(client, model=model_name, mode=mode, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The openai package is required to use the OpenAI provider. "
                "Install it with `pip install openai`."
            )
            raise import_err from None

    elif provider == "anthropic":
        try:
            import anthropic
            from instructor import from_anthropic

            client = (
                anthropic.AsyncAnthropic() if async_client else anthropic.Anthropic()
            )
            return from_anthropic(client, model=model_name, mode=mode, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The anthropic package is required to use the Anthropic provider. "
                "Install it with `pip install anthropic`."
            )
            raise import_err from None

    elif provider == "google":
        import google.generativeai as genai
        from instructor import from_gemini

        client = genai.GenerativeModel(model=model_name)
        return from_gemini(client, model=model_name, use_async=async_client, **kwargs)

    elif provider == "mistral":
        from mistralai import MistralClient, AsyncMistralClient
        from instructor import from_mistral

        client = AsyncMistralClient() if async_client else MistralClient()
        return from_mistral(client, use_async=async_client, **kwargs)

    elif provider == "cohere":
        import cohere
        from instructor import from_cohere

        client = cohere.AsyncClient() if async_client else cohere.Client()
        return from_cohere(client, **kwargs)

    elif provider == "perplexity":
        import openai
        from instructor import from_perplexity

        client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
        return from_perplexity(client, **kwargs)

    elif provider == "groq":
        import groq
        from instructor import from_groq

        client = groq.AsyncGroq() if async_client else groq.Groq()
        return from_groq(client, **kwargs)

    elif provider == "writer":
        from writerai import AsyncWriter, Writer
        from instructor import from_writer

        client = AsyncWriter() if async_client else Writer()
        return from_writer(client, **kwargs)

    elif provider == "bedrock":
        import boto3
        from instructor import from_bedrock

        client = boto3.client("bedrock-runtime")
        return from_bedrock(client, **kwargs)

    elif provider == "cerebras":
        from cerebras.cloud.sdk import AsyncCerebras, Cerebras
        from instructor import from_cerebras

        client = AsyncCerebras() if async_client else Cerebras()
        return from_cerebras(client, **kwargs)

    elif provider == "fireworks":
        from fireworks.client import AsyncFireworks, Fireworks
        from instructor import from_fireworks

        client = AsyncFireworks() if async_client else Fireworks()
        return from_fireworks(client, **kwargs)

    elif provider == "vertexai":
        import vertexai.generative_models as gm
        from instructor import from_vertexai

        client = gm.GenerativeModel(model=model_name)
        return from_vertexai(client, _async=async_client, **kwargs)

    elif provider == "genai":
        from google.genai import Client
        from instructor import from_genai

        client = Client()
        return from_genai(client, use_async=async_client, **kwargs)

    else:
        supported_providers = ", ".join(sorted(DEFAULT_MODES.keys()))
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )
