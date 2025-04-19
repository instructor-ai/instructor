from __future__ import annotations
from typing import Any, Union, Literal, overload
from instructor.client import AsyncInstructor, Instructor
import instructor

# Type alias for the return type
InstructorType = Union[Instructor, AsyncInstructor]

# List of supported providers
supported_providers = [
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "perplexity",
    "groq",
    "writer",
    "bedrock",
    "cerebras",
    "fireworks",
    "vertexai",
    "genai",
]


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
    mode: instructor.Mode | None = None,  # noqa: ARG001
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

    if provider == "openai":
        try:
            import openai
            from instructor import from_openai

            client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
            return from_openai(client, model=model_name, **kwargs)
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
            return from_anthropic(client, model=model_name, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The anthropic package is required to use the Anthropic provider. "
                "Install it with `pip install anthropic`."
            )
            raise import_err from None

    elif provider == "google":
        try:
            import google.generativeai as genai  # type: ignore
            from instructor import from_gemini

            client = genai.GenerativeModel(model_name=model_name)  # type: ignore
            if async_client:
                return from_gemini(client, use_async=True, **kwargs)  # type: ignore
            else:
                return from_gemini(client, **kwargs)  # type: ignore
        except ImportError:
            import_err = ImportError(
                "The google-generativeai package is required to use the Google provider. "
                "Install it with `pip install google-generativeai`."
            )
            raise import_err from None

    elif provider == "mistral":
        try:
            from mistralai import MistralClient, AsyncMistralClient  # type: ignore
            from instructor import from_mistral

            client = AsyncMistralClient() if async_client else MistralClient()  # type: ignore
            if async_client:
                return from_mistral(client, use_async=True, **kwargs)
            else:
                return from_mistral(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The mistralai package is required to use the Mistral provider. "
                "Install it with `pip install mistralai`."
            )
            raise import_err from None

    elif provider == "cohere":
        try:
            import cohere
            from instructor import from_cohere

            client = cohere.AsyncClient() if async_client else cohere.Client()
            return from_cohere(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The cohere package is required to use the Cohere provider. "
                "Install it with `pip install cohere`."
            )
            raise import_err from None

    elif provider == "perplexity":
        try:
            import openai
            from instructor import from_perplexity

            client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
            return from_perplexity(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The openai package is required to use the Perplexity provider. "
                "Install it with `pip install openai`."
            )
            raise import_err from None

    elif provider == "groq":
        try:
            import groq
            from instructor import from_groq

            client = groq.AsyncGroq() if async_client else groq.Groq()
            return from_groq(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The groq package is required to use the Groq provider. "
                "Install it with `pip install groq`."
            )
            raise import_err from None

    elif provider == "writer":
        try:
            from writerai import AsyncWriter, Writer
            from instructor import from_writer

            client = AsyncWriter() if async_client else Writer()
            return from_writer(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The writerai package is required to use the Writer provider. "
                "Install it with `pip install writerai`."
            )
            raise import_err from None

    elif provider == "bedrock":
        try:
            import boto3
            from instructor import from_bedrock

            client = boto3.client("bedrock-runtime")
            return from_bedrock(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The boto3 package is required to use the AWS Bedrock provider. "
                "Install it with `pip install boto3`."
            )
            raise import_err from None

    elif provider == "cerebras":
        try:
            from cerebras.cloud.sdk import AsyncCerebras, Cerebras
            from instructor import from_cerebras

            client = AsyncCerebras() if async_client else Cerebras()
            return from_cerebras(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The cerebras package is required to use the Cerebras provider. "
                "Install it with `pip install cerebras`."
            )
            raise import_err from None

    elif provider == "fireworks":
        try:
            from fireworks.client import AsyncFireworks, Fireworks
            from instructor import from_fireworks

            client = AsyncFireworks() if async_client else Fireworks()
            return from_fireworks(client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The fireworks-ai package is required to use the Fireworks provider. "
                "Install it with `pip install fireworks-ai`."
            )
            raise import_err from None

    elif provider == "vertexai":
        try:
            import vertexai.generative_models as gm
            from instructor import from_vertexai

            client = gm.GenerativeModel(model_name=model_name)
            return from_vertexai(client, _async=async_client, **kwargs)
        except ImportError:
            import_err = ImportError(
                "The google-cloud-aiplatform package is required to use the VertexAI provider. "
                "Install it with `pip install google-cloud-aiplatform`."
            )
            raise import_err from None

    elif provider == "genai":
        try:
            from google.genai import Client
            from instructor import from_genai

            client = Client()
            if async_client:
                return from_genai(client, use_async=True, **kwargs)  # type: ignore
            else:
                return from_genai(client, **kwargs)  # type: ignore
        except ImportError:
            import_err = ImportError(
                "The google-genai package is required to use the Google GenAI provider. "
                "Install it with `pip install google-genai`."
            )
            raise import_err from None

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )
