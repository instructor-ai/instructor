from __future__ import annotations
from typing import Any, Union, Literal, overload
from .core.client import AsyncInstructor, Instructor
import instructor
from instructor.models import KnownModelName
from instructor.cache import BaseCache
import warnings
import logging

# Type alias for the return type
InstructorType = Union[Instructor, AsyncInstructor]

logger = logging.getLogger("instructor.auto_client")


# List of supported providers
supported_providers = [
    "openai",
    "azure_openai",
    "anthropic",
    "google",
    "generative-ai",
    "vertexai",
    "mistral",
    "cohere",
    "perplexity",
    "groq",
    "writer",
    "bedrock",
    "cerebras",
    "deepseek",
    "fireworks",
    "ollama",
    "openrouter",
    "xai",
    "litellm",
]


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: KnownModelName,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[True] = True,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> AsyncInstructor: ...


@overload
def from_provider(
    model: str,
    async_client: Literal[False] = False,
    cache: BaseCache | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> Instructor: ...


def from_provider(
    model: Union[str, KnownModelName],  # noqa: UP007
    async_client: bool = False,
    cache: BaseCache | None = None,
    mode: Union[instructor.Mode, None] = None,  # noqa: ARG001, UP007
    **kwargs: Any,
) -> Union[Instructor, AsyncInstructor]:  # noqa: UP007
    """Create an Instructor client from a model string.

    Args:
        model: String in format "provider/model-name"
              (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet", "google/gemini-pro")
        async_client: Whether to return an async client
        cache: Optional cache adapter (e.g., ``AutoCache`` or ``RedisCache``)
               to enable transparent response caching. Automatically flows through
               **kwargs to all provider implementations.
        mode: Override the default mode for the provider. If not specified, uses the
              recommended default mode for each provider.
        **kwargs: Additional arguments passed to the provider client functions.
                 This includes the cache parameter and any provider-specific options.

    Returns:
        Instructor or AsyncInstructor instance

    Raises:
        ValueError: If provider is not supported or model string is invalid
        ImportError: If required package for provider is not installed

    Examples:
        >>> import instructor
        >>> from instructor.cache import AutoCache
        >>>
        >>> # Basic usage
        >>> client = instructor.from_provider("openai/gpt-4")
        >>> client = instructor.from_provider("anthropic/claude-3-sonnet")
        >>>
        >>> # With caching
        >>> cache = AutoCache(maxsize=1000)
        >>> client = instructor.from_provider("openai/gpt-4", cache=cache)
        >>>
        >>> # Async clients
        >>> async_client = instructor.from_provider("openai/gpt-4", async_client=True)
    """
    # Add cache to kwargs if provided so it flows through to provider functions
    if cache is not None:
        kwargs["cache"] = cache

    try:
        provider, model_name = model.split("/", 1)
    except ValueError:
        from .core.exceptions import ConfigurationError

        raise ConfigurationError(
            'Model string must be in format "provider/model-name" '
            '(e.g. "openai/gpt-4" or "anthropic/claude-3-sonnet")'
        ) from None

    provider_info = {"provider": provider, "operation": "initialize"}
    logger.info(
        "Initializing %s provider with model %s",
        provider,
        model_name,
        extra=provider_info,
    )
    logger.debug(
        "Provider configuration: async_client=%s, mode=%s",
        async_client,
        mode,
        extra=provider_info,
    )
    api_key = None
    if "api_key" in kwargs:
        api_key = kwargs.pop("api_key")
        if api_key:
            logger.debug(
                "API key provided for %s provider (length: %d characters)",
                provider,
                len(api_key),
                extra=provider_info,
            )

    if provider == "openai":
        try:
            import openai
            from instructor import from_openai

            client = (
                openai.AsyncOpenAI(api_key=api_key)
                if async_client
                else openai.OpenAI(api_key=api_key)
            )
            result = from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the OpenAI provider. "
                "Install it with `pip install openai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "azure_openai":
        try:
            import os
            from openai import AzureOpenAI, AsyncAzureOpenAI
            from instructor import from_openai

            # Get required Azure OpenAI configuration from environment
            api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            azure_endpoint = kwargs.pop(
                "azure_endpoint", os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
            api_version = kwargs.pop("api_version", "2024-02-01")

            if not api_key:
                from .core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "AZURE_OPENAI_API_KEY is not set. "
                    "Set it with `export AZURE_OPENAI_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
                )

            if not azure_endpoint:
                from .core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "AZURE_OPENAI_ENDPOINT is not set. "
                    "Set it with `export AZURE_OPENAI_ENDPOINT=<your-endpoint>` or pass it as kwarg azure_endpoint=<your-endpoint>"
                )

            client = (
                AsyncAzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                )
                if async_client
                else AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                )
            )
            result = from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the Azure OpenAI provider. "
                "Install it with `pip install openai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "anthropic":
        try:
            import anthropic
            from instructor import from_anthropic

            client = (
                anthropic.AsyncAnthropic(api_key=api_key)
                if async_client
                else anthropic.Anthropic(api_key=api_key)
            )
            max_tokens = kwargs.pop("max_tokens", 4096)
            result = from_anthropic(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.ANTHROPIC_TOOLS,
                max_tokens=max_tokens,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The anthropic package is required to use the Anthropic provider. "
                "Install it with `pip install anthropic`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "google":
        try:
            import google.genai as genai
            from instructor import from_genai
            import os

            # Remove vertexai from kwargs if present to avoid passing it twice
            vertexai_flag = kwargs.pop("vertexai", False)

            # Get API key from kwargs or environment
            api_key = api_key or os.environ.get("GOOGLE_API_KEY")

            # Extract client-specific parameters
            client_kwargs = {}
            for key in [
                "debug_config",
                "http_options",
                "credentials",
                "project",
                "location",
            ]:
                if key in kwargs:
                    client_kwargs[key] = kwargs.pop(key)

            client = genai.Client(
                vertexai=vertexai_flag,
                api_key=api_key,
                **client_kwargs,
            )  # type: ignore
            if async_client:
                result = from_genai(client, use_async=True, model=model_name, **kwargs)  # type: ignore
            else:
                result = from_genai(client, model=model_name, **kwargs)  # type: ignore
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The google-genai package is required to use the Google provider. "
                "Install it with `pip install google-genai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "mistral":
        try:
            from mistralai import Mistral
            from instructor import from_mistral
            import os

            api_key = api_key or os.environ.get("MISTRAL_API_KEY")

            if api_key:
                client = Mistral(api_key=api_key)
            else:
                raise ValueError(
                    "MISTRAL_API_KEY is not set. "
                    "Set it with `export MISTRAL_API_KEY=<your-api-key>`."
                )

            if async_client:
                result = from_mistral(
                    client, model=model_name, use_async=True, **kwargs
                )
            else:
                result = from_mistral(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The mistralai package is required to use the Mistral provider. "
                "Install it with `pip install mistralai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "cohere":
        try:
            import cohere
            from instructor import from_cohere

            client = (
                cohere.AsyncClientV2(api_key=api_key)
                if async_client
                else cohere.ClientV2(api_key=api_key)
            )
            result = from_cohere(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The cohere package is required to use the Cohere provider. "
                "Install it with `pip install cohere`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "perplexity":
        try:
            import openai
            from instructor import from_perplexity
            import os

            api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
            if not api_key:
                raise ValueError(
                    "PERPLEXITY_API_KEY is not set. "
                    "Set it with `export PERPLEXITY_API_KEY=<your-api-key>` or pass it as a kwarg api_key=<your-api-key>"
                )

            client = (
                openai.AsyncOpenAI(
                    api_key=api_key, base_url="https://api.perplexity.ai"
                )
                if async_client
                else openai.OpenAI(
                    api_key=api_key, base_url="https://api.perplexity.ai"
                )
            )
            result = from_perplexity(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the Perplexity provider. "
                "Install it with `pip install openai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "groq":
        try:
            import groq
            from instructor import from_groq

            client = (
                groq.AsyncGroq(api_key=api_key)
                if async_client
                else groq.Groq(api_key=api_key)
            )
            result = from_groq(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The groq package is required to use the Groq provider. "
                "Install it with `pip install groq`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "writer":
        try:
            from writerai import AsyncWriter, Writer
            from instructor import from_writer

            client = (
                AsyncWriter(api_key=api_key)
                if async_client
                else Writer(api_key=api_key)
            )
            result = from_writer(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The writerai package is required to use the Writer provider. "
                "Install it with `pip install writer-sdk`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "bedrock":
        try:
            import os
            import boto3
            from instructor import from_bedrock

            # Get AWS configuration from environment or kwargs
            if "region" in kwargs:
                region = kwargs.pop("region")
            else:
                logger.debug(
                    "AWS_DEFAULT_REGION is not set. Using default region us-east-1"
                )
                region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

            # Extract AWS-specific parameters
            # Dictionary to collect AWS credentials and session parameters for boto3 client
            aws_kwargs = {}
            for key in [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
            ]:
                if key in kwargs:
                    aws_kwargs[key] = kwargs.pop(key)
                elif key.upper() in os.environ:
                    logger.debug(f"Using {key.upper()} from environment variable")
                    aws_kwargs[key] = os.environ[key.upper()]

            # Add region to client configuration
            aws_kwargs["region_name"] = region

            # Create bedrock-runtime client
            client = boto3.client("bedrock-runtime", **aws_kwargs)

            # Determine default mode based on model
            if mode is None:
                # Anthropic models (Claude) support tools, others use JSON
                if model_name and (
                    "anthropic" in model_name.lower() or "claude" in model_name.lower()
                ):
                    default_mode = instructor.Mode.BEDROCK_TOOLS
                else:
                    default_mode = instructor.Mode.BEDROCK_JSON
            else:
                default_mode = mode

            result = from_bedrock(
                client,
                mode=default_mode,
                async_client=async_client,
                _async=async_client,  # for backward compatibility
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The boto3 package is required to use the AWS Bedrock provider. "
                "Install it with `pip install boto3`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "cerebras":
        try:
            from cerebras.cloud.sdk import AsyncCerebras, Cerebras
            from instructor import from_cerebras

            client = (
                AsyncCerebras(api_key=api_key)
                if async_client
                else Cerebras(api_key=api_key)
            )
            result = from_cerebras(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The cerebras package is required to use the Cerebras provider. "
                "Install it with `pip install cerebras`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "fireworks":
        try:
            from fireworks.client import AsyncFireworks, Fireworks
            from instructor import from_fireworks

            client = (
                AsyncFireworks(api_key=api_key)
                if async_client
                else Fireworks(api_key=api_key)
            )
            result = from_fireworks(client, model=model_name, **kwargs)
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The fireworks-ai package is required to use the Fireworks provider. "
                "Install it with `pip install fireworks-ai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "vertexai":
        warnings.warn(
            "The 'vertexai' provider is deprecated. Use 'google' provider with vertexai=True instead. "
            "Example: instructor.from_provider('google/gemini-pro', vertexai=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import google.genai as genai  # type: ignore
            from instructor import from_genai
            import os

            # Get project and location from kwargs or environment
            project = kwargs.pop("project", os.environ.get("GOOGLE_CLOUD_PROJECT"))
            location = kwargs.pop(
                "location", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            )

            if not project:
                raise ValueError(
                    "Project ID is required for Vertex AI. "
                    "Set it with `export GOOGLE_CLOUD_PROJECT=<your-project-id>` "
                    "or pass it as kwarg project=<your-project-id>"
                )

            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                **kwargs,
            )  # type: ignore
            kwargs["model"] = model_name  # Pass model as part of kwargs
            if async_client:
                result = from_genai(client, use_async=True, **kwargs)  # type: ignore
            else:
                result = from_genai(client, **kwargs)  # type: ignore
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The google-genai package is required to use the VertexAI provider. "
                "Install it with `pip install google-genai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "generative-ai":
        warnings.warn(
            "The 'generative-ai' provider is deprecated. Use 'google' provider instead. "
            "Example: instructor.from_provider('google/gemini-pro')",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from google import genai
            from instructor import from_genai
            import os

            # Get API key from kwargs or environment
            api_key = api_key or os.environ.get("GOOGLE_API_KEY")

            client = genai.Client(vertexai=False, api_key=api_key)
            if async_client:
                result = from_genai(client, use_async=True, model=model_name, **kwargs)  # type: ignore
            else:
                result = from_genai(client, model=model_name, **kwargs)  # type: ignore
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The google-genai package is required to use the Google GenAI provider. "
                "Install it with `pip install google-genai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "ollama":
        try:
            import openai
            from instructor import from_openai

            # Get base_url from kwargs or use default
            base_url = kwargs.pop("base_url", "http://localhost:11434/v1")
            api_key = kwargs.pop("api_key", "ollama")  # required but unused

            client = (
                openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
                if async_client
                else openai.OpenAI(base_url=base_url, api_key=api_key)
            )

            # Models that support function calling (tools mode)
            tool_capable_models = {
                "llama3.1",
                "llama3.2",
                "llama4",
                "mistral-nemo",
                "firefunction-v2",
                "command-a",
                "command-r",
                "command-r-plus",
                "command-r7b",
                "qwen2.5",
                "qwen2.5-coder",
                "qwen3",
                "devstral",
            }

            # Check if model supports tools by looking at model name
            supports_tools = any(
                capable_model in model_name.lower()
                for capable_model in tool_capable_models
            )

            default_mode = (
                instructor.Mode.TOOLS if supports_tools else instructor.Mode.JSON
            )

            result = from_openai(
                client,
                model=model_name,
                mode=mode if mode else default_mode,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the Ollama provider. "
                "Install it with `pip install openai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "deepseek":
        try:
            import openai
            from instructor import from_openai
            import os

            # Get API key from kwargs or environment
            api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

            if not api_key:
                from .core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "DEEPSEEK_API_KEY is not set. "
                    "Set it with `export DEEPSEEK_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
                )

            # DeepSeek uses OpenAI-compatible API
            base_url = kwargs.pop("base_url", "https://api.deepseek.com")

            client = (
                openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                if async_client
                else openai.OpenAI(api_key=api_key, base_url=base_url)
            )

            result = from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the DeepSeek provider. "
                "Install it with `pip install openai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "xai":
        try:
            from xai_sdk.sync.client import Client as SyncClient
            from xai_sdk.aio.client import Client as AsyncClient
            from instructor import from_xai

            client = (
                AsyncClient(api_key=api_key)
                if async_client
                else SyncClient(api_key=api_key)
            )
            result = from_xai(
                client,
                mode=mode if mode else instructor.Mode.XAI_JSON,
                model=model_name,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The xai-sdk package is required to use the xAI provider. "
                "Install it with `pip install xai-sdk`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "openrouter":
        try:
            import openai
            from instructor import from_openai
            import os

            # Get API key from kwargs or environment
            api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

            if not api_key:
                from .core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "OPENROUTER_API_KEY is not set. "
                    "Set it with `export OPENROUTER_API_KEY=<your-api-key>` or pass it as kwarg api_key=<your-api-key>"
                )

            # OpenRouter uses OpenAI-compatible API
            base_url = kwargs.pop("base_url", "https://openrouter.ai/api/v1")

            client = (
                openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                if async_client
                else openai.OpenAI(api_key=api_key, base_url=base_url)
            )

            result = from_openai(
                client,
                model=model_name,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The openai package is required to use the OpenRouter provider. "
                "Install it with `pip install openai`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    elif provider == "litellm":
        try:
            from litellm import completion, acompletion
            from instructor import from_litellm

            completion_func = acompletion if async_client else completion
            result = from_litellm(
                completion_func,
                mode=mode if mode else instructor.Mode.TOOLS,
                **kwargs,
            )
            logger.info(
                "Client initialized",
                extra={**provider_info, "status": "success"},
            )
            return result
        except ImportError:
            from .core.exceptions import ConfigurationError

            raise ConfigurationError(
                "The litellm package is required to use the LiteLLM provider. "
                "Install it with `pip install litellm`."
            ) from None
        except Exception as e:
            logger.error(
                "Error initializing %s client: %s",
                provider,
                e,
                exc_info=True,
                extra={**provider_info, "status": "error"},
            )
            raise

    else:
        from .core.exceptions import ConfigurationError

        logger.error(
            "Error initializing %s client: unsupported provider",
            provider,
            extra={**provider_info, "status": "error"},
        )
        raise ConfigurationError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: {supported_providers}"
        )
