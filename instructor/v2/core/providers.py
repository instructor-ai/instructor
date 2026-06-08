"""Provider detection and registry utilities.

This module contains provider-related enums and detection logic.
"""

from enum import Enum

from instructor.v2.core.mode import Mode


class Provider(Enum):
    """Supported provider identifiers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    GENAI = "genai"
    DATABRICKS = "databricks"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    WRITER = "writer"
    XAI = "xai"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    GOOGLE = "google"
    GENERATIVE_AI = "generative-ai"
    UNKNOWN = "unknown"
    BEDROCK = "bedrock"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"


def provider_from_mode(mode: Mode, default: Provider = Provider.OPENAI) -> Provider:
    """Infer provider from a provider-specific Mode."""
    from instructor.v2.core.provider_specs import PROVIDER_SPECS

    owners = {
        spec.canonical_provider
        for spec in PROVIDER_SPECS.values()
        if mode in spec.legacy_modes
    }
    return owners.pop() if len(owners) == 1 else default


def normalize_mode_for_provider(mode: Mode, provider: Provider) -> Mode:
    """Apply provider-specific mode overrides before registry lookup."""
    from instructor.v2.core.registry import normalize_mode

    return normalize_mode(provider, mode)


def get_provider(base_url: str) -> Provider:
    """
    Detect the provider based on the base URL.

    Args:
        base_url: The base URL to analyze

    Returns:
        Provider: The detected provider enum value
    """
    normalized = str(base_url).lower()
    providers = (
        ("azure", Provider.AZURE_OPENAI),
        ("anyscale", Provider.ANYSCALE),
        ("together", Provider.TOGETHER),
        ("anthropic", Provider.ANTHROPIC),
        ("cerebras", Provider.CEREBRAS),
        ("fireworks", Provider.FIREWORKS),
        ("groq", Provider.GROQ),
        ("openai", Provider.OPENAI),
        ("mistral", Provider.MISTRAL),
        ("cohere", Provider.COHERE),
        ("generative-ai", Provider.GENERATIVE_AI),
        ("vertexai", Provider.VERTEXAI),
        ("gemini", Provider.GEMINI),
        ("genai", Provider.GENAI),
        ("google", Provider.GOOGLE),
        ("databricks", Provider.DATABRICKS),
        ("deepseek", Provider.DEEPSEEK),
        ("bedrock", Provider.BEDROCK),
        ("writer", Provider.WRITER),
        ("perplexity", Provider.PERPLEXITY),
        ("ollama", Provider.OLLAMA),
        ("localhost:11434", Provider.OLLAMA),
        ("litellm", Provider.LITELLM),
        ("openrouter", Provider.OPENROUTER),
        ("x.ai", Provider.XAI),
        ("xai", Provider.XAI),
    )
    for token, provider in providers:
        if token in normalized:
            return provider
    return Provider.UNKNOWN
