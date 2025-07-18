"""Provider detection and registry utilities.

This module contains provider-related enums and detection logic.
"""

from enum import Enum


class Provider(Enum):
    OPENAI = "openai"
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
    UNKNOWN = "unknown"
    BEDROCK = "bedrock"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"


def get_provider(base_url: str) -> Provider:
    """
    Detect the provider based on the base URL.

    Args:
        base_url: The base URL to analyze

    Returns:
        Provider: The detected provider enum value
    """
    if "anyscale" in str(base_url):
        return Provider.ANYSCALE
    elif "together" in str(base_url):
        return Provider.TOGETHER
    elif "anthropic" in str(base_url):
        return Provider.ANTHROPIC
    elif "cerebras" in str(base_url):
        return Provider.CEREBRAS
    elif "fireworks" in str(base_url):
        return Provider.FIREWORKS
    elif "groq" in str(base_url):
        return Provider.GROQ
    elif "openai" in str(base_url):
        return Provider.OPENAI
    elif "mistral" in str(base_url):
        return Provider.MISTRAL
    elif "cohere" in str(base_url):
        return Provider.COHERE
    elif "gemini" in str(base_url):
        return Provider.GEMINI
    elif "databricks" in str(base_url):
        return Provider.DATABRICKS
    elif "deepseek" in str(base_url):
        return Provider.DEEPSEEK
    elif "vertexai" in str(base_url):
        return Provider.VERTEXAI
    elif "writer" in str(base_url):
        return Provider.WRITER
    elif "perplexity" in str(base_url):
        return Provider.PERPLEXITY
    elif "x.ai" in str(base_url) or "xai" in str(base_url):
        return Provider.XAI
    elif "openrouter" in str(base_url):
        return Provider.OPENROUTER
    return Provider.UNKNOWN
