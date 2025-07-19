"""Provider implementations for instructor."""

import importlib

__all__ = []

# Conditional imports based on installed packages
if importlib.util.find_spec("anthropic") is not None:
    from .anthropic.client import from_anthropic  # noqa: F401

    __all__.append("from_anthropic")

if importlib.util.find_spec("boto3") is not None:
    from .bedrock.client import from_bedrock  # noqa: F401

    __all__.append("from_bedrock")

if importlib.util.find_spec("cerebras") is not None:
    from .cerebras.client import from_cerebras  # noqa: F401

    __all__.append("from_cerebras")

if importlib.util.find_spec("cohere") is not None:
    from .cohere.client import from_cohere  # noqa: F401

    __all__.append("from_cohere")

if importlib.util.find_spec("fireworks") is not None:
    from .fireworks.client import from_fireworks  # noqa: F401

    __all__.append("from_fireworks")

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    from .gemini.client import from_gemini  # noqa: F401

    __all__.append("from_gemini")

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.genai") is not None
):
    from .genai.client import from_genai  # noqa: F401

    __all__.append("from_genai")

if importlib.util.find_spec("groq") is not None:
    from .groq.client import from_groq  # noqa: F401

    __all__.append("from_groq")

if importlib.util.find_spec("mistralai") is not None:
    from .mistral.client import from_mistral  # noqa: F401

    __all__.append("from_mistral")

if importlib.util.find_spec("openai") is not None:
    from .perplexity.client import from_perplexity  # noqa: F401

    __all__.append("from_perplexity")

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    from .vertexai.client import from_vertexai  # noqa: F401

    __all__.append("from_vertexai")

if importlib.util.find_spec("writerai") is not None:
    from .writer.client import from_writer  # noqa: F401

    __all__.append("from_writer")

if importlib.util.find_spec("xai_sdk") is not None:
    from .xai.client import from_xai  # noqa: F401

    __all__.append("from_xai")
