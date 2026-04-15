"""Provider implementations for instructor.

Provider factory functions are loaded lazily to avoid pulling in heavy SDK
dependencies (anthropic, boto3, google-genai, etc.) at import time.
See https://github.com/567-labs/instructor/issues/2205
"""

import importlib
import importlib.util

_PROVIDER_MAP: dict[str, tuple[str, str | list[str]]] = {
    "from_anthropic": (".anthropic.client", "anthropic"),
    "from_bedrock": (".bedrock.client", "boto3"),
    "from_cerebras": (".cerebras.client", "cerebras"),
    "from_cohere": (".cohere.client", "cohere"),
    "from_fireworks": (".fireworks.client", "fireworks"),
    "from_gemini": (".gemini.client", ["google", "google.generativeai"]),
    "from_genai": (".genai.client", ["google", "google.genai"]),
    "from_groq": (".groq.client", "groq"),
    "from_mistral": (".mistral.client", "mistralai"),
    "from_perplexity": (".perplexity.client", "openai"),
    "from_vertexai": (".vertexai.client", ["vertexai", "jsonref"]),
    "from_writer": (".writer.client", "writerai"),
    "from_xai": (".xai.client", "xai_sdk"),
}

__all__: list[str] = []
for _name, (_, _specs) in _PROVIDER_MAP.items():
    _spec_list = [_specs] if isinstance(_specs, str) else _specs
    if all(importlib.util.find_spec(s) is not None for s in _spec_list):
        __all__.append(_name)


def __getattr__(name: str):
    if name in _PROVIDER_MAP:
        module_path, specs = _PROVIDER_MAP[name]
        spec_list = [specs] if isinstance(specs, str) else specs
        try:
            specs_found = all(
                importlib.util.find_spec(s) is not None for s in spec_list
            )
        except (ValueError, ModuleNotFoundError):
            specs_found = False
        if not specs_found:
            raise AttributeError(
                f"module 'instructor.providers' has no attribute {name!r}"
            )
        try:
            mod = importlib.import_module(module_path, package=__name__)
            attr = getattr(mod, name)
        except Exception as exc:
            raise AttributeError(
                f"module 'instructor.providers' has no attribute {name!r}"
            ) from exc
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'instructor.providers' has no attribute {name!r}")
