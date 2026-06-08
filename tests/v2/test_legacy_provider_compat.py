from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from instructor.v2.core.provider_specs import PROVIDER_SPECS
from instructor.v2.core.providers import Provider


_ROOT = Path(__file__).parents[2]
_LEGACY_OPTIONAL_CLIENTS = (
    Provider.ANTHROPIC,
    Provider.BEDROCK,
    Provider.CEREBRAS,
    Provider.COHERE,
    Provider.FIREWORKS,
    Provider.GEMINI,
    Provider.GENAI,
    Provider.GROQ,
    Provider.MISTRAL,
    Provider.VERTEXAI,
    Provider.WRITER,
    Provider.XAI,
)


def test_legacy_provider_modules_remain_importable() -> None:
    modules = [
        "instructor.providers",
        "instructor.providers.anthropic.client",
        "instructor.providers.anthropic.utils",
        "instructor.providers.bedrock.client",
        "instructor.providers.bedrock.utils",
        "instructor.providers.cerebras.client",
        "instructor.providers.cerebras.utils",
        "instructor.providers.cohere.client",
        "instructor.providers.cohere.utils",
        "instructor.providers.fireworks.client",
        "instructor.providers.gemini.client",
        "instructor.providers.gemini.utils",
        "instructor.providers.genai.client",
        "instructor.providers.groq.client",
        "instructor.providers.mistral.client",
        "instructor.providers.mistral.utils",
        "instructor.providers.openai.utils",
        "instructor.providers.perplexity.client",
        "instructor.providers.perplexity.utils",
        "instructor.providers.vertexai.client",
        "instructor.providers.writer.client",
        "instructor.providers.writer.utils",
        "instructor.providers.xai.client",
        "instructor.providers.xai.utils",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_legacy_provider_utils_forward_to_v2_symbols() -> None:
    openai_utils = importlib.import_module("instructor.providers.openai.utils")
    anthropic_utils = importlib.import_module("instructor.providers.anthropic.utils")
    gemini_utils = importlib.import_module("instructor.providers.gemini.utils")

    assert (
        openai_utils.reask_tools
        is importlib.import_module(
            "instructor.v2.providers.openai.handlers"
        ).reask_tools
    )
    assert (
        anthropic_utils.combine_system_messages
        is importlib.import_module(
            "instructor.v2.providers.anthropic.handlers"
        ).combine_system_messages
    )
    assert (
        gemini_utils.map_to_gemini_function_schema
        is importlib.import_module(
            "instructor.v2.providers.gemini.utils"
        ).map_to_gemini_function_schema
    )


@pytest.mark.parametrize("provider", _LEGACY_OPTIONAL_CLIENTS)
def test_legacy_client_facades_remain_lazy_without_optional_sdks(
    provider: Provider,
) -> None:
    spec = PROVIDER_SPECS[provider]
    assert spec.sdk_module is not None
    blocked_sdk = spec.sdk_module
    legacy_module = f"instructor.providers.{provider.value}.client"
    implementation_module = f"instructor.v2.providers.{provider.value}.client"
    probe = """
import builtins
import importlib
import sys

blocked, facade, implementation = sys.argv[1:]
original_import = builtins.__import__

def block_sdk(name, globals=None, locals=None, fromlist=(), level=0):
    if name == blocked or name.startswith(blocked + "."):
        raise ImportError(f"blocked optional SDK: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_sdk
importlib.import_module(facade)
assert implementation not in sys.modules, implementation
"""
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            blocked_sdk,
            legacy_module,
            implementation_module,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
