from pathlib import Path
import importlib
from typing import Any

try:
    tomllib: Any = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.9/3.10
    tomllib = importlib.import_module("tomli")

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from openai import OpenAI

from instructor import from_openai


def test_openai_v3_is_supported_by_package_metadata() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    requirement = next(
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if dependency.startswith("openai")
    )

    specifier = SpecifierSet(requirement.removeprefix("openai"))
    assert Version("3.0.0") in specifier


def test_openai_client_can_be_wrapped_without_api_call() -> None:
    client = OpenAI(api_key="test-key")

    wrapped_client = from_openai(client)

    assert wrapped_client.client is client
