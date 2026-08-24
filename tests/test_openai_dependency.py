from pathlib import Path
import tomllib

from packaging.specifiers import SpecifierSet
from packaging.version import Version


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
