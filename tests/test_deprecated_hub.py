from typing import Any
from typer.testing import CliRunner, Result
from instructor.cli.deprecated_hub import app
from typer import Typer

runner: CliRunner = CliRunner()

def test_deprecated_hub_command() -> None:
    """Test that the hub command returns a deprecation message and exits with code 1"""
    result: Result[Any] = runner.invoke(app)
    assert result.exit_code == 1
    assert "instructor hub has been deprecated" in result.stdout
    assert "https://python.useinstructor.com/examples/" in result.stdout
