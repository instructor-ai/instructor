import os
import pytest
import instructor

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from litellm import acompletion, completion
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("litellm package is not installed", allow_module_level=True)


def test_litellm_create():
    client = instructor.from_litellm(completion)

    assert isinstance(client, instructor.Instructor)


def test_async_litellm_create():
    client = instructor.from_litellm(acompletion)

    assert isinstance(client, instructor.AsyncInstructor)
