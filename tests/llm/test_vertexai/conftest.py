import os
import pytest

if not os.getenv("GOOGLE_API_KEY"):
    pytest.skip(
        "GOOGLE_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    import vertexai  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip(
        "google-cloud-aiplatform package is not installed", allow_module_level=True
    )
