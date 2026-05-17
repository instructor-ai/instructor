import os
from pathlib import Path
import pytest

SKIP_REASON = None

if not os.getenv("GOOGLE_API_KEY"):
    SKIP_REASON = "GOOGLE_API_KEY environment variable not set"
else:
    try:
        import vertexai  # noqa: F401
    except ImportError:  # pragma: no cover - optional dependency
        SKIP_REASON = "google-cloud-aiplatform package is not installed"
    else:
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError

            _, project_id = google.auth.default()
            if not project_id:
                SKIP_REASON = (
                    "Google application default credentials are missing a project ID"
                )
        except DefaultCredentialsError:
            SKIP_REASON = "Google application default credentials are not configured"


BASE_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if not SKIP_REASON:
        return
    skip_marker = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        if BASE_DIR in item_path.parents:
            item.add_marker(skip_marker)
