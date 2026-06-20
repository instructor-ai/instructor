import os

import pytest

if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
    pytest.skip(
        "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables not set",  # ty: ignore[too-many-positional-arguments]
        allow_module_level=True,
    )
