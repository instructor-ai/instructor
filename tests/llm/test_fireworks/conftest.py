import os
import pytest

if not os.getenv("FIREWORKS_API_KEY"):
    pytest.skip(
        "FIREWORKS_API_KEY environment variable not set",
        allow_module_level=True,
    )
