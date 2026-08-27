"""Provider-specific tests for Sarvam v2 client factory."""

from __future__ import annotations

import pytest

from instructor import Mode


class TestSarvamClient:
    """Sarvam uses the OpenAI SDK with a custom base URL."""

    def test_from_sarvam_with_invalid_client(self):
        from instructor.core.exceptions import ClientError
        from instructor.v2.providers.openai.client import from_sarvam

        with pytest.raises(ClientError, match="must be an instance"):
            from_sarvam(123)  # ty: ignore[no-matching-overload]

    def test_from_sarvam_with_invalid_mode(self):
        import openai

        from instructor.core.exceptions import ModeError
        from instructor.v2.providers.openai.client import from_sarvam

        client = openai.OpenAI(api_key="fake-key", base_url="https://api.sarvam.ai/v1")

        with pytest.raises(ModeError):
            from_sarvam(client, mode=Mode.PARALLEL_TOOLS)
