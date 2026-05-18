"""Provider-specific tests for Bedrock v2 client factory."""

from __future__ import annotations

import pytest

from instructor import Mode


class TestBedrockClientWithSDK:
    """Tests for Bedrock client factory that require botocore."""

    @pytest.fixture
    def bedrock_available(self):
        """Check if botocore is available."""
        try:
            from botocore.client import BaseClient  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_bedrock_raises_without_sdk(self, bedrock_available):
        """from_bedrock should raise when botocore is missing."""
        if bedrock_available:
            pytest.skip("botocore is installed")

        from instructor.v2.providers.bedrock.client import from_bedrock
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="botocore is not installed"):
            from_bedrock(None)  # type: ignore[arg-type]

    def test_from_bedrock_with_invalid_client(self, bedrock_available):
        """from_bedrock should reject non-BaseClient objects."""
        if not bedrock_available:
            pytest.skip("botocore not installed")

        from instructor.v2.providers.bedrock.client import from_bedrock
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="BaseClient"):
            from_bedrock("not a client")  # type: ignore[arg-type]

    def test_from_bedrock_with_invalid_mode(self, bedrock_available):
        """from_bedrock should raise for unsupported modes."""
        if not bedrock_available:
            pytest.skip("botocore not installed")

        from botocore.client import BaseClient
        from instructor.v2.providers.bedrock.client import from_bedrock
        from instructor.core.exceptions import ModeError

        def _converse(**_kwargs):
            return {}

        client = BaseClient.__new__(BaseClient)
        client.converse = _converse  # type: ignore[assignment]

        with pytest.raises(ModeError):
            from_bedrock(client, mode=Mode.JSON_SCHEMA)
