"""Unit tests for Cerebras v2 client factory.

These tests verify client factory behavior without requiring API keys.
"""

from __future__ import annotations

import pytest

from instructor import Mode


# ============================================================================
# Integration Tests (require Cerebras SDK but not API key)
# ============================================================================


class TestCerebrasClientWithSDK:
    """Tests that require Cerebras SDK but not API key."""

    @pytest.fixture
    def cerebras_available(self):
        """Check if cerebras SDK is available."""
        try:
            from cerebras.cloud.sdk import Cerebras  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_cerebras_raises_without_sdk(self, cerebras_available):
        """Test from_cerebras raises error when cerebras not installed."""
        if cerebras_available:
            pytest.skip(
                "cerebras is installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.cerebras.client import from_cerebras
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="cerebras is not installed"):
            from_cerebras("not a client")  # ty: ignore[no-matching-overload]

    def test_from_cerebras_with_invalid_client(self, cerebras_available):
        """Test from_cerebras raises error with invalid client."""
        if not cerebras_available:
            pytest.skip(
                "cerebras not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.cerebras.client import from_cerebras
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_cerebras("not a client")  # ty: ignore[no-matching-overload]

    def test_from_cerebras_with_invalid_mode(self, cerebras_available):
        """Test from_cerebras raises error with invalid mode."""
        if not cerebras_available:
            pytest.skip(
                "cerebras not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from cerebras.cloud.sdk import Cerebras

        from instructor.v2.providers.cerebras.client import from_cerebras
        from instructor.core.exceptions import ModeError

        client = Cerebras(api_key="fake-key")

        with pytest.raises(ModeError):
            from_cerebras(client, mode=Mode.RESPONSES_TOOLS)
