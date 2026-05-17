"""Provider-specific tests for Fireworks v2 client factory.

Note: Common tests (mode normalization, registry, imports) are unified in
test_client_unified.py. This file only contains Fireworks-specific tests.
"""

from __future__ import annotations

import pytest

from instructor import Mode


# ============================================================================
# Provider-Specific Integration Tests
# ============================================================================
# Note: Common SDK availability tests are in test_client_unified.py


class TestFireworksClientWithSDK:
    """Tests that require Fireworks SDK but not API key."""

    @pytest.fixture
    def fireworks_available(self):
        """Check if fireworks SDK is available."""
        try:
            from fireworks.client import Fireworks  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_fireworks_raises_without_sdk(self, fireworks_available):
        """Test from_fireworks raises error when fireworks not installed."""
        if fireworks_available:
            pytest.skip("fireworks is installed")

        from instructor.v2.providers.fireworks.client import from_fireworks
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="fireworks is not installed"):
            from_fireworks("not a client")  # type: ignore[arg-type]

    def test_from_fireworks_with_invalid_client(self, fireworks_available):
        """Test from_fireworks raises error with invalid client."""
        if not fireworks_available:
            pytest.skip("fireworks not installed")

        from instructor.v2.providers.fireworks.client import from_fireworks
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_fireworks("not a client")  # type: ignore[arg-type]

    def test_from_fireworks_with_invalid_mode(self, fireworks_available):
        """Test from_fireworks raises error with invalid mode."""
        if not fireworks_available:
            pytest.skip("fireworks not installed")

        from fireworks.client import Fireworks

        from instructor.v2.providers.fireworks.client import from_fireworks
        from instructor.core.exceptions import ModeError

        client = Fireworks(api_key="fake-key")

        with pytest.raises(ModeError):
            from_fireworks(client, mode=Mode.RESPONSES_TOOLS)
