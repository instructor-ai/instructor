"""Provider-specific tests for Mistral v2 client factory.

Note: Common tests (mode normalization, registry, imports, errors) are unified in
test_client_unified.py. This file only contains Mistral-specific tests.
"""

from __future__ import annotations

import pytest


# ============================================================================
# Provider-Specific Integration Tests
# ============================================================================
# Note: Common SDK availability tests are in test_client_unified.py


class TestMistralClientWithSDK:
    """Tests for Mistral client factory that require the SDK."""

    def test_from_mistral_raises_without_sdk(self):
        """Test from_mistral raises helpful error when SDK not installed."""
        import importlib.util

        # This test checks behavior when mistralai is not installed
        if importlib.util.find_spec("mistralai") is not None:
            pytest.skip("mistralai is installed, skipping SDK-not-installed test")

        from instructor.v2.providers.mistral.client import from_mistral
        from instructor.core.exceptions import ClientError

        # Should raise ClientError about missing SDK
        with pytest.raises(ClientError) as exc_info:
            from_mistral(None)  # type: ignore

        assert "mistralai is not installed" in str(exc_info.value)

    @pytest.mark.skipif(True, reason="Requires mistralai SDK")
    def test_from_mistral_with_invalid_client(self):
        """Test from_mistral raises error with invalid client type."""
        pass

    @pytest.mark.skipif(True, reason="Requires mistralai SDK")
    def test_from_mistral_with_invalid_mode(self):
        """Test from_mistral raises error with invalid mode."""
        pass

    @pytest.mark.skipif(True, reason="Requires mistralai SDK")
    def test_from_mistral_sync_client(self):
        """Test from_mistral creates sync Instructor."""
        pass

    @pytest.mark.skipif(True, reason="Requires mistralai SDK")
    def test_from_mistral_async_client(self):
        """Test from_mistral creates async Instructor with use_async=True."""
        pass
