"""Provider-specific tests for Groq v2 client factory.

Note: Common tests (mode normalization, registry, imports) are unified in
test_client_unified.py. This file only contains Groq-specific tests.
"""

from __future__ import annotations

import pytest

from instructor import Mode


# ============================================================================
# Provider-Specific Integration Tests
# ============================================================================
# Note: Common SDK availability tests are in test_client_unified.py


class TestGroqClientWithSDK:
    """Tests that require Groq SDK but not API key."""

    @pytest.fixture
    def groq_available(self):
        """Check if groq SDK is available."""
        try:
            import groq  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_groq_raises_without_sdk(self, groq_available):
        """Test from_groq raises error when groq not installed."""
        if groq_available:
            pytest.skip(
                "groq is installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.groq.client import from_groq
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="groq is not installed"):
            from_groq("not a client")  # ty: ignore[no-matching-overload]

    def test_from_groq_with_invalid_client(self, groq_available):
        """Test from_groq raises error with invalid client."""
        if not groq_available:
            pytest.skip(
                "groq not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.groq.client import from_groq
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_groq("not a client")  # ty: ignore[no-matching-overload]

    def test_from_groq_with_invalid_mode(self, groq_available):
        """Test from_groq raises error with invalid mode."""
        if not groq_available:
            pytest.skip(
                "groq not installed"  # ty: ignore[too-many-positional-arguments]
            )

        import groq

        from instructor.v2.providers.groq.client import from_groq
        from instructor.core.exceptions import ModeError

        client = groq.Groq(api_key="fake-key")

        with pytest.raises(ModeError):
            from_groq(client, mode=Mode.RESPONSES_TOOLS)
