"""Unit tests for Writer v2 client factory.

These tests verify client factory behavior without requiring API keys.
"""

from __future__ import annotations

import pytest

from instructor import Mode


# ============================================================================
# Import Tests
# ============================================================================


class TestWriterImports:
    """Tests for Writer module imports."""

    def test_handlers_importable(self):
        """Test Writer handlers are importable."""
        from instructor.v2.providers.writer.handlers import (
            WriterJSONSchemaHandler,
            WriterMDJSONHandler,
            WriterToolsHandler,
        )

        assert WriterToolsHandler is not None
        assert WriterJSONSchemaHandler is not None
        assert WriterMDJSONHandler is not None


# ============================================================================
# Integration Tests (require Writer SDK but not API key)
# ============================================================================


class TestWriterClientWithSDK:
    """Tests that require Writer SDK but not API key."""

    @pytest.fixture
    def writer_available(self):
        """Check if writerai SDK is available."""
        try:
            from writerai import Writer  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_writer_raises_without_sdk(self, writer_available):
        """Test from_writer raises error when writerai not installed."""
        if writer_available:
            pytest.skip(
                "writerai is installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.writer.client import from_writer
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="writerai is not installed"):
            from_writer("not a client")  # ty: ignore[no-matching-overload]

    def test_from_writer_with_invalid_client(self, writer_available):
        """Test from_writer raises error with invalid client."""
        if not writer_available:
            pytest.skip(
                "writerai not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from instructor.v2.providers.writer.client import from_writer
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_writer("not a client")  # ty: ignore[no-matching-overload]

    def test_from_writer_with_invalid_mode(self, writer_available):
        """Test from_writer raises error with invalid mode."""
        if not writer_available:
            pytest.skip(
                "writerai not installed"  # ty: ignore[too-many-positional-arguments]
            )

        from writerai import Writer

        from instructor.v2.providers.writer.client import from_writer
        from instructor.core.exceptions import ModeError

        client = Writer(api_key="fake-key")

        with pytest.raises(ModeError):
            from_writer(client, mode=Mode.RESPONSES_TOOLS)
