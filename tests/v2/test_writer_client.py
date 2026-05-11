"""Unit tests for Writer v2 client factory.

These tests verify client factory behavior without requiring API keys.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


# ============================================================================
# Mode Normalization Tests
# ============================================================================


class TestWriterModeNormalization:
    """Tests for Writer mode normalization."""

    def test_mode_normalization_generic_tools(self):
        """Test generic TOOLS mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.TOOLS)

        assert result == Mode.TOOLS

    def test_mode_normalization_generic_md_json(self):
        """Test generic MD_JSON mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.MD_JSON)

        assert result == Mode.MD_JSON

    def test_mode_normalization_writer_tools(self):
        """Legacy WRITER_TOOLS remains accepted via normalization."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.WRITER_TOOLS)

        assert result == Mode.TOOLS
        assert mode_registry.is_registered(Provider.WRITER, Mode.WRITER_TOOLS)

    def test_mode_normalization_writer_json(self):
        """Legacy WRITER_JSON remains accepted via normalization."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.WRITER_JSON)

        assert result == Mode.MD_JSON
        assert mode_registry.is_registered(Provider.WRITER, Mode.WRITER_JSON)


# ============================================================================
# Mode Registry Tests for Writer
# ============================================================================


class TestWriterModeRegistry:
    """Tests for Writer mode registration in the v2 registry."""

    def test_tools_mode_registered(self):
        """Test TOOLS mode is registered for Writer."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.WRITER, Mode.TOOLS)

    def test_md_json_mode_registered(self):
        """Test MD_JSON mode is registered for Writer."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.WRITER, Mode.MD_JSON)

    def test_json_schema_registered(self):
        """Test JSON_SCHEMA mode is registered for Writer."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.WRITER, Mode.JSON_SCHEMA)

    def test_get_modes_for_writer(self):
        """Test getting all modes for Writer provider."""
        from instructor.v2.core.registry import mode_registry

        modes = mode_registry.get_modes_for_provider(Provider.WRITER)

        assert Mode.TOOLS in modes
        assert Mode.MD_JSON in modes
        assert Mode.JSON_SCHEMA in modes

    def test_writer_in_providers_for_tools(self):
        """Test Writer is listed as provider for TOOLS mode."""
        from instructor.v2.core.registry import mode_registry

        providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

        assert Provider.WRITER in providers


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestWriterClientErrors:
    """Tests for error handling in Writer client."""

    def test_json_schema_supported(self):
        """Test JSON_SCHEMA mode is supported by Writer."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.WRITER, Mode.JSON_SCHEMA)

    def test_parallel_tools_not_supported(self):
        """Test PARALLEL_TOOLS is not supported by Writer."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.WRITER, Mode.PARALLEL_TOOLS)

    def test_responses_tools_not_supported(self):
        """Test RESPONSES_TOOLS is not supported by Writer."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.WRITER, Mode.RESPONSES_TOOLS)


# ============================================================================
# Import Tests
# ============================================================================


class TestWriterImports:
    """Tests for Writer module imports."""

    def test_from_writer_importable_from_v2(self):
        """Test from_writer is importable from instructor.v2."""
        from instructor.v2 import from_writer

        # Should be None if writerai not installed, or a function if installed
        assert from_writer is None or callable(from_writer)

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
            pytest.skip("writerai is installed")

        from instructor.v2.providers.writer.client import from_writer
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="writerai is not installed"):
            from_writer("not a client")  # type: ignore[arg-type]

    def test_from_writer_with_invalid_client(self, writer_available):
        """Test from_writer raises error with invalid client."""
        if not writer_available:
            pytest.skip("writerai not installed")

        from instructor.v2.providers.writer.client import from_writer
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_writer("not a client")  # type: ignore[arg-type]

    def test_from_writer_with_invalid_mode(self, writer_available):
        """Test from_writer raises error with invalid mode."""
        if not writer_available:
            pytest.skip("writerai not installed")

        from writerai import Writer

        from instructor.v2.providers.writer.client import from_writer
        from instructor.core.exceptions import ModeError

        client = Writer(api_key="fake-key")

        with pytest.raises(ModeError):
            from_writer(client, mode=Mode.RESPONSES_TOOLS)
