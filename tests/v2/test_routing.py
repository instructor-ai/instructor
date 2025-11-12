"""Tests for from_provider() routing to v2.

Verifies that from_provider("anthropic/...") routes to v2 implementation.
"""

import warnings

import pytest


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_from_provider_routes_to_v2():
    """Test that from_provider() routes Anthropic to v2."""
    import instructor

    # from_provider should route to v2 for Anthropic
    client = instructor.from_provider("anthropic/claude-3-5-sonnet-20241022")

    assert client is not None
    # Verify it's using v2 by checking the mode is a tuple
    assert isinstance(client.mode, tuple)
    assert len(client.mode) == 2


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_from_provider_anthropic_async():
    """Test that from_provider() routes async Anthropic to v2."""
    import instructor

    client = instructor.from_provider(
        "anthropic/claude-3-5-sonnet-20241022", async_client=True
    )

    assert client is not None
    from instructor import AsyncInstructor

    assert isinstance(client, AsyncInstructor)
    # Verify it's using v2
    assert isinstance(client.mode, tuple)


def test_old_from_anthropic_deprecation_warning():
    """Test that old from_anthropic() emits deprecation warning."""
    import anthropic
    from instructor import from_anthropic

    client = anthropic.Anthropic()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instructor_client = from_anthropic(client)

        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "v2" in str(w[0].message)


@pytest.mark.skip(reason="Requires Anthropic API key")
def test_from_provider_with_mode_compatibility():
    """Test that from_provider() handles v1 Mode enum for compatibility."""
    import instructor

    # Passing v1 Mode should still work (gets converted to v2 Mode)
    client = instructor.from_provider(
        "anthropic/claude-3-5-sonnet-20241022", mode=instructor.Mode.TOOLS
    )

    assert client is not None
    # Should be converted to v2 tuple mode
    assert isinstance(client.mode, tuple)
