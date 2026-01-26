"""Tests for from_provider() routing to v2.

Verifies that from_provider("anthropic/...") routes to v2 implementation.
"""

import importlib.util
import warnings

import pytest


@pytest.mark.skip(reason="Requires Anthropic API key")
@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
def test_from_provider_routes_to_v2(async_client: bool):
    """Test that from_provider() routes Anthropic to v2."""
    import instructor

    # from_provider should route to v2 for Anthropic
    client = instructor.from_provider(
        "anthropic/claude-3-5-sonnet-20241022",
        async_client=async_client,
    )

    assert client is not None
    # Verify it's using v2 by checking the mode is a tuple
    assert isinstance(client.mode, tuple)
    assert len(client.mode) == 2

    if async_client:
        from instructor import AsyncInstructor

        assert isinstance(client, AsyncInstructor)


@pytest.mark.parametrize(
    "client_class_name",
    ["Anthropic", "AsyncAnthropic"],
    ids=["sync", "async"],
)
def test_old_from_anthropic_deprecation_warning(client_class_name: str):
    """Test that old from_anthropic() emits deprecation warning with correct v2 example.
    """
    if importlib.util.find_spec("anthropic") is None:
        pytest.skip("anthropic package is not installed")
    import anthropic
    from instructor import from_anthropic

    client_class = getattr(anthropic, client_class_name)
    client = client_class()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        instructor_client = from_anthropic(client)  # noqa: F841

        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        expected_warning = (
            "from_anthropic() is deprecated and will be removed in v2.0. "
            "Use instructor.v2.from_anthropic() with Mode instead:\n"
            "  from instructor.v2.providers.anthropic import from_anthropic\n"
            "  from instructor import Mode\n"
            "  client = from_anthropic(anthropic_client, mode=Mode.TOOLS)\n"
            "Or use from_provider() which automatically routes to v2:\n"
            "  client = instructor.from_provider('anthropic/claude-3-sonnet')"
        )
        assert str(w[0].message) == expected_warning


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
