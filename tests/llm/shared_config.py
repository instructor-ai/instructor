"""
Shared configuration for multi-provider tests.

This module provides common test configuration for running the same tests
across multiple providers (OpenAI, Anthropic, Google).
"""

import os
import pytest
import instructor
from typing import List, Tuple


# Provider configurations: (model_string, mode, required_env_var, required_package)
PROVIDER_CONFIGS = [
    (
        "openai/gpt-5-nano",
        instructor.Mode.TOOLS,
        "OPENAI_API_KEY",
        "openai",
    ),
    (
        "anthropic/claude-3-7-sonnet-latest",
        instructor.Mode.ANTHROPIC_TOOLS,
        "ANTHROPIC_API_KEY",
        "anthropic",
    ),
    (
        "google/gemini-2.0-flash-exp",
        instructor.Mode.GENAI_TOOLS,
        "GOOGLE_API_KEY",
        "google.genai",
    ),
]


def get_available_providers() -> List[Tuple[str, instructor.Mode]]:
    """
    Get list of available providers based on API keys and installed packages.

    Returns:
        List of tuples (model_string, mode) for available providers
    """
    available = []

    for model, mode, env_var, package in PROVIDER_CONFIGS:
        # Check if API key is set
        if not os.getenv(env_var):
            continue

        # Check if package is installed
        try:
            parts = package.split(".")
            if len(parts) > 1:
                __import__(parts[0])
                # For nested imports like google.genai
                __import__(package)
            else:
                __import__(package)
            available.append((model, mode))
        except ImportError:
            continue

    return available


def pytest_generate_tests(metafunc):
    """
    Pytest hook to generate parametrized tests for available providers.

    This is used in test files that have 'provider_config' as a parameter.
    """
    if "provider_config" in metafunc.fixturenames:
        available = get_available_providers()
        if not available:
            pytest.skip("No providers available (missing API keys or packages)")

        # Generate test IDs like "openai" "anthropic" "google"
        ids = [model.split("/")[0] for model, _ in available]
        metafunc.parametrize("provider_config", available, ids=ids)


def pytest_configure(config):
    """Register custom markers for provider-specific tests."""
    config.addinivalue_line(
        "markers", "openai: mark test as requiring OpenAI provider"
    )
    config.addinivalue_line(
        "markers", "anthropic: mark test as requiring Anthropic provider"
    )
    config.addinivalue_line(
        "markers", "google: mark test as requiring Google provider"
    )


# Convenience function to skip if specific provider not available
def skip_if_provider_unavailable(provider_name: str):
    """
    Skip test if specific provider is not available.

    Args:
        provider_name: One of "openai", "anthropic", "google"
    """
    config_map = {
        "openai": ("OPENAI_API_KEY", "openai"),
        "anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
        "google": ("GOOGLE_API_KEY", "google.genai"),
    }

    if provider_name not in config_map:
        pytest.skip(f"Unknown provider: {provider_name}")

    env_var, package = config_map[provider_name]

    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set")

    try:
        __import__(package)
    except ImportError:
        pytest.skip(f"{package} package not installed")
