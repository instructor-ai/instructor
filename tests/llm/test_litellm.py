import os
import pytest
import instructor
from unittest.mock import patch

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY environment variable not set",
        allow_module_level=True,
    )

try:
    from litellm import acompletion, completion
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("litellm package is not installed", allow_module_level=True)


def test_litellm_create():
    client = instructor.from_litellm(completion)

    assert isinstance(client, instructor.Instructor)


def test_async_litellm_create():
    client = instructor.from_litellm(acompletion)

    assert isinstance(client, instructor.AsyncInstructor)


def test_litellm_hosted_vllm_with_parameters():
    """Test LiteLLM with hosted VLLM using direct parameters"""
    with patch.dict(os.environ, {}, clear=True):
        client = instructor.from_provider(
            "litellm/hosted_vllm/my-model",
            api_base="https://my-vllm-server.com",
            api_key="test-key"
        )
        
        assert isinstance(client, instructor.Instructor)
        assert os.environ.get("HOSTED_VLLM_API_BASE") == "https://my-vllm-server.com"
        assert os.environ.get("HOSTED_VLLM_API_KEY") == "test-key"


def test_litellm_hosted_vllm_with_base_url():
    """Test LiteLLM with hosted VLLM using base_url parameter"""
    with patch.dict(os.environ, {}, clear=True):
        client = instructor.from_provider(
            "litellm/hosted_vllm/my-model",
            base_url="https://my-vllm-server.com",
            api_key="test-key"
        )
        
        assert isinstance(client, instructor.Instructor)
        assert os.environ.get("HOSTED_VLLM_API_BASE") == "https://my-vllm-server.com"


def test_litellm_hosted_vllm_with_env_vars():
    """Test LiteLLM with hosted VLLM using environment variables"""
    with patch.dict(os.environ, {
        "HOSTED_VLLM_API_BASE": "https://env-vllm-server.com",
        "HOSTED_VLLM_API_KEY": "env-test-key"
    }):
        client = instructor.from_provider("litellm/hosted_vllm/my-model")
        
        assert isinstance(client, instructor.Instructor)


def test_async_litellm_hosted_vllm():
    """Test async LiteLLM with hosted VLLM"""
    with patch.dict(os.environ, {}, clear=True):
        client = instructor.from_provider(
            "litellm/hosted_vllm/my-model",
            async_client=True,
            api_base="https://my-vllm-server.com",
            api_key="test-key"
        )
        
        assert isinstance(client, instructor.AsyncInstructor)


def test_litellm_backward_compatibility():
    """Test that existing LiteLLM usage still works"""
    client = instructor.from_provider("litellm/gpt-3.5-turbo")
    assert isinstance(client, instructor.Instructor)
