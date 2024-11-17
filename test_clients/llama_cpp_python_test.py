import pytest
from .llama_wrapper import LlamaWrapper, CompletionResponse, StreamingResponse
from typing import Generator, Dict, Any

def test_llama_completion():
    """Test basic completion functionality"""
    llama = LlamaWrapper(model_path="/home/ubuntu/instructor/models/llama-2-7b-chat.gguf")

    # Test synchronous completion
    response = llama.create(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        stream=False
    )
    assert isinstance(response, CompletionResponse)
    assert isinstance(response.choices[0].delta.get("content", ""), str)

def test_llama_streaming():
    """Test streaming functionality"""
    llama = LlamaWrapper(model_path="/home/ubuntu/instructor/models/llama-2-7b-chat.gguf")

    # Test streaming completion
    stream = llama.create(
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True
    )
    assert isinstance(stream, Generator)

    responses = list(stream)
    assert len(responses) > 0
    assert all(isinstance(r, StreamingResponse) for r in responses)
    assert all(isinstance(r.choices[0].delta.get("content", ""), str) for r in responses)

if __name__ == "__main__":
    pytest.main([__file__])
