import pytest
import instructor
from pydantic import BaseModel
from llama_cpp import Llama
from test_clients import LlamaWrapper
import logging
import os
import time

logging.basicConfig(level=logging.DEBUG)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "llama-2-7b-chat.Q4_K_M.gguf")

class User(BaseModel):
    name: str
    age: int

def test_llama_cpp_basic():
    """Test basic functionality with llama-cpp-python"""
    try:
        # Create wrapper with model path and smaller context
        wrapped_llm = LlamaWrapper(
            MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=256,  # Keep small context
            n_batch=32,  # Match GGML_KQ_MASK_PAD requirement
            verbose=True,
            seed=42  # Add deterministic seed
        )

        # Enable instructor patches with our custom patch method
        client = instructor.patch(wrapped_llm)

        # Add timeout for inference
        start_time = time.time()
        timeout = 60  # Increased timeout

        response = None
        while time.time() - start_time < timeout:
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
                    response_model=User,
                    max_tokens=200,  # Increased max tokens
                    temperature=0.1,  # Keep low temperature
                    top_p=0.1,  # Add top_p for more focused sampling
                    repeat_penalty=1.1  # Add repeat penalty
                )
                break
            except Exception as e:
                logging.error(f"Attempt failed: {str(e)}")
                time.sleep(1)

        if response is None:
            pytest.fail("Model inference timed out")

        assert isinstance(response, User)
        assert response.name == "Jason"
        assert response.age == 25
    except Exception as e:
        pytest.fail(f"llama-cpp-python test failed: {str(e)}")

def test_llama_cpp_streaming():
    """Test streaming functionality with llama-cpp-python"""
    try:
        # Create wrapper with model path and smaller context
        wrapped_llm = LlamaWrapper(
            MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=256,
            n_batch=32,
            verbose=True,
            seed=42
        )

        # Enable instructor patches
        client = instructor.patch(wrapped_llm)

        start_time = time.time()
        timeout = 60

        responses = []
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
            response_model=User,
            max_tokens=200,
            temperature=0.1,
            top_p=0.1,
            repeat_penalty=1.1,
            stream=True
        )

        for response in stream:
            if time.time() - start_time > timeout:
                pytest.fail("Streaming timed out")
            responses.append(response)
            logging.debug(f"Received streaming response: {response}")

        assert len(responses) > 0
        final_responses = [r for r in responses if isinstance(r, User)]
        assert len(final_responses) >= 1
        assert any(u.name == "Jason" and u.age == 25 for u in final_responses)
    except Exception as e:
        pytest.fail(f"llama-cpp-python streaming test failed: {str(e)}")

def test_llama_cpp_nested():
    """Test nested object handling with llama-cpp-python"""
    from typing import List

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class UserWithAddresses(BaseModel):
        name: str
        age: int
        addresses: List[Address]

    try:
        # Create wrapper with model path and smaller context
        wrapped_llm = LlamaWrapper(
            MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=256,
            n_batch=32,
            verbose=True,
            seed=42
        )

        # Enable instructor patches
        client = instructor.patch(wrapped_llm)

        start_time = time.time()
        timeout = 60

        response = None
        while time.time() - start_time < timeout:
            try:
                response = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": """
                            Extract: Jason is 25 years old.
                            He lives at 123 Main St, New York, USA
                            and has a summer house at 456 Beach Rd, Miami, USA
                        """
                    }],
                    response_model=UserWithAddresses,
                    max_tokens=200,
                    temperature=0.1,
                    top_p=0.1,
                    repeat_penalty=1.1
                )
                break
            except Exception as e:
                logging.error(f"Attempt failed: {str(e)}")
                time.sleep(1)

        if response is None:
            pytest.fail("Model inference timed out")

        assert isinstance(response, UserWithAddresses)
        assert response.name == "Jason"
        assert response.age == 25
        assert len(response.addresses) == 2
        assert response.addresses[0].city == "New York"
        assert response.addresses[1].city == "Miami"
    except Exception as e:
        pytest.fail(f"llama-cpp-python nested object test failed: {str(e)}")

def test_llama_cpp_iterable():
    """Test iterable response handling with llama-cpp-python"""
    try:
        # Create wrapper with model path and smaller context
        wrapped_llm = LlamaWrapper(
            MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=256,
            n_batch=32,
            verbose=True,
            seed=42
        )

        # Enable instructor patches
        client = instructor.patch(wrapped_llm)

        start_time = time.time()
        timeout = 60

        responses = []
        stream = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": """
                    Extract users:
                    1. Jason is 25 years old
                    2. Sarah is 30 years old
                    3. Mike is 28 years old
                """
            }],
            response_model=User,
            max_tokens=200,
            temperature=0.1,
            top_p=0.1,
            repeat_penalty=1.1,
            stream=True
        )

        for response in stream:
            if time.time() - start_time > timeout:
                pytest.fail("Streaming timed out")
            responses.append(response)
            logging.debug(f"Received streaming response: {response}")

        assert len(responses) > 0
        final_responses = [r for r in responses if isinstance(r, User)]
        assert len(final_responses) >= 1
        assert any(u.name == "Jason" and u.age == 25 for u in final_responses)
    except Exception as e:
        pytest.fail(f"llama-cpp-python iterable test failed: {str(e)}")
