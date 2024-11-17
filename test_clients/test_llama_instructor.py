import os
from llama_cpp import Llama
import instructor
from pydantic import BaseModel
import logging
import time
from typing import Optional
from instructor.llama_wrapper import LlamaWrapper

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "llama-2-7b-chat.Q4_K_M.gguf")

class SimpleUser(BaseModel):
    """A simple user model for testing"""
    name: str
    age: Optional[int] = None

def test_instructor_basic():
    """Test basic instructor integration"""
    try:
        logger.info("Initializing Llama model...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=256,
            n_batch=32,
            verbose=True,
            seed=42
        )

        # Create wrapper and patch with instructor
        wrapped_llm = LlamaWrapper(llm)
        client = instructor.patch(wrapped_llm)

        logger.info("Model initialized and patched with instructor, starting completion...")
        start_time = time.time()

        # Simple extraction test
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
            response_model=SimpleUser,
            max_tokens=100,
            temperature=0.1,
            timeout=60  # 60 second timeout
        )

        duration = time.time() - start_time
        logger.info(f"Completion finished in {duration:.2f} seconds")
        logger.info(f"Response: {response}")

        # Test streaming
        logger.info("Testing streaming capability...")
        start_time = time.time()

        stream_response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
            response_model=SimpleUser,
            max_tokens=100,
            temperature=0.1,
            stream=True,
            timeout=60
        )

        # Try to get first chunk
        try:
            first_chunk = next(stream_response)
            logger.info(f"Streaming works! First chunk: {first_chunk}")

            # Try to get all chunks
            chunks = []
            for chunk in stream_response:
                chunks.append(chunk)
                logger.info(f"Got chunk: {chunk}")

            logger.info(f"Successfully received {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            logger.info("Streaming is not supported or failed")

        duration = time.time() - start_time
        logger.info(f"Streaming test finished in {duration:.2f} seconds")

        return {
            "basic_test": "success" if response else "failed",
            "streaming_test": "success" if chunks else "failed",
            "duration": duration
        }

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = test_instructor_basic()
    print("\nTest Results:")
    print(f"Basic Test: {results.get('basic_test', 'failed')}")
    print(f"Streaming Test: {results.get('streaming_test', 'failed')}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
