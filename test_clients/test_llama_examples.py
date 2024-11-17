import logging
import time
import signal
from pathlib import Path
from typing import Generator, Any, TypeVar
from pydantic import BaseModel

from llama_cpp import Llama
from instructor import patch
from instructor.llama_wrapper import LlamaWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T', bound=BaseModel)
LlamaType = Llama  # Note: AsyncLlama is not supported in current version
ClientType = Any  # Type returned by patch()
ResponseType = dict[str, Any]

# Test timeout in seconds
TEST_TIMEOUT = 60

class TimeoutException(Exception):
    """Exception raised when a test times out."""
    pass

def timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for test timeouts."""
    raise TimeoutException("Test timed out")

# Test classes from documentation
class User(BaseModel):
    """User model for testing basic extraction."""
    name: str
    age: int

class Address(BaseModel):
    street: str
    city: str
    country: str

class UserWithAddresses(BaseModel):
    name: str
    age: int
    addresses: list[Address]

def test_sync_example() -> None:
    """Test basic synchronous extraction."""
    start_time = time.time()

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TEST_TIMEOUT)

        # Initialize the model with larger context window
        llm: Llama = Llama(
            model_path=str(Path(__file__).parent.parent / "models" / "llama-2-7b-chat.Q4_K_M.gguf"),
            n_ctx=2048,
            n_batch=32,
            verbose=False
        )

        # Create wrapper and enable instructor patches
        wrapped_llm: LlamaWrapper = LlamaWrapper(llm)
        client: ClientType = patch(wrapped_llm)

        # Test extraction with simple prompt
        user: User = client.chat.create(
            messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
            response_model=User,
            max_tokens=100,
            temperature=0.1
        )

        logger.info(f"Sync example result: {user}")
        logger.info(f"Sync example took {time.time() - start_time:.2f} seconds")

        # Assert the extracted data is correct
        assert user.name == "Jason"
        assert user.age == 25

    except TimeoutException:
        logger.error("Sync example timed out")
        raise AssertionError("Test timed out")
    except Exception as e:
        logger.error(f"Sync example failed: {str(e)}")
        raise AssertionError(f"Test failed: {str(e)}")
    finally:
        signal.alarm(0)

def test_nested_example() -> None:
    """Test nested object extraction."""
    start_time = time.time()

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TEST_TIMEOUT)

        # Initialize the model
        llm: Llama = Llama(
            model_path=str(Path(__file__).parent.parent / "models" / "llama-2-7b-chat.Q4_K_M.gguf"),
            n_ctx=2048,
            n_batch=32
        )

        # Create wrapper and enable instructor patches
        wrapped_llm: LlamaWrapper = LlamaWrapper(llm)
        client: ClientType = patch(wrapped_llm)

        # Test nested extraction with shorter prompt
        user: UserWithAddresses = client.chat.create(
            messages=[{
                "role": "user",
                "content": "Extract: Jason is 25 years old and lives at 123 Main St, New York, USA"
            }],
            response_model=UserWithAddresses,
            max_tokens=200,
            temperature=0.1
        )

        logger.info(f"Nested example result: {user}")
        logger.info(f"Nested example took {time.time() - start_time:.2f} seconds")

        # Assert the extracted data is correct
        assert user.name == "Jason"
        assert user.age == 25
        assert len(user.addresses) > 0

    except TimeoutException:
        logger.error("Nested example timed out")
        raise AssertionError("Test timed out")
    except Exception as e:
        logger.error(f"Nested example failed: {str(e)}")
        raise AssertionError(f"Test failed: {str(e)}")
    finally:
        signal.alarm(0)

def test_streaming_example() -> None:
    """Test streaming functionality."""
    start_time = time.time()

    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TEST_TIMEOUT)

        # Initialize the model
        llm: Llama = Llama(
            model_path=str(Path(__file__).parent.parent / "models" / "llama-2-7b-chat.Q4_K_M.gguf"),
            n_ctx=2048,
            n_batch=32
        )

        # Create wrapper and enable instructor patches
        wrapped_llm: LlamaWrapper = LlamaWrapper(llm)
        client: ClientType = patch(wrapped_llm)

        # Test streaming with simple prompt
        stream: Generator[ResponseType, None, None] = client.chat.create(
            messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
            response_model=User,
            max_tokens=100,
            temperature=0.1,
            stream=True
        )

        for chunk in stream:
            logger.info(f"Streaming chunk: {chunk}")

        logger.info(f"Streaming example took {time.time() - start_time:.2f} seconds")

    except TimeoutException:
        logger.error("Streaming example timed out")
        raise AssertionError("Test timed out")
    except Exception as e:
        logger.error(f"Streaming example failed: {str(e)}")
        raise AssertionError(f"Test failed: {str(e)}")
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    # Run tests
    logger.info("Testing sync example...")
    test_sync_example()

    logger.info("Testing nested example...")
    test_nested_example()

    logger.info("Testing streaming example...")
    test_streaming_example()

    # Print results
    logger.info("\nTest Results Summary:")
    logger.info("All tests completed successfully.")
