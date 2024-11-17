import os
from llama_cpp import Llama
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "llama-2-7b-chat.Q4_K_M.gguf")

def test_basic_completion():
    """Test basic completion without instructor integration"""
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

        logger.info("Model initialized, starting completion...")
        start_time = time.time()

        # Simple completion test
        prompt = "Extract the name and age from this text: Jason is 25 years old"

        response = llm.create_completion(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.1,
            repeat_penalty=1.1,
            stop=["</s>"]
        )

        duration = time.time() - start_time
        logger.info(f"Completion finished in {duration:.2f} seconds")
        logger.info(f"Response: {response}")

        return response

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_basic_completion()
