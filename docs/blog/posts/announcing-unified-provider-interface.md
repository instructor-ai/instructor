---
title: "Instructor's Unified Provider Interface"
date: 2025-05-08
authors:
  - jxnl
description: "Learn about Instructor's new `from_provider` function, a unified interface that simplifies working with multiple LLM providers like OpenAI, Anthropic, Google, and more. Easily switch models, compare outputs, and streamline your multi-provider workflows."
categories:
  - new-features
  - integrations
---

We are pleased to introduce a significant enhancement to Instructor: the **`from_provider()`** function. While Instructor has always focused on providing robust structured outputs, we've observed that many users work with multiple LLM providers. This often involves repetitive setup for each client. 

The `from_provider()` function aims to simplify this process, making it easier to initialize clients and experiment across different models.

This new feature offers a streamlined, string-based method to initialize an Instructor-enhanced client for a variety of popular LLM providers.

## What is `from_provider()`?

The `from_provider()` function serves as a smart factory for creating LLM clients. By providing a model string identifier, such as `"openai/gpt-4o"` or `"anthropic/claude-3-opus-20240229"`, the function handles the necessary setup:

*   **Automatic SDK Detection**: It identifies the targeted provider (e.g., OpenAI, Anthropic, Google, Mistral, Cohere).
*   **Client Initialization**: It dynamically imports the required provider-specific SDK and initializes the native client (like `openai.OpenAI()` or `anthropic.Anthropic()`).
*   **Instructor Patching**: It automatically applies the Instructor patch to the client, enabling structured outputs, validation, and retry mechanisms.
*   **Sensible Defaults**: It uses recommended `instructor.Mode` settings for each provider, optimized for performance and capabilities such as tool use or JSON mode, where applicable.
*   **Sync and Async Support**: Users can obtain either a synchronous or an asynchronous client by setting the `async_client=True` flag.

## Key Benefits

The `from_provider()` function is designed to streamline several common workflows:

*   **Model Comparison**: Facilitates quick switching between different models or providers to evaluate performance, cost, or output quality for specific tasks.
*   **Multi-Provider Strategies**: Simplifies the implementation of fallback mechanisms or routing queries to different LLMs based on criteria like complexity or cost, reducing client management overhead.
*   **Rapid Prototyping**: Allows for faster setup when starting with a new provider or model.
*   **Simplified Configuration**: Reduces boilerplate code in projects that integrate with multiple LLM providers.

## How it Works: A Look Under the Hood

Internally, `from_provider()` (located in `instructor/auto_client.py`) parses the model string (e.g., `"openai/gpt-4o-mini"`) to identify the provider and model name. It then uses conditional logic to import the correct libraries, instantiate the client, and apply the appropriate Instructor patch. For instance, the conceptual handling for an OpenAI client would involve importing the `openai` SDK and `instructor.from_openai`.

```python
# Conceptual illustration of internal logic for OpenAI:
# (Actual implementation is in instructor/auto_client.py)

# if provider == "openai":
#     import openai
#     from instructor import from_openai, Mode
#
#     # 'async_client', 'model_name', 'kwargs' are determined by from_provider
#     native_client = openai.AsyncOpenAI() if async_client else openai.OpenAI()
#     
#     return from_openai(
#         native_client,
#         model=model_name,
#         mode=Mode.TOOLS,  # Default mode for OpenAI
#         **kwargs,
#     )
```

The function also manages dependencies by alerting users to install missing packages (e.g., via `uv pip install openai`) if they are not found.

## Example Usage

Here's a self-contained example demonstrating how `from_provider()` can be used to retrieve structured output from different models. Ensure your API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are configured as environment variables to run this code.

```python
import instructor
from pydantic import BaseModel
import os
import asyncio

# Ensure your API keys are set as environment variables
# e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

class UserDetail(BaseModel):
    name: str
    age: int
    country: str

async def extract_user_info(client_identifier: str, text: str) -> UserDetail:
    print(f"\n--- Testing {client_identifier} ---")
    try:
        # Initialize client using the provider string
        # The 'mode' will be set by instructor based on the provider
        client = instructor.from_provider(client_identifier, async_client=True)
        
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": f"Extract user details: {text}"}],
            response_model=UserDetail,
        )
        print(f"Provider: {client_identifier}, User: {response.name}, Age: {response.age}, Country: {response.country}")
        return response
    except Exception as e:
        print(f"Error with {client_identifier}: {e}")
        # Return a dummy object or raise error as per your error handling strategy
        return UserDetail(name="Error", age=0, country="Error")

async def main():
    sample_text = "John Doe is 30 years old and lives in the USA. Alice Smith is 25 and resides in Canada."
    
    models_to_test = []
    if os.getenv("OPENAI_API_KEY"):
        models_to_test.append("openai/gpt-4o-mini")
    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_test.append("anthropic/claude-3-haiku-20240307")
    # Example for Google (ensure GOOGLE_API_KEY is set if uncommented)
    # if os.getenv("GOOGLE_API_KEY"):
    #     models_to_test.append("google/gemini-1.5-flash-latest")

    if not models_to_test:
        print("No API keys found for testing. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.")
        return

    tasks = [extract_user_info(model_id, sample_text) for model_id in models_to_test]
    results = await asyncio.gather(*tasks)
    
    print("\n--- All Results ---")
    for res in results:
        if res.name != "Error":
            print(f"Successfully extracted: {res.model_dump_json()}")

if __name__ == "__main__":
    asyncio.run(main())
```

This example, drawing inspiration from `examples/automodel/run.py`, illustrates the ease of iterating through different providers using `from_provider`.

## Path Forward

The `from_provider()` function offers a convenient method for client initialization. Instructor remains a lightweight wrapper around your chosen LLM provider's client, and users always retain the flexibility to initialize and patch clients manually for more granular control or when using providers not yet covered by this utility.

This unified interface is intended to balance ease of use for common tasks with the underlying flexibility of Instructor, aiming to make multi-provider LLM development more accessible and efficient. However, there is still much to do to further streamline multi-provider workflows. Future efforts could focus on:

*   **Unified Prompt Caching API**: While Instructor supports prompt caching for providers like [Anthropic](../integrations/anthropic.md#caching) (see also our [blog post on Anthropic prompt caching](../blog/posts/anthropic-prompt-caching.md) and the general [Prompt Caching concepts](../concepts/prompt_caching.md)), a more standardized, cross-provider API for managing cache behavior could significantly simplify optimizing costs and latency.
*   **Unified Multimodal Object Handling**: Instructor already provides a robust way to work with [multimodal inputs like Images, Audio, and PDFs](../concepts/multimodal.md) across different providers. However, a higher-level unified API could further abstract provider-specific nuances for these types, making it even simpler to build applications that seamlessly switch between, for example, OpenAI's vision capabilities and Anthropic's, without changing how media objects are passed.

These are areas where `instructor` can continue to reduce friction for developers working in an increasingly diverse LLM ecosystem.

We encourage you to try `from_provider()` in your projects, particularly when experimenting with multiple LLMs. Feedback and suggestions for additional providers or features are always welcome. 