---
title: Instructor Provider Integrations
description: Connect Instructor to a wide variety of LLM providers including OpenAI, Anthropic, Google, open-source models, and more.
---

# Provider Integrations

Instructor works with many different LLM providers, allowing you to use structured outputs with your preferred models.

<div class="grid cards" markdown>

- :material-openai: **OpenAI Compatible**

    Connect to OpenAI's powerful models or Azure-hosted versions

    [:octicons-arrow-right-16: OpenAI](./openai.md) · [:octicons-arrow-right-16: Azure](./azure.md)

- :material-cloud: **Cloud Providers**

    Use major commercial AI providers with enterprise features

    [:octicons-arrow-right-16: Anthropic](./anthropic.md) · [:octicons-arrow-right-16: Google](./google.md) · [:octicons-arrow-right-16: Vertex AI](./vertex.md) · [:octicons-arrow-right-16: Cohere](./cohere.md) · [:octicons-arrow-right-16: Mistral](./mistral.md) · [:octicons-arrow-right-16: DeepSeek](./deepseek.md)

- :material-server: **Fast Inference**

    High-performance inference platforms for production

    [:octicons-arrow-right-16: Groq](./groq.md) · [:octicons-arrow-right-16: Fireworks](./fireworks.md) · [:octicons-arrow-right-16: Cerebras](./cerebras.md) · [:octicons-arrow-right-16: Writer](./writer.md)

- :material-open-source-initiative: **Open Source**

    Run open-source models locally or in the cloud

    [:octicons-arrow-right-16: Ollama](./ollama.md) · [:octicons-arrow-right-16: llama-cpp-python](./llama-cpp-python.md) · [:octicons-arrow-right-16: Together AI](./together.md)
    
- :material-router-wireless: **Routing**

    Unified interfaces for multiple providers

    [:octicons-arrow-right-16: LiteLLM](./litellm.md)

</div>

## Common Features

All integrations support these core features:

| Feature | Description | Documentation |
|---------|-------------|---------------|
| **Model Patching** | Enhance provider clients with structured output capabilities | [Patching](../concepts/patching.md) |
| **Response Models** | Define expected response schema with Pydantic | [Models](../concepts/models.md) |
| **Validation** | Ensure responses match your schema definition | [Validation](../concepts/validation.md) |
| **Streaming** | Stream partial or iterative responses | [Partial](../concepts/partial.md), [Iterable](../concepts/iterable.md) |
| **Hooks** | Add callbacks for monitoring and debugging | [Hooks](../concepts/hooks.md) |

However, each provider has different capabilities and limitations. Refer to the specific provider documentation for details.

## Provider Modes

Providers support different methods for generating structured outputs:

| Mode | Description | Providers |
|------|-------------|-----------|
| `TOOLS` | Uses OpenAI-style tools/function calling | OpenAI, Anthropic, Mistral |
| `PARALLEL_TOOLS` | Multiple simultaneous tool calls | OpenAI |
| `JSON` | Direct JSON response generation | OpenAI, Gemini, Cohere |
| `MD_JSON` | JSON embedded in markdown | Most providers |

See the [Modes Comparison](../modes-comparison.md) guide for details.

## Getting Started

To use a provider with Instructor:

1. Install the required dependencies:
   ```bash
   pip install "instructor[provider]"  # e.g., instructor[anthropic]
   ```

2. Import the provider client and patch it with Instructor:
   ```python
   import instructor
   from provider_package import Client
   
   client = instructor.from_provider(Client())
   ```

3. Use the patched client with your Pydantic model:
   ```python
   response = client.chat.completions.create(
       response_model=YourModel,
       messages=[{"role": "user", "content": "Your prompt"}]
   )
   ```

For provider-specific setup and examples, visit each provider's documentation page.

## Need Help?

If you need assistance with a specific integration:

1. Check the provider-specific documentation
2. Browse the [examples](../examples/index.md) and [cookbooks](../examples/index.md)
3. Search existing [GitHub issues](https://github.com/jxnl/instructor/issues)
4. Join our [Discord community](https://discord.gg/bD9YE9JArw)
