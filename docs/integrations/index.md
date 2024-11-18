# Structured Output Integrations

Welcome to the Instructor integrations guide. This section provides detailed information about using structured outputs with various AI model providers.

## Supported Providers

Instructor supports a wide range of AI model providers, each with their own capabilities and features:

### OpenAI-Compatible Models
- [OpenAI](./openai.md) - GPT-3.5, GPT-4, and other OpenAI models
- [Azure OpenAI](./azure.md) - Microsoft's Azure-hosted OpenAI models

### Open Source & Self-Hosted Models
- [Ollama](./ollama.md) - Run open-source models locally
- [llama-cpp-python](./llama-cpp-python.md) - Python bindings for llama.cpp
- [Together AI](./together.md) - Host and run open source models

### Cloud AI Providers
- [Anthropic](./anthropic.md) - Claude and Claude 2 models
- [Google](./google.md) - PaLM and Gemini models
- [Vertex AI](./vertex.md) - Google Cloud's AI platform
- [Cohere](./cohere.md) - Command and other Cohere models
- [Anyscale](./anyscale.md) - Hosted open source models
- [Groq](./groq.md) - High-performance inference platform
- [Mistral](./mistral.md) - Mistral's hosted models
- [Fireworks](./fireworks.md) - High-performance model inference
- [Cerebras](./cerebras.md) - AI accelerator platform

### Model Management
- [LiteLLM](./litellm.md) - Unified interface for multiple providers

## Features Support Matrix

Not all providers support all features. Here's a quick overview:

| Provider | Streaming | Function Calling | Vision | RAG Support |
|----------|-----------|------------------|---------|-------------|
| OpenAI | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | ✅ |
| Google | ✅ | ✅ | ✅ | ✅ |
| Vertex AI | ✅ | ✅ | ✅ | ✅ |
| Cohere | ❌ | ✅ | ❌ | ✅ |
| Ollama | ✅ | ✅ | ✅ | ✅ |
| llama-cpp | ✅ | ✅ | ❌ | ✅ |
| Together | ✅ | ✅ | ❌ | ✅ |
| Anyscale | ✅ | ✅ | ❌ | ✅ |
| Groq | ✅ | ✅ | ❌ | ✅ |
| Mistral | ✅ | ✅ | ❌ | ✅ |
| Fireworks | ⚠️ | ✅ | ❌ | ✅ |
| Cerebras | ❌ | ✅ | ❌ | ✅ |
| LiteLLM | ⚠️ | ✅ | ⚠️ | ✅ |

Legend:
- ✅ Full support
- ⚠️ Limited support (provider/model dependent)
- ❌ Not supported

## Getting Started

To get started with any provider:

1. Install the required dependencies
2. Set up your API credentials
3. Initialize the client with Instructor
4. Define your Pydantic models
5. Make API calls with structured outputs

For detailed instructions, click on any provider in the list above.

## Common Concepts

All integrations share some common concepts:

- [Data Validation](../concepts/validation.md)
- [Streaming Support](../concepts/partial.md)
- [Model Validation](../concepts/models.md)
- [Instructor Hooks](../concepts/hooks.md)

## Need Help?

If you need help with a specific integration:

1. Check the provider-specific documentation
2. Look at the [examples](../examples/index.md)
3. Check our [GitHub issues](https://github.com/jxnl/instructor/issues)
4. Join our [Discord community](https://discord.gg/CV8sPM5k5Y)
