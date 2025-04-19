---
title: Instructor Provider Integrations
description: Connect Instructor to a wide variety of LLM providers including OpenAI, Anthropic, Google, open-source models, and more.
---

# Provider Integrations

Instructor works with many different LLM providers, allowing you to use structured outputs with your preferred models.

<div class="grid cards" markdown>

- :material-cloud: **Major Cloud Providers**

    Leading AI providers with comprehensive features

    [:octicons-arrow-right-16: OpenAI](./openai.md)          · 
    [:octicons-arrow-right-16: Azure](./azure.md)            · 
    [:octicons-arrow-right-16: Anthropic](./anthropic.md)    · 
    [:octicons-arrow-right-16: Google.GenerativeAI](./google.md)          · 
    [:octicons-arrow-right-16: Vertex AI](./vertex.md)       · 
    [:octicons-arrow-right-16: AWS Bedrock](./bedrock.md)    · 
    [:octicons-arrow-right-16: Google.GenAI](./genai.md)

- :material-cloud-outline: **Additional Cloud Providers**

    Other commercial AI providers with specialized offerings

    [:octicons-arrow-right-16: Cohere](./cohere.md)          · 
    [:octicons-arrow-right-16: Mistral](./mistral.md)        · 
    [:octicons-arrow-right-16: DeepSeek](./deepseek.md)      · 
    [:octicons-arrow-right-16: Together AI](./together.md)    · 
    [:octicons-arrow-right-16: Groq](./groq.md)              · 
    [:octicons-arrow-right-16: Fireworks](./fireworks.md)    · 
    [:octicons-arrow-right-16: Cerebras](./cerebras.md)      · 
    [:octicons-arrow-right-16: Writer](./writer.md)          · 
    [:octicons-arrow-right-16: Perplexity](./perplexity.md)
    [:octicons-arrow-right-16: Sambanova](./sambanova.md)

- :material-open-source-initiative: **Open Source**

    Run open-source models locally or in the cloud

    [:octicons-arrow-right-16: Ollama](./ollama.md)                  · 
    [:octicons-arrow-right-16: llama-cpp-python](./llama-cpp-python.md)
    
- :material-router-wireless: **Routing**

    Unified interfaces for multiple providers

    [:octicons-arrow-right-16: LiteLLM](./litellm.md)
    [:octicons-arrow-right-16: OpenRouter](./openrouter.md)

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
| `JSON` | Direct JSON response generation | OpenAI, Gemini, Cohere, Perplexity |
| `MD_JSON` | JSON embedded in markdown | Most providers |
| `BEDROCK_TOOLS` | AWS Bedrock function calling | AWS Bedrock |
| `BEDROCK_JSON` | AWS Bedrock JSON generation | AWS Bedrock |
| `PERPLEXITY_JSON` | Perplexity JSON generation | Perplexity |

See the [Modes Comparison](../modes-comparison.md) guide for details.

## Getting Started

There are two ways to use providers with Instructor:

### 1. Using Provider Initialization (Recommended)

The simplest way to get started is using the provider initialization:

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Initialize any provider with a simple string
client = instructor.from_provider("openai/gpt-4")
# Or use async client
async_client = instructor.from_provider("anthropic/claude-3-sonnet", async_client=True)

# Use the same interface for all providers
response = client.chat.completions.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Your prompt"}]
)
```

Supported provider strings:
- `openai/model-name`: OpenAI models
- `anthropic/model-name`: Anthropic models
- `google/model-name`: Google models
- `mistral/model-name`: Mistral models
- `cohere/model-name`: Cohere models
- `perplexity/model-name`: Perplexity models
- `groq/model-name`: Groq models
- `writer/model-name`: Writer models
- `bedrock/model-name`: AWS Bedrock models
- `cerebras/model-name`: Cerebras models
- `fireworks/model-name`: Fireworks models
- `vertexai/model-name`: Vertex AI models
- `genai/model-name`: Google GenAI models

### 2. Manual Client Setup

Alternatively, you can manually set up the client:

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
