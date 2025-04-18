---
draft: false
date: 2025-04-15
authors:
  - jxnl
categories:
  - Tutorial
---

# Unified Provider Interface with String-Based Initialization

Instructor now offers a simplified way to initialize any supported LLM provider with a single consistent interface. This approach makes it easier than ever to switch between different LLM providers while maintaining the same structured output functionality you rely on.

## The Problem

As the number of LLM providers grows, so does the complexity of initializing and working with different client libraries. Each provider has its own initialization patterns, API structures, and quirks. This leads to code that isn't portable between providers and requires significant refactoring when you want to try a new model.

## The Solution: String-Based Initialization

We've introduced a new unified interface that allows you to initialize any supported provider with a simple string format:

```python
import instructor
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Initialize any provider with a single consistent interface
client = instructor.from_provider("openai/gpt-4")
client = instructor.from_provider("anthropic/claude-3-sonnet")
client = instructor.from_provider("google/gemini-pro")
client = instructor.from_provider("mistral/mistral-large")
```

The `from_provider` function takes a string in the format `"provider/model-name"` and handles all the details of setting up the appropriate client with the right model. This provides several key benefits:

- **Simplified Initialization**: No need to manually create provider-specific clients
- **Consistent Interface**: Same syntax works across all providers
- **Reduced Dependency Exposure**: You don't need to import specific provider libraries in your application code
- **Easy Experimentation**: Switch between providers with a single line change

## Supported Providers

The string-based initialization currently supports all major providers in the ecosystem:

- OpenAI: `"openai/model-name"`
- Anthropic: `"anthropic/model-name"`
- Google Gemini: `"google/model-name"`
- Mistral: `"mistral/model-name"`
- Cohere: `"cohere/model-name"`
- Perplexity: `"perplexity/model-name"`
- Groq: `"groq/model-name"`
- Writer: `"writer/model-name"`
- AWS Bedrock: `"bedrock/model-name"`
- Cerebras: `"cerebras/model-name"`
- Fireworks: `"fireworks/model-name"`
- Vertex AI: `"vertexai/model-name"`
- Google GenAI: `"genai/model-name"`

Each provider will be initialized with sensible defaults, but you can also pass additional keyword arguments to customize the configuration.

## Async Support

The unified interface fully supports both synchronous and asynchronous clients:

```python
# Synchronous client (default)
client = instructor.from_provider("openai/gpt-4")

# Asynchronous client
async_client = instructor.from_provider("anthropic/claude-3-sonnet", async_client=True)

# Use like any other async client
response = await async_client.chat.completions.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Extract information about John who is 30 years old"}]
)
```

## Mode Selection

You can also specify which structured output mode to use with the provider:

```python
import instructor
from instructor import Mode

# Override the default mode for a provider
client = instructor.from_provider(
    "anthropic/claude-3-sonnet", 
    mode=Mode.ANTHROPIC_TOOLS
)

# Use JSON mode instead of the default tools mode
client = instructor.from_provider(
    "mistral/mistral-large", 
    mode=Mode.MISTRAL_STRUCTURED_OUTPUTS
)

# Use reasoning tools instead of regular tools for Anthropic
client = instructor.from_provider(
    "anthropic/claude-3-opus", 
    mode=Mode.ANTHROPIC_REASONING_TOOLS
)
```

If not specified, each provider will use its recommended default mode:

- OpenAI: `Mode.OPENAI_FUNCTIONS`
- Anthropic: `Mode.ANTHROPIC_TOOLS`
- Google Gemini: `Mode.GEMINI_JSON`
- Mistral: `Mode.MISTRAL_TOOLS`
- Cohere: `Mode.COHERE_TOOLS`
- Perplexity: `Mode.JSON`
- Groq: `Mode.GROQ_TOOLS`
- Writer: `Mode.WRITER_JSON`
- Bedrock: `Mode.ANTHROPIC_TOOLS` (for Claude on Bedrock)
- Vertex AI: `Mode.VERTEXAI_TOOLS`

You can always customize this based on your specific needs and model capabilities.

## Error Handling

The `from_provider` function includes robust error handling to help you quickly identify and fix issues:

```python
# Missing dependency
try:
    client = instructor.from_provider("anthropic/claude-3-sonnet")
except ImportError as e:
    print("Error: Install the anthropic package first")
    # pip install anthropic

# Invalid provider format
try:
    client = instructor.from_provider("invalid-format")
except ValueError as e:
    print(e)  # Model string must be in format "provider/model-name"

# Unsupported provider
try:
    client = instructor.from_provider("unknown/model")
except ValueError as e:
    print(e)  # Unsupported provider: unknown. Supported providers are: ...
```

The function validates the provider string format, checks if the provider is supported, and ensures the necessary packages are installed.

## Environment Variables

Like the native client libraries, `from_provider` respects environment variables set for each provider:

```python
# Set environment variables 
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key" 
os.environ["MISTRAL_API_KEY"] = "your-mistral-key"

# No need to pass API keys directly
client = instructor.from_provider("openai/gpt-4")
```

## Conclusion

String-based initialization is a significant step toward making Instructor even more user-friendly and flexible. It reduces the learning curve for working with multiple providers and makes it easier than ever to experiment with different models.

Benefits include:
- Simplified initialization with a consistent interface
- Automatic selection of appropriate default modes
- Support for both synchronous and asynchronous clients
- Clear error messages when issues arise
- Respect for provider-specific environment variables

Whether you're building a new application or migrating an existing one, the unified provider interface offers a cleaner, more maintainable way to work with structured outputs across the LLM ecosystem.

Try it today with `instructor.from_provider()` and let us know what you think!