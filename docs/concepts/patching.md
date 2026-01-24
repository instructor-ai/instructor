---
title: How Instructor Patches LLM Clients
description: Understand how Instructor enhances LLM client libraries with structured output capabilities through patching.
---

# Patching

Instructor enhances LLM client functionality by patching them with additional capabilities for structured outputs. This page explains how patching works conceptually. For practical usage, see [`from_provider`](./from_provider.md).

!!! tip "Recommended Approach"
    For most use cases, we recommend using [`from_provider`](./from_provider.md) instead of manual patching. It provides a simpler, unified interface that works across all providers. See the [Migration Guide](./migration.md) if you're using older patching patterns.

## What is Patching?

Patching is the process of adding new functionality to existing LLM client objects without modifying their original code. Instructor patches clients to add structured output capabilities:

- **Adds new parameters**: `response_model`, `max_retries`, and `context` to completion methods
- **Enables validation**: Automatically validates responses against Pydantic models
- **Provides retry logic**: Automatically retries when validation fails
- **Maintains compatibility**: The patched client still works with all original methods

## How Patching Works

When Instructor patches a client, it:

1. **Wraps the completion method**: Intercepts calls to `create()` or `chat.completions.create()`
2. **Converts schemas**: Transforms Pydantic models into provider-specific formats (JSON schema, tool definitions, etc.)
3. **Validates responses**: Checks LLM outputs against your Pydantic model
4. **Handles retries**: Automatically retries with validation feedback if needed
5. **Returns typed objects**: Converts validated JSON into Pydantic model instances

## Patching Modes

Different providers support different modes for structured extraction. Instructor automatically selects the best mode for each provider, but you can override it:

### Tool Calling (TOOLS)

Uses the provider's function/tool calling API. This is the default for OpenAI and provides the most reliable structured outputs.

**Supported by**: OpenAI, Anthropic (ANTHROPIC_TOOLS), Google (GENAI_TOOLS), Ollama (for supported models)

### JSON Mode

Instructs the model to return JSON directly. Works with most providers but may be less reliable than tool calling.

**Supported by**: OpenAI, Anthropic, Google, Ollama, and most providers

### Markdown JSON (MD_JSON)

Asks for JSON wrapped in markdown. Not recommended except for specific providers like Databricks.

**Supported by**: Databricks, some vision models

## Default Modes by Provider

Each provider uses a recommended default mode:

- **OpenAI**: `Mode.TOOLS` (function calling)
- **Anthropic**: `Mode.ANTHROPIC_TOOLS` (tool use)
- **Google**: `Mode.GENAI_TOOLS` (function calling)
- **Ollama**: `Mode.TOOLS` (if model supports it) or `Mode.JSON`
- **Others**: Provider-specific defaults

When using `from_provider`, these defaults are applied automatically. You can override them with the `mode` parameter.

## Manual Patching (Advanced)

If you need to patch a client manually (not recommended for most users):

```python
import openai
import instructor

# Create the base client
client = openai.OpenAI()

# Patch it manually
patched_client = instructor.patch(client, mode=instructor.Mode.TOOLS)

# Now use it
response = patched_client.chat.completions.create(
    response_model=YourModel,
    messages=[...]
)
```

However, using `from_provider` is simpler and recommended:

```python
import instructor

# Simpler approach
client = instructor.from_provider("openai/gpt-4o-mini")

response = client.create(
    response_model=YourModel,
    messages=[...]
)
```

## What Gets Patched?

Instructor adds these capabilities to patched clients:

### New Parameters

- **`response_model`**: A Pydantic model or type that defines the expected output structure
- **`max_retries`**: Number of retry attempts if validation fails (default: 0)
- **`context`**: Additional context for validation hooks

### Enhanced Methods

The patched client's `create()` method:
- Accepts `response_model` parameter
- Validates responses automatically
- Retries on validation failures
- Returns typed Pydantic objects instead of raw responses

## Provider-Specific Considerations

### OpenAI

- Default mode: `TOOLS` (function calling)
- Most reliable mode for structured outputs
- Supports streaming with structured outputs

### Anthropic

- Default mode: `ANTHROPIC_TOOLS` (tool use)
- Uses Claude's native tool calling API
- Excellent for complex structured outputs

### Google Gemini

- Default mode: `GENAI_TOOLS` (function calling)
- Requires `jsonref` package for tool calling
- Some limitations with strict validation and enums

### Ollama (Local Models)

- Default mode: `TOOLS` (if model supports it) or `JSON`
- Models like llama3.1, llama3.2, mistral-nemo support tools
- Older models fall back to JSON mode

## When to Use Manual Patching

Manual patching is rarely needed. Use it only if:

1. You need fine-grained control over the patching process
2. You're working with a custom client implementation
3. You're debugging patching behavior

For 99% of use cases, `from_provider` is the better choice.

## Related Documentation

- [from_provider Guide](./from_provider.md) - Recommended way to create patched clients
- [Migration Guide](./migration.md) - Migrating from manual patching to from_provider
- [Modes Comparison](../modes-comparison.md) - Detailed comparison of different modes
- [Integrations](../integrations/index.md) - Provider-specific documentation
