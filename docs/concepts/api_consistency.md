# API Consistency and Migration Guide

This guide explains the new standardized API for creating Instructor instances across all providers and how to migrate from provider-specific parameters.

## Overview

Starting with version 1.7.0, Instructor provides a consistent API across all providers while maintaining full backward compatibility. This means:

- Standardized parameter names across all providers
- Deprecation warnings for legacy parameters
- Consistent error messages
- Unified capability system

## Key Changes

### 1. Async Mode Standardization

**Before (Provider-specific):**
```python
# VertexAI and Bedrock used _async
instructor = from_vertexai(client, _async=True)

# Gemini and Mistral used use_async  
instructor = from_gemini(client, use_async=True)

# OpenAI and others required different client types
from openai import AsyncOpenAI
client = AsyncOpenAI()
instructor = from_openai(client)
```

**After (Standardized):**
```python
# All providers can now use async_mode
instructor = create_instructor("vertexai", client, async_mode=True)
instructor = create_instructor("gemini", client, async_mode="async")
instructor = create_instructor("openai", async_client, async_mode=AsyncMode.ASYNC)

# Legacy parameters still work but show deprecation warnings
instructor = from_vertexai(client, _async=True)  # Works but deprecated
```

### 2. Parameter Name Consistency

**Before:**
```python
# Different retry parameter names
instructor1 = from_openai(client, max_retries=3)
instructor2 = from_anthropic(client, retries=3)  # Inconsistent

# Different model specification
instructor3 = from_openai(client, model="gpt-4")
instructor4 = from_gemini(client, model_name="gemini-pro")  # Inconsistent
```

**After:**
```python
# Consistent parameter names
instructor1 = create_instructor("openai", client, max_retries=3)
instructor2 = create_instructor("anthropic", client, max_retries=3)

# Standardized model specification
instructor3 = create_instructor("openai", client, model="gpt-4")
instructor4 = create_instructor("gemini", client, model="gemini-pro")
```

### 3. Error Message Standardization

**Before:**
```python
# Inconsistent error messages
# OpenAI: "Client must be an OpenAI client"
# Anthropic: "Client must be an instance of {anthropic.Anthropic, ...}"
# Gemini: Different format entirely
```

**After:**
```python
# Standardized error messages
# All providers: "Invalid client type for {provider} provider: {details}"
```

## Migration Examples

### Example 1: Migrating VertexAI Code

**Old Code:**
```python
from vertexai.generative_models import GenerativeModel
import instructor

model = GenerativeModel("gemini-pro")
client = instructor.from_vertexai(model, _async=True)
```

**New Code (Option 1 - Minimal Change):**
```python
# This still works but shows a deprecation warning
client = instructor.from_vertexai(model, _async=True)
```

**New Code (Option 2 - Recommended):**
```python
# Use the new standardized API
client = instructor.create_instructor("vertexai", model, async_mode=True)

# Or use the existing function with new parameter
client = instructor.from_vertexai(model, async_mode=True)
```

### Example 2: Migrating Anthropic Code

**Old Code:**
```python
import anthropic
import instructor

client = anthropic.Anthropic()
instructor_client = instructor.from_anthropic(
    client,
    mode=instructor.Mode.ANTHROPIC_TOOLS,
    enable_prompt_caching=True,
    retries=5
)
```

**New Code:**
```python
# Using standardized parameters
instructor_client = instructor.from_anthropic(
    client,
    mode=instructor.Mode.ANTHROPIC_TOOLS,
    prompt_caching=True,  # New parameter name
    max_retries=5  # Standardized retry parameter
)
```

### Example 3: Using the Builder Pattern

The new API also supports a builder pattern for more complex configurations:

```python
from instructor import ClientBuilder

# Build a client with consistent API
instructor_client = (
    ClientBuilder("openai")
    .with_client(openai_client)
    .with_mode("TOOLS")
    .with_async_mode(False)
    .with_kwargs(
        max_retries=3,
        temperature=0.7,
        max_tokens=2000
    )
    .build()
)
```

## Provider Capabilities

The new API includes a capability system to check what features each provider supports:

```python
from instructor.provider_utils import get_provider_capabilities

# Check what a provider supports
caps = get_provider_capabilities("anthropic")
print(f"Supports tools: {caps.supports_tools}")
print(f"Supports streaming: {caps.supports_streaming}")
print(f"Requires max_tokens: {caps.requires_max_tokens}")
print(f"Default mode: {caps.default_mode}")
```

## Deprecation Timeline

- **Version 1.7.0**: Deprecation warnings introduced for legacy parameters
- **Version 2.0.0**: Legacy parameters will be removed

## Benefits of Migration

1. **Consistency**: Same parameter names across all providers
2. **Discoverability**: Easier to switch between providers
3. **Type Safety**: Better type hints and validation
4. **Future-Proof**: New providers will follow the same pattern
5. **Better Errors**: Consistent, helpful error messages

## Backward Compatibility

All existing code will continue to work in version 1.x. You'll see deprecation warnings for legacy parameters, but functionality remains unchanged. This gives you time to migrate at your own pace.

## Need Help?

If you encounter issues during migration:

1. Check the deprecation warnings for guidance
2. Refer to provider-specific documentation
3. Open an issue on GitHub with the migration tag