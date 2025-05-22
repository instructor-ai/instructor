# Provider Registry Migration Guide

This guide explains how to migrate from the old monolithic `process_response.py` to the new provider-based architecture.

## Overview

The new architecture introduces:
1. A `BaseProvider` abstract class that all providers must implement
2. A `ProviderRegistry` for automatic provider discovery
3. Provider-specific response processing and request preparation

## Benefits

- **Reduced coupling**: Each provider manages its own logic
- **Easier testing**: Test providers in isolation
- **Better extensibility**: Add new providers without modifying core code
- **Cleaner code**: No more 1000+ line files with massive switch statements

## Migration Steps

### 1. Create a Provider Class

```python
from instructor.providers import BaseProvider, ProviderRegistry
from instructor.mode import Mode

@ProviderRegistry.register("my_provider")
class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"
    
    def get_supported_modes(self) -> set[Mode]:
        return {Mode.MY_MODE_1, Mode.MY_MODE_2}
    
    def prepare_request(self, response_model, new_kwargs, mode):
        # Move mode-specific request preparation here
        if mode == Mode.MY_MODE_1:
            return self._handle_mode_1(response_model, new_kwargs)
        # ... etc
```

### 2. Move Handler Functions

Instead of standalone functions in `process_response.py`:

```python
# OLD: process_response.py
def handle_my_provider_mode(response_model, new_kwargs):
    # ... logic ...
    return response_model, new_kwargs
```

Move them to your provider class:

```python
# NEW: providers/my_provider.py
class MyProvider(BaseProvider):
    def _handle_mode_1(self, response_model, new_kwargs):
        # ... same logic ...
        return response_model, new_kwargs
```

### 3. Update Mode Detection

The registry automatically maps modes to providers:

```python
# Automatic provider detection
provider = ProviderRegistry.get_provider_for_mode(Mode.MY_MODE_1)

# Or explicit provider selection
provider = ProviderRegistry.get_provider("my_provider")
```

### 4. Response Processing

Move response processing logic from the monolithic switch statement to provider methods:

```python
class MyProvider(BaseProvider):
    def process_response(self, response, response_model, mode, **kwargs):
        # Provider-specific response processing
        if mode == Mode.MY_MODE_1:
            return self._process_mode_1_response(response, response_model)
        # ... etc
```

## Example: Migrating OpenAI Provider

### Before (in process_response.py):

```python
def handle_tools(response_model, new_kwargs):
    openai_schema = response_model.openai_schema()
    new_kwargs["tools"] = [{"type": "function", "function": openai_schema}]
    new_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": openai_schema["name"]},
    }
    return response_model, new_kwargs

# In handle_response_model():
if mode == Mode.TOOLS:
    response_model, new_kwargs = handle_tools(response_model, new_kwargs)
```

### After (in providers/openai_provider.py):

```python
@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):
    def get_supported_modes(self) -> set[Mode]:
        return {Mode.TOOLS, Mode.JSON, Mode.FUNCTIONS, ...}
    
    def prepare_request(self, response_model, new_kwargs, mode):
        if mode == Mode.TOOLS:
            return self._handle_tools(response_model, new_kwargs)
        # ... other modes
    
    def _handle_tools(self, response_model, new_kwargs):
        openai_schema = response_model.openai_schema()
        new_kwargs["tools"] = [{"type": "function", "function": openai_schema}]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": openai_schema["name"]},
        }
        return response_model, new_kwargs
```

## Testing

The new architecture makes testing much easier:

```python
def test_openai_provider_tools_mode():
    provider = OpenAIProvider()
    assert Mode.TOOLS in provider.get_supported_modes()
    
    # Test request preparation
    model, kwargs = provider.prepare_request(
        MyModel, {"messages": []}, Mode.TOOLS
    )
    assert "tools" in kwargs
    assert kwargs["tool_choice"]["type"] == "function"
```

## Backwards Compatibility

The refactored `process_response.py` maintains the same public API while delegating to providers internally. Existing code will continue to work without changes.

## Adding New Providers

To add a new provider:

1. Create a new file: `providers/new_provider.py`
2. Implement the `BaseProvider` interface
3. Register with `@ProviderRegistry.register("new_provider")`
4. The provider is automatically available!

No need to modify any existing files or update switch statements.