# Providers Directory Structure

This directory contains implementations for all supported LLM providers in the instructor library.

## Provider Organization

Each provider is organized in its own subdirectory with the following structure:

```
providers/
├── provider_name/
│   ├── __init__.py
│   ├── client.py      # Provider-specific client factory (optional)
│   └── utils.py       # Provider-specific utilities (optional)
```

## File Structure Patterns

### Providers with both `client.py` and `utils.py`
- **anthropic**, **bedrock**, **cerebras**, **cohere**, **fireworks**, **gemini**, **mistral**, **perplexity**, **writer**, **xai**
- These providers require custom response handling logic and utility functions
- `client.py`: Contains the `from_<provider>()` factory function
- `utils.py`: Contains provider-specific response handlers, reask functions, and message formatting

### Providers with only `client.py`
- **genai**, **groq**, **vertexai**
- These are simpler providers that use standard response handling from the core
- They don't require custom utility functions

### Special Case: OpenAI (only `utils.py`)
- OpenAI doesn't have a `client.py` because `from_openai()` is defined in `core/client.py`
- This is because OpenAI is the reference implementation that other providers are based on
- OpenAI utilities are still needed by the core processing logic for standard handling

## Adding a New Provider

When adding a new provider:

1. Create a new subdirectory under `providers/`
2. Add an `__init__.py` file (can be minimal)
3. Create `client.py` with a `from_<provider>()` function if needed
4. Create `utils.py` only if you need custom:
   - Response handlers (e.g., `handle_<provider>_json()`)
   - Reask functions (e.g., `reask_<provider>_tools()`)
   - Message formatting (e.g., `convert_to_<provider>_messages()`)
5. Update `providers/__init__.py` to conditionally import your provider
6. Update the main `instructor/__init__.py` to export the factory function

## Import Structure

- Provider modules use relative imports with `...` to access parent modules
- Example: `from ...core.exceptions import ProviderError`
- This maintains clean separation between provider implementations and core functionality