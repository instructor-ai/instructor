# Provider System

Instructor supports multiple Large Language Model (LLM) providers through a standardized provider interface. This documentation explains how the provider system works and how to implement new providers.

## Provider Interface

All providers in Instructor implement a common interface that defines the expected behavior and capabilities. This interface is defined by the `ProviderProtocol` in `instructor.providers.base`.

```python
from instructor.providers.base import ProviderProtocol
```

The protocol requires the following properties and methods:

- `supported_modes`: List of `Mode` values supported by the provider
- `capabilities()`: Dictionary of provider capabilities (streaming, function_calling, etc.)
- `create_client()`: Factory method to create an Instructor client for this provider
- `create()`: Method to create structured output from messages
- `create_stream()`: Method to stream structured output from messages

## Base Provider Implementation

A base implementation of the provider interface is available in `ProviderBase`, which handles common functionality like mode validation:

```python
from instructor.providers.base import ProviderBase
```

## Using Provider Registry

Providers register themselves with the registry using the `register_provider` decorator:

```python
from instructor.providers import register_provider

@register_provider("openai")
class OpenAIProvider(ProviderBase):
    # Implementation...
```

You can list all available providers and get a specific provider:

```python
from instructor.providers import list_providers, get_provider

# List all providers
providers = list_providers()
print(providers)  # ['openai', 'anthropic', ...]

# Get a specific provider
openai_provider = get_provider("openai")
```

## Creating a New Provider

To create a new provider, implement the `ProviderBase` class and register it:

```python
from typing import List, Dict, Any, Type, Union, Iterator, Optional
from pydantic import BaseModel

from instructor.mode import Mode
from instructor.client import Instructor, AsyncInstructor
from instructor.providers import register_provider
from instructor.providers.base import ProviderBase


@register_provider("my_provider")
class MyProvider(ProviderBase):
    """My custom LLM provider."""
    
    _supported_modes = [Mode.JSON]
    
    @classmethod
    def create_client(
        cls, 
        client: Any, 
        provider_id: Optional[str] = None,
        **kwargs
    ) -> Union[Instructor, AsyncInstructor]:
        # Implementation...
        
    def create(
        self,
        response_model: Type[BaseModel], 
        messages: List[Dict[str, Any]], 
        mode: Mode,
        **kwargs
    ) -> BaseModel:
        # Implementation...
    
    def create_stream(
        self,
        response_model: Type[BaseModel], 
        messages: List[Dict[str, Any]],
        mode: Mode,
        **kwargs
    ) -> Iterator[BaseModel]:
        # Implementation...
```

## Provider Capabilities

The `capabilities()` method returns a dictionary of provider capabilities:

```python
{
    "streaming": True,           # Supports streaming responses
    "function_calling": True,    # Supports function calling
    "async": True,               # Supports async operations
    "multimodal": False          # Supports multimodal inputs
}
```

## Mode Support

Each provider indicates which modes it supports through the `supported_modes` property. Common modes include:

- `Mode.JSON`: Provider uses JSON mode for output format
- `Mode.TOOLS`: Provider uses tools/functions for output format
- `Mode.MARKDOWN`: Provider uses markdown for output format

Providers validate that requested modes are supported:

```python
def validate_mode(self, mode: Mode) -> None:
    if mode not in self.supported_modes:
        supported_str = ", ".join(str(m) for m in self.supported_modes)
        raise ValueError(
            f"Mode {mode} not supported by this provider. "
            f"Supported modes: {supported_str}"
        )
```

## Provider-Specific Arguments

Providers can accept additional arguments through `**kwargs` in the `create_client()`, `create()`, and `create_stream()` methods.